import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ibot_ssl import ViTBackbone  # reuse backbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, required=True, help="eval_public/train")
    p.add_argument("--test_dir", type=str, required=True, help="eval_public/test")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to ibot_epochXXX.pt")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def build_dataloader(root, batch_size, train=True):
    # same normalization as pretrain, but deterministic crop
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    dataset = datasets.ImageFolder(root=root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # feature bank should not depend on shuffle
        num_workers=8,
        pin_memory=True,
    )
    return loader, dataset


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load backbone (frozen encoder - no projection head for evaluation)
    backbone = ViTBackbone(img_size=96, patch_size=16, embed_dim=768)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone.load_state_dict(ckpt["student_backbone"])
    backbone.to(device)
    backbone.eval()

    # Data
    train_loader, train_dataset = build_dataloader(args.train_dir, args.batch_size, train=True)
    test_loader, test_dataset = build_dataloader(args.test_dir, args.batch_size, train=False)

    # Build feature bank on train (using frozen backbone CLS token)
    train_feats = []
    train_labels = []
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            cls, _ = backbone(imgs)  # [B, D]
            feats = torch.nn.functional.normalize(cls, dim=-1)  # [B, D], L2-normalized
            train_feats.append(feats)
            train_labels.append(labels.to(device))

    train_feats = torch.cat(train_feats, dim=0)   # [N_train, D]
    train_labels = torch.cat(train_labels, dim=0) # [N_train]

    # k-NN classification on test
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device)

            cls, _ = backbone(imgs)
            feats = torch.nn.functional.normalize(cls, dim=-1)  # [B, D], L2-normalized

            # cosine similarity
            sim = feats @ train_feats.t()  # [B, N_train]
            topk_sim, topk_idx = sim.topk(args.k, dim=-1)  # [B, k]
            topk_labels = train_labels[topk_idx]  # [B, k]

            # majority vote (weighted by similarity)
            B = imgs.size(0)
            preds = []
            for i in range(B):
                labels_i = topk_labels[i]  # [k]
                sims_i = topk_sim[i]      # [k]
                unique_labels = labels_i.unique()
                scores = []
                for c in unique_labels:
                    scores.append((sims_i[labels_i == c]).sum())
                scores = torch.stack(scores)
                pred = unique_labels[scores.argmax()]
                preds.append(pred)

            preds = torch.stack(preds)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100.0
    print(f"k-NN top-1 accuracy (k={args.k}): {acc:.2f}%")


if __name__ == "__main__":
    main()
