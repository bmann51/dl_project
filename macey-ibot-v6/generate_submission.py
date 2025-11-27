"""
Submission generation script for the *macey-ibot-v6* self-supervised model.

This is adapted from the simple DINO submission helper in *brian-folder*,
but modified to work with iBOT checkpoints and the architecture helpers in
`macey-ibot-v6/ibot_ssl.py`.

Example usage (local):
    python generate_submission.py \
        --checkpoint ./checkpoints/ibot_final.pth \
        --data_dir ./data \
        --output submission.csv \
        --arch vit_small \
        --resolution 96 \
        --k 5

Notes
-----
1. The script performs **feature extraction** using the iBOT backbone and
   a simple K-Nearest-Neighbours classifier (cosine distance, distance
   weights) on the extracted CLS-token embeddings.
2. Architecture, embedding dimension and other model-specific hyper-
   parameters are automatically inferred via `get_model_config` below.
   Only change them if you *really* know what you are doing.
3. The script is intentionally self-contained so that it can be executed
   on compute nodes where the full training code may not be available.
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# =============================================================================
#                               MODEL HELPERS
# =============================================================================

class iBOTTokenizer(nn.Module):
    """Lightweight online tokenizer used by iBOT."""

    def __init__(self, embed_dim: int, num_tokens: int = 8192, hidden_dim: int = 512):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.tokenizer(x)


class iBOTHead(nn.Module):
    """Projection head for iBOT (same as in training)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        if norm_last_layer:
            self.last_layer.weight.data = F.normalize(self.last_layer.weight.data, dim=1)
            self.last_layer.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class MultiCropiBOTWrapper(nn.Module):
    """Wrapper that returns the (normalized) CLS token during evaluation."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        feats = self.backbone(x)
        if feats.ndim == 3:  # [B, N+1, D]
            cls = feats[:, 0]
        elif feats.ndim == 2:  # [B, D]
            cls = feats
        else:
            raise ValueError(f"Unexpected feature shape: {feats.shape}")
        return F.normalize(cls, dim=-1, p=2)


# -----------------------------------------------------------------------------
# Architecture helper
# -----------------------------------------------------------------------------

def get_model_config(arch: str) -> Dict[str, Any]:
    cfg = {
        "vit_tiny": {"model_name": "vit_tiny_patch16_224", "embed_dim": 192},
        "vit_small": {"model_name": "vit_small_patch16_224", "embed_dim": 384},
        "vit_base": {"model_name": "vit_base_patch16_224", "embed_dim": 768},
    }
    if arch not in cfg:
        raise ValueError(f"Unknown architecture '{arch}'. Choices: {list(cfg)}")
    return cfg[arch]


def load_ibot_model(
    checkpoint_path: str,
    arch: str = "vit_small",
    img_size: int = 96,
    device: str = "cuda",
) -> Tuple[nn.Module, int]:
    """Load backbone and return model ready for feature extraction."""

    info = get_model_config(arch)
    print(f"Loading iBOT checkpoint from: {checkpoint_path}")
    print(f"Architecture: {arch} ({info['model_name']}) | Img {img_size}px")

    backbone = timm.create_model(
        info["model_name"],
        pretrained=False,
        num_classes=0,
        img_size=img_size,
        global_pool="",  # keep tokens
    )

    model = MultiCropiBOTWrapper(backbone)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("student", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Loaded with {len(missing)} missing keys (ignored)")
    if unexpected:
        print(f"Loaded with {len(unexpected)} unexpected keys (ignored)")

    model.to(device)
    model.eval()
    return model, info["embed_dim"]


# =============================================================================
#                               DATASET
# =============================================================================

class ImageDataset(Dataset):
    """Simple image dataset assuming a `folder/filename.jpg` structure."""

    def __init__(
        self,
        image_dir: Path,
        filenames: list,
        labels: list | None,
        resolution: int = 96,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.filenames = filenames
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.resolution = resolution

    def __len__(self) -> int:  # noqa: D401
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img = self.transform(img)
        if self.labels is None:
            return img
        return img, self.labels[idx]


# =============================================================================
#                       FEATURE EXTRACTION + KNN HELPERS
# =============================================================================

def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    split_name: str = "train",
):
    feats, labs = [], []
    print(f"Extracting {split_name} features ...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, lbls = batch
                imgs = imgs.to(device)
                feat = model(imgs)
                feats.append(feat.cpu().numpy())
                labs.extend(lbls.numpy())
            else:
                imgs = batch.to(device)
                feat = model(imgs)
                feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    labs = np.array(labs) if labs else None
    print(f"  -> {feats.shape[0]} samples, dim {feats.shape[1]}")
    return feats, labs


# =============================================================================
#                                   MAIN
# =============================================================================

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument(
        "--arch",
        type=str,
        default="vit_small",
        choices=["vit_tiny", "vit_small", "vit_base"],
    )
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train_labels.csv")
    val_df = pd.read_csv(data_dir / "val_labels.csv")
    test_df = pd.read_csv(data_dir / "test_images.csv")

    # Datasets & loaders
    train_ds = ImageDataset(data_dir / "train", train_df["filename"].tolist(), train_df["class_id"].tolist(), args.resolution)
    val_ds = ImageDataset(data_dir / "val", val_df["filename"].tolist(), val_df["class_id"].tolist(), args.resolution)
    test_ds = ImageDataset(data_dir / "test", test_df["filename"].tolist(), None, args.resolution)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model, _ = load_ibot_model(args.checkpoint, args.arch, args.resolution, device)

    # Extract embeddings
    train_feats, train_labs = extract_features(model, train_loader, device, "train")
    val_feats, val_labs = extract_features(model, val_loader, device, "val")
    test_feats, _ = extract_features(model, test_loader, device, "test")

    # KNN classifier
    print(f"Training KNN (k={args.k}) ...")
    knn = KNeighborsClassifier(n_neighbors=args.k, metric="cosine", weights="distance", n_jobs=-1)
    knn.fit(train_feats, train_labs)

    train_acc = knn.score(train_feats, train_labs)
    val_acc = knn.score(val_feats, val_labs)
    print(f"Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")

    preds = knn.predict(test_feats)
    submission = pd.DataFrame({"id": test_df["filename"], "class_id": preds})
    submission.to_csv(args.output, index=False)
    print(f"Saved submission to {args.output} | {len(submission)} rows")


if __name__ == "__main__":
    main()
