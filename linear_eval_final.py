#!/usr/bin/env python3
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, image_dir, filenames, labels=None, transform=None):
        self.image_dir = Path(image_dir)
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = self.image_dir / filename
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image, filename


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_transform(image_size=96):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def build_vit_model(arch='vit_small', image_size=96):
    import timm
    arch_mapping = {
        'vit_tiny': 'vit_tiny_patch16_224',
        'vit_small': 'vit_small_patch16_224', 
        'vit_base': 'vit_base_patch16_224',
    }
    model_name = arch_mapping.get(arch, arch)
    model = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=image_size)
    return model


def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ['student', 'teacher', 'model', 'state_dict']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in ['module.', 'backbone.', 'encoder.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    
    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        images, labels = batch[0], batch[1]
        images = images.to(device)
        
        features = model.forward_features(images) if hasattr(model, 'forward_features') else model(images)
        if len(features.shape) == 3:
            features = features[:, 0]
        features = F.normalize(features, dim=1, p=2)
        
        features_list.append(features.cpu())
        labels_list.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))
    
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def extract_features_test(model, dataloader, device):
    model.eval()
    features_list = []
    filenames = []
    
    for images, names in tqdm(dataloader, desc="Extracting test features", leave=False):
        images = images.to(device)
        features = model.forward_features(images) if hasattr(model, 'forward_features') else model(images)
        if len(features.shape) == 3:
            features = features[:, 0]
        features = F.normalize(features, dim=1, p=2)
        features_list.append(features.cpu())
        filenames.extend(names)
    
    return torch.cat(features_list, dim=0), filenames


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)


def train_dataset_specific(train_features, train_labels, val_features, val_labels,
                           num_classes, testset_name, device='cuda'):
    """
    Use dataset-specific winning hyperparameters from previous runs
    """
    print(f"Training on {len(train_features)} samples, validating on {len(val_features)} samples")
    
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    
    # Dataset-specific configs based on your logs
    if testset_name == 'testset_1':
        # Winner: lr=0.00075, try even more configs around this
        configs = [
            (0.00075, 0.0, 0.0, 20000),   # Your winner, run longer
            (0.0007, 0.0, 0.0, 20000),    # Slightly lower
            (0.0008, 0.0, 0.0, 20000),    # Was at 18.99% and still going
            (0.00065, 0.0, 0.0, 20000),   # Even lower
            (0.00075, 0.0, 0.02, 18000),  # Add tiny label smoothing
        ]
    elif testset_name == 'testset_2':
        # Winner: lr=0.001, label_smooth=0.05, still improving at epoch 12500
        configs = [
            (0.001, 0.0, 0.05, 20000),    # Your winner, run longer
            (0.001, 0.0, 0.06, 18000),    # Slightly more smoothing
            (0.001, 0.0, 0.04, 18000),    # Slightly less smoothing
            (0.0009, 0.0, 0.05, 18000),   # Slightly lower lr
            (0.0011, 0.0, 0.05, 18000),   # Slightly higher lr
        ]
    elif testset_name == 'testset_3':
        # Winner: lr=0.001, label_smooth=0.05, but plateaus early
        # Try configs that might help with the early plateau
        configs = [
            (0.001, 0.0, 0.05, 15000),    # Your winner
            (0.001, 0.0, 0.08, 15000),    # More smoothing to fight overfitting
            (0.0008, 0.0, 0.05, 15000),   # Lower lr
            (0.001, 1e-5, 0.05, 15000),   # Add weight decay
            (0.0012, 0.0, 0.03, 15000),   # Slightly higher lr, less smoothing
        ]
    else:
        # Fallback
        configs = [(0.001, 0.0, 0.0, 15000)]
    
    best_val_acc = 0
    best_model_state = None
    best_config = None
    
    for lr, wd, label_smooth, max_epochs in configs:
        print(f"\nTrying: lr={lr}, wd={wd}, label_smooth={label_smooth}, max_epochs={max_epochs}")
        
        classifier = LinearClassifier(train_features.shape[1], num_classes).to(device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr/1000)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        
        current_best_val = 0
        current_best_state = None
        no_improve = 0
        patience = 3000  # More patient
        
        for epoch in range(max_epochs):
            classifier.train()
            logits = classifier(train_features)
            loss = criterion(logits, train_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                classifier.eval()
                with torch.no_grad():
                    val_logits = classifier(val_features)
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == val_labels).float().mean().item() * 100
                    
                    if val_acc > current_best_val:
                        current_best_val = val_acc
                        current_best_state = classifier.state_dict().copy()
                        no_improve = 0
                    else:
                        no_improve += 10
                    
                    if (epoch + 1) % 500 == 0:
                        print(f"  Epoch {epoch+1}: Val={val_acc:.2f}%, Best={current_best_val:.2f}%")
            
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        print(f"  Final: {current_best_val:.2f}%")
        
        if current_best_val > best_val_acc:
            best_val_acc = current_best_val
            best_model_state = current_best_state
            best_config = (lr, wd, label_smooth)
    
    print(f"\n{'='*60}")
    print(f"Best config: lr={best_config[0]}, wd={best_config[1]}, label_smooth={best_config[2]}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")
    
    classifier = LinearClassifier(train_features.shape[1], num_classes).to(device)
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
    
    return classifier, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    data_dir = Path(f"/gpfs/scratch/bm3772/fall2025_finalproject/{args.testset}/data")
    
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"Dataset: {args.testset}")
    print(f"Train: {len(train_df)} images")
    print(f"Val: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    print(f"Classes: {train_df['class_id'].nunique()}")
    
    output_path = Path(args.output)
    output_dir = output_path.parent if output_path.parent != Path('.') else Path('.')
    output_name = output_path.stem
    output_ext = output_path.suffix if output_path.suffix else '.csv'
    final_output = output_dir / f"{output_name}_{args.testset}{output_ext}"
    final_output.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = load_config(args.config)
    arch = config.get('arch', 'vit_small')
    image_size = config.get('image_size', 96)
    
    print(f"\nLoading model: {arch} (image_size={image_size})")
    model = build_vit_model(arch=arch, image_size=image_size)
    model = load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    transform = get_transform(image_size)
    
    print("\n" + "="*60)
    print("EXTRACTING FEATURES")
    print("="*60)
    
    print("\nExtracting training features...")
    train_dataset = ImageDataset(data_dir / 'train', train_df['filename'].tolist(), 
                                 train_df['class_id'].tolist(), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    train_features, train_labels = extract_features(model, train_loader, device)
    num_classes = train_df['class_id'].nunique()
    
    print("Extracting validation features...")
    val_dataset = ImageDataset(data_dir / 'val', val_df['filename'].tolist(),
                               val_df['class_id'].tolist(), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    val_features, val_labels = extract_features(model, val_loader, device)
    
    print(f"Feature dimension: {train_features.shape[1]}")
    
    print("\n" + "="*60)
    print("TRAINING LINEAR PROBE")
    print("="*60)
    
    classifier, best_val_acc = train_dataset_specific(
        train_features, train_labels, val_features, val_labels,
        num_classes, args.testset, device=device
    )
    
    print("\n" + "="*60)
    print(f"BEST VALIDATION ACCURACY: {best_val_acc:.2f}%")
    print("="*60)
    
    print("\nGenerating test predictions...")
    test_dataset = ImageDataset(data_dir / 'test', test_df['filename'].tolist(),
                               labels=None, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    test_features, test_filenames = extract_features_test(model, test_loader, device)
    
    classifier.eval()
    test_features = test_features.to(device)
    with torch.no_grad():
        logits = classifier(test_features)
        predictions = logits.argmax(dim=1).cpu().numpy()
    
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions
    })
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    submission_df.to_csv(final_output, index=False)
    
    print(f"\nSubmission saved to: {final_output}")
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION ACCURACY: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()