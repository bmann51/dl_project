"""
Submission script for iBOT v9 trained models

Usage:
    python generate_submission_ibot.py \
        --checkpoint /gpfs/scratch/bm3772/checkpoints/v9/v9a_ibot_checkpoint/ibot_epoch400.pt \
        --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_2/data \
        --output submission_ibot_v9.csv \
        --arch vit_small \
        --resolution 96 \
        --k 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import argparse
import timm
from ibot_ssl import ViTBackbone


# ============================================================================
#                          SIMPLE DATASET
# ============================================================================

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_list, labels=None, resolution=96):
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_dir / self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image


# ============================================================================
#                          LOAD MODEL
# ============================================================================

def load_v9_model(checkpoint_path, arch='vit_small', img_size=96, device='cuda'):
    """Load v9 iBOT model from checkpoint (backbone only)"""
    
    print(f"Loading v9 iBOT checkpoint: {checkpoint_path}")
    print(f"Architecture: {arch}")
    
    # Create backbone
    if arch == 'vit_small':
        backbone = ViTBackbone(img_size=img_size, patch_size=16, embed_dim=384)
    else:
        raise ValueError(f"Unsupported architecture for v9: {arch}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'student_backbone' in checkpoint:
        backbone.load_state_dict(checkpoint['student_backbone'])
        print("  Loaded from 'student_backbone' key in checkpoint")
    else:
        raise ValueError("Checkpoint must contain 'student_backbone' key")
    
    backbone.to(device)
    backbone.eval()
    
    # Create wrapper that returns normalized CLS token
    class ModelWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        
        def forward(self, x):
            cls, _ = self.backbone(x)  # [B, D]
            return F.normalize(cls, dim=-1, p=2)  # L2-normalized CLS token
    
    model = ModelWrapper(backbone)
    print(f"  Model loaded successfully")
    
    return model


# ============================================================================
#                          EXTRACT FEATURES
# ============================================================================

def extract_features(model, dataloader, device, split_name='train'):
    """Extract features from dataset"""
    all_features = []
    all_labels = []
    
    print(f"\nExtracting {split_name} features...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # Handle both train/val (with labels) and test (without labels)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                images = images.to(device)
                features = model(images)
                
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
            else:
                images = batch.to(device)
                features = model(images)
                
                all_features.append(features.cpu().numpy())
    
    if len(all_features) == 0:
        raise ValueError(f"No features extracted for {split_name} split!")
    
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels) if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    return features, labels


# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to iBOT v9 checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory with train/val/test folders')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV file')
    parser.add_argument('--arch', type=str, default='vit_small',
                        choices=['vit_small'],
                        help='Model architecture')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Image resolution')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    data_dir = Path(args.data_dir)
    
    # Load CSV files
    print("Loading dataset...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    # Create datasets
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist(),
        args.resolution
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist(),
        args.resolution
    )
    
    test_dataset = ImageDataset(
        data_dir / 'test',
        test_df['filename'].tolist(),
        None,
        args.resolution
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    # Load model
    model = load_v9_model(
        args.checkpoint, args.arch, args.resolution, device
    )
    
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device, 'train')
    val_features, val_labels = extract_features(model, val_loader, device, 'val')
    test_features, _ = extract_features(model, test_loader, device, 'test')
    
    # Check feature statistics
    print(f"\nFeature statistics:")
    print(f"  Train features - mean: {train_features.mean():.6f}, std: {train_features.std():.6f}")
    print(f"  Train features - min: {train_features.min():.6f}, max: {train_features.max():.6f}")
    
    train_feature_norms = np.linalg.norm(train_features, axis=1)
    print(f"  Train feature norms - mean: {train_feature_norms.mean():.6f}, std: {train_feature_norms.std():.6f}")
    
    # Train KNN
    print(f"\nTraining KNN (k={args.k})...")
    knn = KNeighborsClassifier(n_neighbors=args.k, weights='distance', 
                               metric='cosine', n_jobs=-1)
    knn.fit(train_features, train_labels)
    
    # Evaluate
    train_acc = knn.score(train_features, train_labels)
    val_acc = knn.score(val_features, val_labels)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Predict on test set
    print(f"\nGenerating predictions...")
    predictions = knn.predict(test_features)
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': test_df['filename'],
        'class_id': predictions
    })
    
    submission_df.to_csv(args.output, index=False)
    
    print(f"\n✓ Submission saved to: {args.output}")
    print(f"  Total predictions: {len(submission_df)}")


if __name__ == "__main__":
    main()

