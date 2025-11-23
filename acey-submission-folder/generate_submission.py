"""
Generate Kaggle submission using YOUR trained DINO checkpoint
==============================================================

This script loads your trained DINO model and creates a submission CSV file
for the SSL competition using k-NN evaluation.

Usage examples:
    # For vit_tiny trained model
    python generate_submission.py \
        --checkpoint /path/to/checkpoint/final_checkpoint.pth \
        --data_dir ./data \
        --output submission.csv \
        --arch vit_tiny \
        --resolution 96 \
        --k 20

    # For vit_small trained model with custom dimensions
    python generate_submission.py \
        --checkpoint /path/to/checkpoint/final_checkpoint.pth \
        --data_dir ./data \
        --output submission.csv \
        --arch vit_small \
        --resolution 96 \
        --out_dim 4096 \
        --bottleneck_dim 128 \
        --k 20
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
import sys

# Import from local dino_ssl module
from dino_ssl import get_backbone, DINOHead, MultiCropWrapper, count_parameters


# ============================================================================
#                          DATASET
# ============================================================================

class ImageDataset(Dataset):
    """Dataset for loading images from CUB-200 or similar structure"""
    
    def __init__(self, image_dir, image_list, labels=None, resolution=96):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
            resolution: Image resolution (96 for competition)
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
        
        # ImageNet normalization (same as training)
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_dir / self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image


# ============================================================================
#                          LOAD YOUR MODEL
# ============================================================================

def load_your_model(checkpoint_path, arch='vit_small', img_size=96, 
                    out_dim=8192, bottleneck_dim=256, device='cuda'):
    """
    Load YOUR trained DINO model from checkpoint.
    
    This function reconstructs the exact model architecture used during training
    and loads the weights from the checkpoint.
    """
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Architecture: {arch}")
    
    # Get model configuration
    backbone, embed_dim = get_backbone(arch, img_size=img_size, drop_path_rate=0.0)
    
    print(f"  Embed dim: {embed_dim}")
    print(f"  Image size: {img_size}")
    print(f"  Out dim: {out_dim}")
    print(f"  Bottleneck dim: {bottleneck_dim}")
    
    # Create the same model structure you trained with
    # Use untied heads (global + local) as in acey-folder training
    global_head = DINOHead(
        in_dim=embed_dim, 
        out_dim=out_dim, 
        hidden_dim=2048, 
        bottleneck_dim=bottleneck_dim, 
        nlayers=3,
        norm_last_layer=True
    )
    
    local_head = DINOHead(
        in_dim=embed_dim, 
        out_dim=out_dim, 
        hidden_dim=2048, 
        bottleneck_dim=bottleneck_dim, 
        nlayers=3,
        norm_last_layer=False  # Local head doesn't use norm_last_layer
    )
    
    # Wrap with MultiCropWrapper (untied heads)
    model = MultiCropWrapper(backbone, global_head, local_head=local_head)
    
    # Load weights
    print(f"\nLoading checkpoint weights...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract student state dict (teacher is used for evaluation)
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
        print("  Found 'student' key in checkpoint")
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
        print("  Found 'teacher' key in checkpoint (using teacher weights)")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("  Found 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        print("  Loading checkpoint directly")
    
    # Try to load args from checkpoint to verify dimensions
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        print(f"\n  Checkpoint training config:")
        if hasattr(ckpt_args, 'arch'):
            print(f"    arch: {ckpt_args.arch}")
            if ckpt_args.arch != arch:
                print(f"    WARNING: Checkpoint arch ({ckpt_args.arch}) != specified arch ({arch})")
        if hasattr(ckpt_args, 'out_dim'):
            print(f"    out_dim: {ckpt_args.out_dim}")
            if ckpt_args.out_dim != out_dim:
                print(f"    WARNING: Checkpoint was trained with out_dim={ckpt_args.out_dim}, but you specified {out_dim}")
        if hasattr(ckpt_args, 'bottleneck_dim'):
            print(f"    bottleneck_dim: {ckpt_args.bottleneck_dim}")
            if ckpt_args.bottleneck_dim != bottleneck_dim:
                print(f"    WARNING: Checkpoint was trained with bottleneck_dim={ckpt_args.bottleneck_dim}, but you specified {bottleneck_dim}")
    
    # Load state dict (strict=False to handle missing keys gracefully)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"\n  WARNING: Missing keys when loading model:")
        for key in missing_keys[:10]:  # Show first 10
            print(f"    - {key}")
        if len(missing_keys) > 10:
            print(f"    ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"\n  WARNING: Unexpected keys in checkpoint:")
        for key in unexpected_keys[:10]:  # Show first 10
            print(f"    - {key}")
        if len(unexpected_keys) > 10:
            print(f"    ... and {len(unexpected_keys) - 10} more")
    
    # Freeze model for evaluation
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    
    # Verify parameter count
    total_params = count_parameters(model)
    print(f"\n  Total model parameters: {total_params:,}")
    if total_params >= 100_000_000:
        print(f"  WARNING: Model has {total_params:,} parameters (limit is 100M)")
    
    print("\n✓ Model loaded successfully!")
    return model


# ============================================================================
#                          EXTRACT FEATURES
# ============================================================================

def extract_features(model, dataloader, device, split_name='train'):
    """
    Extract features from dataset using frozen backbone.
    
    Args:
        model: Frozen DINO model
        dataloader: DataLoader for the dataset
        device: Device to run on
        split_name: Name of split (for progress bar)
    
    Returns:
        features: numpy array (N, feature_dim)
        labels: numpy array (N,) or None for test set
    """
    all_features = []
    all_labels = []
    
    print(f"\nExtracting {split_name} features...")
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{split_name} features"):
            # Handle both train/val (with labels) and test (without labels)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # train/val: (images, labels)
                images, labels = batch
                images = images.to(device)
                features = model(images)  # MultiCropWrapper returns normalized backbone features
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
            else:
                # test: just images
                images = batch.to(device)
                features = model(images)
                all_features.append(features.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels) if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    # Verify normalization
    feature_norms = np.linalg.norm(features, axis=1)
    print(f"  Feature L2 norms - min: {feature_norms.min():.3f}, max: {feature_norms.max():.3f}, mean: {feature_norms.mean():.3f}")
    
    return features, labels


# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Kaggle submission using trained DINO model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with vit_tiny
  python generate_submission.py \\
      --checkpoint ./checkpoints/final_checkpoint.pth \\
      --data_dir ./data \\
      --output submission.csv \\
      --arch vit_tiny \\
      --k 20

  # With custom dimensions (must match training!)
  python generate_submission.py \\
      --checkpoint ./checkpoints/final_checkpoint.pth \\
      --data_dir ./data \\
      --output submission.csv \\
      --arch vit_small \\
      --out_dim 4096 \\
      --bottleneck_dim 128 \\
      --k 20
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to your trained checkpoint (final_checkpoint.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory containing train/val/test folders and CSV files')
    
    # Model arguments (must match training config!)
    parser.add_argument('--arch', type=str, default='vit_small',
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'resnet50'],
                        help='Model architecture (MUST match training)')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Image resolution (96 for competition)')
    parser.add_argument('--out_dim', type=int, default=8192,
                        help='DINO head output dim (MUST match training! Default: 8192)')
    parser.add_argument('--bottleneck_dim', type=int, default=256,
                        help='DINO head bottleneck dim (MUST match training! Default: 256)')
    
    # Evaluation arguments
    parser.add_argument('--k', type=int, default=20,
                        help='Number of neighbors for k-NN (default: 20)')
    
    # Output
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV file path')
    
    # Misc
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    print("=" * 80)
    print("DINO Submission Generator")
    print("=" * 80)
    print(f"Using device: {device}\n")
    
    data_dir = Path(args.data_dir)
    
    # Verify data directory structure
    required_files = ['train_labels.csv', 'val_labels.csv', 'test_images.csv']
    required_dirs = ['train', 'val', 'test']
    
    for f in required_files:
        if not (data_dir / f).exists():
            print(f"ERROR: Required file not found: {data_dir / f}")
            sys.exit(1)
    
    for d in required_dirs:
        if not (data_dir / d).exists():
            print(f"ERROR: Required directory not found: {data_dir / d}")
            sys.exit(1)
    
    # Load CSV files
    print("Loading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")
    
    # Create datasets
    print(f"\nCreating datasets (resolution={args.resolution}px)...")
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
        None,  # No labels for test
        args.resolution
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load your model
    model = load_your_model(
        args.checkpoint, 
        args.arch, 
        args.resolution, 
        args.out_dim, 
        args.bottleneck_dim, 
        device
    )
    
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device, 'train')
    val_features, val_labels = extract_features(model, val_loader, device, 'val')
    test_features, _ = extract_features(model, test_loader, device, 'test')
    
    # Train k-NN classifier
    print(f"\n{'=' * 80}")
    print(f"Training k-NN classifier (k={args.k})...")
    print(f"{'=' * 80}")
    
    knn = KNeighborsClassifier(
        n_neighbors=args.k, 
        weights='distance',  # Weight by inverse distance
        metric='cosine',  # Cosine similarity for normalized embeddings
        n_jobs=-1
    )
    
    knn.fit(train_features, train_labels)
    
    # Evaluate on train and val
    train_acc = knn.score(train_features, train_labels)
    val_acc = knn.score(val_features, val_labels)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Predict on test set
    print(f"\nGenerating predictions on test set...")
    predictions = knn.predict(test_features)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['filename'],
        'class_id': predictions
    })
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, f"Invalid class_id < 0: {submission_df['class_id'].min()}"
    assert submission_df['class_id'].max() <= 199, f"Invalid class_id > 199: {submission_df['class_id'].max()}"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")
    
    # Save submission
    submission_df.to_csv(args.output, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Submission saved to: {args.output}")
    print(f"{'=' * 80}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nClass distribution in predictions:")
    print(submission_df['class_id'].value_counts().head(10))
    print(f"\n{'=' * 80}")
    print("Done! Upload your submission.csv to Kaggle.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

