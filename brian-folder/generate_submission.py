"""
Simple submission script using YOUR trained DINO checkpoint

Usage:
    python submission_simple.py \
        --checkpoint /gpfs/scratch/bm3772/checkpoints_base/final_checkpoint.pth \
        --data_dir ./data \
        --output submission.csv \
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


# ============================================================================
#                          YOUR MODEL CLASSES
# ============================================================================

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, 
                 nlayers=3, norm_last_layer=True):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        if norm_last_layer:
            self.last_layer.weight.data = F.normalize(self.last_layer.weight.data, dim=1)
            self.last_layer.weight.requires_grad = False
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head, local_head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.local_head = local_head
    
    def forward(self, x):
        # For inference, just return backbone features
        return self.backbone(x)


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
#                          LOAD YOUR MODEL
# ============================================================================

def load_your_model(checkpoint_path, model_name='vit_base_patch16_224', 
                    embed_dim=768, img_size=96, device='cuda'):
    """Load YOUR trained model from checkpoint"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create the same model structure you trained with
    # IMPORTANT: Must specify img_size to match training!
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0, 
                                img_size=img_size)
    
    # Use out_dim=8192 to match your training (not 65536!)
    head = DINOHead(in_dim=embed_dim, out_dim=8192, hidden_dim=2048, 
                    bottleneck_dim=256, nlayers=3)
    
    local_head = DINOHead(in_dim=embed_dim, out_dim=8192, hidden_dim=2048,
                         bottleneck_dim=256, nlayers=3)
    
    model = MultiCropWrapper(backbone, head, local_head)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    print("✓ Model loaded!")
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
        for batch in tqdm(dataloader):
            # Handle both train/val (with labels) and test (without labels)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # train/val: (images, labels)
                images, labels = batch
                images = images.to(device)
                features = model(images)
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
            else:
                # test: just images
                images = batch.to(device)
                features = model(images)
                all_features.append(features.cpu().numpy())
    
    features = np.concatenate(all_features)
    labels = np.array(all_labels) if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    return features, labels


# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to your checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory with train/val/test folders')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV file')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='Model architecture')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension (768 for base, 384 for small)')
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
    
    # Load your model
    model = load_your_model(args.checkpoint, args.model_name, args.embed_dim, 
                           args.resolution, device)
    
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device, 'train')
    val_features, val_labels = extract_features(model, val_loader, device, 'val')
    test_features, _ = extract_features(model, test_loader, device, 'test')
    
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