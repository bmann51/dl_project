"""
Submission script for iBOT trained models

Usage:
    python generate_submission_ibot.py \
        --checkpoint /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v3/checkpoint_0099.pth \
        --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_2/data \
        --output submission_ibot.csv \
        --arch vit_tiny \
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
#                          iBOT MODEL CLASSES
# ============================================================================

class iBOTTokenizer(nn.Module):
    """Online tokenizer for iBOT"""
    def __init__(self, embed_dim, num_tokens=8192, hidden_dim=512):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokenizer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens)
        )
    
    def forward(self, x):
        return self.tokenizer(x)


class iBOTHead(nn.Module):
    """Projection head for iBOT - FIXED to match training architecture with BatchNorm"""
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, 
                 nlayers=3, norm_last_layer=True):
        super().__init__()
        
        # Build MLP with BatchNorm layers (this matches the checkpoint)
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
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


class MultiCropiBOTWrapper(nn.Module):
    """Wrapper for iBOT - for evaluation, returns normalized backbone CLS token"""
    def __init__(self, backbone, head, tokenizer, local_head=None, local_tokenizer=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.local_head = local_head
        self.local_tokenizer = local_tokenizer
    
    def forward(self, x):
        """For evaluation: return normalized CLS token from backbone"""
        features = self.backbone(x)  # Get backbone output
        
        # Handle different output formats from timm
        if isinstance(features, torch.Tensor):
            # If features is a tensor, check its shape
            if len(features.shape) == 3:
                # [batch, num_patches+1, embed_dim] - extract CLS token
                cls_token = features[:, 0]
            elif len(features.shape) == 2:
                # [batch, embed_dim] - already pooled, use directly
                cls_token = features
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
        else:
            raise ValueError(f"Unexpected feature type: {type(features)}")
        
        # Normalize
        cls_token = F.normalize(cls_token, dim=-1, p=2)
        return cls_token


# ============================================================================
#                          ARCHITECTURE CONFIG
# ============================================================================

def get_model_config(arch):
    """Get model configuration based on architecture"""
    configs = {
        'vit_tiny': {
            'model_name': 'vit_tiny_patch16_224',
            'embed_dim': 192,
        },
        'vit_small': {
            'model_name': 'vit_small_patch16_224',
            'embed_dim': 384,
        },
        'vit_base': {
            'model_name': 'vit_base_patch16_224',
            'embed_dim': 768,
        },
    }
    
    if arch not in configs:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return configs[arch]


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
#                          INFER BOTTLENECK DIM
# ============================================================================

def infer_bottleneck_dim_from_checkpoint(checkpoint_path: str) -> int:
    """Infer bottleneck_dim from checkpoint state_dict by finding the last Linear layer in head.mlp"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get state dict
        if 'student' in checkpoint:
            state_dict = checkpoint['student']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Find the last Linear layer in head.mlp by checking all mlp.*.weight keys
        # The last Linear layer should be: hidden_dim -> bottleneck_dim
        # So its weight shape is [bottleneck_dim, hidden_dim]
        mlp_linear_keys = []
        for key in state_dict.keys():
            if 'head.mlp.' in key and '.weight' in key and 'last_layer' not in key:
                # Extract layer index
                parts = key.split('.')
                try:
                    layer_idx = int(parts[2])  # head.mlp.{idx}.weight
                    mlp_linear_keys.append((layer_idx, key))
                except (ValueError, IndexError):
                    continue
        
        if mlp_linear_keys:
            # Sort by layer index and get the last one (highest index)
            mlp_linear_keys.sort(key=lambda x: x[0])
            last_layer_idx, last_key = mlp_linear_keys[-1]
            weight_shape = state_dict[last_key].shape
            
            # weight_shape is [out_features, in_features] = [bottleneck_dim, hidden_dim]
            if len(weight_shape) == 2:
                bottleneck_dim = weight_shape[0]  # out_features dimension
                hidden_dim = weight_shape[1]  # in_features dimension
                # Verify it's reasonable (hidden_dim should be 2048)
                if hidden_dim == 2048:
                    print(f"    Inferred bottleneck_dim={bottleneck_dim} from checkpoint (last MLP layer: {last_key}, shape: {weight_shape})")
                    return bottleneck_dim
    except Exception as e:
        print(f"    Could not infer bottleneck_dim from checkpoint: {e}")
    
    return None


# ============================================================================
#                          LOAD iBOT MODEL
# ============================================================================

def load_ibot_model(checkpoint_path, arch='vit_tiny', img_size=96, 
                    out_dim=4096, bottleneck_dim=128, num_tokens=8192, device='cuda'):
    """Load iBOT trained model from checkpoint"""
    
    print(f"Loading iBOT checkpoint: {checkpoint_path}")
    print(f"Architecture: {arch}")
    
    # Infer bottleneck_dim from checkpoint if not explicitly provided or incorrect
    inferred_bottleneck_dim = infer_bottleneck_dim_from_checkpoint(checkpoint_path)
    if inferred_bottleneck_dim is not None and inferred_bottleneck_dim != bottleneck_dim:
        print(f"    WARNING: Overriding bottleneck_dim from {bottleneck_dim} to inferred {inferred_bottleneck_dim}")
        bottleneck_dim = inferred_bottleneck_dim
    
    # Get model configuration
    config = get_model_config(arch)
    model_name = config['model_name']
    embed_dim = config['embed_dim']
    
    print(f"  Timm model name: {model_name}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Image size: {img_size}")
    print(f"  Out dim: {out_dim}")
    print(f"  Bottleneck dim: {bottleneck_dim}")
    print(f"  Num tokens: {num_tokens}")
    
    # Create backbone - IMPORTANT: use global_pool='' to get all tokens like in training
    backbone = timm.create_model(
        model_name, 
        pretrained=False, 
        num_classes=0,
        img_size=img_size,
        global_pool=''  # Returns [batch, num_patches+1, embed_dim] instead of [batch, embed_dim]
    )
    
    # Create iBOT heads - FIXED: Now includes BatchNorm layers
    head = iBOTHead(in_dim=embed_dim, out_dim=out_dim, hidden_dim=2048, 
                    bottleneck_dim=bottleneck_dim, nlayers=3)
    
    tokenizer = iBOTTokenizer(embed_dim=embed_dim, num_tokens=num_tokens, hidden_dim=512)
    
    # Create local heads if using untied architecture
    local_head = iBOTHead(in_dim=embed_dim, out_dim=out_dim, hidden_dim=2048,
                         bottleneck_dim=bottleneck_dim, nlayers=3)
    local_tokenizer = iBOTTokenizer(embed_dim=embed_dim, num_tokens=num_tokens, hidden_dim=512)
    
    model = MultiCropiBOTWrapper(backbone, head, tokenizer, local_head, local_tokenizer)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
        print("\n  Loading from 'student' key in checkpoint")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("\n  Loading from 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        print("\n  Loading directly from checkpoint")
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    
    model.eval()
    model.to(device)
    
    print("\n✓ iBOT model loaded successfully!")
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
                
                # Debug first batch
                if i == 0:
                    print(f"\n  First batch - images shape: {images.shape}")
                    print(f"  First batch - features shape: {features.shape}")
                    print(f"  First batch - features type: {type(features)}")
                
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
            else:
                images = batch.to(device)
                features = model(images)
                
                # Debug first batch
                if i == 0:
                    print(f"\n  First batch - images shape: {images.shape}")
                    print(f"  First batch - features shape: {features.shape}")
                    print(f"  First batch - features type: {type(features)}")
                
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
                        help='Path to iBOT checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory with train/val/test folders')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV file')
    parser.add_argument('--arch', type=str, default='vit_tiny',
                        choices=['vit_tiny', 'vit_small', 'vit_base'],
                        help='Model architecture')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Image resolution')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dim', type=int, default=4096,
                        help='iBOT head output dim')
    parser.add_argument('--bottleneck_dim', type=int, default=128,
                        help='iBOT head bottleneck dim')
    parser.add_argument('--num_tokens', type=int, default=8192,
                        help='iBOT tokenizer vocabulary size')
    
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
    
    # Load iBOT model
    model = load_ibot_model(
        args.checkpoint, args.arch, args.resolution, 
        args.out_dim, args.bottleneck_dim, args.num_tokens, device
    )
    
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device, 'train')
    val_features, val_labels = extract_features(model, val_loader, device, 'val')
    test_features, _ = extract_features(model, test_loader, device, 'test')
    
    # Check feature statistics to detect collapsed embeddings
    print(f"\nFeature statistics:")
    print(f"  Train features - mean: {train_features.mean():.6f}, std: {train_features.std():.6f}")
    print(f"  Train features - min: {train_features.min():.6f}, max: {train_features.max():.6f}")
    
    # Check if embeddings are collapsed (all nearly identical)
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