import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import timm
from ibot_ssl import iBOTTokenizer


# Define BatchNorm version of iBOTHead to match training checkpoints
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


# Define simpler wrapper for evaluation (matches generate_submission_ibot.py)
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


class LabeledImageDataset(Dataset):
    """Dataset for labeled evaluation images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}
        
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, idx))
        
        print(f"Found {len(self.samples)} images in {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


@torch.no_grad()
def extract_features(model, data_loader, device):
    """Extract features using the frozen backbone"""
    model.eval()
    features_list = []
    labels_list = []
    
    for i, (images, labels) in enumerate(tqdm(data_loader, desc="Extracting features")):
        try:
            images = images.to(device)
            
            # Extract features (backbone only, no masking for evaluation)
            # Model returns normalized CLS token for single images
            features = model(images)
            
            # Debug first batch
            if i == 0:
                print(f"\n  First batch - images shape: {images.shape}")
                print(f"  First batch - features shape: {features.shape}")
                print(f"  First batch - features type: {type(features)}")
            
            # Features are already normalized in the forward pass
            features_list.append(features.cpu())
            labels_list.append(labels)
        except Exception as e:
            print(f"\nERROR in extract_features at batch {i}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    if len(features_list) == 0:
        raise ValueError("No features extracted! Check if dataloader is empty or model forward is failing.")
    
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    print(f"\nExtracted features - shape: {features.shape}")
    feature_norms = np.linalg.norm(features, axis=1)
    print(f"Feature L2 norms - min: {feature_norms.min():.3f}, max: {feature_norms.max():.3f}, mean: {feature_norms.mean():.3f}")
    
    return features, labels


def knn_evaluation(train_features, train_labels, test_features, test_labels, k=20):
    """k-NN evaluation on extracted features"""
    print(f"\nRunning k-NN with k={k}...")
    
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_features, train_labels)
    
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    
    print(f"k-NN (k={k}) Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main(args):
    print("=" * 80)
    print("iBOT Evaluation")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms (no augmentation for evaluation)
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    print(f"\nLoading train dataset from {args.train_path}")
    train_dataset = LabeledImageDataset(args.train_path, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nLoading test dataset from {args.test_path}")
    test_dataset = LabeledImageDataset(args.test_path, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.class_to_idx)
    print(f"\nNumber of classes: {num_classes}")
    
    # Load model
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Get model configuration
    config = get_model_config(args.arch)
    model_name = config['model_name']
    embed_dim = config['embed_dim']
    
    # Build backbone - IMPORTANT: use global_pool='' to get all tokens like in training
    backbone = timm.create_model(
        model_name, 
        pretrained=False, 
        num_classes=0,
        img_size=args.image_size,
        global_pool=''  # Returns [batch, num_patches+1, embed_dim] instead of [batch, embed_dim]
    )
    
    # Create heads and tokenizers (needed for model structure, but we only use backbone for eval)
    global_head = iBOTHead(embed_dim, args.out_dim, bottleneck_dim=args.bottleneck_dim)
    global_tokenizer = iBOTTokenizer(embed_dim, num_tokens=args.num_tokens)
    
    # Create model wrapper
    model = MultiCropiBOTWrapper(backbone, global_head, global_tokenizer)
    
    # Load weights - use same approach as generate_submission_ibot.py
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
        print("  Found 'student' key in checkpoint")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("  Found 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        print("  Loading checkpoint directly")
    
    # Load state dict - same as submission script
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        if len(missing) > 0:
            print(f"  First few missing: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        if len(unexpected) > 0:
            print(f"  First few unexpected: {unexpected[:5]}")
    
    model.eval()
    model.to(device)
    
    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    
    print("Model loaded and frozen")
    
    # Test model forward first
    print("\nTesting model forward...")
    try:
        test_batch = next(iter(train_loader))
        test_images, test_labels = test_batch
        test_images = test_images.to(device)
        print(f"  Test batch images shape: {test_images.shape}")
        test_features = model(test_images)
        print(f"  Test batch features shape: {test_features.shape}")
        print("  ✓ Model forward works!")
    except Exception as e:
        print(f"  ✗ Model forward failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Extract features
    print("\nExtracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    
    print("\nExtracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)
    
    print(f"\nTrain features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    # k-NN Evaluation
    knn_acc = knn_evaluation(train_features, train_labels, 
                            test_features, test_labels, k=args.k)
    
    # Print final results
    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"k-NN (k={args.k}) Accuracy: {knn_acc * 100:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT Evaluation')
    
    # Data parameters
    parser.add_argument('--train_path', default='/mnt/user-data/uploads/eval_public/train',
                       type=str, help='Path to train data')
    parser.add_argument('--test_path', default='/mnt/user-data/uploads/eval_public/test',
                       type=str, help='Path to test data')
    parser.add_argument('--image_size', default=96, type=int, help='Image size')
    
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                       choices=['vit_tiny', 'vit_small', 'vit_base'],
                       help='Architecture')
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to checkpoint')
    parser.add_argument('--out_dim', default=8192, type=int,
                       help='Output dimension (must match training)')
    parser.add_argument('--bottleneck_dim', default=256, type=int,
                       help='Bottleneck dimension (must match training)')
    parser.add_argument('--num_tokens', default=8192, type=int,
                       help='Number of tokens (must match training)')
    
    # k-NN parameters
    parser.add_argument('--k', default=20, type=int, help='k for k-NN')
    
    # Misc
    parser.add_argument('--batch_size', default=256, type=int,
                       help='Batch size for feature extraction')
    parser.add_argument('--num_workers', default=4, type=int,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)

