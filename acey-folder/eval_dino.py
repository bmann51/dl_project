import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dino_ssl import get_backbone, MultiCropWrapper, DINOHead


class LabeledImageDataset(Dataset):
    """
    Dataset for labeled evaluation images
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all images organized by class folders
        self.samples = []
        self.class_to_idx = {}
        
        # Assume structure: root_dir/class_name/image.jpg
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
    """
    Extract features using the frozen backbone
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # Debug: check a sample batch
    sample_checked = False
    
    for images, labels in tqdm(data_loader, desc="Extracting features"):
        images = images.to(device)
        
        # Debug: print first batch info
        if not sample_checked:
            print(f"Sample batch - images shape: {images.shape}, dtype: {images.dtype}")
            print(f"Sample batch - images min: {images.min().item():.3f}, max: {images.max().item():.3f}, mean: {images.mean().item():.3f}")
            sample_checked = True
        
        # Get features from backbone
        features = model(images)
        
        # Debug: check feature statistics on first batch
        if not sample_checked:
            print(f"Sample features - shape: {features.shape}, dtype: {features.dtype}")
            print(f"Sample features - min: {features.min().item():.3f}, max: {features.max().item():.3f}, mean: {features.mean().item():.3f}, std: {features.std().item():.3f}")
            sample_checked = True
        
        # Normalize features (important for k-NN)
        features = nn.functional.normalize(features, dim=1, p=2)
        
        features_list.append(features.cpu())
        labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    print(f"\nExtracted features - shape: {features.shape}, dtype: {features.dtype}")
    print(f"Extracted features - min: {features.min():.3f}, max: {features.max():.3f}, mean: {features.mean():.3f}, std: {features.std():.3f}")
    print(f"Extracted labels - shape: {labels.shape}, unique labels: {np.unique(labels)}")
    
    return features, labels


def knn_evaluation(train_features, train_labels, test_features, test_labels, k=20):
    """
    k-NN evaluation on extracted features
    """
    print(f"\nRunning k-NN with k={k}...")
    
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_features, train_labels)
    
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    
    print(f"k-NN (k={k}) Accuracy: {accuracy * 100:.2f}%")
    return accuracy


class LinearProbe(nn.Module):
    """
    Linear classifier for linear probing evaluation
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
    
    def forward(self, x):
        return self.linear(x)


def linear_probe_evaluation(train_features, train_labels, test_features, test_labels,
                           num_classes, lr=0.1, epochs=100, batch_size=256):
    """
    Linear probe evaluation on extracted features
    """
    print(f"\nTraining linear probe for {epochs} epochs...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_labels).long()
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_features).float(),
        torch.from_numpy(test_labels).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create linear classifier
    in_dim = train_features.shape[1]
    linear_probe = LinearProbe(in_dim, num_classes).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.SGD(linear_probe.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0
    for epoch in range(epochs):
        linear_probe.train()
        train_loss = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = linear_probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            linear_probe.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = linear_probe(features)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {acc:.2f}%")
    
    print(f"Best Linear Probe Accuracy: {best_acc:.2f}%")
    return best_acc / 100.


def main(args):
    print("=" * 80)
    print("DINO Evaluation")
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
    
    # Build backbone
    backbone, embed_dim = get_backbone(args.arch, img_size=args.image_size)
    
    # Load weights
    if 'teacher' in checkpoint:
        # Full checkpoint with student/teacher
        state_dict = checkpoint['teacher']
    else:
        # Just the model state dict
        state_dict = checkpoint
    
    # Extract backbone weights (remove 'backbone.' prefix if present)
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if 'backbone' in k:
            # Remove 'backbone.' prefix if it exists
            new_key = k.replace('backbone.', '')
            backbone_state_dict[new_key] = v
        elif 'head' not in k and 'last_layer' not in k:
            # If keys don't have 'backbone' prefix, try loading directly
            # (this handles cases where state dict is just the backbone)
            backbone_state_dict[k] = v
    
    # Debug: print some keys to verify loading
    print(f"State dict keys (first 5): {list(state_dict.keys())[:5]}")
    print(f"Backbone state dict keys (first 5): {list(backbone_state_dict.keys())[:5]}")
    print(f"Total keys in state_dict: {len(state_dict)}, in backbone_state_dict: {len(backbone_state_dict)}")
    
    if len(backbone_state_dict) == 0:
        print("WARNING: No backbone weights found! Trying to load all non-head weights...")
        # Fallback: try loading all weights that aren't from the head
        backbone_state_dict = {k: v for k, v in state_dict.items() 
                              if 'head' not in k and 'last_layer' not in k}
        print(f"Fallback: Found {len(backbone_state_dict)} keys")
    
    missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_dict, strict=False)
    if missing_keys:
        print(f"WARNING: Missing keys when loading backbone: {missing_keys[:5]}... (showing first 5)")
    if unexpected_keys:
        print(f"WARNING: Unexpected keys when loading backbone: {unexpected_keys[:5]}... (showing first 5)")
    backbone = backbone.to(device)
    backbone.eval()
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    print("Backbone loaded and frozen")
    
    # Extract features
    print("\nExtracting train features...")
    train_features, train_labels = extract_features(backbone, train_loader, device)
    
    print("\nExtracting test features...")
    test_features, test_labels = extract_features(backbone, test_loader, device)
    
    print(f"\nTrain features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    # k-NN Evaluation
    knn_acc = knn_evaluation(train_features, train_labels, 
                            test_features, test_labels, k=args.k)
    
    # Linear Probe Evaluation (optional)
    if args.run_linear_probe:
        linear_acc = linear_probe_evaluation(
            train_features, train_labels,
            test_features, test_labels,
            num_classes,
            lr=args.lr,
            epochs=args.linear_epochs,
            batch_size=args.batch_size
        )
    
    # Print final results
    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"k-NN (k={args.k}) Accuracy: {knn_acc * 100:.2f}%")
    if args.run_linear_probe:
        print(f"Linear Probe Accuracy: {linear_acc * 100:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO Evaluation')
    
    # Data parameters
    parser.add_argument('--train_path', default='/mnt/user-data/uploads/eval_public/train',
                       type=str, help='Path to train data')
    parser.add_argument('--test_path', default='/mnt/user-data/uploads/eval_public/test',
                       type=str, help='Path to test data')
    parser.add_argument('--image_size', default=96, type=int, help='Image size')
    
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                       choices=['vit_tiny', 'vit_small', 'vit_base', 'resnet50'],
                       help='Architecture')
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to checkpoint')
    
    # k-NN parameters
    parser.add_argument('--k', default=20, type=int, help='k for k-NN')
    
    # Linear probe parameters
    parser.add_argument('--run_linear_probe', action='store_true',
                       help='Whether to run linear probe evaluation')
    parser.add_argument('--linear_epochs', default=100, type=int,
                       help='Number of epochs for linear probe')
    parser.add_argument('--lr', default=0.1, type=float,
                       help='Learning rate for linear probe')
    
    # Misc
    parser.add_argument('--batch_size', default=256, type=int,
                       help='Batch size for feature extraction')
    parser.add_argument('--num_workers', default=4, type=int,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)
