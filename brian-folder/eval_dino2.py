#Uses the head/bottleneck

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
from dino_ssl import get_backbone, DINOHead


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


class FeatureExtractor(nn.Module):
    """
    Wrapper to extract bottleneck features from DINO model
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        # Extract backbone features
        x = self.backbone(x)
        # Go through head MLP to get bottleneck features
        x = self.head.mlp(x)
        # Normalize
        x = F.normalize(x, dim=-1, p=2)
        return x


@torch.no_grad()
def extract_features(model, data_loader, device):
    """
    Extract features using the frozen model
    """
    model.eval()
    features_list = []
    labels_list = []
    
    for images, labels in tqdm(data_loader, desc="Extracting features"):
        images = images.to(device)
        
        # Get features
        features = model(images)
        
        features_list.append(features.cpu())
        labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
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
    print("DINO Evaluation (Using Bottleneck Features)")
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
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Get checkpoint args
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        out_dim = ckpt_args.out_dim if hasattr(ckpt_args, 'out_dim') else args.out_dim
        bottleneck_dim = ckpt_args.bottleneck_dim if hasattr(ckpt_args, 'bottleneck_dim') else args.bottleneck_dim
        print(f"Checkpoint config: out_dim={out_dim}, bottleneck_dim={bottleneck_dim}")
    else:
        out_dim = args.out_dim
        bottleneck_dim = args.bottleneck_dim
    
    # Build model
    backbone, embed_dim = get_backbone(args.arch, img_size=args.image_size)
    head = DINOHead(
        in_dim=embed_dim,
        out_dim=out_dim,
        hidden_dim=2048,
        bottleneck_dim=bottleneck_dim,
        nlayers=3
    )
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(backbone, head)
    
    # Load weights
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    else:
        state_dict = checkpoint
    
    # Load state dict
    feature_extractor.load_state_dict(state_dict, strict=False)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Freeze model
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    print(f"Model loaded and frozen (using {bottleneck_dim}-dim bottleneck features)")
    
    # Extract features
    print("\nExtracting train features...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    
    print("\nExtracting test features...")
    test_features, test_labels = extract_features(feature_extractor, test_loader, device)
    
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
    parser = argparse.ArgumentParser('DINO Evaluation with Bottleneck Features')
    
    # Data parameters
    parser.add_argument('--train_path', required=True, type=str,
                       help='Path to train data')
    parser.add_argument('--test_path', required=True, type=str,
                       help='Path to test data')
    parser.add_argument('--image_size', default=96, type=int, help='Image size')
    
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                       choices=['vit_tiny', 'vit_small', 'vit_base', 'resnet50'],
                       help='Architecture')
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to checkpoint')
    parser.add_argument('--out_dim', default=8192, type=int,
                       help='DINO head output dim')
    parser.add_argument('--bottleneck_dim', default=256, type=int,
                       help='DINO head bottleneck dim')
    
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