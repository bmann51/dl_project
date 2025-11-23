"""
Quick test script for generate_submission.py
Tests the code structure without needing real data or checkpoints
"""

import torch
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    try:
        from dino_ssl import get_backbone, DINOHead, MultiCropWrapper, count_parameters
        print("✓ dino_ssl imports work")
        
        from generate_submission import load_your_model, ImageDataset
        print("✓ generate_submission imports work")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_model_creation():
    """Test that models can be created"""
    print("\nTesting model creation...")
    try:
        from dino_ssl import get_backbone, DINOHead, MultiCropWrapper
        
        # Test vit_tiny
        backbone, embed_dim = get_backbone('vit_tiny', img_size=96)
        print(f"✓ Created vit_tiny backbone (embed_dim={embed_dim})")
        
        head = DINOHead(embed_dim, out_dim=8192, bottleneck_dim=256)
        print("✓ Created DINOHead")
        
        model = MultiCropWrapper(backbone, head, local_head=head)
        print("✓ Created MultiCropWrapper")
        
        # Test forward pass
        x = torch.randn(2, 3, 96, 96)
        with torch.no_grad():
            features = model(x)
        print(f"✓ Forward pass works (output shape: {features.shape})")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_loading():
    """Test checkpoint loading (if test checkpoint exists)"""
    print("\nTesting checkpoint loading...")
    
    test_checkpoint = 'test_checkpoint.pth'
    if not os.path.exists(test_checkpoint):
        print(f"⚠ Test checkpoint not found: {test_checkpoint}")
        print("  Run: python create_test_checkpoint.py")
        return True  # Not a failure, just missing file
    
    try:
        from generate_submission import load_your_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        model = load_your_model(
            test_checkpoint,
            arch='vit_tiny',
            img_size=96,
            out_dim=8192,
            bottleneck_dim=256,
            device=device
        )
        print("✓ Checkpoint loaded successfully")
        
        # Test feature extraction
        x = torch.randn(2, 3, 96, 96).to(device)
        with torch.no_grad():
            features = model(x)
        print(f"✓ Feature extraction works (output shape: {features.shape})")
        
        return True
    except Exception as e:
        print(f"✗ Checkpoint loading error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Testing acey-submission-folder Code")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Checkpoint Loading", test_checkpoint_loading()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. Check errors above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

