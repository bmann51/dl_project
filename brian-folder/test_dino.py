"""
Quick sanity check script to verify DINO implementation works correctly
"""
import torch
import sys
from dino_ssl import (
    get_backbone, DINOHead, MultiCropWrapper, DINOLoss,
    DataAugmentation, count_parameters
)


def test_backbone():
    """Test backbone creation and forward pass"""
    print("\n" + "="*60)
    print("Testing Backbone...")
    print("="*60)
    
    for arch in ['vit_tiny', 'vit_small']:
        print(f"\nTesting {arch}:")
        backbone, embed_dim = get_backbone(arch, img_size=96)
        print(f"  Embed dim: {embed_dim}")
        print(f"  Parameters: {count_parameters(backbone):,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 96, 96)
        with torch.no_grad():
            out = backbone(x)
        print(f"  Output shape: {out.shape}")
        assert out.shape == (2, embed_dim), f"Expected shape (2, {embed_dim}), got {out.shape}"
    
    print("✓ Backbone tests passed!")


def test_head():
    """Test projection head"""
    print("\n" + "="*60)
    print("Testing Projection Head...")
    print("="*60)
    
    embed_dim = 384
    out_dim = 8192
    bottleneck_dim = 256
    
    head = DINOHead(embed_dim, out_dim, bottleneck_dim=bottleneck_dim)
    print(f"  Input dim: {embed_dim}")
    print(f"  Output dim: {out_dim}")
    print(f"  Bottleneck dim: {bottleneck_dim}")
    print(f"  Parameters: {count_parameters(head):,}")
    
    # Test forward pass
    x = torch.randn(4, embed_dim)
    with torch.no_grad():
        out = head(x)
    print(f"  Output shape: {out.shape}")
    assert out.shape == (4, out_dim), f"Expected shape (4, {out_dim}), got {out.shape}"
    
    print("✓ Head tests passed!")


def test_full_model():
    """Test full student/teacher model"""
    print("\n" + "="*60)
    print("Testing Full Model...")
    print("="*60)
    
    backbone, embed_dim = get_backbone('vit_small', img_size=96)
    head = DINOHead(embed_dim, 8192, bottleneck_dim=256)
    model = MultiCropWrapper(backbone, head)
    
    total_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    
    if total_params >= 100_000_000:
        print(f"  ⚠️  WARNING: Model exceeds 100M parameter limit!")
        sys.exit(1)
    else:
        print(f"  ✓ Model is under 100M parameter limit")
    
    # Test forward pass with single input
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        out = model(x)
    print(f"  Single input output shape: {out.shape}")
    
    # Test forward pass with multiple crops
    crops = [torch.randn(2, 3, 96, 96) for _ in range(8)]
    with torch.no_grad():
        out = model(crops)
    print(f"  Multi-crop output shape: {out.shape}")
    
    print("✓ Full model tests passed!")


def test_loss():
    """Test DINO loss"""
    print("\n" + "="*60)
    print("Testing DINO Loss...")
    print("="*60)
    
    batch_size = 4
    out_dim = 8192
    ncrops = 8
    
    loss_fn = DINOLoss(out_dim, ncrops=ncrops, nepochs=100)
    print(f"  Output dim: {out_dim}")
    print(f"  Number of crops: {ncrops}")
    
    # Create dummy student and teacher outputs
    student_output = torch.randn(batch_size * ncrops, out_dim)
    teacher_output = torch.randn(batch_size * 2, out_dim)  # Only 2 global views
    
    loss = loss_fn(student_output, teacher_output, epoch=0)
    print(f"  Loss value: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite!"
    
    print("✓ Loss tests passed!")


def test_augmentation():
    """Test data augmentation"""
    print("\n" + "="*60)
    print("Testing Data Augmentation...")
    print("="*60)
    
    from PIL import Image
    import numpy as np
    
    # Create a dummy image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    aug = DataAugmentation(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=6,
        size=96
    )
    
    crops = aug(img)
    print(f"  Number of crops: {len(crops)}")
    print(f"  Global crop 1 shape: {crops[0].shape}")
    print(f"  Global crop 2 shape: {crops[1].shape}")
    print(f"  Local crop shape: {crops[2].shape}")
    
    assert len(crops) == 8, f"Expected 8 crops, got {len(crops)}"
    for i, crop in enumerate(crops):
        assert crop.shape == (3, 96, 96), f"Crop {i} has wrong shape: {crop.shape}"
    
    print("✓ Augmentation tests passed!")


def test_momentum_update():
    """Test momentum update function"""
    print("\n" + "="*60)
    print("Testing Momentum Update...")
    print("="*60)
    
    from dino_ssl import update_momentum
    
    # Create simple models
    student = torch.nn.Linear(10, 10)
    teacher = torch.nn.Linear(10, 10)
    
    # Initialize with same weights
    teacher.load_state_dict(student.state_dict())
    
    # Get initial weights
    initial_teacher_weight = teacher.weight.data.clone()
    
    # Update student
    student.weight.data += 0.1
    
    # Apply momentum update
    m = 0.99
    update_momentum(student, teacher, m)
    
    # Check teacher was updated
    assert not torch.allclose(teacher.weight.data, initial_teacher_weight), \
        "Teacher weights did not update"
    
    print(f"  Momentum coefficient: {m}")
    print("✓ Momentum update tests passed!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("DINO IMPLEMENTATION SANITY CHECK")
    print("="*60)
    
    try:
        test_backbone()
        test_head()
        test_full_model()
        test_loss()
        test_augmentation()
        test_momentum_update()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour DINO implementation is ready to use!")
        print("Next steps:")
        print("  1. Prepare your data (unlabeled pretrain/ folder)")
        print("  2. Run: python train_dino.py --data_path /path/to/pretrain/")
        print("  3. Evaluate: python eval_dino.py --checkpoint /path/to/checkpoint.pth")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()