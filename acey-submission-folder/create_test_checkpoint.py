"""
Create a dummy checkpoint for testing the submission code
This creates a minimal checkpoint with the same structure as train_dino.py saves
"""

import torch
import torch.nn as nn
from dino_ssl import get_backbone, DINOHead, MultiCropWrapper
import argparse
import os

def create_test_checkpoint(arch='vit_tiny', img_size=96, out_dim=8192, bottleneck_dim=256, output_path='test_checkpoint.pth'):
    """
    Create a dummy checkpoint for testing
    """
    print(f"Creating test checkpoint: {output_path}")
    print(f"  Architecture: {arch}")
    print(f"  Image size: {img_size}")
    print(f"  Out dim: {out_dim}")
    print(f"  Bottleneck dim: {bottleneck_dim}")
    
    # Create model (same structure as training)
    backbone, embed_dim = get_backbone(arch, img_size=img_size, drop_path_rate=0.0)
    
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
        norm_last_layer=False
    )
    
    student = MultiCropWrapper(backbone, global_head, local_head=local_head)
    teacher = MultiCropWrapper(backbone, global_head, local_head=local_head)
    
    # Create dummy args as a namespace (pickle-able)
    from types import SimpleNamespace
    args = SimpleNamespace(
        arch=arch,
        image_size=img_size,
        out_dim=out_dim,
        bottleneck_dim=bottleneck_dim
    )
    
    # Save checkpoint (same format as train_dino.py)
    checkpoint = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'args': args,
        'epoch': 100,
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n✓ Test checkpoint saved to: {output_path}")
    print(f"  You can now test generate_submission.py with this checkpoint")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='vit_tiny', choices=['vit_tiny', 'vit_small', 'vit_base'],
                       help='Architecture')
    parser.add_argument('--out_dim', type=int, default=8192, help='Out dim')
    parser.add_argument('--bottleneck_dim', type=int, default=256, help='Bottleneck dim')
    parser.add_argument('--output', default='test_checkpoint.pth', help='Output path')
    
    args = parser.parse_args()
    
    create_test_checkpoint(
        arch=args.arch,
        img_size=96,
        out_dim=args.out_dim,
        bottleneck_dim=args.bottleneck_dim,
        output_path=args.output
    )

