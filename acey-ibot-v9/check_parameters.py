#!/usr/bin/env python3
"""
Script to verify backbone parameter counts for competition compliance.
Assignment requirement: Backbone must be strictly < 100M parameters.
"""

import torch
from ibot_ssl import ViTBackbone

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_backbone_params(img_size=96):
    """Check parameter count for ViT-Base backbone."""
    backbone = ViTBackbone(img_size=img_size, patch_size=16, embed_dim=384)
    param_count = count_parameters(backbone)
    
    print(f"\n{'='*60}")
    print(f"Architecture: ViT-Base/16")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Embedding dimension: 384")
    print(f"Backbone parameters: {param_count:,}")
    print(f"Under 100M limit: {'✅ YES' if param_count < 100_000_000 else '❌ NO'}")
    print(f"{'='*60}")
    
    return param_count

if __name__ == "__main__":
    print("Checking ViT-Base backbone parameter counts for competition compliance...")
    print("Assignment requirement: Backbone must be strictly < 100M parameters\n")
    
    param_count = check_backbone_params()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    status = "✅ OK" if param_count < 100_000_000 else "❌ EXCEEDS LIMIT"
    print(f"ViT-Base/16: {param_count:>12,} params  {status}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if param_count < 100_000_000:
        print("✅ ViT-Base is SAFE to use (< 100M)")
    else:
        print("❌ ViT-Base EXCEEDS limit!")
    
    print("\nNote: The assignment requires BACKBONE parameters < 100M.")
    print("Heads and tokenizers are NOT counted in the backbone limit.")

