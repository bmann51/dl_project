#!/usr/bin/env python3
"""
Script to verify backbone parameter counts for competition compliance.
Assignment requirement: Backbone must be strictly < 100M parameters.
"""

import torch
import timm
from ibot_ssl import get_backbone, count_parameters

def check_backbone_params(arch, img_size=96):
    """Check parameter count for a given architecture."""
    backbone, embed_dim = get_backbone(arch, img_size=img_size)
    param_count = count_parameters(backbone)
    
    print(f"\n{'='*60}")
    print(f"Architecture: {arch}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Backbone parameters: {param_count:,}")
    print(f"Under 100M limit: {'✅ YES' if param_count < 100_000_000 else '❌ NO'}")
    print(f"{'='*60}")
    
    return param_count

if __name__ == "__main__":
    print("Checking ViT backbone parameter counts for competition compliance...")
    print("Assignment requirement: Backbone must be strictly < 100M parameters\n")
    
    # Check all ViT variants
    architectures = ['vit_tiny', 'vit_small', 'vit_base']
    results = {}
    
    for arch in architectures:
        param_count = check_backbone_params(arch)
        results[arch] = param_count
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for arch, params in results.items():
        status = "✅ OK" if params < 100_000_000 else "❌ EXCEEDS LIMIT"
        print(f"{arch:15s}: {params:>12,} params  {status}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if results['vit_tiny'] < 100_000_000:
        print("✅ vit_tiny is SAFE to use (< 100M)")
    else:
        print("❌ vit_tiny EXCEEDS limit!")
    
    if results['vit_small'] < 100_000_000:
        print("✅ vit_small is SAFE to use (< 100M)")
    else:
        print("❌ vit_small EXCEEDS limit!")
    
    if results['vit_base'] < 100_000_000:
        print("✅ vit_base is SAFE to use (< 100M)")
    else:
        print("❌ vit_base EXCEEDS limit!")
    
    print("\nNote: The assignment requires BACKBONE parameters < 100M.")
    print("Heads and tokenizers are NOT counted in the backbone limit.")

