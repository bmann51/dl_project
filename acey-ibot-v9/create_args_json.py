#!/usr/bin/env python3
"""
Create args.json files for v9 checkpoint directories
This script creates args.json files based on the training configuration
without needing to restart training.
"""

import json
import os
from pathlib import Path

# Base checkpoint path
BASE_CHECKPOINT_PATH = "/gpfs/scratch/bm3772/checkpoints"

# v9a configuration
v9a_config = {
    "data_root": "/gpfs/scratch/bm3772/fall2025_data/train",
    "output_dir": "/gpfs/scratch/bm3772/checkpoints/v9/v9a_ibot_checkpoint",
    "epochs": 400,
    "batch_size": 256,
    "lr": 0.0005,
    "weight_decay": 0.04,
    "mask_ratio": 0.3,
    "device": "cuda",
    
    # Hardcoded model configuration
        "arch": "vit_small",
        "image_size": 96,
        "patch_size": 16,
        "embed_dim": 384,
    "out_dim": 8192,
    "hidden_dim": 2048,
    
    # Hardcoded training configuration
    "optimizer": "adamw",
    "betas": [0.9, 0.95],
    "num_workers": 8,
    "pin_memory": True,
    "drop_last": True,
    
    # Hardcoded loss configuration
    "student_temp_cls": 0.1,
    "student_temp_patch": 0.1,
    "teacher_temp_cls_start": 0.04,
    "teacher_temp_cls_end": 0.07,
    "teacher_temp_patch_start": 0.04,
    "teacher_temp_patch_end": 0.07,
    "warmup_teacher_temp_epochs": 30,
    "center_momentum_cls": 0.9,
    "center_momentum_patch": 0.9,
    
    # Hardcoded EMA configuration
    "momentum_teacher_start": 0.996,
    "momentum_teacher_end": 0.999,
    
    # Hardcoded augmentation
    "num_global_crops": 2,
    "num_local_crops": 0,
    "mask_type": "random",
    
    # LR schedule
    "lr_schedule": "cosine",
    "warmup_epochs": 0,
    "min_lr": 0.0,
    
    # Save configuration
    "save_freq": 50,
}

# v9_higher_mask configuration (same as v9a but with mask_ratio=0.4)
v9_higher_mask_config = v9a_config.copy()
v9_higher_mask_config.update({
    "output_dir": "/gpfs/scratch/bm3772/checkpoints/v9/v9_higher_mask_ibot_checkpoint",
    "mask_ratio": 0.4,
})

def create_args_json(config, checkpoint_dir):
    """Create args.json file in checkpoint directory"""
    checkpoint_path = Path(checkpoint_dir)
    
    # Check if directory exists (or create it)
    if not checkpoint_path.exists():
        print(f"Warning: Directory {checkpoint_path} does not exist. Creating it...")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    args_json_path = checkpoint_path / "args.json"
    
    # Write args.json
    with open(args_json_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Created {args_json_path}")
    return args_json_path

if __name__ == "__main__":
    print("Creating args.json files for v9 checkpoints...")
    print()
    
    # Create args.json for v9a
    create_args_json(v9a_config, v9a_config["output_dir"])
    
    # Create args.json for v9_higher_mask
    create_args_json(v9_higher_mask_config, v9_higher_mask_config["output_dir"])
    
    print()
    print("Done!")

