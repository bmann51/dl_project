#!/usr/bin/env python3
"""
Helper script to manually add checkpoint paths to the batch submission system.

Usage:
    python manual_add_checkpoints.py --model acey-ibot-v3 --checkpoint /path/to/checkpoint.pth
"""

import argparse
import json
from pathlib import Path

MANIFEST_FILE = Path(__file__).parent / "submission_manifest.json"


def add_checkpoint(model_name: str, checkpoint_path: str, testsets: list = None):
    """Add a checkpoint to the manifest"""
    
    if testsets is None:
        testsets = ["testset_1", "testset_2", "testset_3"]
    
    # Load existing manifest
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
    else:
        manifest = []
    
    # Add jobs for this checkpoint
    from generate_batch_submissions import get_model_configs
    
    configs = get_model_configs()
    if model_name not in configs:
        print(f"Error: Model '{model_name}' not found in configurations")
        print(f"Available models: {', '.join(configs.keys())}")
        return
    
    config = configs[model_name]
    checkpoint_name = Path(checkpoint_path).stem
    
    for testset in testsets:
        output_name = f"submission_{model_name}_{testset}_{checkpoint_name}.csv"
        
        job = {
            "model": model_name,
            "checkpoint": checkpoint_path,
            "testset": testset,
            "output": output_name,
            "config": config,
        }
        
        # Check if already exists
        exists = any(
            j["model"] == model_name and
            j["checkpoint"] == checkpoint_path and
            j["testset"] == testset
            for j in manifest
        )
        
        if not exists:
            manifest.append(job)
            print(f"Added: {model_name} - {testset} - {Path(checkpoint_path).name}")
        else:
            print(f"Skipped (already exists): {model_name} - {testset}")
    
    # Save manifest
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Updated manifest: {MANIFEST_FILE}")
    print(f"Total jobs: {len(manifest)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually add checkpoints to batch submission")
    parser.add_argument("--model", required=True, help="Model name (e.g., acey-ibot-v3)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--testsets", nargs="+", default=["testset_1", "testset_2", "testset_3"],
                       help="Test sets to run (default: all)")
    
    args = parser.parse_args()
    add_checkpoint(args.model, args.checkpoint, args.testsets)

