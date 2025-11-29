#!/usr/bin/env python3
"""
Batch evaluation script for assessing accuracy of all checkpoints

This script:
1. Discovers all checkpoints (same as generate_batch_submissions.py)
2. Runs evaluation on each checkpoint using val set
3. Generates a results summary CSV comparing all models

Usage:
    python batch_evaluate_all.py
    sbatch batch_submissions/batch_evaluate_all.sh
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import subprocess

# Import from generate_batch_submissions to reuse discovery logic
from generate_batch_submissions import (
    BASE_DATA_PATH, BASE_CHECKPOINT_PATH, PROJECT_ROOT, OUTPUT_DIR,
    find_checkpoints, find_checkpoint_directories, read_checkpoint_args,
    get_model_configs, get_args_from_checkpoint
)

TESTSETS = ["testset_1", "testset_2", "testset_3"]


def run_evaluation(model_name: str, config: Dict, checkpoint: str, testset: str) -> Dict:
    """
    Run evaluation script for a checkpoint and return accuracy metrics.
    
    Returns dict with accuracy results or None if evaluation fails.
    """
    folder = config["folder"]
    script_name = config["script"]
    
    # Determine evaluation script based on model type
    if "ibot" in script_name or "ibot" in model_name:
        eval_script = "eval_ibot.py"
    else:
        eval_script = "eval_dino2.py"  # or eval_dino.py
    
    # Get args
    if "args" in config:
        args = config["args"]
    elif "default_args" in config:
        args = get_args_from_checkpoint(checkpoint, config["default_args"])
    else:
        args = {}
    
    # Build command
    testset_data_path = f"{BASE_DATA_PATH}/{testset}/data"
    train_path = f"{testset_data_path}/train"
    val_path = f"{testset_data_path}/val"
    
    cmd = [
        "python", eval_script,
        "--checkpoint", checkpoint,
        "--train_path", train_path,
        "--test_path", val_path,  # Evaluate on validation set
        "--arch", str(args.get("arch", "vit_tiny")),
        "--image_size", str(args.get("resolution", 96)),
        "--k", str(args.get("k", 20)),
    ]
    
    # Add model-specific args
    if "out_dim" in args:
        cmd.extend(["--out_dim", str(args["out_dim"])])
    if "bottleneck_dim" in args:
        cmd.extend(["--bottleneck_dim", str(args["bottleneck_dim"])])
    if "num_tokens" in args:
        cmd.extend(["--num_tokens", str(args["num_tokens"])])
    
    # Change to model folder
    folder_path = PROJECT_ROOT / folder
    if not (folder_path / eval_script).exists():
        print(f"  ⚠️  Evaluation script not found: {folder_path / eval_script}")
        return None
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(folder_path),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            # Print full error for debugging
            error_msg = result.stderr if result.stderr else result.stdout
            print(f"  ✗ Evaluation failed:")
            # Print last 500 chars to see the actual error
            if error_msg:
                print(f"    {error_msg[-500:]}")
            return None
        
        # Parse accuracy from output
        output = result.stdout
        accuracy = None
        
        # Look for accuracy in output (format: "k-NN (k=20) Accuracy: XX.XX%")
        for line in output.split('\n'):
            if "Accuracy:" in line and "%" in line:
                try:
                    # Extract number before %
                    acc_str = line.split("Accuracy:")[1].split("%")[0].strip()
                    accuracy = float(acc_str) / 100.0
                    break
                except:
                    pass
        
        return {
            "model": model_name,
            "checkpoint": checkpoint,
            "testset": testset,
            "accuracy": accuracy,
            "status": "success" if accuracy is not None else "failed"
        }
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ Evaluation timed out")
        return None
    except Exception as e:
        print(f"  ✗ Evaluation error: {e}")
        return None


def main():
    """Main function to batch evaluate all checkpoints"""
    
    print("=" * 80)
    print("Batch Accuracy Evaluation")
    print("=" * 80)
    print()
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    model_configs = get_model_configs()
    all_results = []
    
    # Discover checkpoints (same logic as generate_batch_submissions.py)
    print("Discovering checkpoints...")
    for model_name, config in model_configs.items():
        print(f"\n{model_name}:")
        found_any = False
        
        # Handle variant-based discovery
        if config.get("use_variants", False):
            pattern = config.get("checkpoint_pattern", "")
            if pattern:
                matching_dirs = find_checkpoint_directories(pattern)
                
                if not matching_dirs:
                    version = model_name.split("-")[-1] if "-" in model_name else ""
                    if version and version.startswith("v"):
                        version_num = version[1:]
                        variant_patterns = [
                            f"checkpoints_ibot_conservative_{version_num}",
                            f"checkpoints_ibot_feedback_{version_num}",
                            f"checkpoints_ibot_aggressive_{version_num}",
                        ]
                        for vp in variant_patterns:
                            matching_dirs.extend(find_checkpoint_directories(vp))
                
                for ckpt_dir in sorted(set(matching_dirs)):
                    dir_name = ckpt_dir.name
                    
                    if "conservative" in dir_name:
                        variant_name = "conservative"
                    elif "feedback" in dir_name:
                        variant_name = "feedback"
                    elif "aggressive" in dir_name:
                        variant_name = "aggressive"
                    else:
                        parts = dir_name.replace("checkpoints_ibot_", "").split("_")
                        variant_name = parts[0] if parts else "unknown"
                    
                    checkpoints = find_checkpoints(str(ckpt_dir), max_checkpoints=1)  # Just final checkpoint
                    if checkpoints:
                        found_any = True
                        ckpt = checkpoints[0]  # Use final checkpoint
                        print(f"  {dir_name}: {Path(ckpt).name}")
                        
                        variant_model_name = f"{model_name}-{variant_name}"
                        variant_config = config.copy()
                        variant_config["args"] = get_args_from_checkpoint(ckpt, config["default_args"])
                        
                        # Evaluate on each testset
                        for testset in TESTSETS:
                            print(f"    Evaluating on {testset}...", end=" ", flush=True)
                            result = run_evaluation(variant_model_name, variant_config, ckpt, testset)
                            if result:
                                all_results.append(result)
                                print(f"✓ {result['accuracy']*100:.2f}%")
                            else:
                                print("✗ Failed")
        
        # Handle regular checkpoint directories
        elif "checkpoint_dirs" in config:
            for checkpoint_dir in config["checkpoint_dirs"]:
                matching_dirs = find_checkpoint_directories(checkpoint_dir)
                
                if not matching_dirs:
                    full_path = Path(BASE_CHECKPOINT_PATH) / checkpoint_dir
                    if full_path.exists():
                        matching_dirs = [full_path]
                
                for ckpt_dir in matching_dirs:
                    checkpoints = find_checkpoints(str(ckpt_dir), max_checkpoints=1)
                    
                    if checkpoints:
                        found_any = True
                        ckpt = checkpoints[0]
                        print(f"  {ckpt_dir.name}: {Path(ckpt).name}")
                        
                        for testset in TESTSETS:
                            print(f"    Evaluating on {testset}...", end=" ", flush=True)
                            result = run_evaluation(model_name, config, ckpt, testset)
                            if result:
                                all_results.append(result)
                                print(f"✓ {result['accuracy']*100:.2f}%")
                            else:
                                print("✗ Failed")
    
    # Create results DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Pivot to show results by testset
        summary = df.pivot_table(
            index=['model', 'checkpoint'],
            columns='testset',
            values='accuracy',
            aggfunc='first'
        )
        
        # Add average across testsets
        summary['avg_accuracy'] = summary.mean(axis=1)
        summary = summary.sort_values('avg_accuracy', ascending=False)
        
        # Save results
        results_path = OUTPUT_DIR / "evaluation_results.csv"
        summary.to_csv(results_path)
        
        print(f"\n{'='*80}")
        print("Evaluation Results Summary")
        print(f"{'='*80}")
        print(f"\nTotal evaluations: {len(all_results)}")
        print(f"Results saved to: {results_path}")
        print(f"\nTop 10 models by average accuracy:")
        print(summary.head(10).to_string())
        print(f"\n{'='*80}")
    else:
        print("\n⚠️  No evaluation results collected")


if __name__ == "__main__":
    main()

