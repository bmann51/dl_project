#!/usr/bin/env python3
"""
Batch submission generator for all model variations

This script generates submission commands for all checkpoints across:
- acey (DINO and iBOT versions)
- brian (DINO)
- macey (iBOT v6)

It creates a bash script that can be submitted to SLURM to run all submissions.
"""

import os
from pathlib import Path
import json
from typing import List, Dict, Tuple

# Base paths
BASE_DATA_PATH = "/gpfs/scratch/bm3772/fall2025_finalproject"
BASE_CHECKPOINT_PATH = "/gpfs/scratch/bm3772"
OUTPUT_DIR = Path(__file__).parent / "batch_submissions"

# Test sets
TESTSETS = ["testset_1", "testset_2", "testset_3"]

# Model configurations
MODEL_CONFIGS = {
    # Acey DINO
    "acey-dino": {
        "folder": "acey-submission-folder",
        "script": "generate_submission.py",
        "type": "dino",
        "checkpoints": [
            # Add checkpoint paths here - you'll need to update these
            # Example: f"{BASE_CHECKPOINT_PATH}/checkpoints_vit_tiny/final_checkpoint.pth"
        ],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 8192,
            "bottleneck_dim": 256,
        }
    },
    
    # Acey iBOT versions
    "acey-ibot-v1": {
        "folder": "acey-ibot-folder",
        "script": "generate_submission_ibot.py",
        "type": "ibot",
        "checkpoints": [
            # Example: f"{BASE_CHECKPOINT_PATH}/checkpoints_ibot_vit_tiny_v1/final_checkpoint.pth"
        ],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 4096,
            "bottleneck_dim": 128,
            "num_tokens": 8192,
        }
    },
    "acey-ibot-v2": {
        "folder": "acey-ibot-v2",
        "script": "generate_submission_ibot.py",
        "type": "ibot",
        "checkpoints": [],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 4096,
            "bottleneck_dim": 128,
            "num_tokens": 8192,
        }
    },
    "acey-ibot-v3": {
        "folder": "acey-ibot-v3",
        "script": "generate_submission_ibot.py",
        "type": "ibot",
        "checkpoints": [],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 4096,
            "bottleneck_dim": 128,
            "num_tokens": 8192,
        }
    },
    "acey-ibot-v4": {
        "folder": "acey-ibot-v4",
        "script": "generate_submission_ibot.py",
        "type": "ibot",
        "checkpoints": [],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 4096,
            "bottleneck_dim": 128,
            "num_tokens": 8192,
        }
    },
    "acey-ibot-v5": {
        "folder": "acey-ibot-v5",
        "script": "generate_submission_ibot.py",
        "type": "ibot",
        "checkpoints": [],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 4096,
            "bottleneck_dim": 128,
            "num_tokens": 8192,
        }
    },
    "acey-ibot-v7": {
        "folder": "acey-ibot-v7",
        "script": "generate_submission_ibot.py",
        "type": "ibot",
        "checkpoints": [],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 4096,
            "bottleneck_dim": 128,
            "num_tokens": 8192,
        }
    },
    "acey-ibot-v8": {
        "folder": "acey-ibot-v8",
        "script": "generate_submission_ibot.py",
        "type": "ibot",
        "checkpoints": [],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 4096,
            "bottleneck_dim": 128,
            "num_tokens": 8192,
        }
    },
    
    # Brian DINO
    "brian-dino": {
        "folder": "brian-folder",
        "script": "generate_submission2.py",
        "type": "dino",
        "checkpoints": [],
        "default_args": {
            "arch": "vit_base",
            "resolution": 96,
            "k": 5,
            "out_dim": 4096,
            "bottleneck_dim": 128,
        }
    },
    
    # Macey iBOT
    "macey-ibot-v6": {
        "folder": "macey-ibot-v6",
        "script": "generate_submission.py",
        "type": "ibot",
        "checkpoints": [],
        "default_args": {
            "arch": "vit_tiny",
            "resolution": 96,
            "k": 20,
            "out_dim": 4096,
            "bottleneck_dim": 128,
            "num_tokens": 8192,
        }
    },
}


def discover_checkpoints(base_path: str, pattern: str = "checkpoint*.pth") -> List[str]:
    """
    Discover all checkpoint files matching pattern in base_path.
    Returns list of full paths to checkpoints.
    """
    checkpoints = []
    base = Path(base_path)
    
    if not base.exists():
        return checkpoints
    
    # Look for final_checkpoint.pth first
    final_ckpt = base / "final_checkpoint.pth"
    if final_ckpt.exists():
        checkpoints.append(str(final_ckpt))
    
    # Look for numbered checkpoints
    for ckpt in base.glob(pattern):
        if ckpt.name != "final_checkpoint.pth":
            checkpoints.append(str(ckpt))
    
    # Sort by epoch number if possible
    def get_epoch(path_str):
        try:
            # Extract epoch from checkpoint_XXXX.pth
            parts = Path(path_str).stem.split('_')
            for part in parts:
                if part.isdigit():
                    return int(part)
        except:
            pass
        return 0
    
    checkpoints.sort(key=get_epoch, reverse=True)  # Latest first
    return checkpoints


def generate_submission_command(
    model_name: str,
    config: Dict,
    checkpoint: str,
    testset: str,
    output_name: str,
    base_dir: Path
) -> Tuple[str, str]:
    """
    Generate a submission command for a specific checkpoint and testset.
    Returns (command_string, output_file_path)
    """
    folder = config["folder"]
    script = config["script"]
    args = config["default_args"].copy()
    
    # Build command
    cmd_parts = [
        f"cd {base_dir / folder}",
        "&&",
        "python", script,
        "--checkpoint", checkpoint,
        "--data_dir", f"{BASE_DATA_PATH}/{testset}/data",
        "--output", output_name,
    ]
    
    # Add all default args
    for key, value in args.items():
        cmd_parts.extend([f"--{key}", str(value)])
    
    command = " ".join(cmd_parts)
    
    return command, output_name


def main():
    """Generate batch submission script"""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Discover checkpoints for each model
    print("Discovering checkpoints...")
    for model_name, config in MODEL_CONFIGS.items():
        # Try to discover checkpoints from common locations
        folder = config["folder"]
        
        # Common checkpoint directory patterns
        checkpoint_patterns = [
            f"{BASE_CHECKPOINT_PATH}/checkpoints_{model_name.replace('-', '_')}",
            f"{BASE_CHECKPOINT_PATH}/checkpoints_{folder}",
            f"{BASE_CHECKPOINT_PATH}/{model_name}",
        ]
        
        # Also check if there's a specific pattern in the folder name
        if "ibot" in model_name:
            version = model_name.split("-")[-1] if "-" in model_name else ""
            checkpoint_patterns.extend([
                f"{BASE_CHECKPOINT_PATH}/checkpoints_ibot_vit_tiny_{version}",
                f"{BASE_CHECKPOINT_PATH}/checkpoints_ibot_{version}",
            ])
        
        # Try to find checkpoints
        found_checkpoints = []
        for pattern in checkpoint_patterns:
            ckpts = discover_checkpoints(pattern)
            if ckpts:
                found_checkpoints.extend(ckpts)
                print(f"  {model_name}: Found {len(ckpts)} checkpoints in {pattern}")
                break
        
        if not found_checkpoints:
            print(f"  {model_name}: No checkpoints found (will use manual config)")
        else:
            config["checkpoints"] = found_checkpoints[:5]  # Limit to 5 most recent
    
    # Generate submission script
    script_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=batch_submit",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64GB",
        "#SBATCH --time=12:00:00",
        "#SBATCH --gres=gpu:a100:1",
        "#SBATCH --partition=a100_short",
        f"#SBATCH --output={OUTPUT_DIR}/batch_submit_%j.out",
        f"#SBATCH --error={OUTPUT_DIR}/batch_submit_%j.err",
        "",
        "# Load modules",
        "module load cuda/11.8",
        "",
        "# Activate environment",
        "source ~/.bashrc",
        "conda activate dino_new",
        "",
        "# Print info",
        "echo 'Job started at $(date)'",
        "echo 'Running on node: $(hostname)'",
        "echo 'GPU info:'",
        "nvidia-smi",
        "",
        "# Base directory",
        f"BASE_DIR={Path(__file__).parent}",
        "",
        "# Create output directory",
        f"mkdir -p {OUTPUT_DIR}",
        "",
        "echo ''",
        "echo 'Starting batch submissions...'",
        "echo ''",
        "",
    ]
    
    # Generate commands for each combination
    total_jobs = 0
    job_log = []
    
    for model_name, config in MODEL_CONFIGS.items():
        if not config["checkpoints"]:
            print(f"Skipping {model_name}: No checkpoints configured")
            continue
        
        folder = config["folder"]
        base_dir = Path(__file__).parent
        
        for checkpoint in config["checkpoints"]:
            # Check if checkpoint exists
            if not Path(checkpoint).exists():
                print(f"Warning: Checkpoint not found: {checkpoint}")
                continue
            
            for testset in TESTSETS:
                # Generate output filename
                ckpt_name = Path(checkpoint).stem
                output_name = f"submission_{model_name}_{testset}_{ckpt_name}.csv"
                
                # Generate command
                cmd, output_file = generate_submission_command(
                    model_name, config, checkpoint, testset, output_name, base_dir
                )
                
                # Add to script
                script_lines.append(f"# {model_name} - {testset} - {Path(checkpoint).name}")
                script_lines.append("echo ''")
                script_lines.append(f"echo 'Running: {model_name} on {testset}'")
                script_lines.append(cmd)
                script_lines.append(f"if [ $? -eq 0 ]; then")
                script_lines.append(f"    echo '✓ Success: {output_file}'")
                script_lines.append(f"else")
                script_lines.append(f"    echo '✗ Failed: {output_file}'")
                script_lines.append(f"fi")
                script_lines.append("")
                
                total_jobs += 1
                job_log.append({
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "testset": testset,
                    "output": output_file,
                    "folder": folder,
                })
    
    script_lines.extend([
        "echo ''",
        "echo 'Job finished at $(date)'",
        "echo 'All submissions completed!'",
    ])
    
    # Write script
    script_path = OUTPUT_DIR / "batch_submit_all.sh"
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines))
    
    os.chmod(script_path, 0o755)
    
    # Write job log
    log_path = OUTPUT_DIR / "submission_jobs.json"
    with open(log_path, "w") as f:
        json.dump(job_log, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Generated batch submission script!")
    print(f"{'='*80}")
    print(f"Total jobs: {total_jobs}")
    print(f"Script: {script_path}")
    print(f"Job log: {log_path}")
    print(f"\nTo run:")
    print(f"  sbatch {script_path}")
    print(f"\nTo check status:")
    print(f"  squeue -u bm3772")
    print(f"  tail -f {OUTPUT_DIR}/batch_submit_*.out")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

