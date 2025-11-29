#!/usr/bin/env python3
"""
Generate batch submission script for all model variations

This script:
1. Discovers all checkpoints in configured directories
2. Generates a SLURM batch script to run all submissions
3. Creates a job manifest for tracking

Usage:
    python generate_batch_submissions.py
    sbatch batch_submissions/batch_submit_all.sh
"""

import os
import json
from pathlib import Path
from typing import List, Dict

# Configuration
BASE_DATA_PATH = "/gpfs/scratch/bm3772/fall2025_finalproject"
BASE_CHECKPOINT_PATH = "/gpfs/scratch/bm3772/checkpoints"  # Updated: checkpoints are in /checkpoints subdirectory
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "batch_submissions"
OUTPUT_DIR.mkdir(exist_ok=True)

TESTSETS = ["testset_1", "testset_2", "testset_3"]


def find_checkpoints(checkpoint_dir: str, max_checkpoints: int = 5) -> List[str]:
    """Find all checkpoint files in a directory, prioritizing final_checkpoint.pth"""
    checkpoints = []
    base = Path(checkpoint_dir)
    
    if not base.exists():
        return checkpoints
    
    # Look for final_checkpoint.pth first
    final_ckpt = base / "final_checkpoint.pth"
    if final_ckpt.exists():
        checkpoints.append(str(final_ckpt))
    
    # Find numbered checkpoints (checkpoint_XXXX.pth)
    numbered_ckpts = sorted(base.glob("checkpoint_*.pth"), reverse=True)
    for ckpt in numbered_ckpts:
        if str(ckpt) not in checkpoints:
            checkpoints.append(str(ckpt))
    
    return checkpoints[:max_checkpoints]


def read_checkpoint_args(checkpoint_dir: str) -> Dict:
    """Read args.json from checkpoint directory to get model configuration"""
    args_file = Path(checkpoint_dir) / "args.json"
    if args_file.exists():
        try:
            with open(args_file, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def find_checkpoint_directories(pattern: str, base_path: Path = None) -> List[Path]:
    """
    Find checkpoint directories matching pattern, checking version subdirectories.
    
    Structure: BASE_CHECKPOINT_PATH/v*/checkpoints_ibot_*_*/
    Example: /gpfs/scratch/bm3772/checkpoints/v8/checkpoints_ibot_conservative_8/
    
    Looks in:
    - BASE_CHECKPOINT_PATH/v*/pattern (v2, v5, v6, v7, v8, etc.)
    - BASE_CHECKPOINT_PATH/pattern (fallback for non-versioned)
    """
    if base_path is None:
        base_path = Path(BASE_CHECKPOINT_PATH)
    
    found_dirs = []
    
    # Primary: Check version subdirectories (v2, v5, v6, v7, v8, etc.)
    # Structure: checkpoints/v8/checkpoints_ibot_conservative_8/
    for version_dir in sorted(base_path.glob("v*")):
        if version_dir.is_dir():
            # Look for pattern inside version directory
            found_dirs.extend(version_dir.glob(pattern))
            
            # Also try variant-specific patterns if pattern is generic
            if "*" in pattern:
                # For patterns like "checkpoints_ibot_*_v8", try direct variant names
                version_num = version_dir.name[1:] if version_dir.name.startswith("v") else version_dir.name
                variant_patterns = [
                    f"checkpoints_ibot_conservative_{version_num}",
                    f"checkpoints_ibot_feedback_{version_num}",
                    f"checkpoints_ibot_aggressive_{version_num}",
                ]
                for vp in variant_patterns:
                    found_dirs.extend(version_dir.glob(vp))
    
    # Fallback: Direct match in base (for non-versioned checkpoints)
    found_dirs.extend(base_path.glob(pattern))
    
    # Also check parent directory (for checkpoints at /gpfs/scratch/bm3772/checkpoints_*)
    parent_base = base_path.parent
    if parent_base.exists():
        found_dirs.extend(parent_base.glob(pattern))
        for version_dir in parent_base.glob("v*"):
            if version_dir.is_dir():
                found_dirs.extend(version_dir.glob(pattern))
    
    # Return unique directories
    return list(set([d for d in found_dirs if d.is_dir()]))


def get_model_configs() -> Dict:
    """Get all model configurations"""
    return {
        # Acey DINO
        "acey-dino-tiny": {
            "folder": "acey-submission-folder",
            "script": "generate_submission.py",
            "checkpoint_dirs": ["checkpoints_vit_tiny"],
            "args": {"arch": "vit_tiny", "resolution": 96, "k": 20, "out_dim": 8192, "bottleneck_dim": 256},
        },
        "acey-dino-small": {
            "folder": "acey-submission-folder",
            "script": "generate_submission.py",
            "checkpoint_dirs": ["checkpoints_vit_small"],
            "args": {"arch": "vit_small", "resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128},
        },
        "acey-dino-base": {
            "folder": "acey-submission-folder",
            "script": "generate_submission.py",
            "checkpoint_dirs": ["checkpoints_vit_base"],
            "args": {"arch": "vit_base", "resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128},
        },
        
        # Acey iBOT v1
        "acey-ibot-v1": {
            "folder": "acey-ibot-folder",
            "script": "generate_submission_ibot.py",
            "checkpoint_dirs": ["checkpoints_ibot_vit_tiny_v1"],
            "args": {"arch": "vit_tiny", "resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        
        # Acey iBOT v2
        "acey-ibot-v2": {
            "folder": "acey-ibot-v2",
            "script": "generate_submission_ibot.py",
            "checkpoint_dirs": ["checkpoints_ibot_vit_tiny_v2"],
            "args": {"arch": "vit_tiny", "resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        
        # Acey iBOT v3 (with variants: conservative, feedback, aggressive)
        "acey-ibot-v3": {
            "folder": "acey-ibot-v3",
            "script": "generate_submission_ibot.py",
            "checkpoint_pattern": "checkpoints_ibot_*_v3",  # Will discover variants
            "use_variants": True,
            "default_args": {"resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        
        # Acey iBOT v4
        "acey-ibot-v4-tiny": {
            "folder": "acey-ibot-v4",
            "script": "generate_submission_ibot.py",
            "checkpoint_dirs": ["checkpoints_ibot_vit_tiny_v4"],
            "args": {"arch": "vit_tiny", "resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        "acey-ibot-v4-resnet": {
            "folder": "acey-ibot-v4",
            "script": "generate_submission_ibot.py",
            "checkpoint_dirs": ["checkpoints_ibot_resnet_v4"],
            "args": {"arch": "resnet50", "resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        "acey-ibot-v4-convnext": {
            "folder": "acey-ibot-v4",
            "script": "generate_submission_ibot.py",
            "checkpoint_dirs": ["checkpoints_ibot_convnext_v4"],
            "args": {"arch": "convnext_tiny", "resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        
        # Acey iBOT v5 (with variants)
        "acey-ibot-v5": {
            "folder": "acey-ibot-v5",
            "script": "generate_submission_ibot.py",
            "checkpoint_pattern": "checkpoints_ibot_*_v5",
            "use_variants": True,
            "default_args": {"resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        
        # Acey iBOT v7 (with variants)
        "acey-ibot-v7": {
            "folder": "acey-ibot-v7",
            "script": "generate_submission_ibot.py",
            "checkpoint_pattern": "checkpoints_ibot_*_v7",
            "use_variants": True,
            "default_args": {"resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        
        # Acey iBOT v8 (with variants)
        "acey-ibot-v8": {
            "folder": "acey-ibot-v8",
            "script": "generate_submission_ibot.py",
            "checkpoint_pattern": "checkpoints_ibot_*_v8",
            "use_variants": True,
            "default_args": {"resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
        
        # Brian DINO
        "brian-dino-base": {
            "folder": "brian-folder",
            "script": "generate_submission2.py",
            "checkpoint_dirs": ["checkpoints_base"],
            "args": {"arch": "vit_base", "resolution": 96, "k": 5, "out_dim": 4096, "bottleneck_dim": 128},
        },
        "brian-dino-small": {
            "folder": "brian-folder",
            "script": "generate_submission2.py",
            "checkpoint_dirs": ["checkpoints_small"],
            "args": {"arch": "vit_small", "resolution": 96, "k": 5, "out_dim": 4096, "bottleneck_dim": 128},
        },
        
        # Macey iBOT v6
        "macey-ibot-v6": {
            "folder": "macey-ibot-v6",
            "script": "generate_submission.py",
            "checkpoint_dirs": ["checkpoints_v6"],
            "args": {"arch": "vit_tiny", "resolution": 96, "k": 20, "out_dim": 4096, "bottleneck_dim": 128, "num_tokens": 8192},
        },
    }


def get_args_from_checkpoint(checkpoint_path: str, default_args: Dict) -> Dict:
    """Extract args from checkpoint directory's args.json, merge with defaults"""
    checkpoint_dir = Path(checkpoint_path).parent
    args_file = checkpoint_dir / "args.json"
    
    args = default_args.copy()
    
    if args_file.exists():
        try:
            with open(args_file, "r") as f:
                ckpt_args = json.load(f)
                
                # Map checkpoint args to submission args
                if "arch" in ckpt_args:
                    args["arch"] = ckpt_args["arch"]
                if "out_dim" in ckpt_args:
                    args["out_dim"] = ckpt_args["out_dim"]
                if "bottleneck_dim" in ckpt_args:
                    args["bottleneck_dim"] = ckpt_args["bottleneck_dim"]
                if "num_tokens" in ckpt_args and "num_tokens" in args:
                    args["num_tokens"] = ckpt_args["num_tokens"]
        except Exception as e:
            print(f"    Warning: Could not read args.json from {checkpoint_dir}: {e}")
    
    return args


def generate_submission_command(
    model_name: str,
    config: Dict,
    checkpoint: str,
    testset: str,
    output_name: str,
    output_path: str = None
) -> str:
    """Generate a single submission command"""
    folder = config["folder"]
    script = config["script"]
    
    # Get args - use checkpoint args if available, otherwise use config args
    if "args" in config:
        args = config["args"]
    elif "default_args" in config:
        args = get_args_from_checkpoint(checkpoint, config["default_args"])
    else:
        args = {}
    
    # Use provided output_path or construct it
    if output_path is None:
        output_path = f"{OUTPUT_DIR}/{output_name}"
    
    cmd_parts = [
        f"cd {PROJECT_ROOT / folder}",
        "&&",
        "python", script,
        "--checkpoint", checkpoint,
        "--data_dir", f"{BASE_DATA_PATH}/{testset}/data",
        "--output", output_path,
    ]
    
    # Add all args
    for key, value in args.items():
        cmd_parts.extend([f"--{key}", str(value)])
    
    return " ".join(cmd_parts)


def main():
    """Main function to generate batch submission script"""
    
    print("=" * 80)
    print("Batch Submission Generator")
    print("=" * 80)
    print()
    
    model_configs = get_model_configs()
    all_jobs = []
    
    # Discover checkpoints and generate jobs
    print("Discovering checkpoints...")
    for model_name, config in model_configs.items():
        print(f"\n{model_name}:")
        found_any = False
        
        # Handle variant-based discovery (v3, v5, v7, v8)
        if config.get("use_variants", False):
            pattern = config.get("checkpoint_pattern", "")
            if pattern:
                # Find directories using flexible search
                matching_dirs = find_checkpoint_directories(pattern)
                
                # Also try variant-specific patterns if main pattern doesn't match
                if not matching_dirs:
                    # Extract version from model name (e.g., "v8" from "acey-ibot-v8")
                    version = model_name.split("-")[-1] if "-" in model_name else ""
                    if version and version.startswith("v"):
                        version_num = version[1:]  # "8" from "v8"
                        # Try patterns like: checkpoints_ibot_conservative_8
                        variant_patterns = [
                            f"checkpoints_ibot_conservative_{version_num}",
                            f"checkpoints_ibot_feedback_{version_num}",
                            f"checkpoints_ibot_aggressive_{version_num}",
                        ]
                        for vp in variant_patterns:
                            matching_dirs.extend(find_checkpoint_directories(vp))
                        
                        # Also try with "v" prefix in pattern
                        variant_patterns_v = [
                            f"checkpoints_ibot_conservative_v{version_num}",
                            f"checkpoints_ibot_feedback_v{version_num}",
                            f"checkpoints_ibot_aggressive_v{version_num}",
                        ]
                        for vp in variant_patterns_v:
                            matching_dirs.extend(find_checkpoint_directories(vp))
                
                for ckpt_dir in sorted(set(matching_dirs)):
                    dir_name = ckpt_dir.name
                    
                    # Extract variant name (conservative, feedback, aggressive, etc.)
                    if "conservative" in dir_name:
                        variant_name = "conservative"
                    elif "feedback" in dir_name:
                        variant_name = "feedback"
                    elif "aggressive" in dir_name:
                        variant_name = "aggressive"
                    else:
                        # Try to extract from pattern
                        parts = dir_name.replace("checkpoints_ibot_", "").split("_")
                        variant_name = parts[0] if parts else "unknown"
                    
                    checkpoints = find_checkpoints(str(ckpt_dir), max_checkpoints=3)
                    if checkpoints:
                        found_any = True
                        print(f"  {dir_name}: Found {len(checkpoints)} checkpoints")
                        
                        # Read args from checkpoint directory to get architecture
                        ckpt_args = read_checkpoint_args(str(ckpt_dir))
                        arch = ckpt_args.get("arch", "vit_tiny")
                        
                        for ckpt in checkpoints:
                            print(f"    - {Path(ckpt).name} (arch: {arch})")
                            
                            # Create variant-specific model name
                            variant_model_name = f"{model_name}-{variant_name}"
                            
                            # Generate jobs for each testset
                            for testset in TESTSETS:
                                ckpt_name = Path(ckpt).stem
                                output_name = f"submission_{variant_model_name}_{testset}_{ckpt_name}.csv"
                                output_path = f"{OUTPUT_DIR}/{output_name}"
                                
                                # Create config with checkpoint-specific args
                                variant_config = config.copy()
                                variant_config["args"] = get_args_from_checkpoint(ckpt, config["default_args"])
                                
                                all_jobs.append({
                                    "model": variant_model_name,
                                    "checkpoint": ckpt,
                                    "testset": testset,
                                    "output": output_name,
                                    "output_path": output_path,
                                    "config": variant_config,
                                })
        
        # Handle regular checkpoint directories
        elif "checkpoint_dirs" in config:
            for checkpoint_dir in config["checkpoint_dirs"]:
                # Try flexible search (checks v* subdirectories)
                matching_dirs = find_checkpoint_directories(checkpoint_dir)
                
                if not matching_dirs:
                    # Fallback: try direct path in base
                    full_path = Path(BASE_CHECKPOINT_PATH) / checkpoint_dir
                    if full_path.exists():
                        matching_dirs = [full_path]
                    
                    # Also try in version subdirectories
                    base_path = Path(BASE_CHECKPOINT_PATH)
                    for version_dir in base_path.glob("v*"):
                        if version_dir.is_dir():
                            version_path = version_dir / checkpoint_dir
                            if version_path.exists():
                                matching_dirs.append(version_path)
                    
                    # Also try in parent (for checkpoints at /gpfs/scratch/bm3772/checkpoints_*)
                    parent_path = Path(BASE_CHECKPOINT_PATH).parent / checkpoint_dir
                    if parent_path.exists():
                        matching_dirs.append(parent_path)
                
                for ckpt_dir in matching_dirs:
                    checkpoints = find_checkpoints(str(ckpt_dir), max_checkpoints=3)
                    
                    if checkpoints:
                        found_any = True
                        print(f"  {ckpt_dir.name}: Found {len(checkpoints)} checkpoints")
                        for ckpt in checkpoints:
                            print(f"    - {Path(ckpt).name}")
                            
                            # Generate jobs for each testset
                            for testset in TESTSETS:
                                ckpt_name = Path(ckpt).stem
                                output_name = f"submission_{model_name}_{testset}_{ckpt_name}.csv"
                                output_path = f"{OUTPUT_DIR}/{output_name}"
                                
                                all_jobs.append({
                                    "model": model_name,
                                    "checkpoint": ckpt,
                                    "testset": testset,
                                    "output": output_name,
                                    "output_path": output_path,
                                    "config": config,
                                })
                    else:
                        print(f"  {ckpt_dir.name}: No checkpoints found")
        
        if not found_any:
            print(f"  ⚠️  No checkpoints found for {model_name}")
    
    print(f"\n{'='*80}")
    print(f"Total jobs to run: {len(all_jobs)}")
    print(f"{'='*80}\n")
    
    # Generate SLURM script
    script_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=batch_submit",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64GB",
        "#SBATCH --time=24:00:00",
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
        "echo '========================================'",
        "echo 'Batch Submission Job Started'",
        "echo '========================================'",
        "echo 'Job started at: $(date)'",
        "echo 'Running on node: $(hostname)'",
        "echo 'GPU info:'",
        "nvidia-smi",
        "echo ''",
        "",
        f"# Create output directory",
        f"mkdir -p {OUTPUT_DIR}",
        "",
        "# Counter for tracking",
        "SUCCESS=0",
        "FAILED=0",
        "",
        "echo 'Starting batch submissions...'",
        "echo ''",
        "",
    ]
    
    # Add commands for each job
    for i, job in enumerate(all_jobs, 1):
        model_name = job["model"]
        checkpoint = job["checkpoint"]
        testset = job["testset"]
        output_name = job["output"]
        config = job["config"]
        
        # Get output path
        output_path = job.get("output_path", f"{OUTPUT_DIR}/{output_name}")
        
        cmd = generate_submission_command(model_name, config, checkpoint, testset, output_name, output_path)
        
        script_lines.extend([
            f"# Job {i}/{len(all_jobs)}: {model_name} - {testset}",
            f"echo ''",
            f"echo '--- Job {i}/{len(all_jobs)}: {model_name} on {testset} ---'",
            f"echo 'Checkpoint: {Path(checkpoint).name}'",
            f"echo 'Output: {output_name}'",
            cmd,
            "if [ $? -eq 0 ]; then",
            f"    echo '✓ SUCCESS: {output_name}'",
            "    SUCCESS=$((SUCCESS + 1))",
            "else",
            f"    echo '✗ FAILED: {output_name}'",
            "    FAILED=$((FAILED + 1))",
            "fi",
            "echo ''",
            "",
        ])
    
    # Add summary
    script_lines.extend([
        "echo ''",
        "echo '========================================'",
        "echo 'Batch Submission Job Completed'",
        "echo '========================================'",
        "echo 'Job finished at: $(date)'",
        "echo \"Total jobs: $((SUCCESS + FAILED))\"",
        "echo \"Successful: $SUCCESS\"",
        "echo \"Failed: $FAILED\"",
        "echo '========================================'",
    ])
    
    # Write script
    script_path = OUTPUT_DIR / "batch_submit_all.sh"
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines))
    
    os.chmod(script_path, 0o755)
    
    # Write job manifest
    manifest_path = OUTPUT_DIR / "submission_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_jobs, f, indent=2)
    
    print(f"✓ Generated batch submission script: {script_path}")
    print(f"✓ Generated job manifest: {manifest_path}")
    print(f"\nTo run:")
    print(f"  sbatch {script_path}")
    print(f"\nTo check status:")
    print(f"  squeue -u bm3772")
    print(f"  tail -f {OUTPUT_DIR}/batch_submit_*.out")
    print(f"\nOutput files will be in each model's folder")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

