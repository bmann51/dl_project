#!/usr/bin/env python3
"""
Launch multiple DINO training experiments with different hyperparameter configurations.
Supports both local execution and SLURM job submission.
"""
import json
import os
import subprocess
import sys
import argparse
from pathlib import Path


def load_config(config_path):
    """Load hyperparameter configurations from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def build_command(exp_name, exp_config, shared_args, use_slurm=False):
    """Build training command for a single experiment."""
    cmd = []
    
    if use_slurm:
        # SLURM submission
        cmd = ['sbatch', '--job-name', f'dino_{exp_name}', '--parsable']
        # Add SLURM directives from shared_args if present
        if 'slurm_directives' in shared_args:
            for directive, value in shared_args['slurm_directives'].items():
                cmd.extend([f'--{directive}', str(value)])
        cmd.append('train_job.sh')
        # Training command will be in train_job.sh
        return cmd, None
    else:
        # Local execution
        cmd = ['python', 'train_dino.py']
    
    # Add shared arguments
    cmd.extend(['--data_path', shared_args.get('data_path', '/mnt/user-data/uploads/pretrain/')])
    output_dir = os.path.join(shared_args.get('output_dir_base', './experiments'), exp_name)
    cmd.extend(['--output_dir', output_dir])
    cmd.extend(['--num_workers', str(shared_args.get('num_workers', 4))])
    cmd.extend(['--save_freq', str(shared_args.get('save_freq', 10))])
    
    if shared_args.get('use_fp16', True):
        cmd.append('--use_fp16')
    
    # Add experiment-specific arguments
    for key, value in exp_config.items():
        if key == 'use_fp16':  # Handle boolean
            if value:
                cmd.append('--use_fp16')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    return cmd, output_dir


def launch_local(experiments, shared_args, max_parallel=None):
    """Launch experiments locally (optionally in parallel)."""
    processes = []
    output_dirs = []
    
    print("=" * 80)
    print("Launching Local Training Experiments")
    print("=" * 80)
    
    for exp_name, exp_config in experiments.items():
        cmd, output_dir = build_command(exp_name, exp_config, shared_args, use_slurm=False)
        output_dirs.append((exp_name, output_dir))
        
        print(f"\nExperiment: {exp_name}")
        print(f"  Output dir: {output_dir}")
        print(f"  Command: {' '.join(cmd)}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Launch process
        log_file = open(os.path.join(output_dir, 'training.log'), 'w')
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append((exp_name, process, log_file))
        
        if max_parallel and len(processes) >= max_parallel:
            # Wait for one to finish before starting next
            print(f"\nWaiting for a process to finish (max_parallel={max_parallel})...")
            while len(processes) >= max_parallel:
                for i, (name, proc, log_f) in enumerate(processes):
                    if proc.poll() is not None:  # Process finished
                        print(f"  {name} finished with code {proc.returncode}")
                        log_f.close()
                        processes.pop(i)
                        break
                else:
                    import time
                    time.sleep(1)
    
    # Wait for all processes
    print("\n" + "=" * 80)
    print("Waiting for all experiments to complete...")
    print("=" * 80)
    
    for exp_name, process, log_file in processes:
        return_code = process.wait()
        log_file.close()
        status = "✓ SUCCESS" if return_code == 0 else "✗ FAILED"
        print(f"{status}: {exp_name} (exit code: {return_code})")
    
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    for exp_name, output_dir in output_dirs:
        print(f"  {exp_name}: {output_dir}")


def launch_slurm(experiments, shared_args):
    """Launch experiments as SLURM jobs."""
    print("=" * 80)
    print("Launching SLURM Training Jobs")
    print("=" * 80)
    
    job_ids = []
    
    # Create a temporary script for each experiment
    for exp_name, exp_config in experiments.items():
        output_dir = os.path.join(shared_args.get('output_dir_base', './experiments'), exp_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create experiment-specific training script
        script_path = f'train_job_{exp_name}.sh'
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=dino_" + exp_name + "\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --cpus-per-task=8\n")
            f.write("#SBATCH --ntasks-per-node=1\n")
            f.write("#SBATCH --gres=gpu:a100:1\n")
            # Set time limit based on experiment (ViT-Base needs more time)
            if exp_config.get('arch') == 'vit_base':
                time_limit = "48:00:00"  # 2 days for ViT-Base
            else:
                time_limit = "36:00:00"  # 1.5 days for ViT-Small
            f.write(f"#SBATCH --time={time_limit}\n")
            f.write("#SBATCH --mem=64G\n")
            f.write("#SBATCH --partition=a100_short\n")
            f.write(f"#SBATCH --output={output_dir}/slurm_%j.out\n")
            f.write(f"#SBATCH --error={output_dir}/slurm_%j.err\n\n")
            f.write("# Load modules\n")
            f.write("module load cuda/11.8\n\n")
            f.write("# Activate environment\n")
            f.write("source ~/.bashrc\n")
            f.write("conda activate dino_new\n\n")
            
            # Build training command
            cmd_parts = ['python', 'train_dino.py']
            cmd_parts.extend(['--data_path', shared_args.get('data_path', '/mnt/user-data/uploads/pretrain/')])
            cmd_parts.extend(['--output_dir', output_dir])
            cmd_parts.extend(['--num_workers', str(shared_args.get('num_workers', 4))])
            cmd_parts.extend(['--save_freq', str(shared_args.get('save_freq', 10))])
            
            if shared_args.get('use_fp16', True):
                cmd_parts.append('--use_fp16')
            
            for key, value in exp_config.items():
                if key == 'use_fp16':
                    if value:
                        cmd_parts.append('--use_fp16')
                else:
                    cmd_parts.extend([f'--{key}', str(value)])
            
            f.write(' '.join(cmd_parts) + '\n')
        
        os.chmod(script_path, 0o755)
        
        # Submit job
        result = subprocess.run(
            ['sbatch', '--parsable', script_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip()
            job_ids.append((exp_name, job_id))
            print(f"  Submitted {exp_name}: Job ID {job_id}")
        else:
            print(f"  ✗ Failed to submit {exp_name}: {result.stderr}")
    
    print("\n" + "=" * 80)
    print("Submitted Jobs")
    print("=" * 80)
    for exp_name, job_id in job_ids:
        print(f"  {exp_name}: {job_id}")
    print(f"\nMonitor with: squeue -u $USER")
    print(f"Cancel all with: scancel {' '.join([j[1] for j in job_ids])}")


def main():
    parser = argparse.ArgumentParser(description='Launch multiple DINO training experiments')
    parser.add_argument('--config', default='hyperparameter_configs.json',
                       help='Path to hyperparameter configuration JSON file')
    parser.add_argument('--mode', choices=['local', 'slurm'], default='local',
                       help='Execution mode: local or slurm')
    parser.add_argument('--max_parallel', type=int, default=None,
                       help='Maximum number of parallel local processes (None = all at once)')
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='Specific experiments to run (default: all)')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    experiments = config['experiments']
    shared_args = config['shared_args']
    
    # Filter experiments if specified
    if args.experiments:
        experiments = {k: v for k, v in experiments.items() if k in args.experiments}
        if not experiments:
            print(f"Error: No matching experiments found. Available: {list(config['experiments'].keys())}")
            sys.exit(1)
    
    print(f"Loaded {len(experiments)} experiment(s) from {args.config}")
    print(f"Mode: {args.mode}")
    
    # Launch experiments
    if args.mode == 'slurm':
        launch_slurm(experiments, shared_args)
    else:
        launch_local(experiments, shared_args, max_parallel=args.max_parallel)


if __name__ == '__main__':
    main()

