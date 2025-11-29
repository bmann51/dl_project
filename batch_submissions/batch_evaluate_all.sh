#!/bin/bash
#SBATCH --job-name=batch_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100_short
#SBATCH --output=batch_evaluate_%j.out
#SBATCH --error=batch_evaluate_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Print info
echo "========================================"
echo "Batch Evaluation Job Started"
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi
echo ""

# Base directory
BASE_DIR=$(dirname $(dirname $(realpath $0)))

# Create output directory
mkdir -p batch_submissions

echo "Starting batch evaluation..."
echo ""

# Run batch evaluation script
cd "$BASE_DIR"
python batch_evaluate_all.py

echo ""
echo "========================================"
echo "Batch Evaluation Job Completed"
echo "========================================"
echo "Job finished at: $(date)"
echo "Check results in: batch_submissions/evaluation_results.csv"
echo "========================================"

