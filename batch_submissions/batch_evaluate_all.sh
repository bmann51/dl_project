#!/bin/bash
#SBATCH --job-name=batch_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100_short
#SBATCH --output=logs/batch_eval_%j.out
#SBATCH --error=logs/batch_eval_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Base directory
BASE_DIR=$(dirname $(dirname $(realpath $0)))

# Create output directories
mkdir -p batch_submissions

echo "Starting batch evaluation..."
echo ""

# Run batch evaluation script
cd "$BASE_DIR"
python batch_evaluate_all.py

echo ""
echo "Batch Evaluation Complete"
echo "Check results in: batch_submissions/evaluation_results.csv"
