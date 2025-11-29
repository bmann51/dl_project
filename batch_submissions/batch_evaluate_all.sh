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

# Base directory - use SLURM_SUBMIT_DIR if available (where sbatch was run from)
# Otherwise fall back to script location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    BASE_DIR="$SLURM_SUBMIT_DIR"
else
    # Get directory where this script is located, then go up one level
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BASE_DIR="$(dirname "$SCRIPT_DIR")"
fi

# Change to base directory first
cd "$BASE_DIR" || exit 1

# Create logs directory in base directory
mkdir -p logs

# Create output directories in base directory
mkdir -p batch_submissions

echo "Starting batch evaluation..."
echo "Base directory: $BASE_DIR"
echo "Working directory: $(pwd)"
echo ""

# Verify the script exists
if [ ! -f "$BASE_DIR/batch_evaluate_all.py" ]; then
    echo "ERROR: batch_evaluate_all.py not found in $BASE_DIR"
    echo "Please make sure you're running sbatch from the project root directory"
    exit 1
fi

# Run batch evaluation script
python "$BASE_DIR/batch_evaluate_all.py"

echo ""
echo "Batch Evaluation Complete"
echo "Check results in: batch_submissions/evaluation_results.csv"
