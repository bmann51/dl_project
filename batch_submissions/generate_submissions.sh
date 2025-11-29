#!/bin/bash
#SBATCH --job-name=gen_submissions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --partition=a100_short
#SBATCH --output=logs/gen_submissions_%j.out
#SBATCH --error=logs/gen_submissions_%j.err

# Parse command-line arguments
K_VALUE=5  # Default k value
while [[ $# -gt 0 ]]; do
    case $1 in
        --k)
            K_VALUE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch generate_submissions.sh [--k K_VALUE]"
            exit 1
            ;;
    esac
done

# Also check environment variable (for SLURM --export)
if [ -n "$K" ]; then
    K_VALUE="$K"
fi

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

echo "Running generate_batch_submissions.py..."
echo "Base directory: $BASE_DIR"
echo "Working directory: $(pwd)"
echo "Using k=$K_VALUE for KNN"
echo ""

# Verify the script exists
if [ ! -f "$BASE_DIR/generate_batch_submissions.py" ]; then
    echo "ERROR: generate_batch_submissions.py not found in $BASE_DIR"
    echo "Please make sure you're running sbatch from the project root directory"
    exit 1
fi

# Run the generator
python "$BASE_DIR/generate_batch_submissions.py" --k "$K_VALUE"

echo ""
echo "Generation Complete"
echo "Next steps:"
echo "  1. Review: batch_submissions/batch_submit_all.sh"
echo "  2. Submit: sbatch batch_submissions/batch_submit_all.sh"
