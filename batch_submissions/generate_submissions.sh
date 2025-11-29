#!/bin/bash
#SBATCH --job-name=gen_submissions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=cpu
#SBATCH --output=generate_submissions_%j.out
#SBATCH --error=generate_submissions_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Print info
echo "========================================"
echo "Generate Batch Submissions"
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Base directory (parent of batch_submissions)
SCRIPT_DIR=$(dirname $(realpath $0))
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd "$BASE_DIR"

# Create output directory
mkdir -p batch_submissions

echo "Running generate_batch_submissions.py..."
echo ""

# Run the generator
python generate_batch_submissions.py

echo ""
echo "========================================"
echo "Generation Complete"
echo "========================================"
echo "Job finished at: $(date)"
echo ""
echo "Next steps:"
echo "  1. Review: batch_submissions/batch_submit_all.sh"
echo "  2. Submit: sbatch batch_submissions/batch_submit_all.sh"
echo "========================================"

