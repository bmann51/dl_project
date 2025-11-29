#!/bin/bash
#SBATCH --job-name=extract_acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=0:30:00
#SBATCH --partition=a100_short
#SBATCH --output=logs/extract_acc_%j.out
#SBATCH --error=logs/extract_acc_%j.err

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

echo "Running extract_accuracies.py..."
echo "Base directory: $BASE_DIR"
echo "Working directory: $(pwd)"
echo ""

# Verify the script exists
if [ ! -f "$BASE_DIR/extract_accuracies.py" ]; then
    echo "ERROR: extract_accuracies.py not found in $BASE_DIR"
    echo "Please make sure you're running sbatch from the project root directory"
    exit 1
fi

# Run the extractor (parse all logs by default)
python "$BASE_DIR/extract_accuracies.py" --all

echo ""
echo "Extraction Complete"
echo "Results saved to:"
echo "  - batch_submissions/accuracy_summary.csv"
echo "  - batch_submissions/accuracy_full_results.csv"
