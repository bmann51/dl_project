#!/bin/bash
#SBATCH --job-name=extract_acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=0:30:00
#SBATCH --partition=cpu
#SBATCH --output=extract_accuracies_%j.out
#SBATCH --error=extract_accuracies_%j.err

# Print info
echo "========================================"
echo "Extract Accuracies from Logs"
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Base directory (parent of batch_submissions)
SCRIPT_DIR=$(dirname $(realpath $0))
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd "$BASE_DIR"

echo "Running extract_accuracies.py..."
echo ""

# Run the extractor (parse all logs by default)
python extract_accuracies.py --all

echo ""
echo "========================================"
echo "Extraction Complete"
echo "========================================"
echo "Job finished at: $(date)"
echo ""
echo "Results saved to:"
echo "  - batch_submissions/accuracy_summary.csv"
echo "  - batch_submissions/accuracy_full_results.csv"
echo ""
echo "View results:"
echo "  cat batch_submissions/accuracy_summary.csv"
echo "========================================"

