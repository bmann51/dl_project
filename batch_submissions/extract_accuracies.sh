#!/bin/bash
#SBATCH --job-name=extract_acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=0:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/extract_acc_%j.out
#SBATCH --error=logs/extract_acc_%j.err

# Create logs directory
mkdir -p logs

# Base directory (parent of batch_submissions)
SCRIPT_DIR=$(dirname $(realpath $0))
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd "$BASE_DIR"

echo "Running extract_accuracies.py..."
echo ""

# Run the extractor (parse all logs by default)
python extract_accuracies.py --all

echo ""
echo "Extraction Complete"
echo "Results saved to:"
echo "  - batch_submissions/accuracy_summary.csv"
echo "  - batch_submissions/accuracy_full_results.csv"
