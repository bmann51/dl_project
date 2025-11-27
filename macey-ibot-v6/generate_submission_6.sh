#!/bin/bash
#SBATCH --job-name=ibot_submission_v6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100_short
#SBATCH --output=logs/submission_v6_%j.out
#SBATCH --error=logs/submission_v6_%j.err

# -----------------------------------------------
#  Submission script for macey-ibot-v6
# -----------------------------------------------
# Adjust the three variables below.
CHECKPOINT="/path/to/final_checkpoint.pth"   # <- update
DATA_DIR="/path/to/data"                     # <- update
OUTPUT="submission_ibot_v6.csv"
ARCH="vit_small"     # one of {vit_small, vit_base}
RES=96
K=5

module load cuda/11.8
source ~/.bashrc
conda activate ibot_env   # <-- your conda env

echo "Starting submission generation at $(date)"

python generate_submission.py \
  --checkpoint "$CHECKPOINT" \
  --data_dir "$DATA_DIR" \
  --output "$OUTPUT" \
  --arch vit_small \
  --resolution $RES \
  --k $K

echo "Finished at $(date). Output saved to $OUTPUT"
