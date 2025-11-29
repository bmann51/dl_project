#!/bin/bash
#SBATCH --job-name=submission
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100_short
#SBATCH --output=logs/submission_%j.out
#SBATCH --error=logs/submission_%j.err



# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Print info
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Set paths
# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints_small/final_checkpoint.pth"
# DATA_DIR="/gpfs/scratch/bm3772/fall2025_finalproject/testset_1/data"  # Update this to your actual data directory
# OUTPUT="submission_small.csv"

# Run submission script
echo ""
echo "Creating submission with checkpoint: $CHECKPOINT"
echo ""


# 1.
python generate_submission_ibot.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v1/checkpoint_0069.pth \
    --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_1/data \
    --output submission_ibot_v1.csv \
    --arch vit_base \
    --resolution 96 \
    --out_dim 4096 \
    --bottleneck_dim 128 \
    --num_tokens 8192 \
    --k 5

python generate_submission_ibot.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v1/checkpoint_0069.pth \
    --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_2/data \
    --output submission_ibot_v1.csv \
    --arch vit_base \
    --resolution 96 \
    --out_dim 4096 \
    --bottleneck_dim 128 \
    --num_tokens 8192 \
    --k 5

# 2
python generate_submission_ibot.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v12/checkpoint_0069.pth \
    --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_1/data \
    --output submission_ibot_v12.csv \
    --arch vit_base \
    --resolution 96 \
    --out_dim 4096 \
    --bottleneck_dim 128 \
    --num_tokens 8192 \
    --k 5

python generate_submission_ibot.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v12/checkpoint_0069.pth \
    --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_2/data \
    --output submission_ibot_v12.csv \
    --arch vit_base \
    --resolution 96 \
    --out_dim 4096 \
    --bottleneck_dim 128 \
    --num_tokens 8192 \
    --k 5


# 3
python generate_submission_ibot.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v3/checkpoint_0059.pth \
    --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_1/data \
    --output submission_ibot_v3.csv \
    --arch vit_base \
    --resolution 96 \
    --out_dim 4096 \
    --bottleneck_dim 128 \
    --num_tokens 8192 \
    --k 5

python generate_submission_ibot.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v3/checkpoint_0059.pth \
    --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_2/data \
    --output submission_ibot_v3.csv \
    --arch vit_base \
    --resolution 96 \
    --out_dim 4096 \
    --bottleneck_dim 128 \
    --num_tokens 8192 \
    --k 5



echo ""
echo "Job finished at $(date)"
echo "Submission file created: $OUTPUT"