#!/bin/bash
#SBATCH --job-name=9_higher_mask_ibot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_higher_mask_9_%j.out
#SBATCH --error=logs/train_ibot_higher_mask_9_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# iBOT v9 Higher Mask: Variant with Aggressive Masking
# ============================================================================
# Philosophy: Clean, competition-ready implementation with higher mask ratio
# 
# Key characteristics:
# - ViT-Base/16 backbone (~86M params, under 100M limit)
# - Pure cross-entropy loss (no entropy/focal terms)
# - Shared projection head for CLS and patch tokens
# - Two global crops only (no local crops)
# - AdamW optimizer with cosine LR schedule
# - Teacher centering + temperature scheduling
# - Explicit image-space masking
# - **Higher mask ratio (0.4) for more challenging MIM task**
#
# Best for: Exploring effect of mask ratio, potentially better local features
# ============================================================================

python train_ibot.py \
    --data_root /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints/v9/v9_higher_mask_ibot_checkpoint \
    --epochs 400 \
    --batch_size 256 \
    --lr 5e-4 \
    --weight_decay 0.04 \
    --mask_ratio 0.4 \
    --device cuda
    # --resume /gpfs/scratch/bm3772/checkpoints/v9/v9_higher_mask_ibot_checkpoint/ibot_epochXXX.pt

