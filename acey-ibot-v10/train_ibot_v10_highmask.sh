#!/bin/bash
#SBATCH --job-name=10_highmask_ibot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_v10_highmask_%j.out
#SBATCH --error=logs/train_ibot_v10_highmask_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# iBOT v10 - High-Mask Multi-Crop Variant
# ============================================================================
# Philosophy: Multi-crop with stronger MIM signal through higher mask ratio
# 
# Key characteristics:
# - ViT-Base/16 backbone (~86M params, under 100M limit)
# - Multi-crop: 2 global (96x96) + 4 local (64x64) crops
# - Higher mask ratio: 0.45 (vs 0.3 in base v10)
# - CLS loss: teacher global -> student global+local
# - MIM loss: only on global crops (patch-level token prediction)
# - Pure cross-entropy loss (no entropy/focal terms)
# - Shared projection head for CLS and patch tokens
# - AdamW optimizer with cosine LR schedule
# - Teacher centering + temperature scheduling
# - Explicit image-space masking (global crops only)
#
# Changes vs base v10:
# - Higher mask ratio: 0.45 (stronger MIM signal)
# - Slightly smaller batch: 160
# - Slightly lower LR: 4e-4
# - 260 epochs
#
# Best for: Stronger MIM learning signal, more challenging reconstruction
# ============================================================================

python train_ibot_v10.py \
    --data_root /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints/v10/v10_ibot_highmask \
    --epochs 260 \
    --batch_size 160 \
    --lr 4e-4 \
    --weight_decay 0.04 \
    --mask_ratio 0.45 \
    --num_local_crops 4 \
    --device cuda
    # --resume /gpfs/scratch/bm3772/checkpoints/v10/v10_ibot_highmask/ibot_v10_epochXXX.pt

