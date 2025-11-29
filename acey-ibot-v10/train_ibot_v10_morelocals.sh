#!/bin/bash
#SBATCH --job-name=10_morelocals_ibot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_v10_morelocals_%j.out
#SBATCH --error=logs/train_ibot_v10_morelocals_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# iBOT v10 - More-Local-Crops Variant
# ============================================================================
# Philosophy: Multi-crop with enhanced local view diversity
# 
# Key characteristics:
# - ViT-Base/16 backbone (~86M params, under 100M limit)
# - Multi-crop: 2 global (96x96) + 6 local (64x64) crops
# - CLS loss: teacher global -> student global+local (more local views)
# - MIM loss: only on global crops (patch-level token prediction)
# - Pure cross-entropy loss (no entropy/focal terms)
# - Shared projection head for CLS and patch tokens
# - AdamW optimizer with cosine LR schedule
# - Teacher centering + temperature scheduling
# - Explicit image-space masking (global crops only)
#
# Changes vs base v10:
# - More locals: 6 local crops instead of 4
# - Batch size: 160 (to help with memory)
# - 250 epochs
#
# Best for: Enhanced multi-scale feature learning with more local views
# ============================================================================

python train_ibot_v10.py \
    --data_root /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints/v10/v10_ibot_morelocals \
    --epochs 250 \
    --batch_size 160 \
    --lr 5e-4 \
    --weight_decay 0.04 \
    --mask_ratio 0.3 \
    --num_local_crops 6 \
    --device cuda
    # --resume /gpfs/scratch/bm3772/checkpoints/v10/v10_ibot_morelocals/ibot_v10_epochXXX.pt

