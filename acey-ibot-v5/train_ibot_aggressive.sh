#!/bin/bash
#SBATCH --job-name=ibot_aggressive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_aggressive_%j.out
#SBATCH --error=logs/train_ibot_aggressive_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# AGGRESSIVE CONFIGURATION: High-Capacity LARS Training
# ============================================================================
# Philosophy: Push the limits with larger model and aggressive settings
# 
# Key characteristics:
# - LARS optimizer (better for large batch, higher LR)
# - High learning rate (0.1) for faster convergence
# - Higher mask ratio (0.4) for more challenging MIM task
# - Blockwise masking (structured, harder to predict)
# - Emphasis on MIM loss (1.5x weight) for better local features
# - More local crops (8) for richer augmentation
# - ViT-small architecture (more capacity, still under 100M)
# - Higher weight decay (0.05 -> 0.2) for regularization
# - More aggressive drop path (0.25) for regularization
#
# Best for: Exploring model capacity, faster training, better local features
# ============================================================================

python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints/v5/checkpoints_ibot_aggressive \
    --arch vit_small \
    --optimizer lars \
    --batch_size 128 \
    --lr 0.1 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --weight_decay_end 0.2 \
    --clip_grad 1.0 \
    --epochs 100 \
    --warmup_epochs 10 \
    --drop_path_rate 0.25 \
    --local_crops_number 8 \
    --bottleneck_dim 256 \
    --out_dim 8192 \
    --num_tokens 8192 \
    --mask_ratio 0.4 \
    --mask_type blockwise \
    --block_size 2 \
    --mim_loss_weight 1.5 \
    --cls_loss_weight 1.0 \
    --koleo_weight 0.0 \
    --mim_temp 0.15 \
    --lars_trust_coefficient 0.001 \
    --lars_eta 0.001 \
    --momentum 0.9 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_aggressive/checkpoint_XXXX.pth

