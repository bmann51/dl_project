#!/bin/bash
#SBATCH --job-name=ibot_conservative
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_conservative_%j.out
#SBATCH --error=logs/train_ibot_conservative_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# CONSERVATIVE CONFIGURATION: Stable AdamW Training
# ============================================================================
# Philosophy: Stability and reliability over aggressive optimization
# 
# Key characteristics:
# - AdamW optimizer (more stable than LARS for this setup)
# - Low learning rate (0.0003) for stable convergence
# - Lower mask ratio (0.3) for easier learning task
# - Random masking (simpler, more diverse patterns)
# - Balanced loss weights (equal MIM and CLS)
# - Conservative weight decay (0.04 -> 0.1)
# - Gradient clipping (1.0) for stability
# - ViT-tiny architecture (fast, well under 100M limit)
# - Moderate augmentation (6 local crops)
#
# Best for: Establishing baseline, stable training, avoiding instability
# ============================================================================

python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints/v5/checkpoints_ibot_conservative \
    --arch vit_small \
    --optimizer adamw \
    --batch_size 128 \
    --lr 0.0003 \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.1 \
    --clip_grad 1.0 \
    --epochs 100 \
    --warmup_epochs 10 \
    --drop_path_rate 0.2 \
    --local_crops_number 6 \
    --bottleneck_dim 128 \
    --out_dim 4096 \
    --num_tokens 8192 \
    --mask_ratio 0.3 \
    --mask_type random \
    --mim_loss_weight 1.0 \
    --cls_loss_weight 1.0 \
    --koleo_weight 0.0 \
    --mim_temp 0.15 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_conservative/checkpoint_XXXX.pth

