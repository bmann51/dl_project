#!/bin/bash
#SBATCH --job-name=ibot_feedback
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_feedback_%j.out
#SBATCH --error=logs/train_ibot_feedback_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# FEEDBACK CONFIGURATION: Strictly Follows Training Feedback
# ============================================================================
# Philosophy: Follow feedback recommendations exactly for stable training
# 
# This configuration strictly implements all feedback recommendations:
# 1. Lower learning rate: base_lr=0.0003, max_lr=0.001 (peak after warmup)
# 2. Reduced weight decay: 0.04 -> 0.1 (not ramping to 0.4+)
# 3. Gradient clipping: max_norm=1.0 (conservative)
# 4. Smaller batch size: 96 (down from 128 for more stable updates)
# 5. Lower mask ratio: 0.3 (easier learning task)
# 6. Loss monitoring: Built into training script
# 7. AdamW optimizer (not LARS - LARS was going to 0.05 which is 50x higher)
#
# Expected loss behavior:
# - Should start around 4-6 and decrease to 1-3
# - NOT start at 12 and increase to 8+
#
# Best for: Following feedback exactly, most stable training
# ============================================================================

python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints/v8/checkpoints_ibot_feedback \
    --arch vit_base \
    --optimizer adamw \
    --batch_size 96 \
    --lr 0.0003 \
    --max_lr 0.001 \
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
    --mask_ratio 0.4 \
    --mask_type random \
    --mim_loss_weight 1.0 \
    --cls_loss_weight 1.0 \
    --koleo_weight 0.0 \
    --warmup_teacher_temp_epochs 10 \
    --mim_temp 0.2 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_feedback/checkpoint_XXXX.pth

