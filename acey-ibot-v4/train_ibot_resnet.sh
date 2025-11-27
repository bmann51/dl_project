#!/bin/bash
#SBATCH --job-name=ibot_resnet18
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_resnet18_%j.out
#SBATCH --error=logs/train_ibot_resnet18_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# RESNET EXPERIMENT: CNN Backbone with iBOT
# ============================================================================
# Philosophy: Test CNN backbone (ResNet18) vs ViT for iBOT
# 
# Key characteristics:
# - ResNet18 backbone (~11M params, well under 100M limit)
# - CNN feature extractor converts feature maps to patch-like tokens
# - Same iBOT training procedure (MIM + self-distillation)
# - Conservative settings for stable training
#
# Why ResNet18:
# - Small, fast to train
# - Well-understood architecture
# - Good baseline for CNN experiments
# - Inductive biases (translation equivariance) might help
#
# Best for: Comparing CNN vs ViT architectures
# ============================================================================

python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints/v4/checkpoints_ibot_resnet18 \
    --arch resnet18 \
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
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_resnet18/checkpoint_XXXX.pth

