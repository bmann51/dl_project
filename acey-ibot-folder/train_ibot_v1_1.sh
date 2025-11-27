#!/bin/bash
#SBATCH --job-name=ibot_vit_tiny_v1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_v1_%j.out
#SBATCH --error=logs/train_ibot_v1_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Version 1: Base configuration with LARS
# - Standard masking ratio (0.4)
# - Balanced loss weights
# - Standard regularization
python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v1_1 \
    --arch vit_tiny \
    --optimizer lars \
    --batch_size 128 \
    --lr 0.1 \
    --momentum 0.9 \
    --lars_trust_coefficient 0.001 \
    --drop_path_rate 0.2 \
    --weight_decay 0.05 \
    --weight_decay_end 0.45 \
    --epochs 100 \
    --warmup_epochs 10 \
    --local_crops_number 6 \
    --bottleneck_dim 128 \
    --out_dim 4096 \
    --num_tokens 8192 \
    --mask_ratio 0.4 \
    --mask_type random \
    --mim_loss_weight 1.0 \
    --cls_loss_weight 1.0 \
    --koleo_weight 0.001 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v1/checkpoint_XXXX.pth

