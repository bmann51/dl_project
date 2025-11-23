#!/bin/bash
#SBATCH --job-name=dino_strong_reg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_strong_reg_%j.out
#SBATCH --error=logs/train_strong_reg_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Run training with strong regularization to reduce overfitting
# Strategy: Higher drop_path_rate, higher weight decay, lower learning rate
python train_dino.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints_strong_reg \
    --arch vit_small \
    --batch_size 64 \
    --lr 0.0003 \
    --drop_path_rate 0.3 \
    --weight_decay 0.06 \
    --weight_decay_end 0.5 \
    --epochs 100 \
    --warmup_epochs 10 \
    --local_crops_number 6 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_strong_reg/checkpoint_XXXX.pth

