#!/bin/bash
#SBATCH --job-name=dino_longer_warmup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_longer_warmup_%j.out
#SBATCH --error=logs/train_longer_warmup_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Run training with longer warmup and slower learning for more stable training
# Strategy: Longer warmup, lower initial LR, different momentum schedule
python train_dino.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints_longer_warmup \
    --arch vit_small \
    --batch_size 64 \
    --lr 0.0004 \
    --min_lr 1e-7 \
    --epochs 100 \
    --warmup_epochs 20 \
    --local_crops_number 6 \
    --momentum_teacher 0.995 \
    --freeze_last_layer 2 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_longer_warmup/checkpoint_XXXX.pth

