#!/bin/bash
#SBATCH --job-name=dino_high_lr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_high_lr_%j.out
#SBATCH --error=logs/train_high_lr_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Run training
python train_dino.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints_high_lr \
    --arch vit_small \
    --batch_size 64 \
    --lr 0.00075 \
    --drop_path_rate 0.2 \
    --epochs 100 \
    --warmup_epochs 10 \
    --local_crops_number 6 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_high_lr/checkpoint_XXXX.pth

