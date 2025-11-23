#!/bin/bash
#SBATCH --job-name=dino_more_crops
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_more_crops_%j.out
#SBATCH --error=logs/train_more_crops_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Run training with more local crops for better augmentation diversity
# Strategy: More local crops, longer warmup, adjusted temperatures
python train_dino.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints_more_crops \
    --arch vit_small \
    --batch_size 64 \
    --lr 0.0005 \
    --epochs 100 \
    --warmup_epochs 15 \
    --local_crops_number 10 \
    --warmup_teacher_temp 0.05 \
    --teacher_temp 0.05 \
    --warmup_teacher_temp_epochs 40 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_more_crops/checkpoint_XXXX.pth

