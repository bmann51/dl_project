#!/bin/bash
#SBATCH --job-name=dino_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Run training
# python train_dino.py \
#     --data_path /gpfs/scratch/bm3772/data/pretrain/ \
#     --output_dir /gpfs/scratch/bm3772/checkpoints \
#     --arch vit_small \
#     --batch_size 64 \
#     --epochs 100 \
#     --num_workers 8 \
#     --save_freq 10

python train_dino.py \
    --data_path ./data/pretrain/ \
    --output_dir ./checkpoints_quick \
    --arch vit_tiny \
    --batch_size 256 \
    --epochs 10 \
    --warmup_teacher_temp_epochs 2 \
    --warmup_epochs 2 \
    --num_workers 4