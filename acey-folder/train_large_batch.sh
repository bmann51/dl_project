#!/bin/bash
#SBATCH --job-name=dino_large_batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_large_batch_%j.out
#SBATCH --error=logs/train_large_batch_%j.err

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
    --output_dir /gpfs/scratch/bm3772/checkpoints_large_batch \
    --arch vit_small \
    --batch_size 128 \
    --lr 0.0005 \
    --drop_path_rate 0.2 \
    --epochs 100 \
    --warmup_epochs 10 \
    --local_crops_number 6 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_large_batch/checkpoint_XXXX.pth

