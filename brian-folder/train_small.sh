#!/bin/bash
#SBATCH --job-name=dino_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_small_%j.out
#SBATCH --error=logs/train_small_%j.err

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
    --output_dir /gpfs/scratch/bm3772/checkpoints_small200 \
    --arch vit_small \
    --batch_size 64 \
    --epochs 200 \
    --num_workers 8 \
    --save_freq 10 \
    --resume /gpfs/scratch/bm3772/checkpoints_small200/checkpoint_0149.pth


# python eval_dino.py \
#     --checkpoint /gpfs/scratch/bm3772/checkpoints/final_checkpoint.pth \
#     --arch vit_small \
#     --train_path ./data/eval_public/train \
#     --test_path ./data/eval_public/test \
#     --k 20