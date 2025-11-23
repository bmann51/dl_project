#!/bin/bash
#SBATCH --job-name=dino_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_base_%j.out
#SBATCH --error=logs/train_base_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Run training
# python train_dino.py \
#     --data_path /gpfs/scratch/bm3772/fall2025_data/train \
#     --output_dir /gpfs/scratch/bm3772/checkpoints_base50 \
#     --arch vit_base \
#     --batch_size 64 \
#     --epochs 50 \
#     --num_workers 8 \
#     --save_freq 10
    # --resume /gpfs/scratch/bm3772/checkpoints_base/checkpoint.pth


python eval_dino.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_baseline/final_checkpoint.pth \
    --arch vit_base \
    --train_path /gpfs/scratch/bm3772/cifar10_dino/eval_public/train \
    --test_path /gpfs/scratch/bm3772/cifar10_dino/eval_public/test \
    --image_size 96 \
    --k 20

    # --checkpoint /gpfs/scratch/bm3772/checkpoints_base200/checkpoint_0179.pth \

    #./data/eval_public/test