#!/bin/bash
#SBATCH --job-name=dino_baseline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_baseline_%j.out
#SBATCH --error=logs/train_baseline_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Run training
python train_dino.py \
    --data_path /mnt/user-data/uploads/pretrain/ \
    --output_dir ./experiments/baseline \
    --arch vit_small \
    --batch_size 64 \
    --lr 0.0005 \
    --drop_path_rate 0.1 \
    --epochs 100 \
    --warmup_epochs 10 \
    --local_crops_number 6 \
    --num_workers 4 \
    --save_freq 10 \
    --use_fp16
    # --resume ./experiments/baseline/checkpoint_XXXX.pth

