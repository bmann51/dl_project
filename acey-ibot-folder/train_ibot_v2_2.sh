#!/bin/bash
#SBATCH --job-name=ibot_vit_tiny_v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_v2_%j.out
#SBATCH --error=logs/train_ibot_v2_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Version 2: Higher masking ratio + stronger MIM focus
# - Higher masking ratio (0.5) for more challenging MIM task
# - Higher MIM loss weight to emphasize masked prediction
# - Lower CLS loss weight
# - Stronger regularization (higher drop_path_rate)
python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v2_2 \
    --arch vit_tiny \
    --optimizer lars \
    --batch_size 128 \
    --lr 0.1 \
    --momentum 0.9 \
    --lars_trust_coefficient 0.001 \
    --drop_path_rate 0.25 \
    --weight_decay 0.06 \
    --weight_decay_end 0.5 \
    --epochs 100 \
    --warmup_epochs 10 \
    --local_crops_number 6 \
    --bottleneck_dim 128 \
    --out_dim 4096 \
    --num_tokens 8192 \
    --mask_ratio 0.5 \
    --mask_type random \
    --mim_loss_weight 1.5 \
    --cls_loss_weight 0.8 \
    --koleo_weight 0.001 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_vit_tiny_v2/checkpoint_XXXX.pth

