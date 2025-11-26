#!/bin/bash
#SBATCH --job-name=ibot_balanced
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_balanced_%j.out
#SBATCH --error=logs/train_ibot_balanced_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Balanced Configuration: Slightly more aggressive but still stable
# - AdamW optimizer with moderate learning rate (0.0005)
# - Moderate mask ratio (0.35) - between stable and aggressive
# - Slightly higher weight decay (0.05 -> 0.15)
# - Gradient clipping (1.0) for stability
# - More local crops (8) for better augmentation
# - Higher MIM loss weight to emphasize masked prediction
# - Focus on better representation learning
python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints_ibot_balanced \
    --arch vit_tiny \
    --optimizer adamw \
    --batch_size 128 \
    --lr 0.0005 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --weight_decay_end 0.15 \
    --clip_grad 1.0 \
    --epochs 100 \
    --warmup_epochs 10 \
    --drop_path_rate 0.2 \
    --local_crops_number 8 \
    --bottleneck_dim 128 \
    --out_dim 4096 \
    --num_tokens 8192 \
    --mask_ratio 0.35 \
    --mask_type random \
    --mim_loss_weight 1.2 \
    --cls_loss_weight 1.0 \
    --koleo_weight 0.001 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_balanced/checkpoint_XXXX.pth

