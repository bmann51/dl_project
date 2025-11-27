#!/bin/bash
#SBATCH --job-name=ibot_v6_liberal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_v6_liberal_%j.out
#SBATCH --error=logs/train_ibot_v6_liberal_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# V6 "LIBERAL" CONFIGURATION: Stable + Curriculum + Focal iBOT
# (formerly the optimized conservative config)
# ============================================================================
# Philosophy:
#   - Keep the small, safe ViT-tiny + AdamW setup from v5
#   - Add *progressive masking* so the model starts with an easier task
#   - Use focal MIM loss to focus on hard masked patches
#   - Use a peak LR schedule (0.0003 -> 0.001 -> 1e-6) for smoother optimization
# ============================================================================

python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train \
    --output_dir /gpfs/scratch/bm3772/checkpoints_ibot_v6_liberal \
    --arch vit_tiny \
    --optimizer adamw \
    --batch_size 128 \
    --lr 0.0003 \
    --max_lr 0.001 \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.1 \
    --clip_grad 1.0 \
    --epochs 100 \
    --warmup_epochs 10 \
    --drop_path_rate 0.2 \
    --local_crops_number 6 \
    --bottleneck_dim 256 \
    --out_dim 4096 \
    --num_tokens 8192 \
    --mask_ratio 0.35 \
    --mask_type random \
    --progressive_masking \
    --mask_ratio_start 0.20 \
    --mim_loss_weight 1.0 \
    --cls_loss_weight 1.0 \
    --koleo_weight 0.0 \
    --mim_temp 0.15 \
    --use_focal_loss \
    --warmup_teacher_temp_epochs 10 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_v6_liberal/checkpoint_XXXX.pth


