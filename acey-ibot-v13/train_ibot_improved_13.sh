#!/bin/bash
#SBATCH --job-name=13_improved_ibot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_stable_%j.out
#SBATCH --error=logs/train_ibot_stable_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Stable / mixed-domain configuration (500k general + 70k birds)
# - ViT-S backbone (<100M params)
# - AdamW with conservative base LR + peak LR scheduler
# - Moderate mask ratio (0.3) for 96x96
# - More weight on CLS self-distillation than MIM
# - KoLeo regularization for feature diversity
# - Mean-pool evaluation for better fine-grained + generic performance
python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train2 \  # or /train to remove birds
    --output_dir /gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_improved_13 \
    --arch vit_small \
    --optimizer adamw \
    --batch_size 128 \
    --lr 0.0003 \
    --max_lr 0.0008 \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.1 \
    --clip_grad 1.0 \
    --epochs 400 \
    --warmup_epochs 10 \
    --drop_path_rate 0.2 \
    --local_crops_number 6 \
    --bottleneck_dim 256 \
    --out_dim 65536 \
    --num_tokens 8192 \
    --mask_ratio 0.3 \
    --mask_type random \
    --mim_loss_weight 0.5 \
    --cls_loss_weight 1.5 \
    --koleo_weight 0.001 \
    --mim_temp 0.15 \
    --num_workers 8 \
    --save_freq 10 \
    --use_fp16 \
    --eval_pool mean
    # --resume /gpfs/scratch/bm3772/checkpoints_ibot_stable/checkpoint_XXXX.pth
