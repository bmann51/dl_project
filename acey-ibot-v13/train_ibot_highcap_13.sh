#!/bin/bash
#SBATCH --job-name=13_highcap_ibot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/train_ibot_highcap_%j.out
#SBATCH --error=logs/train_ibot_highcap_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# Higher-capacity / longer-training configuration
# - Same ViT-S backbone and head sizes as improved_13
# - More epochs (200) for better convergence
# - Slightly higher LR + peak LR
# - Slightly higher mask ratio (0.35) for a bit more MIM pressure
# - More local crops (8) to exploit fine-grained details
# - Slightly stronger KoLeo for feature diversity
python train_ibot.py \
    --data_path /gpfs/scratch/bm3772/fall2025_data/train2 \  # or /train to remove birds
    --output_dir /gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_highcap_13 \
    --arch vit_small \
    --optimizer adamw \
    --batch_size 128 \
    --lr 0.0004 \
    --max_lr 0.0010 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --weight_decay_end 0.12 \
    --clip_grad 1.0 \
    --epochs 200 \
    --warmup_epochs 20 \
    --drop_path_rate 0.25 \
    --local_crops_number 8 \
    --bottleneck_dim 256 \
    --out_dim 65536 \
    --num_tokens 8192 \
    --mask_ratio 0.35 \
    --mask_type random \
    --mim_loss_weight 0.5 \
    --cls_loss_weight 1.5 \
    --koleo_weight 0.0015 \
    --mim_temp 0.15 \
    --num_workers 8 \
    --save_freq 20 \
    --use_fp16 \
    --eval_pool mean
    # To resume:
    # --resume /gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_highcap_13/checkpoint_XXXX.pth
