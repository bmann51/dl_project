#!/bin/bash
#SBATCH --job-name=10_multicrop_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --partition=a100_short
#SBATCH --output=logs/eval_ibot_v10_%j.out
#SBATCH --error=logs/eval_ibot_v10_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate dino_new

# Create logs directory
mkdir -p logs

# ============================================================================
# k-NN EVALUATION - iBOT v10 (Multi-Crop)
# ============================================================================
# Philosophy: Evaluate frozen backbone features using k-NN classification
# 
# Key characteristics:
# - Uses frozen ViT-Base/16 backbone (no fine-tuning)
# - Extracts normalized CLS token features
# - k-NN classification with cosine similarity
# - Majority vote weighted by similarity scores
#
# Note: Evaluation uses only the backbone CLS token, same as v9.
# Multi-crop is only used during training.
#
# Best for: Competition evaluation, feature quality assessment
# ============================================================================

python eval_knn.py \
    --train_dir /gpfs/scratch/bm3772/fall2025_data/eval_public/train \
    --test_dir /gpfs/scratch/bm3772/fall2025_data/eval_public/test \
    --checkpoint /gpfs/scratch/bm3772/checkpoints/v10/v10_ibot_multicrop/ibot_v10_epoch300.pt \
    --k 20 \
    --batch_size 256 \
    --device cuda

