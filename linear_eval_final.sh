#!/bin/bash
#SBATCH --job-name=linear
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/linear%j.out
#SBATCH --error=logs/linear%j.err
#SBATCH --partition=gpu4_dev
#SBATCH --time=2:00:00

# #SBATCH --partition=gpu4_dev #other option not being used
# #SBATCH --partition=gpu4_short

# V2
# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/final_checkpoint.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_linear"

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0259.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_259"

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0249.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
# OUTPUT="submissions/v2_stable_249_train_1and2_final"


# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0009.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_009"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0019.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_019"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0029.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_029"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0039.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_039"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0049.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_049"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0099.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_099"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT


# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0149.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_149"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT


# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/checkpoint_0199.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2/args.json"
# OUTPUT="submissions/v2_stable_199"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0209.pth"
CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
OUTPUT="submissions1v2_stable_209"

python linear_eval_final.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --testset testset_1 \
    --output $OUTPUT

python linear_eval_final.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --testset testset_2 \
    --output $OUTPUT

python linear_eval_final.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --testset testset_3 \
    --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0219.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
# OUTPUT="submissions1v2_stable_219"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0229.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
# OUTPUT="submissions/v2_stable_229"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0239.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
# OUTPUT="submissions/v2_stable_239"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0249.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
# OUTPUT="submissions/v2_stable_249"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0259.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
# OUTPUT="submissions/v2_stable_259"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0269.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
# OUTPUT="submissions/v2_stable_269"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT

# CHECKPOINT="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/checkpoint_0279.pth"
# CONFIG="/gpfs/scratch/bm3772/checkpoints/v2/checkpoints_ibot_stable_2_small_train2_continue/args.json"
# OUTPUT="submissions/v2_stable_279"

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_1 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_2 \
#     --output $OUTPUT

# python linear_eval_final.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --testset testset_3 \
#     --output $OUTPUT