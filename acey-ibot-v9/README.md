# iBOT v9 - Clean Competition-Ready Implementation

This is a clean, self-contained iBOT-style self-supervised learning implementation that fixes the issues from previous versions.

## Key Features

- **ViT-Small/16 backbone** (~22M params, random init, 96×96 resolution)
- **Pure cross-entropy loss** (no entropy/focal terms - MIM loss never goes negative)
- **Shared projection head** for CLS and patch tokens (as in iBOT paper)
- **Teacher centering + temperature scheduling** for both CLS and patches
- **EMA teacher updates** every step
- **Explicit masking** tracked per view
- **Two global crops** (no local crops for cleaner resolution story)

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train on unlabeled pretraining data:

```bash
python train_ibot.py \
  --data_root /path/to/pretrain \
  --output_dir /path/to/checkpoints \
  --epochs 400 \
  --batch_size 256 \
  --mask_ratio 0.3
```

### Arguments

- `--data_root`: Path to directory containing unlabeled pretraining images
- `--output_dir`: Directory to save checkpoints
- `--epochs`: Number of training epochs (default: 400)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 5e-4)
- `--weight_decay`: Weight decay (default: 0.04)
- `--mask_ratio`: Ratio of patches to mask (default: 0.3)
- `--device`: Device to use (default: "cuda")

## Evaluation

Run k-NN evaluation on labeled data:

```bash
python eval_knn.py \
  --train_dir /path/to/eval_public/train \
  --test_dir /path/to/eval_public/test \
  --checkpoint /path/to/checkpoints/ibot_epoch400.pt \
  --k 20
```

### Arguments

- `--train_dir`: Path to training set (ImageFolder structure)
- `--test_dir`: Path to test set (ImageFolder structure)
- `--checkpoint`: Path to trained checkpoint
- `--k`: Number of nearest neighbors (default: 20)
- `--batch_size`: Batch size for evaluation (default: 256)
- `--device`: Device to use (default: "cuda")

## Architecture

- **Backbone**: ViT-Small/16 (384 dim, ~22M params)
- **Projection Head**: 3-layer MLP (384 → 2048 → 2048 → 8192) with L2 normalization
- **Masking**: Random patch masking in image space (mask_ratio=0.3)
- **Augmentation**: Strong SimCLR/DINO-style augmentations

## Loss Function

- **CLS Loss**: Cross-view self-distillation (DINO-style)
  - Student view 0 vs Teacher view 1
  - Student view 1 vs Teacher view 0
  
- **MIM Loss**: Masked patch prediction (in-view)
  - Only computed on masked patches
  - Pure cross-entropy (no entropy/focal terms)

## Key Differences from Previous Versions

1. **No entropy regularization** - MIM loss is pure CE, never negative
2. **No focal loss** - Simpler, more stable training
3. **Shared projection head** - CLS and patches use same head (as in paper)
4. **Cleaner masking** - Explicit image-space masking, tracked per view
5. **Two global crops only** - No local crops for cleaner implementation

## Competition Compliance

- ✅ Backbone < 100M parameters (ViT-Base: ~86M)
- ✅ 96×96 image resolution
- ✅ Random initialization (no pretrained weights)
- ✅ Frozen encoder for evaluation (k-NN)

## Files

- `ibot_ssl.py`: Core iBOT implementation (models, loss, dataset, masking)
- `train_ibot.py`: Main training script
- `eval_knn.py`: k-NN evaluation script
- `check_parameters.py`: Script to verify backbone parameter count (< 100M)
- `train_ibot.sh`: SLURM training script
- `eval_knn.sh`: SLURM evaluation script
- `requirements.txt`: Python dependencies
- `README.md`: This file

