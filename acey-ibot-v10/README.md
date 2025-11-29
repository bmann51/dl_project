# iBOT v10 - Multi-Crop Implementation

This is a multi-crop variant of the iBOT self-supervised learning implementation, building on the clean v9 foundation with enhanced augmentation diversity through multi-scale crops.

## Key Features

- **ViT-Base/16 backbone** (< 100M params, random init, 96×96 resolution)
- **Multi-crop augmentation**: 2 global crops (96×96) + 4 local crops (64×64)
- **Pure cross-entropy loss** (no entropy/focal terms - MIM loss never goes negative)
- **Shared projection head** for CLS and patch tokens (as in iBOT paper)
- **Teacher centering + temperature scheduling** for both CLS and patches
- **EMA teacher updates** every step
- **Explicit masking** on global crops only (image-space, blockwise)
- **Cross-view distillation**: Teacher global views → Student global+local views

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train on unlabeled pretraining data:

```bash
python train_ibot_v10.py \
  --data_root /path/to/pretrain \
  --output_dir /path/to/checkpoints \
  --epochs 300 \
  --batch_size 192 \
  --mask_ratio 0.3 \
  --num_local_crops 4
```

### Arguments

- `--data_root`: Path to directory containing unlabeled pretraining images
- `--output_dir`: Directory to save checkpoints
- `--epochs`: Number of training epochs (default: 300)
- `--batch_size`: Batch size (default: 192, reduced for multi-crop compute)
- `--lr`: Learning rate (default: 5e-4)
- `--weight_decay`: Weight decay (default: 0.04)
- `--mask_ratio`: Ratio of patches to mask on global crops (default: 0.3)
- `--num_local_crops`: Number of local crops per image (default: 4)
- `--device`: Device to use (default: "cuda")

## Evaluation

Run k-NN evaluation on labeled data (uses frozen backbone CLS token, same as v9):

```bash
python eval_knn.py \
  --train_dir /path/to/eval_public/train \
  --test_dir /path/to/eval_public/test \
  --checkpoint /path/to/checkpoints/ibot_v10_epoch300.pt \
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

- **Backbone**: ViT-Base/16 (768 dim, ~86M params)
- **Projection Head**: 3-layer MLP (768 → 2048 → 2048 → 8192) with L2 normalization
- **Masking**: Random patch masking in image space on global crops only (mask_ratio=0.3)
- **Augmentation**: 
  - Global crops: Strong SimCLR/DINO-style augmentations (96×96)
  - Local crops: More aggressive cropping with smaller scale (64×64)

## Loss Function

- **CLS Loss**: Cross-view self-distillation (DINO-style)
  - Teacher global views → Student global+local views
  - Avoids matching same-index global views (prevents trivial identity)
  
- **MIM Loss**: Masked patch prediction (global crops only)
  - Only computed on masked patches of global crops
  - Pure cross-entropy (no entropy/focal terms)
  - Local crops do NOT contribute to MIM loss

## Key Differences from v9

1. **Multi-crop augmentation**: 2 global + 4 local crops (vs 2 global only in v9)
2. **CLS loss uses all student views**: Global + local crops participate in CLS distillation
3. **MIM only on globals**: Local crops are too small for meaningful patch prediction
4. **Reduced batch size**: 192 (vs 256 in v9) to account for multi-crop compute overhead
5. **Fewer epochs**: 300 (vs 400 in v9) as multi-crop provides more learning signal per epoch

## Competition Compliance

- ✅ Backbone < 100M parameters (ViT-Base: ~86M)
- ✅ 96×96 image resolution (global crops)
- ✅ Random initialization (no pretrained weights)
- ✅ Frozen encoder for evaluation (k-NN)

## Files

- `ibot_ssl_v10.py`: Core iBOT implementation (models, loss, dataset, masking)
- `train_ibot_v10.py`: Main training script
- `eval_knn.py`: k-NN evaluation script (reuses backbone from v9)
- `check_parameters.py`: Script to verify backbone parameter count (< 100M)
- `train_ibot_v10.sh`: SLURM training script
- `eval_knn.sh`: SLURM evaluation script
- `requirements.txt`: Python dependencies
- `README.md`: This file

