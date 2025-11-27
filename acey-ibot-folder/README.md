# iBOT Implementation

This folder contains an iBOT (Image BERT Pre-Training with Online Tokenizer) implementation for self-supervised learning.

## Overview

iBOT combines:
- **Masked Image Modeling (MIM)**: Predicts discrete tokens for masked patches
- **Self-Distillation**: DINO-style consistency on visible patches
- **Online Tokenizer**: Learns visual tokens during training

## Key Differences from DINO

1. **Masking**: Randomly masks 40% of patches during training
2. **Dual Loss**: 
   - MIM loss: Token prediction for masked patches
   - Self-distillation loss: Consistency on CLS tokens (visible patches)
   - KoLeo regularization: Feature diversity (from DINOv2)
3. **Tokenizer**: Learns discrete visual tokens (vocabulary size: 8192 by default)

---

## Detailed Implementation Explanation

### Architecture Overview

iBOT uses a **teacher-student** framework with Vision Transformers (ViT):

```
Input Image (96x96)
    ↓
[Patch Embedding] → 36 patches (6x6 grid, patch_size=16)
    ↓
[ViT Backbone] → Patch embeddings [batch, 37, embed_dim] (36 patches + 1 CLS token)
    ↓
    ├─→ [CLS Token] → [iBOTHead] → Global features (for self-distillation)
    └─→ [All Tokens] → [iBOTTokenizer] → Token predictions (for MIM)
```

### Core Components

#### 1. **Masking Strategy** (`RandomMaskingGenerator` / `BlockwiseMaskingGenerator`)

**How it works:**
- For 96x96 images with patch_size=16: 6×6 = 36 patches
- Randomly selects patches to mask (default: 40% = ~14 patches)
- Generates binary mask: `1` = visible, `0` = masked
- Same mask applied to all crops of the same image in a batch

**Example:**
```python
# For 36 patches with mask_ratio=0.4:
mask = [1, 0, 1, 0, 1, 1, 0, 1, ...]  # 14 zeros (masked), 22 ones (visible)
```

**Why it matters:**
- Forces model to learn from context (visible patches predict masked ones)
- Encourages understanding of spatial relationships
- Similar to BERT's masked language modeling

#### 2. **Online Tokenizer** (`iBOTTokenizer`)

**Purpose:** Learn discrete visual tokens during training (no pre-training needed)

**Architecture:**
```
Patch Embedding [embed_dim] 
    ↓
Linear(embed_dim → hidden_dim) + GELU
    ↓
Linear(hidden_dim → num_tokens)
    ↓
Token Logits [num_tokens]  # Vocabulary size: 8192
```

**How it works:**
- Takes patch embeddings from ViT backbone
- Maps each patch to a probability distribution over 8192 discrete tokens
- Teacher produces soft token targets (softmax with temperature)
- Student predicts tokens for masked patches
- Loss: Cross-entropy between teacher soft targets and student predictions

**Key insight:** The tokenizer learns visual "words" (discrete representations) that capture semantic patterns in patches.

#### 3. **Projection Head** (`iBOTHead`)

**Purpose:** Project CLS token for self-distillation (DINO-style)

**Architecture:**
```
CLS Token [embed_dim]
    ↓
Linear(embed_dim → hidden_dim) + GELU
    ↓
[Repeat hidden layers]
    ↓
Linear(hidden_dim → bottleneck_dim)  # Bottleneck (256 dim)
    ↓
L2 Normalize
    ↓
Linear(bottleneck_dim → out_dim)  # Final projection (4096 dim)
```

**Features:**
- Bottleneck architecture (from DINOv2) - reduces overfitting
- Weight normalization on last layer (prevents collapse)
- Separate heads for global/local crops (untied heads)

#### 4. **Model Wrapper** (`MultiCropiBOTWrapper`)

**Handles:**
- Multiple crops: 2 global + 6 local crops
- Extracts CLS token for self-distillation
- Extracts all tokens (patches + CLS) for tokenizer
- For evaluation: returns normalized CLS token features

**Forward pass:**
```python
# Training:
features = backbone(images)  # [batch, 37, embed_dim]
cls_token = features[:, 0]    # [batch, embed_dim]
head_output = head(cls_token)  # [batch, out_dim]
token_logits = tokenizer(features)  # [batch, 37, num_tokens]

# Evaluation:
features = backbone(images)
cls_token = features[:, 0]
cls_token = normalize(cls_token)  # Return normalized features
```

#### 5. **Loss Function** (`iBOTLoss`)

Combines three loss components:

**A. MIM Loss (Masked Image Modeling)**
- **Input:** Token predictions for masked patches
- **Process:**
  1. Teacher processes full image → produces token logits for all patches
  2. Student processes same image → produces token logits for all patches
  3. Extract predictions only for masked patches (where mask == 0)
  4. Teacher soft targets: `softmax(teacher_tokens / 0.07)`
  5. Student predictions: `log_softmax(student_tokens)`
  6. Loss: Cross-entropy between teacher soft targets and student predictions
- **Formula:** `L_MIM = -sum(teacher_soft * log(student_soft))` (only on masked patches)

**B. Self-Distillation Loss (DINO-style)**
- **Input:** CLS token projections
- **Process:**
  1. Teacher sees 2 global crops → produces 2 CLS projections
  2. Student sees all crops (2 global + 6 local) → produces 8 CLS projections
  3. Teacher centering: subtract running mean (prevents collapse)
  4. Temperature sharpening: `softmax((teacher - center) / temp)`
  5. Loss: Cross-entropy between teacher and student CLS projections
  6. Skip cases where teacher and student see the same view
- **Formula:** `L_CLS = -sum(teacher_soft * log(student_soft))` (on CLS tokens)

**C. KoLeo Regularization (from DINOv2)**
- **Purpose:** Encourage feature diversity (prevent collapse)
- **Process:**
  1. Normalize teacher CLS features
  2. Compute pairwise L2 distances between all features in batch
  3. Apply negative log: `-log(pairwise_distances + eps)`
  4. Average over all pairs (excluding self-distances)
- **Formula:** `L_KoLeo = -mean(log(||f_i - f_j||^2 + eps))`
- **Effect:** Encourages features to spread out in embedding space

**Total Loss:**
```
L_total = w_MIM * L_MIM + w_CLS * L_CLS + w_KoLeo * L_KoLeo
```

#### 6. **Training Loop**

**Per iteration:**
1. **Generate mask:** Random mask for each image (40% patches masked)
2. **Forward pass:**
   - Teacher: Full images (no masking) → CLS + token predictions
   - Student: Same images → CLS + token predictions
3. **Compute loss:**
   - MIM loss on masked patches only
   - CLS loss on all crops
   - KoLeo loss on teacher features
4. **Backward pass:** Update student only
5. **Momentum update:** `teacher = m * teacher + (1-m) * student`
6. **Update center:** EMA update of teacher output center

**Key details:**
- Teacher is frozen (no gradients)
- Only student receives gradients
- Teacher updated via exponential moving average (EMA)
- Last layer gradients canceled for first epoch (prevents collapse)

#### 7. **LARS Optimizer**

**Layer-wise Adaptive Rate Scaling:**
- Adapts learning rate per layer based on parameter/gradient norms
- Formula: `local_lr = trust_coeff * ||params|| / (||grads|| + wd * ||params||)`
- Better for large batch training (batch_size=128)
- Uses higher base LR (0.1 vs 0.0005 for AdamW)

**Why LARS:**
- Scales well with large batches
- Common in SSL methods (SimCLR, BYOL)
- More stable training dynamics

### Data Flow Example

**Single training step:**

```
1. Input: Image (96x96x3)
   ↓
2. Augmentation: 8 crops (2 global + 6 local)
   ↓
3. Masking: Generate mask [36 patches] with 14 masked
   ↓
4. Teacher Forward:
   - Full images → ViT → [batch*2, 37, embed_dim]
   - CLS tokens → Head → [batch*2, out_dim]
   - All tokens → Tokenizer → [batch*2, 37, num_tokens]
   ↓
5. Student Forward:
   - Same images → ViT → [batch*8, 37, embed_dim]
   - CLS tokens → Head → [batch*8, out_dim]
   - All tokens → Tokenizer → [batch*8, 37, num_tokens]
   ↓
6. Loss Computation:
   - MIM: Compare student vs teacher tokens on 14 masked patches
   - CLS: Compare student vs teacher CLS on all 8 crops
   - KoLeo: Diversity loss on teacher CLS features
   ↓
7. Backward: Update student only
   ↓
8. EMA Update: teacher = 0.996 * teacher + 0.004 * student
```

### Why This Works

1. **MIM learns local features:** Predicting masked patches forces understanding of patch-level semantics
2. **Self-distillation learns global features:** CLS token consistency learns high-level semantics
3. **Combined = best of both:** Local + global understanding
4. **KoLeo prevents collapse:** Ensures diverse feature representations
5. **Online tokenizer adapts:** Learns visual vocabulary specific to your data

## Files

- `ibot_ssl.py`: Core iBOT implementation
  - `RandomMaskingGenerator`: Random patch masking (default)
  - `BlockwiseMaskingGenerator`: Block-wise masking (alternative)
  - `iBOTTokenizer`: Online tokenizer - MLP mapping patches to discrete tokens
  - `iBOTHead`: Projection head - bottleneck architecture for CLS tokens
  - `MultiCropiBOTWrapper`: Model wrapper - handles crops, extracts CLS/tokens
  - `iBOTLoss`: Combined loss - MIM + self-distillation + KoLeo
  - `LARS`: Layer-wise Adaptive Rate Scaling optimizer
  - `DataAugmentation`: Multi-crop augmentation (2 global + 6 local)

- `train_ibot.py`: Training script with full training loop
- `eval_ibot.py`: Evaluation script (k-NN on frozen features)
- `train_ibot_v1_1.sh`, `train_ibot_v2_2.sh`, `train_ibot_v3_3.sh`: SLURM training scripts (3 variants)
- `requirements.txt`: Dependencies
- `TRAINING_VARIANTS.md`: Detailed comparison of the 3 training variants

## Usage

### Training

```bash
sbatch train_ibot.sh
```

Or run directly:
```bash
python train_ibot.py \
    --data_path /path/to/data \
    --output_dir ./checkpoints \
    --arch vit_tiny \
    --batch_size 128 \
    --mask_ratio 0.4 \
    --mim_loss_weight 1.0 \
    --cls_loss_weight 1.0 \
    --use_fp16
```

### Key Hyperparameters

- `--mask_ratio`: Ratio of patches to mask (default: 0.4)
- `--mask_type`: `random` or `blockwise` (default: `random`)
- `--num_tokens`: Vocabulary size for tokenizer (default: 8192)
- `--mim_loss_weight`: Weight for MIM loss (default: 1.0)
- `--cls_loss_weight`: Weight for self-distillation loss (default: 1.0)
- `--optimizer`: Optimizer type - `lars` or `adamw` (default: `lars`)
- `--lr`: Learning rate (default: 0.1 for LARS, 0.0005 for AdamW)
- `--momentum`: Momentum for LARS (default: 0.9)
- `--lars_trust_coefficient`: Trust coefficient for LARS (default: 0.001)
- `--koleo_weight`: Weight for KoLeo regularization (default: 0.001)
- `--koleo_eps`: Epsilon for KoLeo loss (default: 1e-6)

### Evaluation

```bash
python eval_ibot.py \
    --checkpoint ./checkpoints/final_checkpoint.pth \
    --train_path /path/to/train \
    --test_path /path/to/test \
    --arch vit_tiny \
    --k 20
```

## Architecture

- **Backbone**: ViT (tiny/small/base)
- **Tokenizer**: MLP mapping patch embeddings to discrete tokens
- **Head**: Projection head for CLS token (similar to DINO)

## Training Details

### Architecture
- **Backbone**: ViT-tiny (default, < 100M params)
  - Patch size: 16x16
  - Image size: 96x96 → 6×6 = 36 patches
  - Embedding dimension: 192 (for vit_tiny)
  - Drop path rate: 0.2 (stochastic depth for regularization)

### Optimizer
- **LARS** (Layer-wise Adaptive Rate Scaling)
  - Base LR: 0.1 (much higher than AdamW's 0.0005)
  - Momentum: 0.9
  - Trust coefficient: 0.001 (controls layer-wise LR scaling)
  - Why LARS: Better for large batch training, common in SSL

### Training Process
1. **Teacher-Student Setup:**
   - Teacher: Processes full images (no masking), frozen (no gradients)
   - Student: Processes same images, receives gradients
   - Teacher updated via EMA: `θ_teacher = 0.996 * θ_teacher + 0.004 * θ_student`

2. **Masking:**
   - Generate random mask per image (40% patches masked)
   - Same mask for all crops of same image
   - Loss computed only on masked patches (MIM) and visible patches (CLS)

3. **Multi-Crop Augmentation:**
   - 2 global crops: Strong augmentation, larger scale (0.32-1.0)
   - 6 local crops: Weaker augmentation, smaller scale (0.05-0.32)
   - Teacher sees only global crops
   - Student sees all crops

4. **Learning Rate Schedule:**
   - Cosine decay with warmup
   - Warmup: 10 epochs (linear increase)
   - Decay: Cosine schedule to minimum LR (1e-6)
   - Scaled by batch size: `lr = base_lr * batch_size / 256`

5. **Weight Decay Schedule:**
   - Cosine schedule: 0.05 → 0.45 (increases over training)
   - Helps with generalization

6. **Temperature Schedule:**
   - Teacher temperature: 0.04 → 0.04 (warmup over 30 epochs)
   - Student temperature: 0.1 (fixed)
   - Controls sharpness of softmax distributions

## Implementation Details

### Parameter Counting
- **ViT-tiny**: ~5.5M backbone parameters
- **Tokenizer**: ~1M parameters (per head)
- **Projection head**: ~2M parameters (per head)
- **Total**: ~10-15M parameters (well under 100M limit)

### Memory Considerations
- Batch size: 128 (with 8 crops = 1024 images per batch)
- Mixed precision (FP16): Reduces memory by ~50%
- Gradient checkpointing: Not used (can be added if needed)

### Evaluation
- **Frozen encoder**: Backbone is frozen during k-NN evaluation
- **Feature extraction**: Uses normalized CLS token (L2 normalized)
- **k-NN**: Cosine similarity on normalized features
- **No fine-tuning**: Features are used as-is (competition requirement)

## Notes

- Model must stay under 100M parameters ✅ (ViT-tiny is ~10-15M)
- Image resolution fixed at 96x96 ✅
- Uses frozen encoder for evaluation (k-NN) ✅
- Random initialization (no pretrained weights) ✅
- All paths configured for `/gpfs/scratch/bm3772/` (ready to run)

## Technical Highlights

1. **Untied Heads**: Separate projection heads for global/local crops (better than tied)
2. **Bottleneck Architecture**: Reduces overfitting in projection head
3. **KoLeo Regularization**: Prevents feature collapse (from DINOv2)
4. **Online Tokenizer**: Learns visual vocabulary during training (no pre-training)
5. **LARS Optimizer**: Better for large batch SSL training
6. **Multi-Crop Strategy**: 8 crops per image (2 global + 6 local) for robust learning

