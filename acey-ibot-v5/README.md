# iBOT Implementation (v3 - Enhanced Architecture)

This folder contains an iBOT (Image BERT Pre-Training with Online Tokenizer) implementation for self-supervised learning, with **architectural improvements** and **stable training** configurations.

## Overview

iBOT combines:
- **Masked Image Modeling (MIM)**: Predicts discrete tokens for masked patches
- **Self-Distillation**: DINO-style consistency on visible patches
- **Online Tokenizer**: Learns visual tokens during training

## Key Improvements (v3)

### Architectural Enhancements (NEW in v3)
1. **Enhanced Tokenizer**: LayerNorm, deeper architecture (3 layers), SiLU activation
2. **Enhanced Projection Head**: LayerNorm, SiLU activation, optional dropout
3. **Progressive Masking**: Optional curriculum learning (gradually increases mask ratio)
4. **Enhanced Loss Function**: Optional focal loss for MIM, temperature scheduling for tokenizer

### Training Stability (from v2)
1. **AdamW Optimizer** (instead of LARS) with conservative learning rates (0.0003-0.0005)
2. **Reduced Weight Decay** (0.04→0.1, instead of 0.05→0.45)
3. **Gradient Clipping** (max_norm=1.0) for training stability
4. **Optimized Mask Ratio** (0.3-0.4, adjusted per variant) for balanced learning
5. **KoLeo Loss Disabled** - Disabled to prevent explosion and simplify training (original iBOT doesn't use it)
6. **Loss Computation Checks** to detect multiplication bugs and training issues
7. **Loss Component Monitoring** for better debugging
8. **Parameter Count Verification**: Automatic check that backbone < 100M (assignment requirement)

## Training Scripts

Three diverse training variants are provided for exploration:

- **`train_ibot_conservative_5.sh`**: Stable AdamW baseline (lr=0.0003, mask_ratio=0.3, wd=0.04→0.1, ViT-tiny)
- **`train_ibot_aggressive_5.sh`**: High-capacity LARS training (lr=0.1, mask_ratio=0.4, blockwise masking, ViT-small)
- **`train_ibot_feedback_5.sh`**: Strictly follows training feedback (lr=0.0003, max_lr=0.001, batch_size=96, all recommendations)

See `TRAINING_VARIANTS.md` for detailed comparison.

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
- Randomly selects patches to mask (default: 30-35% = ~11-13 patches)
- Generates binary mask: `1` = visible, `0` = masked
- Same mask applied to all crops of the same image in a batch

**Example:**
```python
# For 36 patches with mask_ratio=0.3:
mask = [1, 0, 1, 1, 0, 1, 1, 1, ...]  # 11 zeros (masked), 25 ones (visible)
```

**Why it matters:**
- Forces model to learn from context (visible patches predict masked ones)
- Encourages understanding of spatial relationships
- Similar to BERT's masked language modeling

#### 2. **Online Tokenizer** (`iBOTTokenizer`) - **ENHANCED in v3**

**Purpose:** Learn discrete visual tokens during training (no pre-training needed)

**Architecture (v3 - Enhanced):**
```
Patch Embedding [embed_dim] 
    ↓
Linear(embed_dim → hidden_dim)
    ↓
LayerNorm(hidden_dim)  # NEW: Training stability
    ↓
SiLU()  # NEW: Better activation than GELU
    ↓
Linear(hidden_dim → hidden_dim)  # NEW: Deeper architecture
    ↓
LayerNorm(hidden_dim)  # NEW
    ↓
SiLU()  # NEW
    ↓
Linear(hidden_dim → num_tokens)
    ↓
Token Logits [num_tokens]  # Vocabulary size: 8192
```

**Improvements in v3:**
- **LayerNorm**: Added after each linear layer for training stability
- **Deeper Architecture**: 3 layers instead of 2 (better representation learning)
- **SiLU Activation**: Often outperforms GELU/ReLU
- **Optional Residual**: Can enable residual connections (when dimensions match)

**How it works:**
- Takes patch embeddings from ViT backbone
- Maps each patch to a probability distribution over 8192 discrete tokens
- Teacher produces soft token targets (softmax with temperature schedule)
- Student predicts tokens for masked patches
- Loss: Cross-entropy (or focal loss) between teacher soft targets and student predictions

**Key insight:** The tokenizer learns visual "words" (discrete representations) that capture semantic patterns in patches.

**Parameters:**
- `use_layernorm=True`: Enable LayerNorm (default: True)
- `activation='silu'`: Activation function ('silu', 'gelu', or 'relu')
- `use_residual=False`: Enable residual connection (default: False)

#### 3. **Projection Head** (`iBOTHead`) - **ENHANCED in v3**

**Purpose:** Project CLS token for self-distillation (DINO-style)

**Architecture (v3 - Enhanced):**
```
CLS Token [embed_dim]
    ↓
Linear(embed_dim → hidden_dim)
    ↓
LayerNorm(hidden_dim)  # NEW: Training stability
    ↓
SiLU()  # NEW: Better activation than GELU
    ↓
Dropout(0.1)  # NEW: Optional regularization
    ↓
[Repeat hidden layers with LayerNorm + SiLU + Dropout]
    ↓
Linear(hidden_dim → bottleneck_dim)  # Bottleneck (256 dim)
    ↓
LayerNorm(bottleneck_dim)  # NEW
    ↓
L2 Normalize
    ↓
Linear(bottleneck_dim → out_dim)  # Final projection (4096 dim)
```

**Improvements in v3:**
- **LayerNorm**: Added after each linear layer for training stability
- **SiLU Activation**: Often outperforms GELU
- **Optional Dropout**: Can add dropout (0.1-0.2) for regularization

**Features:**
- Bottleneck architecture (from DINOv2) - reduces overfitting
- Weight normalization on last layer (prevents collapse)
- Separate heads for global/local crops (untied heads)

**Parameters:**
- `use_layernorm=True`: Enable LayerNorm (default: True)
- `activation='silu'`: Activation function ('silu', 'gelu', or 'relu')
- `dropout=0.0`: Dropout probability (default: 0.0, set to 0.1-0.2 for regularization)

#### 4. **Model Wrapper** (`MultiCropiBOTWrapper`)

**Handles:**
- Multiple crops: 2 global + 6-8 local crops
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

**A. MIM Loss (Masked Image Modeling)** - **ENHANCED in v3**
- **Input:** Token predictions for masked patches
- **Process:**
  1. Teacher processes full image → produces token logits for all patches
  2. Student processes same image → produces token logits for all patches
  3. Extract predictions only for masked patches (where mask == 0)
  4. Teacher soft targets: `softmax(teacher_tokens / temp)` where `temp` follows a schedule (starts easier, gets harder)
  5. Student predictions: `log_softmax(student_tokens)`
  6. Loss: Cross-entropy (or focal loss) between teacher soft targets and student predictions
- **Formula (Standard)**: `L_MIM = -sum(teacher_soft * log(student_soft))` (only on masked patches)
- **Formula (Focal Loss)**: `L_MIM = -sum((1 - p_t)^γ * teacher_soft * log(student_soft))` where `p_t` is probability of correct class, `γ` controls focus on hard examples
- **NEW in v3**: 
  - Temperature schedule for tokenizer (curriculum learning)
  - Optional focal loss to focus on hard examples

**B. Self-Distillation Loss (DINO-style)**
- **Input:** CLS token projections
- **Process:**
  1. Teacher sees 2 global crops → produces 2 CLS projections
  2. Student sees all crops (2 global + 6-8 local) → produces 8-10 CLS projections
  3. Teacher centering: subtract running mean (prevents collapse)
  4. Temperature sharpening: `softmax((teacher - center) / temp)`
  5. Loss: Cross-entropy between teacher and student CLS projections
  6. Skip cases where teacher and student see the same view
- **Formula:** `L_CLS = -sum(teacher_soft * log(student_soft))` (on CLS tokens)

**C. KoLeo Regularization (from DINOv2) - DISABLED**
- **Status:** Disabled by default (`koleo_weight=0.0`)
- **Reason:** Original iBOT doesn't use KoLeo, and it was causing explosion issues
- **Alternative:** Teacher centering in CLS loss already prevents collapse
- **Note:** Can be re-enabled by setting `--koleo_weight > 0` if needed

**Total Loss:**
```
L_total = w_MIM * L_MIM + w_CLS * L_CLS + w_KoLeo * L_KoLeo
```

**Loss Component Monitoring:**
- Returns individual component losses for debugging
- Logs components every 5 epochs or every 100 iterations
- Helps identify which loss component is causing issues

#### 6. **Training Loop**

**Per iteration:**
1. **Generate mask:** Random mask for each image (30-35% patches masked)
2. **Forward pass:**
   - Teacher: Full images (no masking) → CLS + token predictions
   - Student: Same images → CLS + token predictions
3. **Compute loss:**
   - MIM loss on masked patches only
   - CLS loss on all crops
   - KoLeo loss: Disabled (0.0)
4. **Loss validation checks:**
   - Verify loss is in expected range (4-6 initially, not 12+)
   - Check component sum matches total loss
   - Detect batch size multiplication bugs
   - Monitor for unexpected increases
5. **Backward pass:** Update student only (with gradient clipping)
6. **Momentum update:** `teacher = m * teacher + (1-m) * student`
7. **Update center:** EMA update of teacher output center

**Key details:**
- Teacher is frozen (no gradients)
- Only student receives gradients
- Teacher updated via exponential moving average (EMA)
- Last layer gradients canceled for first epoch (prevents collapse)
- **Gradient clipping** at max_norm=1.0 for stability

#### 7. **Optimizer: AdamW (Default)**

**AdamW Configuration:**
- Learning rate: 0.0003-0.0005 (much lower than LARS's 0.1)
- Betas: (0.9, 0.999)
- Epsilon: 1e-8
- Weight decay: 0.04-0.05 (scheduled to 0.1-0.15)

**Why AdamW:**
- More stable for this setup than LARS
- Lower learning rates prevent training instability
- Better for smaller batch sizes
- Standard in modern SSL methods

**Learning Rate Schedules:**
- **Standard**: Cosine schedule with warmup (base_lr → min_lr)
- **Peak LR**: Warmup to peak (0.001), then cosine decay to min_lr

**LARS Optimizer** (still available):
- Layer-wise Adaptive Rate Scaling
- Formula: `local_lr = trust_coeff * ||params|| / (||grads|| + wd * ||params||)`
- Better for large batch training
- Use with `--optimizer lars` flag

### Data Flow Example

**Single training step:**

```
1. Input: Image (96x96x3)
   ↓
2. Augmentation: 8-10 crops (2 global + 6-8 local)
   ↓
3. Masking: Generate mask [36 patches] with 11-13 masked
   ↓
4. Teacher Forward:
   - Full images → ViT → [batch*2, 37, embed_dim]
   - CLS tokens → Head → [batch*2, out_dim]
   - All tokens → Tokenizer → [batch*2, 37, num_tokens]
   ↓
5. Student Forward:
   - Same images → ViT → [batch*8-10, 37, embed_dim]
   - CLS tokens → Head → [batch*8-10, out_dim]
   - All tokens → Tokenizer → [batch*8-10, 37, num_tokens]
   ↓
6. Loss Computation:
   - MIM: Compare student vs teacher tokens on masked patches
   - CLS: Compare student vs teacher CLS on all crops
   - KoLeo: Disabled (not computed)
   ↓
7. Loss Validation:
   - Check loss range (should be 4-6 initially)
   - Verify component sum
   - Detect multiplication bugs
   ↓
8. Backward: Update student only (with gradient clipping)
   ↓
9. EMA Update: teacher = 0.996 * teacher + 0.004 * student
```

### Why This Works

1. **MIM learns local features:** Predicting masked patches forces understanding of patch-level semantics
2. **Self-distillation learns global features:** CLS token consistency learns high-level semantics
3. **Combined = best of both:** Local + global understanding
4. **Teacher centering prevents collapse:** CLS loss has centering mechanism (alternative to KoLeo)
5. **Online tokenizer adapts:** Learns visual vocabulary specific to your data
6. **Stable training:** Conservative hyperparameters prevent instability

## Files

- `ibot_ssl.py`: Core iBOT implementation
  - `RandomMaskingGenerator`: Random patch masking (default)
  - `BlockwiseMaskingGenerator`: Block-wise masking (alternative)
  - `ProgressiveMaskingGenerator`: **NEW in v3** - Curriculum learning (gradually increases mask ratio)
  - `iBOTTokenizer`: **ENHANCED in v3** - Online tokenizer with LayerNorm, deeper architecture, SiLU activation
  - `iBOTHead`: **ENHANCED in v3** - Projection head with LayerNorm, SiLU activation, optional dropout
  - `MultiCropiBOTWrapper`: Model wrapper - handles crops, extracts CLS/tokens
  - `iBOTLoss`: **ENHANCED in v3** - Combined loss with optional focal loss and temperature scheduling
  - `LARS`: Layer-wise Adaptive Rate Scaling optimizer (optional)
  - `DataAugmentation`: Multi-crop augmentation (2 global + 6-8 local)
  - `cosine_scheduler`: Standard cosine LR schedule
  - `cosine_scheduler_with_peak`: Peak LR schedule (warmup to peak, then decay)

- `train_ibot.py`: Training script with full training loop
  - Loss computation validation checks
  - Loss component monitoring
  - Gradient clipping
  - Support for both AdamW and LARS optimizers

- `eval_ibot.py`: Evaluation script (k-NN on frozen features)
- `train_ibot_conservative_5.sh`, `train_ibot_aggressive_5.sh`, `train_ibot_feedback_5.sh`: SLURM training scripts (3 diverse variants)
- `requirements.txt`: Dependencies
- `TRAINING_VARIANTS.md`: Detailed comparison of the 3 training variants

## Usage

### Training

```bash
# Conservative configuration (stable baseline, recommended starting point)
sbatch train_ibot_conservative_5.sh

# Aggressive configuration (high-capacity exploration with LARS and ViT-small)
sbatch train_ibot_aggressive_5.sh

# Feedback configuration (strictly follows all training feedback recommendations)
sbatch train_ibot_feedback_5.sh
```

Or run directly:
```bash
python train_ibot.py \
    --data_path /path/to/data \
    --output_dir ./checkpoints \
    --arch vit_tiny \
    --optimizer adamw \
    --batch_size 128 \
    --lr 0.0003 \
    --max_lr 0.001 \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.1 \
    --clip_grad 1.0 \
    --mask_ratio 0.3 \
    --mim_loss_weight 1.0 \
    --cls_loss_weight 1.0 \
    --use_fp16
```

### Key Hyperparameters

**Optimizer:**
- `--optimizer`: `adamw` (default) or `lars`
- `--lr`: Base learning rate (default: 0.0003 for AdamW, 0.1 for LARS)
- `--max_lr`: Peak LR after warmup (optional, for peak LR schedule)
- `--min_lr`: Minimum learning rate (default: 1e-6)
- `--warmup_epochs`: Number of warmup epochs (default: 10)

**Regularization:**
- `--weight_decay`: Initial weight decay (default: 0.04)
- `--weight_decay_end`: Final weight decay (default: 0.1)
- `--clip_grad`: Gradient clipping max norm (default: 1.0)

**Masking:**
- `--mask_ratio`: Ratio of patches to mask (default: 0.3)
- `--mask_type`: `random` or `blockwise` (default: `random`)

**Loss Weights:**
- `--mim_loss_weight`: Weight for MIM loss (default: 1.0)
- `--cls_loss_weight`: Weight for self-distillation loss (default: 1.0)
- `--koleo_weight`: Weight for KoLeo regularization (default: 0.0, disabled - original iBOT doesn't use it)

**Tokenizer:**
- `--num_tokens`: Vocabulary size for tokenizer (default: 8192)
- `--tokenizer_hidden_dim`: Hidden dimension for tokenizer MLP (default: 512)

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
- **AdamW** (default)
  - Base LR: 0.0003-0.0005 (much lower than LARS's 0.1)
  - Betas: (0.9, 0.999)
  - Epsilon: 1e-8
  - Weight decay: 0.04-0.05 (scheduled to 0.1-0.15)
  - Gradient clipping: max_norm=1.0

### Training Process
1. **Teacher-Student Setup:**
   - Teacher: Processes full images (no masking), frozen (no gradients)
   - Student: Processes same images, receives gradients
   - Teacher updated via EMA: `θ_teacher = 0.996 * θ_teacher + 0.004 * θ_student`

2. **Masking:**
   - Generate random mask per image (30-35% patches masked)
   - Same mask for all crops of same image
   - Loss computed only on masked patches (MIM) and visible patches (CLS)

3. **Multi-Crop Augmentation:**
   - 2 global crops: Strong augmentation, larger scale (0.32-1.0)
   - 6-8 local crops: Weaker augmentation, smaller scale (0.05-0.32)
   - Teacher sees only global crops
   - Student sees all crops

4. **Learning Rate Schedule:**
   - **Standard**: Cosine decay with warmup (base_lr → min_lr)
   - **Peak LR**: Warmup to peak (0.001), then cosine decay to min_lr
   - Warmup: 10 epochs (linear increase)
   - Decay: Cosine schedule to minimum LR (1e-6)
   - **No batch size scaling for AdamW** (unlike LARS)

5. **Weight Decay Schedule:**
   - Cosine schedule: 0.04-0.05 → 0.1-0.15 (increases over training)
   - Much lower than previous versions (was 0.05 → 0.45)

6. **Temperature Schedule:**
   - Teacher temperature: 0.04 → 0.04 (warmup over 30 epochs)
   - Student temperature: 0.1 (fixed)
   - Controls sharpness of softmax distributions

7. **Loss Validation:**
   - Checks loss is in expected range (4-6 initially, not 12+)
   - Verifies component sum matches total loss
   - Detects batch size multiplication bugs
   - Monitors for unexpected increases

## Implementation Details

### Parameter Counting & Competition Compliance

**Assignment Requirement**: Backbone parameters must be strictly < 100M at train time.

**Parameter Breakdown:**
- **ViT-tiny backbone**: ~5.5M parameters ✅ (well under 100M limit)
- **ViT-small backbone**: ~22M parameters ✅ (well under 100M limit)
- **ViT-base backbone**: ~86M parameters ✅ (under 100M limit, but close)
- **Tokenizer**: ~1M parameters (per head) - NOT counted in backbone limit
- **Projection head**: ~2M parameters (per head) - NOT counted in backbone limit
- **Total model**: ~10-15M parameters (for vit_tiny)

**Automatic Verification:**
The training script automatically verifies backbone parameters at startup:
- Checks backbone parameters separately from total model
- Exits with error if backbone >= 100M
- Shows both backbone and total model counts

**Standard ViT Parameter Counts:**
| Architecture | Backbone Parameters | Status |
|--------------|---------------------|--------|
| **ViT-Tiny** | ~5.5M | ✅ **SAFE** (well under 100M) |
| **ViT-Small** | ~22M | ✅ **SAFE** (well under 100M) |
| **ViT-Base** | ~86M | ✅ **SAFE** (under 100M, but close) |
| **ViT-Large** | ~307M | ❌ **EXCEEDS LIMIT** |

**Note**: The assignment restricts **backbone parameters only**, not the total model (backbone + heads + tokenizers).

### Memory Considerations
- Batch size: 128 (with 8-10 crops = 1024-1280 images per batch)
- Mixed precision (FP16): Reduces memory by ~50%
- Gradient clipping: Prevents gradient explosion

### Evaluation
- **Frozen encoder**: Backbone is frozen during k-NN evaluation
- **Feature extraction**: Uses normalized CLS token (L2 normalized)
- **k-NN**: Cosine similarity on normalized features
- **No fine-tuning**: Features are used as-is (competition requirement)

## Loss Computation Checks

The training script includes automatic validation checks:

1. **Expected Range Check**: Warns if initial loss > 10-15 (should be 4-6)
2. **Component Sum Check**: Verifies total loss = sum of weighted components
3. **Batch Size Check**: Detects if loss is accidentally multiplied by batch size
4. **Increase Detection**: Warns if loss increases significantly after initial decrease

These checks help catch the issues mentioned in feedback where "loss should start around 4-6 and decrease to 1-3, not start at 12 and increase to 8+."

## Notes

- **Backbone must stay under 100M parameters** ✅ (ViT-tiny backbone is ~5.5M, well under limit)
- Image resolution fixed at 96x96 ✅
- Uses frozen encoder for evaluation (k-NN) ✅
- Random initialization (no pretrained weights) ✅
- All paths configured for `/gpfs/scratch/bm3772/` (ready to run)
- **Stable training**: Conservative hyperparameters prevent instability ✅
- **Automatic parameter verification**: Training script checks backbone < 100M at startup ✅

## Technical Highlights

1. **Untied Heads**: Separate projection heads for global/local crops (better than tied)
2. **Bottleneck Architecture**: Reduces overfitting in projection head
3. **KoLeo Regularization**: Disabled by default (original iBOT doesn't use it, teacher centering prevents collapse)
4. **Online Tokenizer**: Learns visual vocabulary during training (no pre-training)
5. **AdamW Optimizer**: Stable training with conservative learning rates
6. **Multi-Crop Strategy**: 8-10 crops per image (2 global + 6-8 local) for robust learning
7. **Loss Validation**: Automatic checks to catch computation bugs
8. **Gradient Clipping**: Prevents gradient explosion for stability
9. **Enhanced Architecture (v3)**: LayerNorm, SiLU activation, deeper tokenizer
10. **Progressive Masking (v3)**: Optional curriculum learning
11. **Focal Loss (v3)**: Optional focus on hard examples
12. **Parameter Verification**: Automatic check that backbone < 100M

## Changes from v2 to v3

**Architectural improvements (NEW):**
- Enhanced tokenizer: LayerNorm, deeper architecture (3 layers), SiLU activation
- Enhanced projection head: LayerNorm, SiLU activation, optional dropout
- Progressive masking generator: Curriculum learning option
- Enhanced loss function: Optional focal loss, temperature scheduling for tokenizer
- Parameter count verification: Automatic check that backbone < 100M

**Training stability (from v2):**
- Switched from LARS to AdamW optimizer
- Reduced learning rates (0.0003-0.0005 vs 0.1)
- Reduced weight decay (0.04→0.1 vs 0.05→0.45)
- Added gradient clipping (max_norm=1.0)
- Reduced mask ratio (0.3-0.35 vs 0.4)
- Added loss computation validation checks
- Added loss component monitoring

## Using v3 Enhancements

### Default Behavior
All enhancements are enabled by default with sensible settings:
- LayerNorm: **Enabled** (more stable)
- SiLU activation: **Enabled** (better performance)
- Progressive masking: **Disabled** (use standard masking)
- Focal loss: **Disabled** (use standard CE loss)

### Enable Progressive Masking
```python
# In train_ibot.py, replace RandomMaskingGenerator with:
from ibot_ssl import ProgressiveMaskingGenerator
mask_generator = ProgressiveMaskingGenerator(
    args.image_size,
    mask_ratio_start=0.2,
    mask_ratio_end=args.mask_ratio,
    total_epochs=args.epochs,
    patch_size=patch_size
)
# Update epoch each epoch:
mask_generator.set_epoch(epoch)
```

### Enable Focal Loss
```python
# When creating iBOTLoss:
ibot_loss = iBOTLoss(
    # ... existing parameters ...
    use_focal_loss=True,      # Enable focal loss
    focal_gamma=2.0,          # Focus on hard examples
    mim_temp=0.07             # Tokenizer temperature
)
```

See `ARCHITECTURAL_IMPROVEMENTS.md` for detailed documentation (if created separately).
