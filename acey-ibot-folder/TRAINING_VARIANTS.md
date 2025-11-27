# Training Variants

This folder contains 3 different training configurations to experiment with iBOT training:

---

## Version 1: Base Configuration (`train_ibot_v1_1.sh`)

**Philosophy**: Balanced approach - equal emphasis on MIM and self-distillation

**Key Settings**:
- **Mask ratio: 0.4** - Standard masking (40% of patches)
- **MIM loss weight: 1.0** - Standard weight for masked patch prediction
- **CLS loss weight: 1.0** - Standard weight for self-distillation
- **KoLeo weight: 0.001** - Standard feature diversity regularization
- **Drop path rate: 0.2** - Moderate regularization
- **Local crops: 6** - Standard multi-crop augmentation
- **Weight decay: 0.05 → 0.45** - Standard weight decay schedule

**Justification**:
- This is the **baseline** configuration that balances both learning objectives
- Equal weights (1.0/1.0) let the model learn from both masked patches and global consistency
- Standard masking ratio (0.4) is a proven sweet spot from iBOT/MAE literature
- Good starting point to establish baseline performance

**Expected Outcomes**:
- Balanced local and global feature learning
- Good generalization across different downstream tasks
- Stable training dynamics

---

## Version 2: MIM-Focused (`train_ibot_v2_2.sh`)

**Philosophy**: Emphasize masked image modeling - learn stronger local/patch-level features

**Key Differences from V1**:
- **Mask ratio: 0.5** ⬆️ (50% vs 40%) - More challenging MIM task
- **MIM loss weight: 1.5** ⬆️ (vs 1.0) - Stronger emphasis on masked prediction
- **CLS loss weight: 0.8** ⬇️ (vs 1.0) - Reduced self-distillation weight
- **Drop path rate: 0.25** ⬆️ (vs 0.2) - Stronger regularization to prevent overfitting
- **Weight decay: 0.06 → 0.5** ⬆️ (vs 0.05 → 0.45) - Higher regularization

**Justification**:
- **Higher masking (0.5)**: Forces model to learn from more challenging scenarios where 50% of patches are missing. This encourages better understanding of spatial relationships and local patterns.
- **Higher MIM weight (1.5)**: Prioritizes learning to predict masked patches, which should improve local feature quality.
- **Lower CLS weight (0.8)**: Reduces emphasis on global consistency to allow more focus on local learning.
- **Stronger regularization**: With more challenging masking, we need stronger regularization to prevent overfitting to the MIM task.

**Expected Outcomes**:
- **Better local/patch-level features** - Model learns stronger representations of image patches
- **Better for dense tasks** - Potentially better for segmentation, detection, or other tasks requiring fine-grained understanding
- **May converge slower** - More challenging task might need more epochs

**When to Use**: If you suspect your downstream task benefits more from local features than global semantics

---

## Version 3: Self-Distillation Focused (`train_ibot_v3_3.sh`)

**Philosophy**: Emphasize DINO-style self-distillation - learn stronger global/semantic features

**Key Differences from V1**:
- **Mask ratio: 0.3** ⬇️ (30% vs 40%) - Easier MIM task
- **MIM loss weight: 0.8** ⬇️ (vs 1.0) - Reduced emphasis on masked prediction
- **CLS loss weight: 1.2** ⬆️ (vs 1.0) - Stronger emphasis on self-distillation
- **KoLeo weight: 0.002** ⬆️ (vs 0.001) - More feature diversity
- **Local crops: 8** ⬆️ (vs 6) - More augmentation diversity

**Justification**:
- **Lower masking (0.3)**: Easier MIM task means less focus on local reconstruction, allowing more capacity for global learning.
- **Lower MIM weight (0.8)**: Reduces the importance of masked patch prediction.
- **Higher CLS weight (1.2)**: Prioritizes the self-distillation objective (similar to DINO), which learns global semantic features.
- **Higher KoLeo (0.002)**: Encourages more diverse feature representations, complementing the global learning.
- **More local crops (8)**: Provides more diverse views for self-distillation, similar to DINO's multi-crop strategy.

**Expected Outcomes**:
- **Better global/semantic features** - Model learns stronger high-level representations
- **More DINO-like behavior** - Closer to pure self-distillation approach
- **Better for classification** - Potentially better for tasks requiring semantic understanding

**When to Use**: If you suspect your downstream task benefits more from global semantic features (like classification)

---

## Detailed Comparison Table

| Setting | V1 (Base) | V2 (MIM-Focused) | V3 (Distillation-Focused) |
|---------|-----------|------------------|---------------------------|
| **Mask Ratio** | 0.4 | **0.5** ⬆️ | **0.3** ⬇️ |
| **MIM Loss Weight** | 1.0 | **1.5** ⬆️ | **0.8** ⬇️ |
| **CLS Loss Weight** | 1.0 | **0.8** ⬇️ | **1.2** ⬆️ |
| **KoLeo Weight** | 0.001 | 0.001 | **0.002** ⬆️ |
| **Drop Path Rate** | 0.2 | **0.25** ⬆️ | 0.2 |
| **Local Crops** | 6 | 6 | **8** ⬆️ |
| **Weight Decay Start** | 0.05 | **0.06** ⬆️ | 0.05 |
| **Weight Decay End** | 0.45 | **0.5** ⬆️ | 0.45 |

## Usage

Run any variant:
```bash
sbatch train_ibot_v1_1.sh  # Base configuration (balanced)
sbatch train_ibot_v2_2.sh  # MIM-focused (local features)
sbatch train_ibot_v3_3.sh  # Self-distillation focused (global features)
```

## Which One to Use?

### Start with V1 (Base)
- **Best starting point** - Balanced configuration
- Establishes baseline performance
- Good for general-purpose features

### Try V2 (MIM-Focused) if:
- Your downstream task requires **fine-grained/local understanding**
- Examples: Object detection, semantic segmentation, dense prediction
- You want stronger **patch-level features**

### Try V3 (Distillation-Focused) if:
- Your downstream task requires **semantic/global understanding**
- Examples: Image classification, scene understanding
- You want stronger **high-level features** (more DINO-like)

## Strategy

1. **Run V1 first** to establish baseline
2. **Run V2 and V3 in parallel** to compare
3. **Evaluate on your downstream task** (k-NN on eval set)
4. **Choose the best performing variant** or ensemble features

You can run all 3 in parallel to compare results efficiently!

