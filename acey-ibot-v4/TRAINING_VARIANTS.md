# Training Variants - Architecture Experiments

This folder contains training configurations for **architecture experiments**, comparing different backbone architectures (ViT vs CNN) for iBOT.

---

## ResNet18 Experiment (`train_ibot_resnet.sh`)

**Philosophy**: Test classic CNN backbone (ResNet18) vs ViT for iBOT

**Key Settings**:
- **Architecture**: ResNet18 (~11M backbone params) - Classic CNN
- **Optimizer**: AdamW (stable)
- **Learning rate: 0.0003** - Low base LR for stable convergence
- **Mask ratio: 0.3** - Easier learning task
- **Masking**: Random
- **Weight decay: 0.04 → 0.1** - Conservative regularization
- **Gradient clipping: 1.0** - Conservative clipping
- **Batch size: 128** - Standard batch size
- **MIM loss weight: 1.0** - Balanced
- **CLS loss weight: 1.0** - Balanced
- **Local crops: 6** - Standard augmentation

**Why ResNet18**:
- **Classic architecture**: Well-understood, proven design (2015)
- **Small and fast**: ~11M params, fast training
- **Inductive biases**: Translation equivariance, locality
- **Good baseline**: Compare CNN vs ViT

**Expected Outcomes**:
- **Stable training** - ResNet is well-tested
- **Fast training** - CNN often faster than ViT
- **Baseline performance** - See if CNN can match ViT

**When to Use**: 
- **Start here for CNN experiments** - Classic, reliable CNN baseline
- Compare CNN vs ViT performance
- Test if CNN inductive biases help with 96x96 images

---

## ConvNeXt-Tiny Experiment (`train_ibot_convnext.sh`)

**Philosophy**: Test modern CNN backbone (ConvNeXt) vs ViT for iBOT

**Key Settings**:
- **Architecture**: ConvNeXt-Tiny (~28M backbone params) - Modern CNN
- **Optimizer**: AdamW (stable)
- **Learning rate: 0.0003** - Low base LR for stable convergence
- **Mask ratio: 0.3** - Easier learning task
- **Masking**: Random
- **Weight decay: 0.04 → 0.1** - Conservative regularization
- **Gradient clipping: 1.0** - Conservative clipping
- **Batch size: 128** - Standard batch size
- **MIM loss weight: 1.0** - Balanced
- **CLS loss weight: 1.0** - Balanced
- **Local crops: 6** - Standard augmentation

**Why ConvNeXt-Tiny**:
- **Modern design**: 2022 architecture, ViT-inspired CNN
- **Best performance potential**: Often matches ViT performance
- **More capacity**: ~28M params (still well under 100M)
- **Modern techniques**: Large kernels, LayerNorm, GELU

**Expected Outcomes**:
- **Best CNN performance** - Modern design should perform well
- **Potentially match ViT** - ConvNeXt often competitive with ViT
- **Good balance** - CNN efficiency + modern design

**When to Use**: 
- **Best CNN candidate** - Most likely to match ViT performance
- Test modern CNN design principles
- Compare against ResNet18 and ViT

---

## Architecture Comparison

### ResNet vs ConvNeXt vs ViT

| Feature | ResNet18 | ConvNeXt-Tiny | ViT-tiny (baseline) |
|---------|----------|---------------|---------------------|
| **Year** | 2015 | 2022 | 2020 |
| **Parameters** | ~11M | ~28M | ~5.5M |
| **Design** | Classic CNN | Modern CNN | Vision Transformer |
| **Inductive Biases** | Translation equivariance, locality | Translation equivariance + modern | Minimal (attention-based) |
| **Key Innovation** | Skip connections | Large kernels + LayerNorm | Patch-based attention |
| **Activation** | ReLU | GELU | GELU |
| **Normalization** | Batch Norm | Layer Norm | Layer Norm |
| **Expected Speed** | Fast | Moderate | Moderate |
| **Expected Accuracy** | Baseline | High | High (baseline) |

### Architecture Differences Explained

**ResNet (2015)**:
- Classic residual blocks with skip connections
- Standard 3x3 convolutions
- Batch normalization
- ReLU activation
- **Pros**: Simple, proven, fast
- **Cons**: Older design, less efficient

**ConvNeXt (2022)**:
- Modern CNN with ViT-inspired design
- Large 7x7 kernels (instead of 3x3)
- Layer normalization (like ViT)
- GELU activation (like ViT)
- Depthwise convolutions
- **Pros**: Modern, competitive with ViT, good efficiency
- **Cons**: More parameters, newer (less tested)

**ViT (2020)**:
- Patch-based attention mechanism
- Minimal inductive biases
- Global receptive field from start
- **Pros**: State-of-the-art for SSL, attention mechanism
- **Cons**: Needs more data, less efficient than CNN

## Usage

Run architecture experiments:
```bash
# ResNet18 (classic CNN baseline)
sbatch train_ibot_resnet.sh

# ConvNeXt-Tiny (modern CNN)
sbatch train_ibot_convnext.sh

# For ViT baseline, use scripts from acey-ibot-v3
```

## Experiment Strategy

1. **Start with ResNet18** - Classic CNN baseline
   - Fast to train
   - Well-understood
   - Good comparison point

2. **Try ConvNeXt-Tiny** - Modern CNN
   - Best performance potential
   - Modern design principles
   - Compare against ResNet and ViT

3. **Compare with ViT** - Use v3 scripts as baseline
   - ViT-tiny from v3 as reference
   - See which architecture works best

4. **Evaluate** - Compare on downstream task
   - k-NN accuracy on test set
   - Training stability
   - Training speed

## What to Compare

### Training Metrics
- **Loss convergence**: Which architecture trains most stably?
- **Training speed**: Iterations per second
- **Memory usage**: GPU memory requirements

### Downstream Performance
- **k-NN accuracy**: Which gives best features?
- **Feature quality**: How discriminative are the features?

### Architecture Insights
- **CNN vs ViT**: Do CNN inductive biases help?
- **Modern vs Classic**: Does ConvNeXt outperform ResNet?
- **Efficiency**: Which is most parameter-efficient?

## Expected Results (Hypotheses)

### Performance Ranking (Hypothesis)
1. **ConvNeXt-Tiny**: Best performance (modern design)
2. **ViT-tiny**: Strong baseline (attention mechanism)
3. **ResNet18**: Baseline (classic but proven)

### Training Speed (Hypothesis)
1. **ResNet18**: Fastest (simple CNN)
2. **ConvNeXt-Tiny**: Moderate (more complex)
3. **ViT-tiny**: Moderate (attention overhead)

### Parameter Efficiency (Hypothesis)
1. **ViT-tiny**: Most efficient (~5.5M)
2. **ResNet18**: Efficient (~11M)
3. **ConvNeXt-Tiny**: Less efficient (~28M) but more capacity

## Competition Compliance

All architectures comply with competition requirements:
- ✅ **Backbone < 100M**: 
  - ResNet18: ~11M ✅
  - ConvNeXt-Tiny: ~28M ✅
  - ViT-tiny: ~5.5M ✅
- ✅ **Image resolution**: 96x96 (fixed)
- ✅ **Frozen encoder**: For evaluation (k-NN)
- ✅ **Random initialization**: No pretrained weights

## Training Stability Notes

**KoLeo Loss Disabled**: KoLeo loss has been disabled (set to 0.0) because original iBOT doesn't use it and it was causing explosion issues. Teacher centering in CLS loss already prevents feature collapse.

**Expected Loss Behavior**:
- Initial loss: 4-6 (not 12+ or 17+)
- MIM loss: Should not collapse to 0 (should decrease from ~4-9 to ~0.1-2)
- CLS loss: Should decrease over time (from ~6-8 to ~2-4)
- KoLeo loss: Disabled (0.0) - not computed

## CNN Implementation Details

### How CNN Works with iBOT

CNNs output feature maps `[B, C, H, W]`, not patches. The `CNNFeatureExtractor`:
1. Extracts spatial features from CNN feature maps
2. Uses adaptive pooling to match ViT patch grid (6x6 for 96x96 images)
3. Projects to target embedding dimension
4. Adds CLS token (like ViT)

This makes CNN features compatible with existing iBOT masking and tokenization.

### Supported Architectures

- **ResNet**: `resnet18`, `resnet34`, `resnet50`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`
- **ConvNeXt**: `convnext_tiny`, `convnext_small`
- **ViT**: `vit_tiny`, `vit_small`, `vit_base` (from v3)

All can be used with `--arch` flag in training script.
