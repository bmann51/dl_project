# Training Variants

This folder contains 3 different training configurations for **diverse exploration** of iBOT training:

---

## Conservative Configuration (`train_ibot_conservative_7.sh`)

**Philosophy**: Stability and reliability over aggressive optimization

**Key Settings**:
- **Optimizer**: AdamW (more stable than LARS)
- **Architecture**: ViT-tiny (~5.5M backbone params)
- **Learning rate: 0.0003** - Low base LR for stable convergence
- **Mask ratio: 0.3** - Easier learning task
- **Masking**: Random (simpler, more diverse patterns)
- **Weight decay: 0.04 → 0.1** - Conservative regularization
- **Gradient clipping: 1.0** - Conservative clipping for stability
- **Batch size: 128** - Standard batch size
- **MIM loss weight: 1.0** - Balanced with CLS loss
- **CLS loss weight: 1.0** - Balanced with MIM loss
- **Local crops: 6** - Standard multi-crop augmentation
- **Drop path rate: 0.2** - Moderate regularization

**Justification**:
- **AdamW**: More stable than LARS for this setup
- **Lower LR (0.0003)**: Prevents training instability
- **Lower mask ratio (0.3)**: Easier task initially, allows model to learn more gradually
- **Lower weight decay (0.04→0.1)**: Less aggressive regularization
- **Gradient clipping (1.0)**: Prevents gradient explosion, critical for stability
- **ViT-tiny**: Fast training, well under 100M limit

**Expected Outcomes**:
- **Stable training** - Loss should start around 4-6 and decrease to 1-3
- **Consistent convergence** - No sudden loss spikes or training collapse
- **Good baseline** - Conservative but reliable performance

**When to Use**: 
- **Start here** - Best starting point for stable training
- If previous training was unstable or loss was increasing
- When you want a reliable baseline

---

## Aggressive Configuration (`train_ibot_aggressive_7.sh`)

**Philosophy**: Push the limits with larger model and aggressive settings

**Key Settings**:
- **Optimizer**: LARS (better for large batch, higher LR)
- **Architecture**: ViT-small (~22M backbone params) - More capacity
- **Learning rate: 0.1** - High LR for faster convergence
- **Mask ratio: 0.4** - More challenging MIM task
- **Masking**: Blockwise (structured, harder to predict)
- **Weight decay: 0.05 → 0.2** - More aggressive regularization
- **Gradient clipping: 1.0** - Still conservative
- **Batch size: 128** - Standard batch size
- **MIM loss weight: 1.5** - Emphasis on masked prediction
- **CLS loss weight: 1.0** - Standard weight
- **Local crops: 8** - More augmentation diversity
- **Drop path rate: 0.25** - More aggressive regularization
- **KoLeo weight: 0.0** - Disabled (original iBOT doesn't use it)

**Justification**:
- **LARS**: Better for large batch training with high LR
- **ViT-small**: More representational capacity (still under 100M)
- **Higher LR (0.1)**: Faster convergence, LARS can handle it
- **Higher mask ratio (0.4)**: More challenging task, better local features
- **Blockwise masking**: Structured patterns are harder to predict
- **Higher MIM weight (1.5)**: Emphasizes patch-level learning
- **More local crops (8)**: Richer augmentation for self-distillation

**Expected Outcomes**:
- **Faster convergence** - Higher LR speeds up training
- **Better local features** - Higher MIM weight and mask ratio
- **More capacity** - ViT-small has more parameters for learning
- **Potentially better performance** - If training is stable

**When to Use**: 
- After conservative config works well
- When you want to explore model capacity
- If you want faster training (with LARS)
- When you want to emphasize local/patch-level features

---

## Feedback Configuration (`train_ibot_feedback_7.sh`)

**Philosophy**: Strictly follow training feedback recommendations exactly

**Key Settings**:
- **Optimizer**: AdamW (not LARS - LARS was going to 0.05 which is 50x higher)
- **Architecture**: ViT-tiny (~5.5M backbone params)
- **Learning rate: 0.0003** - Base LR (low)
- **Peak LR: 0.001** - Peak after warmup (as suggested in feedback)
- **Min LR: 1e-6** - Minimum LR
- **Warmup epochs: 10** - Standard warmup
- **Mask ratio: 0.3** - Easier learning task (down from 0.4)
- **Masking**: Random
- **Weight decay: 0.04 → 0.1** - Don't ramp to 0.4+ (was ramping to 0.35+)
- **Gradient clipping: 1.0** - Conservative (not 3.0)
- **Batch size: 96** - Smaller batches for more stable updates (down from 128)
- **MIM loss weight: 1.0** - Standard weight
- **CLS loss weight: 1.0** - Standard weight
- **Local crops: 6** - Standard augmentation
- **Drop path rate: 0.2** - Moderate regularization

**Justification**:
- **Follows feedback exactly**: Implements all 7 feedback recommendations
- **Lower LR (0.0003)**: Prevents instability (LARS was 50x higher)
- **Peak LR schedule**: Warmup to 0.001, then cosine decay
- **Smaller batch (96)**: More gradient updates = more stable (~1.3x longer training)
- **Lower weight decay (0.04→0.1)**: Not ramping to 0.4+ (was too aggressive)
- **Gradient clipping (1.0)**: Conservative, can increase if training too slow
- **Lower mask ratio (0.3)**: Easier learning task initially

**Expected Outcomes**:
- **Most stable training** - Follows all feedback recommendations
- **Loss behavior**: Should start around 4-6 and decrease to 1-3 (NOT start at 12 and increase to 8+)
- **Consistent convergence** - All stability measures in place
- **Best following feedback** - Most closely matches feedback recommendations

**When to Use**: 
- **When you want to follow feedback exactly** - Implements all recommendations
- If previous training was unstable
- When you want the most stable configuration
- As a reference implementation of feedback

---

## Detailed Comparison Table

| Setting | Conservative | Aggressive | Feedback |
|---------|--------------|------------|----------|
| **Optimizer** | AdamW | **LARS** ⬆️ | AdamW |
| **Architecture** | ViT-tiny | **ViT-small** ⬆️ | ViT-tiny |
| **Base LR** | 0.0003 | **0.1** ⬆️ | 0.0003 |
| **Peak LR** | N/A | N/A | **0.001** ⬆️ |
| **LR Schedule** | Standard cosine | Standard cosine | **Peak LR schedule** ⬆️ |
| **Batch Size** | 128 | 128 | **96** ⬇️ |
| **Mask Ratio** | 0.3 | **0.4** ⬆️ | 0.3 |
| **Mask Type** | Random | **Blockwise** ⬆️ | Random |
| **Weight Decay Start** | 0.04 | 0.05 | 0.04 |
| **Weight Decay End** | 0.1 | **0.2** ⬆️ | 0.1 |
| **Gradient Clipping** | 1.0 | 1.0 | 1.0 |
| **MIM Loss Weight** | 1.0 | **1.5** ⬆️ | 1.0 |
| **CLS Loss Weight** | 1.0 | 1.0 | 1.0 |
| **Local Crops** | 6 | **8** ⬆️ | 6 |
| **Drop Path Rate** | 0.2 | **0.25** ⬆️ | 0.2 |
| **KoLeo Weight** | **0.0** (disabled) | **0.0** (disabled) | **0.0** (disabled) |

## Usage

Run any variant:
```bash
sbatch train_ibot_conservative_7.sh    # Stable baseline
sbatch train_ibot_aggressive_7.sh      # High-capacity exploration
sbatch train_ibot_feedback_7.sh        # Follows feedback exactly
```

## Which One to Use?

### Start with Conservative (`train_ibot_conservative_7.sh`)
- **Best starting point** - Most stable, reliable baseline
- Conservative settings prevent instability
- Lowest risk of training collapse
- Use if you want a safe, reliable configuration

### Try Aggressive (`train_ibot_aggressive_7.sh`) if:
- Conservative config works well and you want to explore capacity
- You want faster training (LARS with high LR)
- You want to emphasize local features (higher MIM weight, blockwise masking)
- You want to use a larger model (ViT-small)

### Use Feedback (`train_ibot_feedback_7.sh`) if:
- **You want to follow feedback exactly** - Implements all 7 recommendations
- Previous training was unstable
- You want the most stable configuration
- You want smaller batch size for more stable updates

## Strategy

1. **Start with Conservative** - Establish baseline with stable configuration
2. **If stable works**: Try Aggressive for potentially better performance with more capacity
3. **For maximum stability**: Use Feedback configuration (follows all recommendations)
4. **Compare results**: Evaluate on downstream task (k-NN) to see which performs best
5. **Choose best**: Use the variant that gives best validation performance

You can run all 3 in parallel to compare results efficiently!

## Key Differences

### Conservative vs Aggressive
- **Optimizer**: AdamW (stable) vs LARS (fast)
- **Architecture**: ViT-tiny vs ViT-small (more capacity)
- **Learning rate**: 0.0003 vs 0.1 (100x difference!)
- **Masking**: Random vs Blockwise (structured)
- **Mask ratio**: 0.3 vs 0.4 (easier vs harder)
- **Loss emphasis**: Balanced vs MIM-focused

### Feedback vs Others
- **Strictly follows feedback**: All 7 recommendations implemented
- **Smaller batch**: 96 vs 128 (more stable updates)
- **Peak LR schedule**: Warmup to 0.001, then decay
- **Most conservative**: All stability measures in place

## Expected Loss Behavior

With these configurations, you should see:
- **Initial loss**: 4-6 (not 12+ or 17+)
- **Decreasing trend**: Loss should decrease over time (not increase to 8+)
- **Final loss**: 1-3 after training
- **Stable training**: No sudden spikes or collapses
- **MIM loss**: Should not collapse to 0 (should decrease from ~4-9 to ~0.1-2)
- **CLS loss**: Should decrease over time (from ~6-8 to ~2-4)
- **KoLeo loss**: Disabled (0.0) - not computed

If you see loss > 10 initially or loss increasing, the validation checks will warn you!

**Note**: KoLeo loss has been disabled (set to 0.0) because original iBOT doesn't use it and it was causing explosion issues. Teacher centering in CLS loss already prevents feature collapse.

## Competition Compliance

All three configurations comply with competition requirements:
- ✅ **Backbone < 100M**: ViT-tiny (~5.5M) and ViT-small (~22M) both under limit
- ✅ **Image resolution**: 96x96 (fixed)
- ✅ **Frozen encoder**: For evaluation (k-NN)
- ✅ **Random initialization**: No pretrained weights
