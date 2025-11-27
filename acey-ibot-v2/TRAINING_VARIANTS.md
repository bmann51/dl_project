# Training Variants

This folder contains 3 different training configurations optimized for **stable training** based on feedback:

---

## Stable Configuration (`train_ibot_stable_2.sh`)

**Philosophy**: Conservative settings for maximum stability - follows feedback recommendations closely

**Key Settings**:
- **Optimizer**: AdamW (not LARS)
- **Learning rate: 0.0003** - Low base LR for stability
- **Mask ratio: 0.4** - Standard masking (prevents MIM collapse)
- **Weight decay: 0.04 → 0.1** ⬇️ (vs original 0.05 → 0.45) - Much lower regularization
- **Gradient clipping: 1.0** - Conservative clipping for stability
- **MIM loss weight: 1.0** - Standard weight
- **CLS loss weight: 1.0** - Standard weight
- **KoLeo weight: 0.0** - Disabled (original iBOT doesn't use it)
- **Local crops: 6** - Standard multi-crop augmentation
- **Drop path rate: 0.2** - Moderate regularization

**Justification**:
- **Lower LR (0.0003)**: Prevents training instability (original LARS was going to 0.05, which is 100x higher)
- **Standard mask ratio (0.4)**: Prevents MIM loss collapse to zero, maintains learning signal
- **Lower weight decay (0.04→0.1)**: Original was ramping to 0.45 which is very aggressive
- **Gradient clipping (1.0)**: Prevents gradient explosion, critical for stability
- **KoLeo disabled (0.0)**: Original iBOT doesn't use it, prevents explosion issues
- **Teacher centering**: CLS loss has centering which prevents collapse (alternative to KoLeo)
- **AdamW**: More stable than LARS for this setup

**Expected Outcomes**:
- **Stable training** - Loss should start around 4-6 and decrease to 1-3 (not increase to 8+)
- **Consistent convergence** - No sudden loss spikes or training collapse
- **Good baseline** - Conservative but reliable performance

**When to Use**: 
- **Start here** - Best starting point for stable training
- If previous training was unstable or loss was increasing
- When you want to follow feedback recommendations exactly

---

## Balanced Configuration (`train_ibot_balanced_2.sh`)

**Philosophy**: Moderate settings - slightly more aggressive but still stable

**Key Differences from Stable**:
- **Learning rate: 0.0005** ⬆️ (vs 0.0003) - Slightly higher for faster learning
- **Mask ratio: 0.35** ⬆️ (vs 0.3) - Slightly more challenging
- **Weight decay: 0.05 → 0.15** ⬆️ (vs 0.04 → 0.1) - Slightly higher regularization
- **MIM loss weight: 1.2** ⬆️ (vs 1.0) - More emphasis on masked prediction
- **Local crops: 8** ⬆️ (vs 6) - More augmentation diversity
- **Gradient clipping: 1.0** - Same conservative clipping

**Justification**:
- **Higher LR (0.0005)**: Still conservative but allows faster learning
- **Higher mask ratio (0.35)**: Between stable (0.3) and original (0.4), balanced challenge
- **Higher MIM weight (1.2)**: Emphasizes masked patch prediction for better local features
- **More local crops (8)**: Better augmentation diversity for self-distillation
- **Still stable**: All settings remain conservative enough for stability

**Expected Outcomes**:
- **Faster convergence** - Higher LR may speed up training
- **Better local features** - Higher MIM weight emphasizes patch-level learning
- **Still stable** - Conservative enough to avoid instability
- **Better representation learning** - More crops and balanced masking

**When to Use**: 
- After stable config works well, try this for potentially better performance
- If you want slightly faster training without sacrificing stability
- When you want to emphasize local/patch-level features

---

## Peak LR Configuration (`train_ibot_peak_lr_2.sh`)

**Philosophy**: Follows feedback's peak LR recommendation exactly - warmup to peak, then decay

**Key Differences from Stable**:
- **Learning rate: 0.0003** - Same base LR
- **Peak LR: 0.001** ⬆️ - Warms up to 0.001 (as suggested in feedback)
- **LR Schedule**: Peak LR schedule (warmup to 0.001, then cosine decay to min_lr)
- **All other settings**: Same as stable (mask_ratio=0.3, wd=0.04→0.1, clip_grad=1.0)

**Justification**:
- **Peak LR (0.001)**: Feedback suggested "max_lr = 0.001 peak after warmup"
- **Different schedule**: Instead of using base_lr as peak, warms up to higher peak then decays
- **Best of both**: Higher peak allows faster initial learning, then decays for stability
- **Follows feedback exactly**: Implements the peak LR recommendation precisely

**Expected Outcomes**:
- **Faster initial learning** - Higher peak LR (0.001) allows more aggressive early training
- **Stable later training** - Cosine decay ensures stability in later epochs
- **Best following feedback** - Most closely matches the feedback recommendations
- **Potentially best performance** - Peak LR schedule often works well in practice

**When to Use**: 
- When you want to follow feedback's peak LR recommendation exactly
- If you want faster initial learning with stable later training
- As an alternative to stable config that might converge faster

---

## Detailed Comparison Table

| Setting | Stable | Balanced | Peak LR |
|---------|--------|----------|---------|
| **Optimizer** | AdamW | AdamW | AdamW |
| **Base LR** | 0.0003 | 0.0005 | 0.0003 |
| **Peak LR** | N/A (uses base_lr) | N/A (uses base_lr) | **0.001** ⬆️ |
| **LR Schedule** | Standard cosine | Standard cosine | **Peak LR schedule** ⬆️ |
| **Mask Ratio** | 0.4 | **0.35** ⬆️ | 0.3 |
| **KoLeo Weight** | **0.0** (disabled) | **0.0** (disabled) | **0.0** (disabled) |
| **Weight Decay Start** | 0.04 | **0.05** ⬆️ | 0.04 |
| **Weight Decay End** | 0.1 | **0.15** ⬆️ | 0.1 |
| **Gradient Clipping** | 1.0 | 1.0 | 1.0 |
| **MIM Loss Weight** | 1.0 | **1.2** ⬆️ | 1.0 |
| **CLS Loss Weight** | 1.0 | 1.0 | 1.0 |
| **Local Crops** | 6 | **8** ⬆️ | 6 |
| **Drop Path Rate** | 0.2 | 0.2 | 0.2 |

## Usage

Run any variant:
```bash
sbatch train_ibot_stable_2.sh    # Conservative, most stable
sbatch train_ibot_balanced_2.sh  # Moderate, faster learning
sbatch train_ibot_peak_lr_2.sh  # Peak LR schedule
```

## Which One to Use?

### Start with Stable (`train_ibot_stable_2.sh`)
- **Best starting point** - Most conservative, most stable
- Follows feedback recommendations closely
- Lowest risk of training instability
- Use if previous training was unstable

### Try Balanced (`train_ibot_balanced_2.sh`) if:
- Stable config works well and you want potentially better performance
- You want slightly faster training
- You want to emphasize local/patch-level features (higher MIM weight)
- You want more augmentation diversity (8 local crops)

### Try Peak LR (`train_ibot_peak_lr_2.sh`) if:
- You want to follow feedback's peak LR recommendation exactly
- You want faster initial learning with stable later training
- You want to experiment with the peak LR schedule

## Strategy

1. **Start with Stable** - Establish baseline with most stable configuration
2. **If stable works**: Try Balanced or Peak LR for potentially better performance
3. **Compare results**: Evaluate on downstream task (k-NN) to see which performs best
4. **Choose best**: Use the variant that gives best validation performance

You can run all 3 in parallel to compare results efficiently!

## Key Improvements from Original

All three variants include these stability improvements:

1. **AdamW Optimizer** (instead of LARS) - More stable for this setup
2. **Lower Learning Rates** (0.0003-0.0005 vs 0.1) - Prevents instability
3. **Reduced Weight Decay** (0.04→0.1 vs 0.05→0.45) - Less aggressive regularization
4. **Gradient Clipping** (max_norm=1.0) - Prevents gradient explosion
5. **Optimized Mask Ratio** (0.3-0.4, adjusted per variant) - Balanced learning task
6. **KoLeo Loss Disabled** - Original iBOT doesn't use it, prevents explosion issues
7. **Loss Computation Checks** - Automatic validation to catch bugs
8. **Loss Component Monitoring** - Better debugging and diagnostics

## Expected Loss Behavior

With these improvements, you should see:
- **Initial loss**: 4-6 (not 12+ or 17+)
- **Decreasing trend**: Loss should decrease over time (not increase to 8+)
- **Final loss**: 1-3 after training
- **Stable training**: No sudden spikes or collapses
- **MIM loss**: Should not collapse to 0 (should decrease from ~4-9 to ~0.1-2)
- **CLS loss**: Should decrease over time (from ~6-8 to ~2-4)
- **KoLeo loss**: Disabled (0.0) - not computed

If you see loss > 10 initially or loss increasing, the validation checks will warn you!
