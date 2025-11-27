# iBOT Implementation (macey v6 - Liberal Variant)

This folder contains an iBOT (Image BERT Pre-Training with Online Tokenizer) implementation focused on an **optimized “liberal” configuration** (formerly the conservative v6 config):

- Same core architecture as v5 (ViT teacher–student, online tokenizer, multi-crop).
- Adds **progressive masking** and **optional focal MIM loss** wired into the training loop.
- Uses a **peak LR schedule** for smoother, more stable optimization.

## What’s New in macey v6 (vs v5)

- **Progressive Masking (Curriculum)**
  - Flag: `--progressive_masking` (plus `--mask_ratio_start`).
  - Uses `ProgressiveMaskingGenerator` to ramp the mask ratio from an easier value (e.g. 0.20) up to the target (e.g. 0.35) over training.
  - Helps stability by starting with an easier masked patch prediction task.

- **Focal MIM Loss (Optional)**
  - Flags: `--use_focal_loss`, `--focal_gamma`.
  - When enabled, `iBOTLoss` applies focal loss on the MIM term, putting more weight on hard masked patches and less on trivial ones.
  - Can improve local feature quality without destabilizing training.

- **Peak LR Schedule for AdamW**
  - `train_ibot.py` uses `--max_lr` to optionally switch to a **peak LR schedule**:
    - Warmup from `lr` → `max_lr`, then cosine decay from `max_lr` → `min_lr`.
  - This is enabled in the macey v6 liberal script (`max_lr=0.001`).

- **Conservative but Slightly Stronger Head**
  - Default `bottleneck_dim=256`, `out_dim=4096`:
    - More expressive than v5’s tiny head, but still much smaller than huge 65k heads.

## Liberal Script

The main entry point in this folder is:

- `liberal.sh`: **macey v6 “liberal” configuration** (same hyperparameters as the v6 optimized conservative config).

Key hyperparameters in this script:

- **Backbone**: `vit_tiny` (well under 100M backbone params)
- **Optimizer**: `adamw`
- **Batch size**: `128`
- **LR schedule**:
  - `lr=0.0003`, `max_lr=0.001`, `min_lr=1e-6`
  - `warmup_epochs=10`
- **Masking**:
  - `mask_ratio=0.35`
  - `--progressive_masking --mask_ratio_start 0.20`
  - So mask ratio ramps from 0.20 → 0.35 across epochs
- **Loss weights**:
  - `mim_loss_weight=1.0`, `cls_loss_weight=1.0`
  - `koleo_weight=0.0` (disabled; original iBOT style)
  - `--use_focal_loss` enabled with default `focal_gamma=2.0`

This keeps the **overall character of the v5 conservative config** (small model, AdamW, stable hyperparams) but:

- Adds curriculum on the masking difficulty.
- Uses a smoother LR schedule (peak LR).
- Makes it a bit more expressive and robust without being “aggressive”.

## Files

- `ibot_ssl.py`:
  - Core iBOT components (masking generators, tokenizer, heads, LARS, schedulers).
  - Includes `ProgressiveMaskingGenerator` and the enhanced `iBOTLoss` with focal MIM support.
- `train_ibot.py`:
  - Full training loop with:
    - Progressive masking support (`--progressive_masking`, `--mask_ratio_start`).
    - Optional focal MIM loss (`--use_focal_loss`, `--focal_gamma`).
    - Peak LR schedule when `--max_lr > lr`.
    - Loss sanity checks as in v5.
- `check_parameters.py`:
  - Verifies ViT backbone parameters are `< 100M` (same constraint as previous versions).
- `liberal.sh`:
  - macey v6 “liberal” configuration (see above).
- `requirements.txt`:
  - Dependencies (same as v5/v6).

## How to Run (macey v6 Liberal)

From this folder:

```bash
sbatch liberal.sh
```

Or directly:

```bash
python train_ibot.py \
  --data_path /path/to/data \
  --output_dir ./checkpoints_v6_liberal \
  --arch vit_tiny \
  --optimizer adamw \
  --batch_size 128 \
  --lr 0.0003 \
  --max_lr 0.001 \
  --min_lr 1e-6 \
  --weight_decay 0.04 \
  --weight_decay_end 0.1 \
  --clip_grad 1.0 \
  --epochs 100 \
  --drop_path_rate 0.2 \
  --local_crops_number 6 \
  --bottleneck_dim 256 \
  --out_dim 4096 \
  --num_tokens 8192 \
  --mask_ratio 0.35 \
  --mask_type random \
  --progressive_masking \
  --mask_ratio_start 0.20 \
  --mim_loss_weight 1.0 \
  --cls_loss_weight 1.0 \
  --koleo_weight 0.0 \
  --mim_temp 0.15 \
  --use_focal_loss \
  --use_fp16
```

This should behave very similarly to the v5 conservative config (loss starting around 4–6 and drifting to ~1–3), but with:

- Slightly harder task over time (higher final mask ratio with curriculum).
- Potentially better local patch representations via focal MIM.


