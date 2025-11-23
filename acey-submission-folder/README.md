# Acey's DINO Submission Generator

This folder contains Acey's self-contained submission generation pipeline. It works exactly like `brian-folder/generate_submission2.py` but uses Acey's model architecture (major difference is untied heads).

## Quick Start for Brian

### Step 1: Navigate to the folder
```bash
cd acey-submission-folder
```

### Step 2: Edit `generate_submission.sh`

Open the file and update **lines 39-44** with Acey's checkpoint path and settings:

```bash
python generate_submission.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_vit_tiny/final_checkpoint.pth \  # ← Update this path
    --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_1/data \        # ← Usually same as yours
    --output submission_vit_tiny.csv \                                             # ← Update filename
    --arch vit_tiny \                                                              # ← Update: vit_tiny, vit_small, or vit_base
    --resolution 96 \
    --k 20
```

**What to change:**
- `--checkpoint`: Path to Acey's trained checkpoint
- `--output`: Desired output filename
- `--arch`: Model architecture (must match what Acey trained with)
- `--k`: Number of neighbors (default: 20, can tune)

**If Acey used custom dimensions**, also add:
```bash
    --out_dim 4096 \          # Only if Acey trained with different out_dim
    --bottleneck_dim 128      # Only if Acey trained with different bottleneck_dim
```

### Step 3: Run the job
```bash
sbatch generate_submission.sh
```

### Step 4: Check output
```bash
# Check job status
squeue -u bm3772

# Check logs
tail -f logs/submission_*.out

# Find the submission file
ls -lh submission_*.csv
```

## File Structure

- `generate_submission.py` - Main script (same format as your `generate_submission2.py`)
- `generate_submission.sh` - SLURM script (same format as your `generate_submission.sh`)
- `dino_ssl.py` - Model definitions (self-contained, no imports from other folders)
- `requirements.txt` - Dependencies (same as yours)

## Key Differences from Brian's Code

1. **Model Architecture**: Uses untied heads (global + local) instead of tied heads
2. **Default Dimensions**: `out_dim=8192`, `bottleneck_dim=256` (vs your defaults)
3. **Self-contained**: All code in this folder, no dependencies on `acey-folder/`

## Common Settings

### For vit_tiny:
```bash
--arch vit_tiny --resolution 96 --k 20
```

### For vit_small:
```bash
--arch vit_small --resolution 96 --k 20
# If custom dims: --out_dim 4096 --bottleneck_dim 128
```

### For vit_base:
```bash
--arch vit_base --resolution 96 --k 20
```

## Troubleshooting

### Model loading errors
- **Check architecture**: `--arch` must match what Acey trained (vit_tiny/vit_small/vit_base)
- **Check dimensions**: If Acey used custom `out_dim` or `bottleneck_dim`, add them to the command
- **Check checkpoint**: Make sure the checkpoint path is correct and file exists

### Dimension mismatch warnings
The script will warn if checkpoint args don't match command-line args. Use the checkpoint's saved args to determine correct values.

### CUDA out of memory
Reduce `--batch_size` (add `--batch_size 64` to the command) or use a smaller model.

## Example: Running Acey's vit_tiny model

```bash
# In generate_submission.sh, line 38-44:
python generate_submission.py \
    --checkpoint /gpfs/scratch/bm3772/checkpoints_vit_tiny/final_checkpoint.pth \
    --data_dir /gpfs/scratch/bm3772/fall2025_finalproject/testset_1/data \
    --output submission_acey_vit_tiny.csv \
    --arch vit_tiny \
    --resolution 96 \
    --k 20
```

Then run: `sbatch generate_submission.sh`

## Output

The script will create a CSV file with format:
```csv
id,class_id
00001_image.jpg,5
00002_image.jpg,12
...
```

This is ready to upload to Kaggle!
