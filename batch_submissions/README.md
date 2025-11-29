# Batch Submission System

This system allows you to run submission generation for all model variations across acey, brian, and macey folders.

## Quick Start

1. **Generate the batch submission script:**
   ```bash
   python generate_batch_submissions.py
   ```

2. **Review the generated script:**
   ```bash
   cat batch_submissions/batch_submit_all.sh
   ```

3. **Submit to SLURM:**
   ```bash
   sbatch batch_submissions/batch_submit_all.sh
   ```

4. **Monitor progress:**
   ```bash
   # Check job status
   squeue -u bm3772
   
   # Watch logs
   tail -f batch_submissions/batch_submit_*.out
   ```

## How It Works

The `generate_batch_submissions.py` script:

1. **Discovers checkpoints** in common checkpoint directories:
   - `/gpfs/scratch/bm3772/checkpoints_*`
   - Looks for `final_checkpoint.pth` first, then numbered checkpoints

2. **Generates commands** for each combination of:
   - Model variation (acey-dino, acey-ibot-v1, brian-dino, etc.)
   - Checkpoint file
   - Testset (testset_1, testset_2, testset_3)

3. **Creates a SLURM batch script** that runs all submissions sequentially

## Model Configurations

The script automatically handles:

### Acey Models
- **DINO**: vit_tiny, vit_small, vit_base
- **iBOT v1-v8**: Various architectures and configurations

### Brian Models
- **DINO**: vit_base, vit_small

### Macey Models
- **iBOT v6**: vit_tiny

## Output Files

Each submission generates a CSV file with naming pattern:
```
submission_{model_name}_{testset}_{checkpoint_name}.csv
```

Example:
- `submission_acey-ibot-v3-tiny_testset_1_final_checkpoint.csv`
- `submission_brian-dino-base_testset_2_checkpoint_0099.csv`

## Customizing

### Adding New Checkpoints

If checkpoints aren't auto-discovered, you can manually edit `generate_batch_submissions.py`:

1. Find the `get_model_configs()` function
2. Add your checkpoint directory to the `checkpoint_dirs` list:
   ```python
   "your-model": {
       "checkpoint_dirs": ["checkpoints_your_model"],
       ...
   }
   ```

### Changing k-NN Parameters

Edit the `args` dictionary in `get_model_configs()`:
```python
"args": {
    "k": 20,  # Change this
    ...
}
```

### Running Subset of Models

Comment out models in `get_model_configs()` that you don't want to run.

## Troubleshooting

### No checkpoints found
- Check that checkpoint directories exist at `/gpfs/scratch/bm3772/checkpoints_*`
- Verify checkpoint files are named `final_checkpoint.pth` or `checkpoint_XXXX.pth`

### Script fails on specific model
- Check the logs in `batch_submissions/batch_submit_*.out`
- Verify the model's `generate_submission*.py` script exists
- Check that checkpoint architecture matches the config

### Out of memory
- Reduce batch size in individual submission scripts
- Run fewer models at once (split into multiple batch scripts)

## Job Manifest

After generation, check `batch_submissions/submission_manifest.json` to see all jobs that will be run.

