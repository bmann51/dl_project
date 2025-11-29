#!/usr/bin/env python3
"""
Extract accuracy metrics from submission generation logs

This script parses the output logs from batch_submit_all.sh to extract
accuracy metrics and create a comparison table.

Usage:
    python extract_accuracies.py
    # Or specify log file:
    python extract_accuracies.py --log batch_submissions/batch_submit_12345.out
"""

import re
import pandas as pd
from pathlib import Path
import argparse
import glob

OUTPUT_DIR = Path(__file__).parent / "batch_submissions"


def parse_log_file(log_path: Path) -> list:
    """Parse a log file and extract accuracy metrics"""
    results = []
    
    if not log_path.exists():
        return results
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern to match: "Val Accuracy: 0.8234 (82.34%)"
    # Or: "Val Accuracy: 82.34%"
    accuracy_pattern = r'Val Accuracy:\s*([\d.]+)'
    
    # Find all job sections
    # Pattern: "--- Job X/Y: model_name on testset ---"
    job_pattern = r'--- Job \d+/\d+:\s*([^\s]+)\s+on\s+(testset_\d+) ---'
    
    # Split into job sections
    job_sections = re.split(r'--- Job \d+/\d+:', content)
    
    for section in job_sections[1:]:  # Skip first (header)
        # Extract model and testset from previous line or current section
        job_match = re.search(r'([^\s]+)\s+on\s+(testset_\d+)', section)
        if not job_match:
            continue
        
        model_name = job_match.group(1)
        testset = job_match.group(2)
        
        # Extract checkpoint name
        checkpoint_match = re.search(r'Checkpoint:\s*([^\s]+)', section)
        checkpoint = checkpoint_match.group(1) if checkpoint_match else "unknown"
        
        # Extract output filename
        output_match = re.search(r'Output:\s*([^\s]+)', section)
        output = output_match.group(1) if output_match else "unknown"
        
        # Extract accuracies
        train_acc_match = re.search(r'Train Accuracy:\s*([\d.]+)', section)
        val_acc_match = re.search(r'Val Accuracy:\s*([\d.]+)', section)
        
        if train_acc_match and val_acc_match:
            train_acc = float(train_acc_match.group(1))
            val_acc = float(val_acc_match.group(1))
            
            results.append({
                'model': model_name,
                'testset': testset,
                'checkpoint': checkpoint,
                'output': output,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Extract accuracy from submission logs')
    parser.add_argument('--log', type=str, default=None,
                       help='Specific log file to parse (default: find latest)')
    parser.add_argument('--all', action='store_true',
                       help='Parse all log files and combine results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Accuracy Extraction from Logs")
    print("=" * 80)
    print()
    
    all_results = []
    
    if args.log:
        # Parse specific log file
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = OUTPUT_DIR / log_path
        
        print(f"Parsing: {log_path}")
        results = parse_log_file(log_path)
        all_results.extend(results)
    elif args.all:
        # Parse all log files
        log_files = sorted(OUTPUT_DIR.glob("batch_submit_*.out"), reverse=True)
        print(f"Found {len(log_files)} log files")
        
        for log_file in log_files:
            print(f"Parsing: {log_file.name}")
            results = parse_log_file(log_file)
            all_results.extend(results)
    else:
        # Find latest log file
        log_files = sorted(OUTPUT_DIR.glob("batch_submit_*.out"), reverse=True)
        if log_files:
            log_file = log_files[0]
            print(f"Parsing latest log: {log_file.name}")
            results = parse_log_file(log_file)
            all_results.extend(results)
        else:
            print("No log files found!")
            return
    
    if not all_results:
        print("No accuracy results found in logs")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Create summary pivot table
    summary = df.pivot_table(
        index=['model', 'checkpoint'],
        columns='testset',
        values='val_accuracy',
        aggfunc='first'
    )
    
    # Add average and max
    summary['avg_val_accuracy'] = summary.mean(axis=1)
    summary['max_val_accuracy'] = summary.max(axis=1)
    summary['min_val_accuracy'] = summary.min(axis=1)
    
    # Sort by average accuracy
    summary = summary.sort_values('avg_val_accuracy', ascending=False)
    
    # Save results
    results_path = OUTPUT_DIR / "accuracy_summary.csv"
    summary.to_csv(results_path)
    
    # Also save full results
    full_results_path = OUTPUT_DIR / "accuracy_full_results.csv"
    df.to_csv(full_results_path, index=False)
    
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    print(f"\nTotal evaluations found: {len(all_results)}")
    print(f"Results saved to:")
    print(f"  - Summary: {results_path}")
    print(f"  - Full: {full_results_path}")
    
    print(f"\nTop 15 models by average validation accuracy:")
    print(summary.head(15).to_string())
    
    print(f"\n{'='*80}")
    print("Best model per testset:")
    for testset in ['testset_1', 'testset_2', 'testset_3']:
        if testset in summary.columns:
            best = summary[testset].idxmax()
            best_acc = summary.loc[best, testset]
            print(f"  {testset}: {best[0]} - {best_acc*100:.2f}%")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

