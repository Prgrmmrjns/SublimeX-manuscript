#!/usr/bin/env python3
"""
Script to automatically generate ablation study results table from CSV.
Generates a LaTeX table with mean ± std for scores, time, and features.
Handles incomplete data (e.g., REMC) by showing "-" for missing entries.
"""

import pandas as pd
import os

results_file = '../results/ablation_study.csv'
output_file = '../elsarticle/tables/ablation_results.tex'

# Expected number of folds (K_FOLDS = 5)
EXPECTED_FOLDS = 5

# Variant names in order (baseline first, then variants)
variants = [
    'baseline',  # Default PATX configuration from main evaluation
    'trials_100',
    'no_transforms',
    'control_points_1',
    'linear_model',
    'no_sliding_window',
    'polynomial_pattern',
    'control_points_1_no_sliding'
]

# Dataset information
dataset_info = {
    'mitbih': {'name': 'MITBIH', 'metric': 'Accuracy'},
    'svd': {'name': 'SVD', 'metric': 'Accuracy'},
    'emotions': {'name': 'Emotions', 'metric': 'Accuracy'},
    'mimic': {'name': 'MIMIC-IV', 'metric': 'AUC'},
    'pamap2': {'name': 'PAMAP2', 'metric': 'Accuracy'},
    'remc_E003': {'name': 'REMC', 'metric': 'AUC'}
}

# Read the ablation study CSV
df = pd.read_csv(results_file)

# Read baseline PATX results from main evaluation CSVs
baseline_results_dir = '../results'
baseline_dataset_map = {
    'mitbih': 'mitbih',
    'svd': 'svd',
    'emotions': 'emotions',
    'mimic': 'mimic',
    'pamap2': 'pamap2',
    'remc_E003': 'remc'  # Special handling for REMC
}

baseline_data = {}
for ablation_dataset, main_dataset in baseline_dataset_map.items():
    csv_path = os.path.join(baseline_results_dir, f'{main_dataset}.csv')
    if os.path.exists(csv_path):
        baseline_df = pd.read_csv(csv_path)
        # Filter for PATX approach
        patx_data = baseline_df[baseline_df['approach'] == 'PATX'].copy()
        
        # For REMC, filter for cell_line E003
        if main_dataset == 'remc':
            patx_data = patx_data[patx_data['cell_line'] == 'E003'].copy()
        
        # Map to match ablation study column names (processing_time -> time)
        if 'processing_time' in patx_data.columns:
            patx_data = patx_data.rename(columns={'processing_time': 'time'})
        
        if not patx_data.empty:
            baseline_data[ablation_dataset] = patx_data

# Process results
results = {}

for dataset in dataset_info.keys():
    dataset_data = df[df['dataset'] == dataset]
    if dataset_data.empty:
        continue
    
    dataset_results = {}
    
    # Add baseline results if available
    if dataset in baseline_data:
        baseline_patx = baseline_data[dataset]
        if not baseline_patx.empty:
            # Remove duplicates if any
            baseline_patx = baseline_patx.drop_duplicates(subset=['fold'], keep='first')
            n_folds = len(baseline_patx)
            if n_folds >= EXPECTED_FOLDS:
                dataset_results['baseline'] = {
                    'score': baseline_patx['score'].mean(),
                    'score_std': baseline_patx['score'].std() if n_folds > 1 else 0.0,
                    'time': baseline_patx['time'].mean(),
                    'time_std': baseline_patx['time'].std() if n_folds > 1 else 0.0,
                    'n_features': baseline_patx['n_features'].mean(),
                    'n_features_std': baseline_patx['n_features'].std() if n_folds > 1 else 0.0,
                    'n_folds': n_folds,
                    'complete': True
                }
            else:
                dataset_results['baseline'] = {
                    'score': baseline_patx['score'].mean(),
                    'score_std': baseline_patx['score'].std() if n_folds > 1 else 0.0,
                    'time': baseline_patx['time'].mean(),
                    'time_std': baseline_patx['time'].std() if n_folds > 1 else 0.0,
                    'n_features': baseline_patx['n_features'].mean(),
                    'n_features_std': baseline_patx['n_features'].std() if n_folds > 1 else 0.0,
                    'n_folds': n_folds,
                    'complete': False
                }
        else:
            dataset_results['baseline'] = None
    else:
        dataset_results['baseline'] = None
    
    for variant in variants:
        if variant == 'baseline':
            continue  # Already handled above
        variant_data = dataset_data[dataset_data['approach'] == variant]
        
        if variant_data.empty:
            # No data for this variant
            dataset_results[variant] = None
        else:
            # Remove duplicates if any (keep first occurrence)
            variant_data = variant_data.drop_duplicates(subset=['fold'], keep='first')
            # Check if we have enough folds
            n_folds = len(variant_data)
            if n_folds < EXPECTED_FOLDS:
                # Incomplete data - mark as incomplete but still calculate
                dataset_results[variant] = {
                    'score': variant_data['score'].mean(),
                    'score_std': variant_data['score'].std() if n_folds > 1 else 0.0,
                    'time': variant_data['time'].mean(),
                    'time_std': variant_data['time'].std() if n_folds > 1 else 0.0,
                    'n_features': variant_data['n_features'].mean(),
                    'n_features_std': variant_data['n_features'].std() if n_folds > 1 else 0.0,
                    'n_folds': n_folds,
                    'complete': False
                }
            else:
                # Complete data
                dataset_results[variant] = {
                    'score': variant_data['score'].mean(),
                    'score_std': variant_data['score'].std() if n_folds > 1 else 0.0,
                    'time': variant_data['time'].mean(),
                    'time_std': variant_data['time'].std() if n_folds > 1 else 0.0,
                    'n_features': variant_data['n_features'].mean(),
                    'n_features_std': variant_data['n_features'].std() if n_folds > 1 else 0.0,
                    'n_folds': n_folds,
                    'complete': True
                }
    
    results[dataset] = dataset_results

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Generate LaTeX table (tabular content only, table environment is in main.tex)
with open(output_file, 'w') as f:
    f.write("\\tiny\n")
    f.write("\\renewcommand{\\arraystretch}{0.75}\n")
    
    # Column specification: dataset name + 8 variants (including baseline)
    # Use minimal column spacing (0.5pt) and add vertical lines between columns
    f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{0.5pt}}l|" + 
            "|".join(["c"] * len(variants)) + "@{}}\n")
    f.write("\\toprule\n")
    
    # Header row
    header = "\\textbf{Dataset}"
    variant_labels = {
        'baseline': 'Baseline',
        'trials_100': 'Trials=100',
        'no_transforms': 'No Transforms',
        'control_points_1': 'CP=1',
        'linear_model': 'Linear Model',
        'no_sliding_window': 'No Sliding',
        'polynomial_pattern': 'Polynomial',
        'control_points_1_no_sliding': 'CP=1 + No Sliding'
    }
    for variant in variants:
        label = variant_labels.get(variant, variant.replace('_', ' ').title())
        header += f" & \\textbf{{{label}}}"
    f.write(header + " \\\\\n")
    f.write("\\midrule\n")
    
    # Data rows
    for dataset_idx, (dataset, dataset_data) in enumerate(results.items()):
        info = dataset_info[dataset]
        row_parts = [info['name']]
        
        # Collect scores for ranking (only complete results)
        dataset_scores = []
        for variant in variants:
            if (variant in dataset_data and 
                dataset_data[variant] is not None and 
                dataset_data[variant]['complete']):
                dataset_scores.append((
                    variant, 
                    dataset_data[variant]['score']
                ))
        
        # Sort by score (higher is better for AUC/Accuracy)
        if info['metric'] in ['AUC', 'Accuracy']:
            dataset_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            dataset_scores.sort(key=lambda x: x[1])
        
        # Create ranking dictionary (only for complete results)
        ranking = {variant: i+1 for i, (variant, _) in enumerate(dataset_scores)}
        
        # Build cells for each variant
        for variant in variants:
            if variant in dataset_data and dataset_data[variant] is not None:
                data = dataset_data[variant]
                
                # Check if data is complete
                if not data['complete']:
                    # Incomplete data - show "-"
                    row_parts.append("---")
                else:
                    # Format score
                    score = data['score']
                    score_std = data['score_std']
                    if score_std > 0:
                        score_str = f"{score:.3f} $\\pm$ {score_std:.3f}"
                    else:
                        score_str = f"{score:.3f}"
                    
                    # Format features
                    n_features = data['n_features']
                    n_features_std = data['n_features_std']
                    if n_features_std > 0.05:
                        features_str = f"{n_features:.1f} $\\pm$ {n_features_std:.1f}"
                    else:
                        features_str = f"{n_features:.1f}"
                    
                    # Bold the best score
                    if variant in ranking and ranking[variant] == 1:
                        score_str = f"\\textbf{{{score_str}}}"
                    
                    # Combine score and features in cell
                    # Format: score (top), features (bottom)
                    cell_content = (
                        f"\\makecell{{{score_str} \\\\ "
                        f"Feat: {features_str}}}"
                    )
                    row_parts.append(cell_content)
            else:
                # No data for this variant
                row_parts.append("---")
        
        # Write the row
        f.write(" & ".join(row_parts) + " \\\\\n")
        
        # Add line between datasets (except after the last one)
        if dataset_idx < len(results) - 1:
            f.write("\\midrule\n")
    
    f.write("\\bottomrule\n")
    f.write("\\end{tabular*}\n")

print(f"Generated ablation results table: {output_file}")
