#!/usr/bin/env python3
"""
Script to automatically generate results table from CSV files.
Generates a LaTeX table with mean ± std for scores, features, and runtime.
"""

import pandas as pd
import os

results_dir = '../results'
datasets = ['azt1d', 'emotions', 'mimic', 'mitbih', 'remc', 'pamap2', 'pancancer', 'svd']
results = {}

# Normalize approach names to handle case variations
def normalize_approach_name(name):
    """Normalize approach names to standard format."""
    name_lower = name.lower()
    if name_lower == 'patx':
        return 'PATX'
    elif name_lower == 'tsfresh':
        return 'TSFRESH'
    elif name_lower == 'catch22':
        return 'CATCH22'
    elif name_lower == 'cnn':
        return 'CNN'
    elif name_lower == 'minirocket':
        return 'MiniRocket'
    elif name_lower == 'rdst':
        return 'RDST'
    else:
        return name  # Return as-is if not recognized

for dataset in datasets:
    csv_path = os.path.join(results_dir, f'{dataset}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Normalize approach names in the dataframe
        df['approach_normalized'] = df['approach'].apply(normalize_approach_name)
        
        # Handle different CSV formats
        if 'subject_id' in df.columns:
            # AZT1D format - aggregate across subjects
            dataset_results = {}
            for approach in df['approach_normalized'].unique():
                approach_data = df[df['approach_normalized'] == approach]
                dataset_results[approach] = {
                    'score': approach_data['score'].mean(),
                    'score_std': approach_data['score'].std(),
                    'n_features': approach_data['n_features'].mean(),
                    'n_features_std': approach_data['n_features'].std(),
                    'processing_time': approach_data['processing_time'].mean(),
                    'processing_time_std': approach_data['processing_time'].std()
                }
        elif 'cell_line' in df.columns:
            # REMC format - aggregate across cell lines
            dataset_results = {}
            for approach in df['approach_normalized'].unique():
                approach_data = df[df['approach_normalized'] == approach]
                dataset_results[approach] = {
                    'score': approach_data['score'].mean(),
                    'score_std': approach_data['score'].std(),
                    'n_features': approach_data['n_features'].mean(),
                    'n_features_std': approach_data['n_features'].std(),
                    'processing_time': approach_data['processing_time'].mean(),
                    'processing_time_std': approach_data['processing_time'].std()
                }
        elif 'fold' in df.columns and 'cell_line' not in df.columns:
            # MITBIH/Emotions/MIMIC/Pancancer format - aggregate across folds (K-fold CV)
            dataset_results = {}
            score_col = 'auc' if 'auc' in df.columns else 'score'
            for approach in df['approach_normalized'].unique():
                approach_data = df[df['approach_normalized'] == approach]
                dataset_results[approach] = {
                    'score': approach_data[score_col].mean(),
                    'score_std': approach_data[score_col].std(),
                    'n_features': approach_data['n_features'].mean(),
                    'n_features_std': approach_data['n_features'].std(),
                    'processing_time': approach_data['processing_time'].mean(),
                    'processing_time_std': approach_data['processing_time'].std()
                }
        else:
            # Single fold format (fallback)
            dataset_results = {}
            for approach in df['approach_normalized'].unique():
                approach_data = df[df['approach_normalized'] == approach]
                dataset_results[approach] = {
                    'score': approach_data['score'].iloc[0],
                    'score_std': 0.0,
                    'n_features': approach_data['n_features'].iloc[0],
                    'n_features_std': 0.0,
                    'processing_time': approach_data['processing_time'].iloc[0],
                    'processing_time_std': 0.0
                }
        
        results[dataset] = dataset_results

output_file = '../elsarticle/tables/results_table.tex'
# Dataset information
dataset_info = {
    'azt1d': {'name': 'AZT1D', 'metric': 'RMSE', 'task': 'Glucose Forecasting'},
    'emotions': {'name': 'Emotions', 'metric': 'AUC', 'task': 'EEG Emotion Classification'},
    'mimic': {'name': 'MIMIC-IV', 'metric': 'AUC', 'task': 'ARDS Classification'},
    'mitbih': {'name': 'MITBIH', 'metric': 'Accuracy', 'task': 'Arrhythmia Classification'},
    'remc': {'name': 'REMC', 'metric': 'AUC', 'task': 'Gene Expression Prediction'},
    'pamap2': {'name': 'PAMAP2', 'metric': 'Accuracy', 'task': 'Activity Recognition'},
    'pancancer': {'name': 'Pan-Cancer', 'metric': 'AUC', 'task': 'Cancer Classification'},
    'svd': {'name': 'SVD', 'metric': 'AUC', 'task': 'Voice Pathology Detection'}
}

approaches = ['PATX', 'TSFRESH', 'CATCH22', 'CNN', 'MiniRocket', 'RDST']

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{Performance comparison across biomedical datasets. "
            "Results show mean $\\pm$ standard deviation across subjects/folds. "
            "Best performance per dataset highlighted in bold.}\n")
    f.write("\\label{tab:results}\n")
    f.write("\\tiny\n")
    f.write("\\renewcommand{\\arraystretch}{0.75}\n")
    # Column specification: one for dataset name, then one for each approach
    f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * len(approaches) + "@{}}\n")
    f.write("\\toprule\n")
    # Header row
    header = "\\textbf{Dataset}"
    for approach in approaches:
        header += f" & \\textbf{{{approach}}}"
    f.write(header + " \\\\\n")
    f.write("\\midrule\n")
    
    for dataset_idx, (dataset, dataset_data) in enumerate(results.items()):
        info = dataset_info[dataset]
        dataset_scores = []
        
        # Collect scores for ranking (include all approaches in dataset)
        for approach in dataset_data.keys():
            dataset_scores.append((approach, dataset_data[approach]['score']))
        
        # Sort by score (higher is better for AUC/Accuracy, lower is better for RMSE)
        if info['metric'] in ['AUC', 'Accuracy']:
            dataset_scores.sort(key=lambda x: x[1], reverse=True)
        else:  # RMSE
            dataset_scores.sort(key=lambda x: x[1])
        
        # Create ranking dictionary
        ranking = {approach: i+1 for i, (approach, _) in enumerate(dataset_scores)}
        
        # Build row for this dataset
        row_parts = [info['name']]
        
        for approach in approaches:
            if approach in dataset_data:
                data = dataset_data[approach]
                score = data['score']
                score_std = data['score_std']
                n_features = data['n_features']
                n_features_std = data['n_features_std']
                
                # Format score
                if score_std > 0:
                    score_str = f"{score:.3f} $\\pm$ {score_std:.3f}"
                else:
                    score_str = f"{score:.3f}"
                
                # Format features
                if n_features_std > 0.05:
                    features_str = f"{n_features:.1f} $\\pm$ {n_features_std:.1f}"
                else:
                    features_str = f"{n_features:.1f}"
                
                # Bold the best score
                if ranking[approach] == 1:
                    score_str = f"\\textbf{{{score_str}}}"
                
                # Combine score and features in cell (score on top, features on bottom)
                cell_content = f"\\makecell{{{score_str} \\\\ {features_str}}}"
                row_parts.append(cell_content)
            else:
                # Approach not available for this dataset
                row_parts.append("---")
        
        # Write the row
        f.write(" & ".join(row_parts) + " \\\\\n")
        
        # Add line between datasets (except after the last one)
        if dataset_idx < len(results) - 1:
            f.write("\\midrule\n")
    
    f.write("\\bottomrule\n")
    f.write("\\end{tabular*}\n")
    f.write("\\end{table}\n")