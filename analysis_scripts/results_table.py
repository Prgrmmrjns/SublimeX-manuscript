#!/usr/bin/env python3
"""
Script to automatically generate results table from CSV files.
Generates a LaTeX table with mean ± std for scores, features, and runtime.
"""

import pandas as pd
import numpy as np
import os

def load_and_process_results(results_dir='results'):
    """Load and process all result CSV files."""
    datasets = ['azt1d', 'emotions', 'mimic', 'mitbih', 'remc', 'pamap2', 'pancancer', 'svd']
    results = {}
    
    for dataset in datasets:
        csv_path = os.path.join(results_dir, f'{dataset}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Handle different CSV formats
            if 'subject_id' in df.columns:
                # AZT1D format - aggregate across subjects
                dataset_results = {}
                for approach in df['approach'].unique():
                    approach_data = df[df['approach'] == approach]
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
                for approach in df['approach'].unique():
                    approach_data = df[df['approach'] == approach]
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
                for approach in df['approach'].unique():
                    approach_data = df[df['approach'] == approach]
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
                for approach in df['approach'].unique():
                    approach_data = df[df['approach'] == approach]
                    dataset_results[approach] = {
                        'score': approach_data['score'].iloc[0],
                        'score_std': 0.0,
                        'n_features': approach_data['n_features'].iloc[0],
                        'n_features_std': 0.0,
                        'processing_time': approach_data['processing_time'].iloc[0],
                        'processing_time_std': 0.0
                    }
            
            results[dataset] = dataset_results
    
    return results

def generate_latex_table(results, output_file='../manuscript/tables/results_table.tex'):
    """Generate LaTeX table from processed results."""
    
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
    
    approaches = ['PATX', 'TSFRESH', 'CATCH22', 'CNN']
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance comparison across biomedical datasets. "
                "Results show mean $\\pm$ standard deviation across subjects/folds. "
                "Best performance per dataset highlighted in bold.}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Method} & \\textbf{Score} & \\textbf{Features} & "
                "\\textbf{Time (s)} \\\\\n")
        f.write("\\midrule\n")
        
        for dataset, dataset_data in results.items():
                
            info = dataset_info[dataset]
            dataset_scores = []
            
            # Collect scores for ranking
            for approach in approaches:
                if approach in dataset_data:
                    dataset_scores.append((approach, dataset_data[approach]['score']))
            
            # Sort by score (higher is better for AUC/Accuracy, lower is better for RMSE)
            if info['metric'] in ['AUC', 'Accuracy']:
                dataset_scores.sort(key=lambda x: x[1], reverse=True)
            else:  # RMSE
                dataset_scores.sort(key=lambda x: x[1])
            
            # Create ranking dictionary
            ranking = {approach: i+1 for i, (approach, _) in enumerate(dataset_scores)}
            
            # Write results for this dataset
            for i, approach in enumerate(approaches):
                if approach not in dataset_data:
                    continue
                    
                data = dataset_data[approach]
                score = data['score']
                score_std = data['score_std']
                n_features = data['n_features']
                n_features_std = data['n_features_std']
                time_val = data['processing_time']
                time_std = data['processing_time_std']
                
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
                
                # Format time
                if time_std > 0:
                    time_str = f"{time_val:.1f} $\\pm$ {time_std:.1f}"
                else:
                    time_str = f"{time_val:.1f}"
                
                
                # Bold the best score
                if ranking[approach] == 1:
                    score_str = f"\\textbf{{{score_str}}}"
                
                # Dataset name only on first row
                dataset_name = info['name'] if i == 0 else ""
                
                f.write(f"{dataset_name} & {approach} & {score_str} & {features_str} & "
                       f"{time_str} \\\\\n")
            
            # Add spacing between datasets
            if dataset != list(results.keys())[-1]:
                f.write("\\addlinespace\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\end{table}\n")
    
    print(f"Results table generated: {output_file}")

def main():
    """Main function to generate results table."""
    results = load_and_process_results('../results')
    generate_latex_table(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for dataset, dataset_data in results.items():
        print(f"\n{dataset.upper()}:")
        for approach, data in dataset_data.items():
            print(f"  {approach}: {data['score']:.3f} ± {data['score_std']:.3f} "
                  f"({data['n_features']:.1f} ± {data['n_features_std']:.1f} features, "
                  f"{data['processing_time']:.1f} ± {data['processing_time_std']:.1f}s)")

if __name__ == "__main__":
    main()