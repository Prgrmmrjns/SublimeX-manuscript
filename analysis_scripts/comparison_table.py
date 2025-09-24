import pandas as pd
import numpy as np
import os

def format_mean_std(values, decimals=3, bold=False):
    """Format mean ± std with optional bold formatting"""
    if len(values) == 0:
        return "--"
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    result = f"{mean_val:.{decimals}f} $\\pm$ {std_val:.{decimals}f}"
    return f"\\textbf{{{result}}}" if bold else result

def get_performance_col(dataset_name):
    """Get the performance column name based on dataset"""
    if dataset_name == 'AZT1D':
        return 'test_rmse'
    else:
        return 'score'

# Dataset configuration
datasets = ['AZT1D', 'MITBIH', 'Bonn EEG', 'REMC', 'MIMIC-IV']
file_map = {
    'AZT1D': '../results/azt1d.csv',
    'MITBIH': '../results/mitbih.csv', 
    'Bonn EEG': '../results/bonn_eeg.csv',
    'REMC': '../results/remc.csv',
    'MIMIC-IV': '../results/mimic.csv'
}

# Create LaTeX table
table = r"""\begin{table}[h]
\centering
\caption{Performance comparison across biomedical datasets showing mean $\pm$ standard deviation across folds/subjects.}
\label{tab:results}
\begin{tabular}{|l|l|c|c|c|}
\hline
\textbf{Dataset} & \textbf{Method} & \textbf{Performance} & \textbf{\# Features} & \textbf{Time (s)} \\
\hline
"""

for dataset_name in datasets:
        
    df = pd.read_csv(file_map[dataset_name])
    perf_col = get_performance_col(dataset_name)
    
    # First pass: collect all approach results to find best performance
    approach_results = {}
    for approach in ['PATX', 'TSFRESH', 'CNN']:
        approach_df = df[df['approach'] == approach]
        if approach_df.empty:
            continue
            
        perf_values = approach_df[perf_col].values
        n_features_values = approach_df['n_features'].values  
        time_values = approach_df['processing_time'].values
        
        approach_results[approach] = {
            'perf_mean': np.mean(perf_values),
            'perf_values': perf_values,
            'n_features_values': n_features_values,
            'time_values': time_values
        }
    
    # Find best approach (lowest for RMSE, highest for others)
    if approach_results:
        if dataset_name == 'AZT1D':  # Lower is better for RMSE
            best_approach = min(approach_results.keys(), key=lambda x: approach_results[x]['perf_mean'])
        else:  # Higher is better for accuracy/AUC
            best_approach = max(approach_results.keys(), key=lambda x: approach_results[x]['perf_mean'])
        
        # Second pass: format and add to table
        for i, approach in enumerate(['PATX', 'TSFRESH', 'CNN']):
            if approach not in approach_results:
                continue
                
            is_best = (approach == best_approach)
            
            perf_str = format_mean_std(approach_results[approach]['perf_values'], bold=is_best)
            n_features_str = format_mean_std(approach_results[approach]['n_features_values'], decimals=1, bold=is_best)
            time_str = format_mean_std(approach_results[approach]['time_values'], decimals=1, bold=is_best)
            
            # Add row to table
            if i == 0:
                table += f"{dataset_name} & {approach} & {perf_str} & {n_features_str} & {time_str} \\\\\n"
            else:
                table += f" & {approach} & {perf_str} & {n_features_str} & {time_str} \\\\\n"
    
    table += r"\hline" + "\n"

table += r"""\end{tabular}
\end{table}"""

# Save table
os.makedirs('../manuscript/tables', exist_ok=True)
with open('../manuscript/tables/results_table.tex', 'w') as f:
    f.write(table)

print("Results table generated: ../manuscript/tables/results_table.tex")