import pandas as pd
import os

def format_value(mean_val, decimals=3):
    return f"${mean_val:.{decimals}f} \\pm 0.000$" if mean_val is not None else "--"

def get_metric_col(dataset_name):
    return 'test_rmse' if dataset_name == 'AZT1D' else 'test_score'

datasets = ['AZT1D', 'MITBIH', 'Bonn EEG', 'REMC', 'MIMIC-IV']
file_map = {
    'AZT1D': 'results/azt1d.csv',
    'MITBIH': 'results/mitbih_ecg.csv', 
    'Bonn EEG': 'results/bonn_eeg.csv',
    'REMC': 'results/remc.csv',
    'MIMIC-IV': 'results/mimic.csv'
}

table = r"""\begin{table}[h]
\centering
\caption{Performance comparison across biomedical datasets with processing details.}
\label{tab:results}
\begin{tabular}{|l|l|c|c|c|}
\hline
\textbf{Dataset} & \textbf{Method} & \textbf{Performance} & \textbf{\# Features} & \textbf{Time (s)} \\
\hline
"""

for dataset_name in datasets:
    if not os.path.exists(file_map[dataset_name]):
        continue
        
    df = pd.read_csv(file_map[dataset_name])
    metric_col = get_metric_col(dataset_name)
    
    for i, approach in enumerate(['PATX', 'TSFRESH', 'CNN']):
        approach_df = df[df['approach'] == approach]
        if approach_df.empty:
            continue
            
        perf = approach_df[metric_col].mean()
        perf_str = format_value(perf)
        n_features = int(approach_df['n_features'].mean())
        time = approach_df['processing_time'].mean()
        
        if i == 0:
            table += f"{dataset_name} & {approach} & {perf_str} & {n_features} & {time:.1f} \\\\\n"
        else:
            table += f" & {approach} & {perf_str} & {n_features} & {time:.1f} \\\\\n"
    
    table += r"\hline" + "\n"

table += r"""\end{tabular}
\end{table}"""

os.makedirs('manuscript/tables', exist_ok=True)
with open('manuscript/tables/results_table.tex', 'w') as f:
    f.write(table)