import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

def load_remc_data(cell_line):
    df = pd.read_parquet(f"../processed_datasets/remc/{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in TIME_SERIES], df['target']

json_files = sorted([f for f in os.listdir('../json_files/remc') if f.endswith('.json')])

for json_file in json_files:
    cell_line = json_file.replace('pattern_parameters_', '').replace('.json', '')
    print(f"\nProcessing {cell_line}")
    
    with open(f'../json_files/remc/{json_file}') as f:
        all_patterns = json.load(f)
    
    X_list, y = load_remc_data(cell_line)
    
    fold_1_patterns = all_patterns.get('fold_1', [])
    if not fold_1_patterns:
        continue
    
    n_patterns = len(fold_1_patterns)
    fig, axes = plt.subplots(n_patterns, 1, figsize=(12, 3*n_patterns))
    if n_patterns == 1:
        axes = [axes]
    
    for idx, pattern_info in enumerate(fold_1_patterns):
        ax = axes[idx]
        pattern = np.array(pattern_info['pattern'])
        series_idx = pattern_info['series_idx']
        start = pattern_info['start']
        end = pattern_info.get('end', start + pattern_info['width'])
        
        series_name = TIME_SERIES[series_idx]
        series_data = X_list[series_idx]
        
        pos_samples = series_data[y == 1].values[:30]
        neg_samples = series_data[y == 0].values[:30]
        
        for sample in pos_samples:
            ax.plot(sample, alpha=0.15, color='red', linewidth=0.5)
        for sample in neg_samples:
            ax.plot(sample, alpha=0.15, color='blue', linewidth=0.5)
        
        x_pattern = np.arange(start, end)
        ax.plot(x_pattern, pattern, 'g-', linewidth=3, label='Pattern', zorder=10)
        ax.scatter(x_pattern, pattern, c='green', s=50, zorder=11)
        
        transform = pattern_info.get('transform_type', 'raw')
        ax.set_title(f"Pattern {idx+1}: {series_name} ({transform}), [{start}, {end})", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('../manuscript/images/remc_patterns', exist_ok=True)
    plt.savefig(f'../manuscript/images/remc_patterns/{cell_line}_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization with {n_patterns} patterns")

