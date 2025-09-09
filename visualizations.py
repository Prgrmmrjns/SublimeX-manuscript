import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_patterns(optimizer, pattern_indices, dataset_name, specific_name='patterns'):
    """Visualize selected patterns from a PatternOptimizer instance and save as PNG file"""
    valid_indices = [i for i in pattern_indices if 0 <= i < len(optimizer.pattern_list)]
    n_patterns = len(valid_indices)
    if n_patterns == 0:
        return
    fig, axes = plt.subplots(n_patterns, 2, figsize=(16, 4*n_patterns))
    if n_patterns == 1:
        axes = axes.reshape(1, -1)
    for idx, pattern_idx in enumerate(valid_indices):
        pattern = optimizer.pattern_list[pattern_idx]
        start = optimizer.pattern_starts[pattern_idx]
        end = optimizer.pattern_ends[pattern_idx]
        ax_left = axes[idx, 0]
        ax_left.plot(range(len(pattern)), pattern, 'b-', alpha=0.3, label='Full Pattern')
        active_pattern = pattern[start:end]
        active_x = range(start, end)
        ax_left.plot(active_x, active_pattern, 'r-', linewidth=3, label='Active Region')
        ax_left.scatter(active_x, active_pattern, c='red', s=50, zorder=5)
        title = f'Pattern {pattern_idx} (positions {start}-{end})'
        if optimizer.multiple_series and optimizer.pattern_series_indices:
            title += f', Series {optimizer.pattern_series_indices[pattern_idx]}'
        ax_left.set_title(title)
        ax_left.set_xlabel('Position')
        ax_left.set_ylabel('Pattern Value')
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)
        y_min, y_max = ax_left.get_ylim()
        ax_left.set_ylim(y_min - 0.1, y_max + 0.1)
        ax_right = axes[idx, 1]
        series_idx = optimizer.pattern_series_indices[pattern_idx] if optimizer.multiple_series and optimizer.pattern_series_indices else None
        X_for_pattern = optimizer.X_train
        if optimizer.multiple_series and series_idx is not None and optimizer.X_train.ndim == 3:
            X_for_pattern = optimizer.X_train[:, series_idx, :]
        X_data = np.asarray(X_for_pattern, dtype=np.float32)
        pattern_region = pattern[start:end]
        X_region = X_data[:, start:end]
        mae_values = optimizer.calculate_pattern_mae(X_region, pattern_region.astype(np.float32)).flatten()
        df = pd.DataFrame({'MAE': mae_values,'Target': optimizer.y_train})
        unique_targets = len(np.unique(optimizer.y_train))
        if unique_targets <= 10:
            sns.histplot(data=df, x='MAE', hue='Target', bins=100, alpha=0.7, ax=ax_right)
            ax_right.set_title('MAE Distribution (by Target)')
        else:
            sns.histplot(data=df, x='MAE', bins=100, alpha=0.7, ax=ax_right)
            ax_right.set_title('MAE Distribution')
        ax_right.set_xlabel('MAE')
        ax_right.set_ylabel('Count')
        ax_right.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(f'images/{dataset_name}', exist_ok=True)
    filename = f'images/{dataset_name}/{specific_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


