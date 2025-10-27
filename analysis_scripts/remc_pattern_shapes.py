"""Visualize REMC pattern shapes - one sample per class"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import pywt
from scipy import fft
from scipy.signal import savgol_filter

def apply_transformation(series, transform_type, target_length):
    if transform_type == 'raw': return series
    if transform_type == 'derivative': return np.gradient(series)
    if transform_type == 'diff': return np.diff(series, prepend=series[0])
    if transform_type == 'cumsum': return np.cumsum(series)
    if transform_type == 'log1p': return np.log1p(series - series.min())
    if transform_type == 'savgol': return savgol_filter(series, min(11, len(series) if len(series) % 2 == 1 else len(series) - 1), 2)
    if transform_type == 'fft_magnitude':
        fft_result = fft.fft(series)
        magnitude = np.abs(fft_result)[:len(series)//2]
        return interp1d(np.linspace(0, 1, len(magnitude)), magnitude)(np.linspace(0, 1, target_length))
    if transform_type.startswith('wavelet_'):
        parts = transform_type.split('_')
        wavelet, level = parts[1], int(parts[2][-1])
        coeffs = pywt.wavedec(series, wavelet, level=level, mode='periodization')
        concatenated = np.concatenate(coeffs)
        return interp1d(np.linspace(0, 1, len(concatenated)), concatenated)(np.linspace(0, 1, target_length))
    return series

print("=" * 60)
print("REMC Pattern Shapes - One Sample Per Class")
print("=" * 60)

pattern_file = Path('../json_files/remc/pattern_parameters_E003.json')
with open(pattern_file, 'r') as f:
    data = json.load(f)
patterns = data['fold_1']

df = pd.read_parquet('../processed_datasets/remc/E003.parquet')
TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
X = df[[c for c in df.columns if c != 'target']]
X_list = [X[[c for c in X.columns if c.startswith(f"{s}_")]].values for s in TIME_SERIES]
y = df['target'].values

# Group by transform
transform_groups = {}
for p in patterns:
    t = p['transform_type']
    if t not in transform_groups:
        transform_groups[t] = []
    transform_groups[t].append(p)

top_transforms = sorted(transform_groups.items(), key=lambda x: len(x[1]), reverse=True)[:4]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax_idx, (transform_type, pattern_list) in enumerate(top_transforms):
    ax = axes[ax_idx]
    
    pattern_info = pattern_list[0]
    series_idx = pattern_info['series_idx']
    series_name = TIME_SERIES[series_idx]
    start, end = pattern_info['start'], pattern_info['end']
    
    X_series = X_list[series_idx]
    X_transformed = np.array([apply_transformation(row, transform_type, row.shape[0]) for row in X_series])
    
    pattern = np.array(pattern_info['pattern'])
    
    # Get one sample from each class
    high_expr_idx = np.where(y == 1)[0][0]  # First high expression
    low_expr_idx = np.where(y == 0)[0][0]   # First low expression
    
    # Extract pattern regions and normalize
    high_region = X_transformed[high_expr_idx][start:end]
    low_region = X_transformed[low_expr_idx][start:end]
    pattern_region = pattern[:len(high_region)]
    
    # Z-score normalize
    high_norm = (high_region - high_region.mean()) / (high_region.std() + 1e-10)
    low_norm = (low_region - low_region.mean()) / (low_region.std() + 1e-10)
    pattern_norm = (pattern_region - pattern_region.mean()) / (pattern_region.std() + 1e-10)
    
    # Plot
    ax.plot(range(len(high_norm)), high_norm, color='#3498db', linewidth=2, alpha=0.7, label='High Expression Sample', marker='s', markersize=3, markevery=5)
    ax.plot(range(len(low_norm)), low_norm, color='#e67e22', linewidth=2, alpha=0.7, label='Low Expression Sample', marker='^', markersize=3, markevery=5)
    ax.plot(range(len(pattern_norm)), pattern_norm, color='red', linewidth=3, label='Pattern Shape', zorder=10, linestyle='-', marker='o', markersize=5, markevery=3)
    
    # Flat 1-CP reference (at y=0 after normalization)
    cp_val = pattern_info['control_points'][0]
    ax.axhline(0, color='orange', linewidth=2, linestyle='--', alpha=0.5, label='Flat 1-CP (normalized)', zorder=5)
    
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('Amplitude (Z-scored)', fontsize=10)
    ax.set_title(f'{series_name} - {transform_type}\n(Normalized: one sample per class)', 
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    
    info_text = f'Position: {start}-{end}\nWidth: {len(pattern)}\nCP: {cp_val:.2f}'
    ax.text(0.02, 0.98, info_text, 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('REMC E003 - Pattern Shapes (Z-scored)\nBlue: High Expression, Orange: Low Expression, Red: Pattern, Yellow Dash: Flat 1-CP', 
             fontsize=13, fontweight='bold')
plt.tight_layout()

output_path = '../manuscript/images/remc_pattern_shapes.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved to {output_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
