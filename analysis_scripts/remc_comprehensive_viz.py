"""Comprehensive REMC pattern visualization with RMSE analysis"""
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
    if transform_type.startswith('wavelet_'):
        parts = transform_type.split('_')
        wavelet, level = parts[1], int(parts[2][-1])
        coeffs = pywt.wavedec(series, wavelet, level=level, mode='periodization')
        concatenated = np.concatenate(coeffs)
        return interp1d(np.linspace(0, 1, len(concatenated)), concatenated)(np.linspace(0, 1, target_length))
    return series

def compute_pointwise_rmse(signal, pattern):
    return np.abs(signal - pattern)

print("=" * 60)
print("REMC Comprehensive Pattern Visualization")
print("=" * 60)

# Load patterns
pattern_file = Path('../json_files/remc/pattern_parameters_E003.json')
with open(pattern_file, 'r') as f:
    data = json.load(f)
patterns = data['fold_1']

# Load data
df = pd.read_parquet('../processed_datasets/remc/E003.parquet')
TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
X = df[[c for c in df.columns if c != 'target']]
X_list = [X[[c for c in X.columns if c.startswith(f"{s}_")]].values for s in TIME_SERIES]
y = df['target'].values

# Use first pattern
pattern_info = patterns[0]
series_idx = pattern_info['series_idx']
series_name = TIME_SERIES[series_idx]
transform_type = pattern_info['transform_type']
start, end = pattern_info['start'], pattern_info['end']
pattern = np.array(pattern_info['pattern'])

print(f"Visualizing: {series_name} - {transform_type} @ {start}-{end}")

# Transform data
X_series = X_list[series_idx]
X_transformed = np.array([apply_transformation(row, transform_type, row.shape[0]) for row in X_series])

# Get example samples
high_expr_idx = np.where(y == 1)[0][0]
low_expr_idx = np.where(y == 0)[0][0]

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# --- Panel 1: High expression ---
high_signal = X_transformed[high_expr_idx]
high_region = high_signal[start:end]

ax1.plot(high_signal, color='#2c3e50', linewidth=1.5, alpha=0.7, label='ChIP-seq Signal')
pattern_x = np.arange(start, start + len(pattern))
ax1.plot(pattern_x, pattern, color='#e74c3c', linewidth=3, label='Pattern', zorder=10)
ax1.axvspan(start, end, alpha=0.15, color='#e74c3c')
ax1.axvline(len(high_signal)//2, color='black', linestyle=':', linewidth=2, alpha=0.5, label='TSS')
ax1.set_xlabel('Position (bp from TSS)', fontsize=11)
ax1.set_ylabel(f'Amplitude ({transform_type})', fontsize=11)
ax1.set_title(f'High Expression Example', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(alpha=0.3, linestyle='--')

high_rmse = np.sqrt(np.mean((high_region - pattern[:len(high_region)])**2))
ax1.text(0.02, 0.95, f'Mean RMSE: {high_rmse:.3f}', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# --- Panel 2: Low expression ---
low_signal = X_transformed[low_expr_idx]
low_region = low_signal[start:end]

ax2.plot(low_signal, color='#2c3e50', linewidth=1.5, alpha=0.7, label='ChIP-seq Signal')
ax2.plot(pattern_x, pattern, color='#3498db', linewidth=3, label='Pattern', zorder=10)
ax2.axvspan(start, end, alpha=0.15, color='#3498db')
ax2.axvline(len(low_signal)//2, color='black', linestyle=':', linewidth=2, alpha=0.5, label='TSS')
ax2.set_xlabel('Position (bp from TSS)', fontsize=11)
ax2.set_ylabel(f'Amplitude ({transform_type})', fontsize=11)
ax2.set_title(f'Low Expression Example', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(alpha=0.3, linestyle='--')

low_rmse = np.sqrt(np.mean((low_region - pattern[:len(low_region)])**2))
ax2.text(0.02, 0.95, f'Mean RMSE: {low_rmse:.3f}', 
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# --- Panel 3: RMSE at each position ---
high_pointwise = compute_pointwise_rmse(high_region, pattern[:len(high_region)])
low_pointwise = compute_pointwise_rmse(low_region, pattern[:len(low_region)])

positions = np.arange(len(high_pointwise))

ax3.plot(positions, high_pointwise, color='#e74c3c', linewidth=2.5, label=f'High Expression (mean: {high_rmse:.3f})', marker='o', markersize=5, alpha=0.8)
ax3.plot(positions, low_pointwise, color='#3498db', linewidth=2.5, label=f'Low Expression (mean: {low_rmse:.3f})', marker='s', markersize=5, alpha=0.8)

# Mean lines
ax3.axhline(high_rmse, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.5)
ax3.axhline(low_rmse, color='#3498db', linestyle='--', linewidth=2, alpha=0.5)

# Annotate
ax3.text(len(positions) * 0.98, high_rmse, f'{high_rmse:.3f}', 
         fontsize=10, color='#e74c3c', fontweight='bold', 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#e74c3c'))
ax3.text(len(positions) * 0.98, low_rmse, f'{low_rmse:.3f}', 
         fontsize=10, color='#3498db', fontweight='bold', 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#3498db'))

ax3.set_xlabel('Position within Pattern Window', fontsize=11)
ax3.set_ylabel('Absolute Error', fontsize=11)
ax3.set_title('Pointwise Error Analysis: Pattern Discrimination Power', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(alpha=0.3, linestyle='--')
ax3.fill_between(positions, high_pointwise, alpha=0.2, color='#e74c3c')
ax3.fill_between(positions, low_pointwise, alpha=0.2, color='#3498db')

discrimination = abs(high_rmse - low_rmse)
ax3.text(0.5, 0.98, f'RMSE Difference: {discrimination:.3f}\nDiscrimination: {"GOOD" if discrimination > 20 else "MODERATE"}', 
         transform=ax3.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle(f'REMC E003 Pattern Analysis: {series_name} - {transform_type}\nControl Point: {pattern_info["control_points"][0]:.1f}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = '../manuscript/images/remc_pattern_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved to {output_path}")
plt.close()

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)

