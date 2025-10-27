"""Visualize AZT1D patterns across different transformations"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    if transform_type == 'sqrt': return np.sqrt(np.abs(series))
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
print("AZT1D Pattern Visualization")
print("=" * 60)

# Load patterns
pattern_file = Path('../json_files/azt1d/pattern_parameters_1.json')
with open(pattern_file, 'r') as f:
    data = json.load(f)
patterns = data['patterns']

print(f"Found {len(patterns)} patterns")

# Load data
df = pd.read_parquet('../processed_datasets/azt1d/subject_1.parquet')
TIME_SERIES = ['CGM', 'Insulin', 'Carbs']
cgm_data = df[[col for col in df.columns if col.startswith('CGM_')]]
insulin_data = df[[col for col in df.columns if col.startswith('Insulin_')]]
carbs_data = df[[col for col in df.columns if col.startswith('Carbs_')]]
X_list = [cgm_data.values, insulin_data.values, carbs_data.values]
y = df['target'].values

print(f"Data shape: {len(y)} samples, {[x.shape[1] for x in X_list]} time points per series")

# Group patterns by transform type
transform_groups = {}
for p in patterns:
    t = p['transform_type']
    if t not in transform_groups:
        transform_groups[t] = []
    transform_groups[t].append(p)

print(f"\nTransform usage:")
for t, plist in sorted(transform_groups.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {t:25s}: {len(plist)} patterns")

# Visualize top 4 most-used transforms
top_transforms = sorted(transform_groups.items(), key=lambda x: len(x[1]), reverse=True)[:4]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax_idx, (transform_type, pattern_list) in enumerate(top_transforms):
    ax = axes[ax_idx]
    
    # Use first pattern of this transform
    pattern_info = pattern_list[0]
    series_idx = pattern_info['series_idx']
    series_name = TIME_SERIES[series_idx]
    start, end = pattern_info['start'], pattern_info['end']
    width = end - start
    
    # Get the series data
    X_series = X_list[series_idx]
    
    # Apply transformation
    X_transformed = np.array([apply_transformation(row, transform_type, row.shape[0]) for row in X_series])
    
    # Get pattern shape (already in list)
    pattern = np.array(pattern_info['pattern'])
    
    # Get one sample with high temp and one with low temp
    high_temp_idx = np.argmax(y)
    low_temp_idx = np.argmin(y)
    
    # Extract pattern regions and normalize
    high_region = X_transformed[high_temp_idx][start:end]
    low_region = X_transformed[low_temp_idx][start:end]
    pattern_region = pattern[:len(high_region)]
    
    # Z-score normalize
    high_norm = (high_region - high_region.mean()) / (high_region.std() + 1e-10)
    low_norm = (low_region - low_region.mean()) / (low_region.std() + 1e-10)
    pattern_norm = (pattern_region - pattern_region.mean()) / (pattern_region.std() + 1e-10)
    
    # Plot
    ax.plot(range(len(high_norm)), high_norm, color='#e74c3c', linewidth=2, alpha=0.7, label=f'High Temp ({y[high_temp_idx]:.1f}°C)', marker='s', markersize=3, markevery=2)
    ax.plot(range(len(low_norm)), low_norm, color='#3498db', linewidth=2, alpha=0.7, label=f'Low Temp ({y[low_temp_idx]:.1f}°C)', marker='^', markersize=3, markevery=2)
    ax.plot(range(len(pattern_norm)), pattern_norm, color='darkred', linewidth=3, label='Pattern Shape', zorder=10, linestyle='-', marker='o', markersize=5, markevery=2)
    
    # Flat 1-CP reference
    cp_val = pattern_info['control_points'][0]
    ax.axhline(0, color='orange', linewidth=2, linestyle='--', alpha=0.5, label='Flat 1-CP (norm)', zorder=5)
    
    # Add vertical line for pattern location
    ax.axvline(start, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(start + len(pattern), color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('Amplitude (Z-scored)', fontsize=10)
    ax.set_title(f'{series_name} - {transform_type}\n(Normalized: one sample per class extreme)', 
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    
    info_text = f'Position: {start}-{end}\nWidth: {len(pattern)}\nCP: {cp_val:.2f}'
    ax.text(0.02, 0.98, info_text, 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('AZT1D Temperature Regression - Pattern Shapes (Z-scored)\nRed: High Temp, Blue: Low Temp, Dark Red: Pattern, Orange Dash: Flat 1-CP', 
             fontsize=13, fontweight='bold')
plt.tight_layout()

output_path = '../manuscript/images/azt1d_pattern_interpretation.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved to {output_path}")
plt.close()

# Create transform distribution analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Left: Pattern count by transform
ax_counts = axes2[0]
transforms_sorted = sorted(transform_groups.items(), key=lambda x: len(x[1]), reverse=True)
transform_names = [t for t, _ in transforms_sorted]
counts = [len(p) for _, p in transforms_sorted]

bars = ax_counts.barh(range(len(transform_names)), counts, color='steelblue', alpha=0.8)
ax_counts.set_yticks(range(len(transform_names)))
ax_counts.set_yticklabels(transform_names, fontsize=9)
ax_counts.set_xlabel('Number of Patterns', fontsize=11)
ax_counts.set_title('Pattern Distribution by Transform', fontsize=12, fontweight='bold')
ax_counts.grid(alpha=0.3, axis='x')

for i, (bar, count) in enumerate(zip(bars, counts)):
    ax_counts.text(count + 0.1, i, str(count), va='center', fontsize=9)

# Right: Pattern count by series
ax_series = axes2[1]
series_counts = [0, 0, 0]
for p in patterns:
    series_counts[p['series_idx']] += 1

bars2 = ax_series.bar(TIME_SERIES, series_counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
ax_series.set_ylabel('Number of Patterns', fontsize=11)
ax_series.set_title('Pattern Distribution by Time Series', fontsize=12, fontweight='bold')
ax_series.grid(alpha=0.3, axis='y')

for bar, count in zip(bars2, series_counts):
    height = bar.get_height()
    ax_series.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('AZT1D: Pattern Analysis Statistics', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path2 = '../manuscript/images/azt1d_pattern_stats.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✓ Saved to {output_path2}")
plt.close()

print("\n" + "=" * 60)
print("AZT1D Visualization Complete!")
print("=" * 60)

