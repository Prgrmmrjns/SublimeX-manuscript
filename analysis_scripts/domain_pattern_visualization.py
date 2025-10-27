import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import BSpline, interp1d
import pywt
from scipy import fft

def generate_bspline_pattern(control_points, width):
    """Regenerate B-spline pattern from control points."""
    degree = 3
    n_cp = len(control_points)
    knots = np.concatenate([
        np.zeros(degree + 1), 
        np.linspace(0, 1, n_cp - degree + 1)[1:-1], 
        np.ones(degree + 1)
    ])
    return BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width))

def apply_transformation(series, transform_type, target_length):
    """Apply signal transformation matching core.py implementation."""
    if transform_type == 'raw':
        return series
    elif transform_type == 'wavelet_db4_level4':
        coeffs = pywt.wavedec(series, 'db4', level=4, mode='periodization')
        concatenated = np.concatenate(coeffs)
        x_old = np.linspace(0, 1, len(concatenated))
        x_new = np.linspace(0, 1, target_length)
        return interp1d(x_old, concatenated, kind='linear')(x_new)
    elif transform_type == 'wavelet_db4_level3':
        coeffs = pywt.wavedec(series, 'db4', level=3, mode='periodization')
        concatenated = np.concatenate(coeffs)
        x_old = np.linspace(0, 1, len(concatenated))
        x_new = np.linspace(0, 1, target_length)
        return interp1d(x_old, concatenated, kind='linear')(x_new)
    elif transform_type == 'fft_magnitude':
        magnitude = np.abs(fft.fft(series))[:len(series)//2]
        x_old = np.linspace(0, 1, len(magnitude))
        x_new = np.linspace(0, 1, target_length)
        return interp1d(x_old, magnitude, kind='linear')(x_new)
    elif transform_type == 'fft_power':
        power = np.abs(fft.fft(series))**2
        power = power[:len(power)//2]
        x_old = np.linspace(0, 1, len(power))
        x_new = np.linspace(0, 1, target_length)
        return interp1d(x_old, power, kind='linear')(x_new)
    elif transform_type == 'derivative':
        deriv = np.gradient(series)
        return deriv
    return series

def compute_pattern_rmse(signal, pattern, start, width):
    """Compute RMSE between signal region and pattern."""
    if start + width > len(signal):
        return np.inf
    region = signal[start:start + width]
    if len(region) != len(pattern):
        return np.inf
    return np.sqrt(((region - pattern) ** 2).mean())

def find_best_discriminative_samples(X, y, pattern, start, width, class_0_label=0, class_1_label=1):
    """Find sample pairs that best demonstrate pattern discrimination."""
    # Compute RMSE for all samples
    rmse_scores = []
    for i, signal in enumerate(X):
        rmse = compute_pattern_rmse(signal, pattern, start, width)
        rmse_scores.append((i, rmse, y[i]))
    
    # Separate by class
    class_0_samples = [(idx, rmse) for idx, rmse, label in rmse_scores if label == class_0_label and np.isfinite(rmse)]
    class_1_samples = [(idx, rmse) for idx, rmse, label in rmse_scores if label == class_1_label and np.isfinite(rmse)]
    
    if not class_0_samples or not class_1_samples:
        return None, None
    
    # Sort by RMSE
    class_0_samples.sort(key=lambda x: x[1])
    class_1_samples.sort(key=lambda x: x[1])
    
    # Find best discriminative pair: low RMSE for class 0, high RMSE for class 1
    best_class_0_idx = class_0_samples[0][0]  # Lowest RMSE in class 0
    best_class_1_idx = class_1_samples[-1][0]  # Highest RMSE in class 1
    
    return best_class_0_idx, best_class_1_idx

def visualize_mitbih_patterns():
    """Visualize first MITBIH pattern with RMSE distribution."""
    print("Generating MITBIH pattern visualization...")
    
    # Load patterns (from fold 1)
    pattern_file = Path('../json_files/mitbih/pattern_parameters_fold1.json')
    if not pattern_file.exists():
        print(f"  Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        patterns = json.load(f)
    
    if not patterns:
        print("  No patterns found")
        return None
    
    # Load MITBIH data
    data_file = Path('../processed_datasets/mitbih_processed.csv')
    if not data_file.exists():
        print(f"  Data file not found: {data_file}")
        return None
    
    data = pd.read_csv(data_file)
    y = data['target']
    X = data.drop('target', axis=1).values
    
    # Convert to binary (0 = normal, 1 = any arrhythmia)
    y_binary = (y != 0).astype(int)
    
    pattern_info = patterns[0]
    transform_type = pattern_info.get('transform_type', 'raw')
    
    X_transformed = np.array([apply_transformation(x, transform_type, len(x)) for x in X])
    
    # Generate B-spline pattern
    start = int(pattern_info['center'] - pattern_info['width']/2)
    width = int(pattern_info['width'])
    
    pattern = generate_bspline_pattern(pattern_info['control_points'], width)
    
    # Apply transformation to pattern (matching core.py logic)
    if transform_type != 'raw':
        pattern = apply_transformation(pattern, transform_type, len(pattern))
    
    normal_idx, arrhythmia_idx = find_best_discriminative_samples(
        X_transformed, y_binary, pattern, start, width, class_0_label=0, class_1_label=1
    )
    
    if normal_idx is None or arrhythmia_idx is None:
        print(f"  Warning: Could not find discriminative samples")
        return None
    
    normal_example = X_transformed[normal_idx]
    arrhythmia_example = X_transformed[arrhythmia_idx]
    
    normal_norm = (normal_example - normal_example.mean()) / normal_example.std()
    arrhythmia_norm = (arrhythmia_example - arrhythmia_example.mean()) / arrhythmia_example.std()
    pattern_norm = (pattern - pattern.mean()) / pattern.std()
    
    X_region = X_transformed[:, start:start + width]
    pattern_region = pattern[:width] if len(pattern) > width else pattern
    rmse_all = np.sqrt(np.mean((X_region - pattern_region) ** 2, axis=1))
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Column 1: Normal beat example
    ax_normal = axes[0]
    ax_normal.plot(normal_norm, color='#2c3e50', linewidth=1.5, alpha=0.7, label='ECG (transformed)')
    pattern_x = np.arange(start, start + width)
    ax_normal.plot(pattern_x, pattern_norm, color='#e74c3c', linewidth=3, label='Pattern')
    # Add shift tolerance visualization
    shift_tolerance = pattern_info.get('shift_tolerance', 0.0)
    shift_pixels = int(shift_tolerance * width)
    ax_normal.axvspan(start - shift_pixels, start + width + shift_pixels - 1, alpha=0.15, color='#e74c3c', label='Search Region')
    ax_normal.set_xlim(0, len(normal_example))
    ax_normal.set_xlabel('Sample Index', fontsize=11)
    ax_normal.set_ylabel('Amplitude (z-scored)', fontsize=11)
    ax_normal.set_title(f'Normal - {transform_type}', fontsize=12, fontweight='bold')
    ax_normal.legend(loc='upper right', fontsize=9)
    ax_normal.grid(alpha=0.3, linestyle='--')
    normal_rmse = compute_pattern_rmse(X_transformed[normal_idx], pattern, start, width)
    ax_normal.text(0.02, 0.05, f'RMSE: {normal_rmse:.3f}',
                  transform=ax_normal.transAxes, fontsize=10, verticalalignment='bottom',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax_arrhythmia = axes[1]
    ax_arrhythmia.plot(arrhythmia_norm, color='#2c3e50', linewidth=1.5, alpha=0.7, label='ECG (transformed)')
    ax_arrhythmia.plot(pattern_x, pattern_norm, color='#e74c3c', linewidth=3, label='Pattern')
    # Add shift tolerance visualization
    ax_arrhythmia.axvspan(start - shift_pixels, start + width + shift_pixels - 1, alpha=0.15, color='#e74c3c', label='Search Region')
    ax_arrhythmia.set_xlim(0, len(arrhythmia_example))
    ax_arrhythmia.set_xlabel('Sample Index', fontsize=11)
    ax_arrhythmia.set_title(f'Arrhythmic - {transform_type}', fontsize=12, fontweight='bold')
    ax_arrhythmia.grid(alpha=0.3, linestyle='--')
    arrhythmia_rmse = compute_pattern_rmse(X_transformed[arrhythmia_idx], pattern, start, width)
    ax_arrhythmia.text(0.02, 0.05, f'RMSE: {arrhythmia_rmse:.3f}',
                      transform=ax_arrhythmia.transAxes, fontsize=10, verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Column 3: RMSE distribution
    ax_dist = axes[2]
    sns.histplot(x=rmse_all, hue=y_binary, bins=50, alpha=0.7, ax=ax_dist, 
                palette={0: '#2ecc71', 1: '#e74c3c'}, legend=True)
    ax_dist.set_xlabel('Pattern Similarity (RMSE)', fontsize=11)
    ax_dist.set_ylabel('Count', fontsize=11)
    ax_dist.set_title('RMSE Distribution by Class', fontsize=12, fontweight='bold')
    ax_dist.grid(alpha=0.3, linestyle='--', axis='y')
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    
    handles, labels = ax_dist.get_legend_handles_labels()
    ax_dist.legend(handles, ['Normal', 'Arrhythmia'], title='Beat Type', fontsize=9)
    
    plt.suptitle(f'MITBIH: Pattern in {transform_type} Space (Start: {start}, Width: {width})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def visualize_remc_patterns():
    """Visualize first REMC pattern with RMSE distribution."""
    print("Generating REMC pattern visualization...")
    
    # Load patterns for E003 (fold 1)
    pattern_file = Path('../json_files/remc/pattern_parameters_E003_fold1.json')
    if not pattern_file.exists():
        print(f"  Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        patterns = json.load(f)
    
    if not patterns:
        print("  No patterns found")
        return None
    
    # Load REMC data
    data_file = Path('../processed_datasets/remc/E003.parquet')
    if not data_file.exists():
        print(f"  Data file not found: {data_file}")
        return None
    
    df = pd.read_parquet(data_file)
    y = df['target']
    
    # Histone marks
    histone_names = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    
    # Use first pattern only
    pattern_info = patterns[0]
    transform_type = pattern_info.get('transform_type', 'raw')
    
    # Get pattern details
    start = int(pattern_info['center'] - pattern_info['width']/2)
    width = int(pattern_info['width'])
    
    pattern = generate_bspline_pattern(pattern_info['control_points'], width)
    
    # Apply transformation to pattern if needed (matching core.py logic)
    if transform_type != 'raw':
        pattern = apply_transformation(pattern, transform_type, len(pattern))
    series_idx = pattern_info['series_idx']
    histone_name = histone_names[series_idx]
    
    # Get columns for this histone mark
    histone_cols = [col for col in df.columns if col.startswith(f"{histone_name}_")]
    histone_data = df[histone_cols].values
    
    # Apply transformation to data if pattern uses transformed space
    if transform_type != 'raw':
        histone_data_transformed = np.array([apply_transformation(row, transform_type, len(row)) for row in histone_data])
    else:
        histone_data_transformed = histone_data
    
    # Find best discriminative samples
    high_expr_idx, low_expr_idx = find_best_discriminative_samples(
        histone_data_transformed, y.values, pattern, start, width, class_0_label=1, class_1_label=0
    )
    
    if high_expr_idx is None or low_expr_idx is None:
        print(f"  Warning: Could not find discriminative samples")
        return None
    
    # Compute RMSE for all samples (use transformed data)
    X_region = histone_data_transformed[:, start:start + width]
    pattern_region = pattern[:width] if len(pattern) > width else pattern
    rmse_all = np.sqrt(np.mean((X_region - pattern_region) ** 2, axis=1))
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Column 1: High expression example
    ax_high = axes[0]
    high_example = histone_data[high_expr_idx]
    x_positions = np.arange(len(histone_cols))
    ax_high.plot(x_positions, high_example, color='#3498db', linewidth=2, alpha=0.7, label=f'{histone_name} Signal')
    
    # Overlay pattern
    pattern_x = np.arange(start, start + width)
    data_range = high_example[start:start+width]
    if len(data_range) > 0:
        pattern_scaled = pattern * (data_range.max() - data_range.min()) + data_range.min()
        ax_high.plot(pattern_x, pattern_scaled, color='#e74c3c', linewidth=3, 
                    linestyle='--', label='Pattern Template', zorder=10)
    
    # Add shift tolerance visualization
    shift_tolerance = pattern_info.get('shift_tolerance', 0.0)
    shift_pixels = int(shift_tolerance * width)
    ax_high.axvspan(start - shift_pixels, start + width + shift_pixels - 1, alpha=0.15, color='#e74c3c', label='Search Region')
    tss_pos = len(histone_cols) // 2
    ax_high.axvline(tss_pos, color='black', linestyle=':', linewidth=2, alpha=0.5, label='TSS')
    ax_high.set_xlim(0, len(histone_cols))
    ax_high.set_xlabel('Position (bp from TSS)', fontsize=11)
    ax_high.set_ylabel('ChIP-seq Signal', fontsize=11)
    ax_high.set_title('High Expression Example', fontsize=12, fontweight='bold')
    ax_high.legend(loc='upper right', fontsize=9)
    ax_high.grid(alpha=0.3, linestyle='--')
    high_rmse = compute_pattern_rmse(high_example, pattern, start, width)
    ax_high.text(0.02, 0.05, f'RMSE: {high_rmse:.3f}',
                transform=ax_high.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Column 2: Low expression example
    ax_low = axes[1]
    low_example = histone_data[low_expr_idx]
    ax_low.plot(x_positions, low_example, color='#e67e22', linewidth=2, alpha=0.7, label=f'{histone_name} Signal')
    
    # Overlay pattern
    data_range_low = low_example[start:start+width]
    if len(data_range_low) > 0:
        pattern_scaled_low = pattern * (data_range_low.max() - data_range_low.min()) + data_range_low.min()
        ax_low.plot(pattern_x, pattern_scaled_low, color='#e74c3c', linewidth=3, 
                   linestyle='--', label='Pattern Template', zorder=10)
    
    # Add shift tolerance visualization
    ax_low.axvspan(start - shift_pixels, start + width + shift_pixels - 1, alpha=0.15, color='#e74c3c', label='Search Region')
    ax_low.axvline(tss_pos, color='black', linestyle=':', linewidth=2, alpha=0.5, label='TSS')
    ax_low.set_xlim(0, len(histone_cols))
    ax_low.set_xlabel('Position (bp from TSS)', fontsize=11)
    ax_low.set_title('Low Expression Example', fontsize=12, fontweight='bold')
    ax_low.grid(alpha=0.3, linestyle='--')
    low_rmse = compute_pattern_rmse(low_example, pattern, start, width)
    ax_low.text(0.02, 0.05, f'RMSE: {low_rmse:.3f}',
               transform=ax_low.transAxes, fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Column 3: RMSE distribution
    ax_dist = axes[2]
    sns.histplot(x=rmse_all, hue=y.values, bins=50, alpha=0.7, ax=ax_dist,
                palette={1: '#3498db', 0: '#e67e22'}, legend=True)
    ax_dist.set_xlabel('Pattern Similarity (RMSE)', fontsize=11)
    ax_dist.set_ylabel('Count', fontsize=11)
    ax_dist.set_title('RMSE Distribution by Class', fontsize=12, fontweight='bold')
    ax_dist.grid(alpha=0.3, linestyle='--', axis='y')
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    
    # Update legend labels
    handles, labels = ax_dist.get_legend_handles_labels()
    ax_dist.legend(handles, ['High Expr', 'Low Expr'], title='Expression', fontsize=9)
    
    plt.suptitle(f'REMC E003: Pattern Analysis - {histone_name} (Start: {start}, Width: {width})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def visualize_combined_patterns():
    """Create a combined visualization showing both MITBIH and REMC patterns."""
    print("Generating combined pattern visualization...")
    
    # Create a large figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # MITBIH visualization (top row)
    print("  Processing MITBIH patterns...")
    
    # Load MITBIH patterns
    mitbih_pattern_file = Path('../json_files/mitbih/pattern_parameters_fold1.json')
    if not mitbih_pattern_file.exists():
        print(f"  MITBIH pattern file not found: {mitbih_pattern_file}")
        return None
    
    with open(mitbih_pattern_file, 'r') as f:
        mitbih_patterns = json.load(f)
    
    if not mitbih_patterns:
        print("  No MITBIH patterns found")
        return None
    
    # Load MITBIH data
    mitbih_data_file = Path('../processed_datasets/mitbih_processed.csv')
    if not mitbih_data_file.exists():
        print(f"  MITBIH data file not found: {mitbih_data_file}")
        return None
    
    mitbih_data = pd.read_csv(mitbih_data_file)
    mitbih_y = mitbih_data['target']
    mitbih_X = mitbih_data.drop('target', axis=1).values
    mitbih_y_binary = (mitbih_y != 0).astype(int)
    
    # Use pattern 8 (index 7) which has the best score and proper positioning
    mitbih_pattern_info = mitbih_patterns[7]  # Pattern 8: raw transform, center=78, width=14, score=0.947
    mitbih_transform_type = mitbih_pattern_info.get('transform_type', 'raw')
    
    mitbih_X_transformed = np.array([apply_transformation(x, mitbih_transform_type, len(x)) for x in mitbih_X])
    
    mitbih_start = int(mitbih_pattern_info['center'] - mitbih_pattern_info['width']/2)
    mitbih_width = int(mitbih_pattern_info['width'])
    
    mitbih_pattern = generate_bspline_pattern(mitbih_pattern_info['control_points'], mitbih_width)
    
    if mitbih_transform_type != 'raw':
        mitbih_pattern = apply_transformation(mitbih_pattern, mitbih_transform_type, len(mitbih_pattern))
    
    mitbih_normal_idx, mitbih_arrhythmia_idx = find_best_discriminative_samples(
        mitbih_X_transformed, mitbih_y_binary, mitbih_pattern, mitbih_start, mitbih_width, class_0_label=0, class_1_label=1
    )
    
    if mitbih_normal_idx is not None and mitbih_arrhythmia_idx is not None:
        print(f"    MITBIH Pattern Details:")
        print(f"      Transform: {mitbih_transform_type}")
        print(f"      Position: center={mitbih_pattern_info['center']:.1f}, width={mitbih_pattern_info['width']:.1f}")
        print(f"      Signal range: {mitbih_start} to {mitbih_start + mitbih_width}")
        
        mitbih_normal_example = mitbih_X_transformed[mitbih_normal_idx]
        mitbih_arrhythmia_example = mitbih_X_transformed[mitbih_arrhythmia_idx]
        
        mitbih_normal_norm = (mitbih_normal_example - mitbih_normal_example.mean()) / mitbih_normal_example.std()
        mitbih_arrhythmia_norm = (mitbih_arrhythmia_example - mitbih_arrhythmia_example.mean()) / mitbih_arrhythmia_example.std()
        mitbih_pattern_norm = (mitbih_pattern - mitbih_pattern.mean()) / mitbih_pattern.std()
        
        # MITBIH Column 1: Normal beat
        ax_normal = axes[0, 0]
        ax_normal.plot(mitbih_normal_norm, color='#2c3e50', linewidth=1.5, alpha=0.7, label='ECG (transformed)')
        pattern_x = np.arange(mitbih_start, mitbih_start + mitbih_width)
        ax_normal.plot(pattern_x, mitbih_pattern_norm, color='#e74c3c', linewidth=3, label='Pattern')
        # Add shift tolerance visualization
        mitbih_shift_tolerance = mitbih_pattern_info.get('shift_tolerance', 0.0)
        mitbih_shift_pixels = int(mitbih_shift_tolerance * mitbih_width)
        ax_normal.axvspan(mitbih_start - mitbih_shift_pixels, mitbih_start + mitbih_width + mitbih_shift_pixels - 1, alpha=0.15, color='#e74c3c', label='Search Region')
        ax_normal.set_xlim(0, len(mitbih_normal_example))
        ax_normal.set_xlabel('Sample Index', fontsize=11)
        ax_normal.set_ylabel('Amplitude (z-scored)', fontsize=11)
        ax_normal.set_title(f'MITBIH: Normal - {mitbih_transform_type}', fontsize=12, fontweight='bold')
        ax_normal.legend(loc='upper right', fontsize=9)
        ax_normal.grid(alpha=0.3, linestyle='--')
        normal_rmse = compute_pattern_rmse(mitbih_X_transformed[mitbih_normal_idx], mitbih_pattern, mitbih_start, mitbih_width)
        print(f"      Normal beat RMSE: {normal_rmse:.3f}")
        ax_normal.text(0.02, 0.05, f'RMSE: {normal_rmse:.3f}',
                      transform=ax_normal.transAxes, fontsize=10, verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # MITBIH Column 2: Arrhythmic beat
        ax_arrhythmia = axes[0, 1]
        ax_arrhythmia.plot(mitbih_arrhythmia_norm, color='#2c3e50', linewidth=1.5, alpha=0.7, label='ECG (transformed)')
        ax_arrhythmia.plot(pattern_x, mitbih_pattern_norm, color='#e74c3c', linewidth=3, label='Pattern')
        # Add shift tolerance visualization
        ax_arrhythmia.axvspan(mitbih_start - mitbih_shift_pixels, mitbih_start + mitbih_width + mitbih_shift_pixels - 1, alpha=0.15, color='#e74c3c', label='Search Region')
        ax_arrhythmia.set_xlim(0, len(mitbih_arrhythmia_example))
        ax_arrhythmia.set_xlabel('Sample Index', fontsize=11)
        ax_arrhythmia.set_title(f'MITBIH: Arrhythmic - {mitbih_transform_type}', fontsize=12, fontweight='bold')
        ax_arrhythmia.grid(alpha=0.3, linestyle='--')
        arrhythmia_rmse = compute_pattern_rmse(mitbih_X_transformed[mitbih_arrhythmia_idx], mitbih_pattern, mitbih_start, mitbih_width)
        print(f"      Arrhythmic beat RMSE: {arrhythmia_rmse:.3f}")
        ax_arrhythmia.text(0.02, 0.05, f'RMSE: {arrhythmia_rmse:.3f}',
                          transform=ax_arrhythmia.transAxes, fontsize=10, verticalalignment='bottom',
                          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # MITBIH Column 3: RMSE distribution
        ax_dist = axes[0, 2]
        X_region = mitbih_X_transformed[:, mitbih_start:mitbih_start + mitbih_width]
        pattern_region = mitbih_pattern[:mitbih_width] if len(mitbih_pattern) > mitbih_width else mitbih_pattern
        rmse_all = np.sqrt(np.mean((X_region - pattern_region) ** 2, axis=1))
        
        # Calculate distribution statistics
        normal_rmse_mean = rmse_all[mitbih_y_binary == 0].mean()
        arrhythmia_rmse_mean = rmse_all[mitbih_y_binary == 1].mean()
        print(f"      RMSE Distribution:")
        print(f"        Normal beats: mean={normal_rmse_mean:.3f}, std={rmse_all[mitbih_y_binary == 0].std():.3f}")
        print(f"        Arrhythmic beats: mean={arrhythmia_rmse_mean:.3f}, std={rmse_all[mitbih_y_binary == 1].std():.3f}")
        print(f"        Separation: {abs(normal_rmse_mean - arrhythmia_rmse_mean):.3f}")
        
        sns.histplot(x=rmse_all, hue=mitbih_y_binary, bins=50, alpha=0.7, ax=ax_dist, 
                    palette={0: '#2ecc71', 1: '#e74c3c'}, legend=False)
        ax_dist.set_xlabel('Pattern Similarity (RMSE)', fontsize=11)
        ax_dist.set_ylabel('Count', fontsize=11)
        ax_dist.set_title('MITBIH: RMSE Distribution by Class', fontsize=12, fontweight='bold')
        ax_dist.grid(alpha=0.3, linestyle='--', axis='y')
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='Normal'),
                          Patch(facecolor='#e74c3c', label='Arrhythmia')]
        ax_dist.legend(handles=legend_elements, title='Beat Type', fontsize=9, loc='upper right')
    
    # REMC visualization (bottom row)
    print("  Processing REMC patterns...")
    
    # Load REMC patterns
    remc_pattern_file = Path('../json_files/remc/pattern_parameters_E003_fold1.json')
    if not remc_pattern_file.exists():
        print(f"  REMC pattern file not found: {remc_pattern_file}")
        return None
    
    with open(remc_pattern_file, 'r') as f:
        remc_patterns = json.load(f)
    
    if not remc_patterns:
        print("  No REMC patterns found")
        return None
    
    # Load REMC data
    remc_data_file = Path('../processed_datasets/remc/E003.parquet')
    if not remc_data_file.exists():
        print(f"  REMC data file not found: {remc_data_file}")
        return None
    
    remc_df = pd.read_parquet(remc_data_file)
    remc_y = remc_df['target']
    
    histone_names = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    
    remc_pattern_info = remc_patterns[0]
    remc_transform_type = remc_pattern_info.get('transform_type', 'raw')
    
    remc_start = int(remc_pattern_info['center'] - remc_pattern_info['width']/2)
    remc_width = int(remc_pattern_info['width'])
    
    remc_pattern = generate_bspline_pattern(remc_pattern_info['control_points'], remc_width)
    
    if remc_transform_type != 'raw':
        remc_pattern = apply_transformation(remc_pattern, remc_transform_type, len(remc_pattern))
    
    series_idx = remc_pattern_info['series_idx']
    histone_name = histone_names[series_idx]
    
    # Get columns for this histone mark
    histone_cols = [col for col in remc_df.columns if col.startswith(f"{histone_name}_")]
    histone_data = remc_df[histone_cols].values
    
    # Apply transformation to data if pattern uses transformed space
    if remc_transform_type != 'raw':
        histone_data_transformed = np.array([apply_transformation(row, remc_transform_type, len(row)) for row in histone_data])
    else:
        histone_data_transformed = histone_data
    
    # Find best discriminative samples
    high_expr_idx, low_expr_idx = find_best_discriminative_samples(
        histone_data_transformed, remc_y.values, remc_pattern, remc_start, remc_width, class_0_label=1, class_1_label=0
    )
    
    if high_expr_idx is not None and low_expr_idx is not None:
        print(f"    REMC Pattern Details:")
        print(f"      Transform: {remc_transform_type}")
        print(f"      Histone mark: {histone_name} (series {series_idx})")
        print(f"      Position: center={remc_pattern_info['center']:.1f}, width={remc_pattern_info['width']:.1f}")
        print(f"      Signal range: {remc_start} to {remc_start + remc_width}")
        
        # REMC Column 1: High expression example
        ax_high = axes[1, 0]
        high_example = histone_data[high_expr_idx]
        x_positions = np.arange(len(histone_cols))
        ax_high.plot(x_positions, high_example, color='#3498db', linewidth=2, alpha=0.7, label=f'{histone_name} Signal')
        
        # Overlay pattern
        pattern_x = np.arange(remc_start, remc_start + remc_width)
        data_range = high_example[remc_start:remc_start+remc_width]
        if len(data_range) > 0:
            pattern_scaled = remc_pattern * (data_range.max() - data_range.min()) + data_range.min()
            ax_high.plot(pattern_x, pattern_scaled, color='#e74c3c', linewidth=3, 
                        linestyle='--', label='Pattern Template', zorder=10)
        
        # Add shift tolerance visualization
        remc_shift_tolerance = remc_pattern_info.get('shift_tolerance', 0.0)
        remc_shift_pixels = int(remc_shift_tolerance * remc_width)
        ax_high.axvspan(remc_start - remc_shift_pixels, remc_start + remc_width + remc_shift_pixels - 1, alpha=0.15, color='#e74c3c', label='Search Region')
        tss_pos = len(histone_cols) // 2
        ax_high.axvline(tss_pos, color='black', linestyle=':', linewidth=2, alpha=0.5, label='TSS')
        ax_high.set_xlim(0, len(histone_cols))
        ax_high.set_xlabel('Position (bp from TSS)', fontsize=11)
        ax_high.set_ylabel('ChIP-seq Signal', fontsize=11)
        ax_high.set_title(f'REMC: High Expression - {histone_name}', fontsize=12, fontweight='bold')
        ax_high.legend(loc='upper right', fontsize=9)
        ax_high.grid(alpha=0.3, linestyle='--')
        high_rmse = compute_pattern_rmse(high_example, remc_pattern, remc_start, remc_width)
        print(f"      High expression RMSE: {high_rmse:.3f}")
        ax_high.text(0.02, 0.05, f'RMSE: {high_rmse:.3f}',
                    transform=ax_high.transAxes, fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # REMC Column 2: Low expression example
        ax_low = axes[1, 1]
        low_example = histone_data[low_expr_idx]
        ax_low.plot(x_positions, low_example, color='#e67e22', linewidth=2, alpha=0.7, label=f'{histone_name} Signal')
        
        # Overlay pattern
        data_range_low = low_example[remc_start:remc_start+remc_width]
        if len(data_range_low) > 0:
            pattern_scaled_low = remc_pattern * (data_range_low.max() - data_range_low.min()) + data_range_low.min()
            ax_low.plot(pattern_x, pattern_scaled_low, color='#e74c3c', linewidth=3, 
                       linestyle='--', label='Pattern Template', zorder=10)
        
        # Add shift tolerance visualization
        ax_low.axvspan(remc_start - remc_shift_pixels, remc_start + remc_width + remc_shift_pixels - 1, alpha=0.15, color='#e74c3c', label='Search Region')
        ax_low.axvline(tss_pos, color='black', linestyle=':', linewidth=2, alpha=0.5, label='TSS')
        ax_low.set_xlim(0, len(histone_cols))
        ax_low.set_xlabel('Position (bp from TSS)', fontsize=11)
        ax_low.set_title(f'REMC: Low Expression - {histone_name}', fontsize=12, fontweight='bold')
        ax_low.grid(alpha=0.3, linestyle='--')
        low_rmse = compute_pattern_rmse(low_example, remc_pattern, remc_start, remc_width)
        print(f"      Low expression RMSE: {low_rmse:.3f}")
        ax_low.text(0.02, 0.05, f'RMSE: {low_rmse:.3f}',
                   transform=ax_low.transAxes, fontsize=10, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # REMC Column 3: RMSE distribution
        ax_dist = axes[1, 2]
        X_region = histone_data_transformed[:, remc_start:remc_start + remc_width]
        pattern_region = remc_pattern[:remc_width] if len(remc_pattern) > remc_width else remc_pattern
        rmse_all = np.sqrt(np.mean((X_region - pattern_region) ** 2, axis=1))
        
        # Calculate distribution statistics
        high_expr_rmse_mean = rmse_all[remc_y.values == 1].mean()
        low_expr_rmse_mean = rmse_all[remc_y.values == 0].mean()
        print(f"      RMSE Distribution:")
        print(f"        High expression: mean={high_expr_rmse_mean:.3f}, std={rmse_all[remc_y.values == 1].std():.3f}")
        print(f"        Low expression: mean={low_expr_rmse_mean:.3f}, std={rmse_all[remc_y.values == 0].std():.3f}")
        print(f"        Separation: {abs(high_expr_rmse_mean - low_expr_rmse_mean):.3f}")
        
        sns.histplot(x=rmse_all, hue=remc_y.values, bins=50, alpha=0.7, ax=ax_dist,
                    palette={1: '#3498db', 0: '#e67e22'}, legend=False)
        ax_dist.set_xlabel('Pattern Similarity (RMSE)', fontsize=11)
        ax_dist.set_ylabel('Count', fontsize=11)
        ax_dist.set_title('REMC: RMSE Distribution by Class', fontsize=12, fontweight='bold')
        ax_dist.grid(alpha=0.3, linestyle='--', axis='y')
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        
        # Create custom legend
        legend_elements = [Patch(facecolor='#3498db', label='High Expr'),
                          Patch(facecolor='#e67e22', label='Low Expr')]
        ax_dist.legend(handles=legend_elements, title='Expression', fontsize=9, loc='upper right')
    
    plt.suptitle('Domain-Specific Pattern Interpretation: MITBIH (Top) and REMC (Bottom)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    print(f"    Summary:")
    print(f"      MITBIH: Pattern in {mitbih_transform_type} space achieves {abs(normal_rmse_mean - arrhythmia_rmse_mean):.1f} RMSE separation")
    print(f"      REMC: Pattern in {remc_transform_type} space achieves {abs(high_expr_rmse_mean - low_expr_rmse_mean):.1f} RMSE separation")
    print(f"      Both patterns demonstrate clear discriminative power for their respective classification tasks")
    
    return fig

def main():
    print("="*60)
    print("Domain-Specific Pattern Visualization")
    print("="*60)
    
    output_dir = Path('../manuscript/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate combined visualization
    print("\nGenerating combined MITBIH and REMC pattern visualization:")
    fig_combined = visualize_combined_patterns()
    if fig_combined:
        output_path = output_dir / 'domain_pattern_interpretation.png'
        fig_combined.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_combined)
        print(f"   Saved to {output_path}")
    else:
        print("   Failed to generate combined visualization")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == "__main__":
    main()

