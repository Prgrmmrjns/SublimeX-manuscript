import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import BSpline, interp1d
import pywt
from scipy import fft
from matplotlib.patches import Patch

def generate_bspline_pattern(control_points, width):
    degree = 3
    n_cp = len(control_points)
    knots = np.concatenate([np.zeros(degree + 1), np.linspace(0, 1, n_cp - degree + 1)[1:-1], np.ones(degree + 1)])
    return BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width))

def apply_transformation(series, transform_type, target_length):
    if transform_type == 'raw':
        return series
    elif transform_type == 'wavelet':
        coeffs = pywt.wavedec(series, 'db4', level=4, mode='periodization')
        concatenated = np.concatenate(coeffs)
        return interp1d(np.linspace(0, 1, len(concatenated)), concatenated, kind='linear')(np.linspace(0, 1, target_length))
    elif transform_type == 'fft_power':
        power = (np.abs(fft.fft(series))**2)[:len(series)//2]
        return interp1d(np.linspace(0, 1, len(power)), power, kind='linear')(np.linspace(0, 1, target_length))
    elif transform_type == 'derivative':
        return np.gradient(series)
    elif transform_type == 'cumsum':
        return np.cumsum(series)
    return series

def compute_rmse(signal, pattern, start, width):
    if start + width > len(signal) or len(pattern) != width:
        return np.inf
    return np.sqrt(((signal[start:start + width] - pattern) ** 2).mean())

def find_discriminative_samples(X, y, pattern, start, width, class_0=0, class_1=1):
    rmse_scores = [(i, compute_rmse(signal, pattern, start, width), y[i]) for i, signal in enumerate(X)]
    class_0_samples = [(idx, rmse) for idx, rmse, label in rmse_scores if label == class_0 and np.isfinite(rmse)]
    class_1_samples = [(idx, rmse) for idx, rmse, label in rmse_scores if label == class_1 and np.isfinite(rmse)]
    if not class_0_samples or not class_1_samples:
        return None, None
    return class_0_samples[0][0], class_1_samples[-1][0]

def plot_pattern_example(ax, signal, pattern, start, width, shift_tolerance, title, rmse, color, bg_color):
    signal_norm = (signal - signal.mean()) / signal.std()
    pattern_norm = (pattern - pattern.mean()) / pattern.std()
    ax.plot(signal_norm, color='#2c3e50', linewidth=1.5, alpha=0.7)
    pattern_x = np.arange(start, start + width)
    ax.plot(pattern_x, pattern_norm, color='#e74c3c', linewidth=3)
    shift_pixels = int(shift_tolerance * width)
    ax.axvspan(start - shift_pixels, start + width + shift_pixels - 1, alpha=0.15, color='#e74c3c')
    ax.set_xlim(0, len(signal))
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Amplitude (z-scored)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.text(0.02, 0.05, f'RMSE: {rmse:.3f}', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))

def plot_histone_example(ax, signal, pattern, start, width, shift_tolerance, tss_pos, title, rmse, color, bg_color, histone_name):
    x_pos = np.arange(len(signal))
    ax.plot(x_pos, signal, color=color, linewidth=2, alpha=0.7, label=f'{histone_name} Signal')
    pattern_x = np.arange(start, start + width)
    if len(signal[start:start+width]) > 0:
        data_range = signal[start:start+width]
        pattern_scaled = pattern * (data_range.max() - data_range.min()) + data_range.min()
        ax.plot(pattern_x, pattern_scaled, color='#e74c3c', linewidth=3, linestyle='--', zorder=10)
    shift_pixels = int(shift_tolerance * width)
    ax.axvspan(start - shift_pixels, start + width + shift_pixels - 1, alpha=0.15, color='#e74c3c')
    ax.axvline(tss_pos, color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax.set_xlim(0, len(signal))
    ax.set_xlabel('Position (bp from TSS)', fontsize=11)
    ax.set_ylabel('ChIP-seq Signal', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.text(0.02, 0.05, f'RMSE: {rmse:.3f}', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))

def visualize_combined_patterns():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    print("  Processing MITBIH patterns...")
    with open(Path('../json_files/mitbih/pattern_parameters.json'), 'r') as f:
        mitbih_data = json.load(f)
    mitbih_patterns = mitbih_data['fold_1']
    
    df = pd.read_csv(Path('../processed_datasets/mitbih_processed.csv'))
    mitbih_X, mitbih_y = df.drop('target', axis=1).values, (df['target'] != 0).astype(int)
    
    mitbih_pattern_info = mitbih_patterns[0]
    transform_type = mitbih_pattern_info.get('transform_type', 'raw')
    mitbih_X_transformed = np.array([apply_transformation(x, transform_type, len(x)) for x in mitbih_X])
    
    start = int(mitbih_pattern_info.get('start', mitbih_pattern_info['center'] - mitbih_pattern_info['width']/2))
    width = int(mitbih_pattern_info['width'])
    signal_len = len(mitbih_X_transformed[0])
    if start + width > signal_len:
        width = signal_len - start
    pattern = generate_bspline_pattern(mitbih_pattern_info['control_points'], width)
    if transform_type != 'raw':
        pattern = apply_transformation(pattern, transform_type, len(pattern))
    
    normal_idx, arrhythmia_idx = find_discriminative_samples(mitbih_X_transformed, mitbih_y, pattern, start, width)
    
    if normal_idx is not None and arrhythmia_idx is not None:
        print(f"    MITBIH: {transform_type}, center={mitbih_pattern_info['center']:.1f}, width={mitbih_pattern_info['width']:.1f}")
        normal_rmse = compute_rmse(mitbih_X_transformed[normal_idx], pattern, start, width)
        arrhythmia_rmse = compute_rmse(mitbih_X_transformed[arrhythmia_idx], pattern, start, width)
        
        plot_pattern_example(axes[0, 0], mitbih_X_transformed[normal_idx], pattern, start, width,
                            mitbih_pattern_info.get('shift_tolerance', 0.0), f'MITBIH: Normal - {transform_type}',
                            normal_rmse, '#2c3e50', 'lightgreen')
        plot_pattern_example(axes[0, 1], mitbih_X_transformed[arrhythmia_idx], pattern, start, width,
                            mitbih_pattern_info.get('shift_tolerance', 0.0), f'MITBIH: Arrhythmic - {transform_type}',
                            arrhythmia_rmse, '#2c3e50', 'lightcoral')
        
        X_region = mitbih_X_transformed[:, start:start + width]
        pattern_region = pattern[:width] if len(pattern) > width else pattern
        rmse_all = np.sqrt(np.mean((X_region - pattern_region) ** 2, axis=1))
        normal_mean, arrhythmia_mean = rmse_all[mitbih_y == 0].mean(), rmse_all[mitbih_y == 1].mean()
        print(f"      RMSE: Normal={normal_mean:.3f}, Arrhythmic={arrhythmia_mean:.3f}, Separation={abs(normal_mean - arrhythmia_mean):.3f}")
        
        sns.histplot(x=rmse_all, hue=mitbih_y, bins=50, alpha=0.7, ax=axes[0, 2], palette={0: '#2ecc71', 1: '#e74c3c'}, legend=False)
        axes[0, 2].set_xlabel('Pattern Similarity (RMSE)', fontsize=11)
        axes[0, 2].set_ylabel('Count', fontsize=11)
        axes[0, 2].set_title('MITBIH: RMSE Distribution by Class', fontsize=12, fontweight='bold')
        axes[0, 2].grid(alpha=0.3, linestyle='--', axis='y')
        axes[0, 2].legend(handles=[Patch(facecolor='#2ecc71', label='Normal'), Patch(facecolor='#e74c3c', label='Arrhythmia')], fontsize=9)
    
    print("  Processing REMC patterns...")
    with open(Path('../json_files/remc/pattern_parameters_E003.json'), 'r') as f:
        remc_data = json.load(f)
    remc_patterns = remc_data['fold_1']
    
    remc_df = pd.read_parquet(Path('../processed_datasets/remc/E003.parquet'))
    remc_y = remc_df['target']
    histone_names = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    
    remc_pattern_info = remc_patterns[0]
    transform_type = remc_pattern_info.get('transform_type', 'raw')
    start = int(remc_pattern_info.get('start', remc_pattern_info['center'] - remc_pattern_info['width']/2))
    width = int(remc_pattern_info['width'])
    pattern = generate_bspline_pattern(remc_pattern_info['control_points'], width)
    if transform_type != 'raw':
        pattern = apply_transformation(pattern, transform_type, len(pattern))
    
    series_idx = remc_pattern_info['series_idx']
    histone_name = histone_names[series_idx]
    histone_cols = [col for col in remc_df.columns if col.startswith(f"{histone_name}_")]
    histone_data = remc_df[histone_cols].values
    histone_data_transformed = np.array([apply_transformation(row, transform_type, len(row)) for row in histone_data]) if transform_type != 'raw' else histone_data
    
    high_idx, low_idx = find_discriminative_samples(histone_data_transformed, remc_y.values, pattern, start, width, class_0=1, class_1=0)
    
    if high_idx is not None and low_idx is not None:
        print(f"    REMC: {transform_type}, {histone_name}, center={remc_pattern_info['center']:.1f}, width={remc_pattern_info['width']:.1f}")
        high_rmse = compute_rmse(histone_data[high_idx], pattern, start, width)
        low_rmse = compute_rmse(histone_data[low_idx], pattern, start, width)
        
        tss_pos = len(histone_cols) // 2
        plot_histone_example(axes[1, 0], histone_data[high_idx], pattern, start, width,
                            remc_pattern_info.get('shift_tolerance', 0.0), tss_pos,
                            f'REMC: High Expression - {histone_name}', high_rmse, '#3498db', 'lightblue', histone_name)
        plot_histone_example(axes[1, 1], histone_data[low_idx], pattern, start, width,
                            remc_pattern_info.get('shift_tolerance', 0.0), tss_pos,
                            f'REMC: Low Expression - {histone_name}', low_rmse, '#e67e22', 'lightyellow', histone_name)
        
        X_region = histone_data_transformed[:, start:start + width]
        pattern_region = pattern[:width] if len(pattern) > width else pattern
        rmse_all = np.sqrt(np.mean((X_region - pattern_region) ** 2, axis=1))
        high_mean, low_mean = rmse_all[remc_y.values == 1].mean(), rmse_all[remc_y.values == 0].mean()
        print(f"      RMSE: High={high_mean:.3f}, Low={low_mean:.3f}, Separation={abs(high_mean - low_mean):.3f}")
        
        sns.histplot(x=rmse_all, hue=remc_y.values, bins=50, alpha=0.7, ax=axes[1, 2], palette={1: '#3498db', 0: '#e67e22'}, legend=False)
        axes[1, 2].set_xlabel('Pattern Similarity (RMSE)', fontsize=11)
        axes[1, 2].set_ylabel('Count', fontsize=11)
        axes[1, 2].set_title('REMC: RMSE Distribution by Class', fontsize=12, fontweight='bold')
        axes[1, 2].grid(alpha=0.3, linestyle='--', axis='y')
        axes[1, 2].legend(handles=[Patch(facecolor='#3498db', label='High Expr'), Patch(facecolor='#e67e22', label='Low Expr')], fontsize=9)
    
    plt.suptitle('Domain-Specific Pattern Interpretation: MITBIH (Top) and REMC (Bottom)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    print("="*60)
    print("Domain-Specific Pattern Visualization")
    print("="*60)
    
    output_dir = Path('../manuscript/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig_combined = visualize_combined_patterns()
    if fig_combined:
        output_path = output_dir / 'domain_pattern_interpretation.png'
        fig_combined.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_combined)
        print(f"   Saved to {output_path}")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == "__main__":
    main()
