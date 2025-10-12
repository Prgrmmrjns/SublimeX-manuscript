import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import BSpline

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
    """Visualize top MITBIH patterns on ECG beats."""
    print("Generating MITBIH pattern visualization...")
    
    # Load patterns (from fold 1)
    pattern_file = Path('../json_files/mitbih/pattern_parameters.json')
    if not pattern_file.exists():
        print(f"  Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        all_patterns = json.load(f)
    
    patterns = all_patterns.get('fold_1', [])
    if not patterns:
        print("  No patterns found in fold_1")
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
    
    # Select top 3 patterns
    top_patterns = patterns[:3]
    
    # Create figure with 3 rows (one per pattern), 2 columns (normal vs arrhythmic)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    for i, pattern_info in enumerate(top_patterns):
        # Regenerate pattern
        pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
        start = pattern_info['start']
        width = pattern_info['width']
        
        # Find best discriminative samples
        normal_idx, arrhythmia_idx = find_best_discriminative_samples(
            X, y_binary, pattern, start, width, class_0_label=0, class_1_label=1
        )
        
        if normal_idx is None or arrhythmia_idx is None:
            print(f"  Warning: Could not find discriminative samples for pattern {i+1}")
            continue
        
        normal_example = X[normal_idx]
        arrhythmia_example = X[arrhythmia_idx]
        
        # Z-score normalize for visualization
        normal_norm = (normal_example - normal_example.mean()) / normal_example.std()
        arrhythmia_norm = (arrhythmia_example - arrhythmia_example.mean()) / arrhythmia_example.std()
        
        # Normalize pattern to similar scale
        pattern_norm = (pattern - pattern.mean()) / pattern.std()
        
        # Compute RMSE for these examples
        normal_rmse = compute_pattern_rmse(normal_example, pattern, start, width)
        arrhythmia_rmse = compute_pattern_rmse(arrhythmia_example, pattern, start, width)
        
        # Plot normal beat
        ax_normal = axes[i, 0]
        ax_normal.plot(normal_norm, color='#2c3e50', linewidth=1.5, alpha=0.7, label='ECG Signal')
        
        # Overlay pattern
        pattern_x = np.arange(start, start + width)
        ax_normal.plot(pattern_x, pattern_norm, color='#e74c3c', linewidth=3, label='Pattern Template')
        ax_normal.axvspan(start, start + width - 1, alpha=0.15, color='#e74c3c')
        
        ax_normal.set_xlim(0, len(normal_example))
        ax_normal.set_ylabel('Amplitude (z-scored)', fontsize=10)
        if i == 0:
            ax_normal.set_title('Normal Beat', fontsize=12, fontweight='bold')
        if i == 2:
            ax_normal.set_xlabel('Sample Index', fontsize=10)
        if i == 0:
            ax_normal.legend(loc='upper right', fontsize=9)
        ax_normal.grid(alpha=0.3, linestyle='--')
        ax_normal.text(0.02, 0.95, f'Pattern {i+1}\nStart: {start}, Width: {width}\nRMSE: {normal_rmse:.3f}',
                      transform=ax_normal.transAxes, fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Plot arrhythmic beat
        ax_arrhythmia = axes[i, 1]
        ax_arrhythmia.plot(arrhythmia_norm, color='#2c3e50', linewidth=1.5, alpha=0.7, label='ECG Signal')
        
        # Overlay pattern
        ax_arrhythmia.plot(pattern_x, pattern_norm, color='#e74c3c', linewidth=3, label='Pattern Template')
        ax_arrhythmia.axvspan(start, start + width - 1, alpha=0.15, color='#e74c3c')
        
        ax_arrhythmia.set_xlim(0, len(arrhythmia_example))
        if i == 0:
            ax_arrhythmia.set_title('Arrhythmic Beat', fontsize=12, fontweight='bold')
        if i == 2:
            ax_arrhythmia.set_xlabel('Sample Index', fontsize=10)
        ax_arrhythmia.grid(alpha=0.3, linestyle='--')
        ax_arrhythmia.text(0.02, 0.95, f'RMSE: {arrhythmia_rmse:.3f}',
                          transform=ax_arrhythmia.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.suptitle('MITBIH: Discovered Patterns on ECG Beats', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig

def visualize_remc_patterns():
    """Visualize top REMC patterns on histone modification tracks."""
    print("Generating REMC pattern visualization...")
    
    # Load patterns for E003 (fold 1)
    pattern_file = Path('../json_files/remc/pattern_parameters_E003.json')
    if not pattern_file.exists():
        print(f"  Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        all_patterns = json.load(f)
    
    patterns = all_patterns.get('fold_1', [])
    if not patterns:
        print("  No patterns found in fold_1")
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
    
    # Select top 3 patterns
    top_patterns = patterns[:3]
    
    # Create figure with 3 rows (one per pattern)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for i, pattern_info in enumerate(top_patterns):
        ax = axes[i]
        
        # Get pattern details
        pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
        start = pattern_info['start']
        width = pattern_info['width']
        series_idx = pattern_info['series_idx']
        histone_name = histone_names[series_idx]
        
        # Get columns for this histone mark
        histone_cols = [col for col in df.columns if col.startswith(f"{histone_name}_")]
        histone_data = df[histone_cols].values
        
        # Find best discriminative samples
        high_expr_idx, low_expr_idx = find_best_discriminative_samples(
            histone_data, y.values, pattern, start, width, class_0_label=1, class_1_label=0
        )
        
        if high_expr_idx is None or low_expr_idx is None:
            print(f"  Warning: Could not find discriminative samples for pattern {i+1}")
            continue
        
        # Get all samples for averaging
        high_expr_mask = y == 1
        low_expr_mask = y == 0
        
        # Compute average profiles
        high_avg = histone_data[high_expr_mask].mean(axis=0)
        low_avg = histone_data[low_expr_mask].mean(axis=0)
        
        # Plot average profiles
        x_positions = np.arange(len(histone_cols))
        ax.plot(x_positions, high_avg, color='#3498db', linewidth=2, alpha=0.7, label='High Expression (avg)')
        ax.plot(x_positions, low_avg, color='#e67e22', linewidth=2, alpha=0.7, label='Low Expression (avg)')
        
        # Overlay pattern region
        pattern_x = np.arange(start, start + width)
        
        # Scale pattern to data range for visualization
        data_range = high_avg[start:start+width]
        if len(data_range) > 0:
            pattern_scaled = pattern * (data_range.max() - data_range.min()) + data_range.min()
            ax.plot(pattern_x, pattern_scaled, color='#e74c3c', linewidth=3, 
                   linestyle='--', label='Pattern Template', zorder=10)
        
        ax.axvspan(start, start + width - 1, alpha=0.15, color='#e74c3c')
        
        # Mark TSS (center at position 500 for 1000bp window)
        tss_pos = len(histone_cols) // 2
        ax.axvline(tss_pos, color='black', linestyle=':', linewidth=2, alpha=0.5, label='TSS')
        
        ax.set_xlim(0, len(histone_cols))
        ax.set_ylabel(f'{histone_name}\nSignal', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Add pattern info
        pattern_loc = 'Promoter' if start < tss_pos + 100 and start > tss_pos - 100 else 'Gene Body' if start > tss_pos else 'Upstream'
        ax.text(0.02, 0.95, f'Pattern {i+1}: {histone_name}\nStart: {start}, Width: {width}\nLocation: {pattern_loc}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        if i == 2:
            ax.set_xlabel('Position (bp from TSS)', fontsize=11)
    
    plt.suptitle('REMC E003: Discovered Patterns on Histone Modification Tracks', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig

def main():
    print("="*60)
    print("Domain-Specific Pattern Visualization")
    print("="*60)
    
    output_dir = Path('../manuscript/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate MITBIH visualization
    print("\n1. MITBIH ECG Patterns:")
    fig_mitbih = visualize_mitbih_patterns()
    if fig_mitbih:
        output_path = output_dir / 'mitbih_pattern_interpretation.png'
        fig_mitbih.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_mitbih)
        print(f"   Saved to {output_path}")
    else:
        print("   Failed to generate MITBIH visualization")
    
    # Generate REMC visualization
    print("\n2. REMC Epigenetic Patterns:")
    fig_remc = visualize_remc_patterns()
    if fig_remc:
        output_path = output_dir / 'remc_pattern_interpretation.png'
        fig_remc.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_remc)
        print(f"   Saved to {output_path}")
    else:
        print("   Failed to generate REMC visualization")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == "__main__":
    main()

