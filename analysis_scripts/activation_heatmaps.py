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

def compute_sliding_window_rmse(signal, pattern):
    """Compute RMSE at every possible position in signal."""
    signal = np.asarray(signal)
    pattern = np.asarray(pattern)
    pattern_width = len(pattern)
    signal_len = len(signal)
    
    if pattern_width > signal_len:
        return np.array([])
    
    rmse_values = []
    for start in range(signal_len - pattern_width + 1):
        segment = signal[start:start + pattern_width]
        rmse = np.sqrt(((segment - pattern) ** 2).mean())
        rmse_values.append(rmse)
    
    return np.array(rmse_values)

def create_activation_heatmap_mitbih():
    """Create activation heatmap for MITBIH patterns."""
    print("Generating MITBIH activation heatmap...")
    
    # Load patterns
    pattern_file = Path('../json_files/mitbih/pattern_parameters.json')
    if not pattern_file.exists():
        print(f"  Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        all_patterns = json.load(f)
    
    patterns = all_patterns.get('fold_1', [])
    if not patterns:
        print("  No patterns found")
        return None
    
    # Load data
    data_file = Path('../processed_datasets/mitbih_processed.csv')
    if not data_file.exists():
        print(f"  Data file not found: {data_file}")
        return None
    
    data = pd.read_csv(data_file)
    y = data['target']
    X = data.drop('target', axis=1).values
    
    # Binary classification
    y_binary = (y != 0).astype(int)
    
    # Select top 3 patterns
    top_patterns = patterns[:3]
    
    # Select representative samples (2 normal, 2 arrhythmic)
    normal_idx = np.where(y_binary == 0)[0]
    arrhythmic_idx = np.where(y_binary == 1)[0]
    
    selected_samples = [
        normal_idx[10],
        normal_idx[50],
        arrhythmic_idx[10],
        arrhythmic_idx[50]
    ]
    
    # Create activation maps for each pattern
    n_patterns = len(top_patterns)
    n_samples = len(selected_samples)
    
    fig, axes = plt.subplots(n_patterns, n_samples, figsize=(16, 10))
    
    for i, pattern_info in enumerate(top_patterns):
        # Generate pattern
        pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
        
        for j, sample_idx in enumerate(selected_samples):
            ax = axes[i, j]
            signal = X[sample_idx]
            
            # Compute sliding window RMSE
            activation = compute_sliding_window_rmse(signal, pattern)
            
            if len(activation) == 0:
                ax.axis('off')
                continue
            
            # Normalize signal for display
            signal_norm = (signal - signal.mean()) / signal.std()
            
            # Create heatmap overlay
            # Invert RMSE so high activation = low RMSE (good match)
            activation_inverted = 1 / (activation + 0.1)
            activation_norm = (activation_inverted - activation_inverted.min()) / (activation_inverted.max() - activation_inverted.min())
            
            # Plot signal
            ax.plot(signal_norm, color='black', linewidth=1.5, alpha=0.7)
            
            # Create heatmap overlay
            for k in range(len(activation_norm)):
                color_val = plt.cm.Reds(activation_norm[k])
                ax.axvspan(k, k + pattern_info['width'], alpha=0.3 * activation_norm[k], color=color_val)
            
            # Highlight the actual pattern region
            start = pattern_info['start']
            ax.axvspan(start, start + pattern_info['width'], alpha=0.15, color='blue', edgecolor='blue', linewidth=2)
            
            ax.set_xlim(0, len(signal))
            ax.set_ylim(signal_norm.min() - 0.5, signal_norm.max() + 0.5)
            
            # Labels
            if i == 0:
                sample_type = 'Normal' if y_binary[sample_idx] == 0 else 'Arrhythmic'
                ax.set_title(f'{sample_type} Beat {j+1}', fontsize=10, fontweight='bold')
            
            if j == 0:
                ax.set_ylabel(f'Pattern {i+1}', fontsize=10, fontweight='bold')
            else:
                ax.set_yticks([])
            
            if i == n_patterns - 1:
                ax.set_xlabel('Time (samples)', fontsize=9)
            else:
                ax.set_xticks([])
            
            ax.grid(alpha=0.2)
    
    plt.suptitle('MITBIH: Pattern Activation Heatmaps', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig

def create_activation_heatmap_remc():
    """Create activation heatmap for REMC patterns."""
    print("Generating REMC activation heatmap...")
    
    # Load patterns for E003
    pattern_file = Path('../json_files/remc/pattern_parameters_E003.json')
    if not pattern_file.exists():
        print(f"  Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        all_patterns = json.load(f)
    
    patterns = all_patterns.get('fold_1', [])
    if not patterns:
        print("  No patterns found")
        return None
    
    # Load data
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
    
    # Select representative samples (2 high expression, 2 low expression)
    high_expr_idx = np.where(y == 1)[0]
    low_expr_idx = np.where(y == 0)[0]
    
    selected_samples = [
        high_expr_idx[20],
        high_expr_idx[100],
        low_expr_idx[20],
        low_expr_idx[100]
    ]
    
    # Create activation maps
    n_patterns = len(top_patterns)
    n_samples = len(selected_samples)
    
    fig, axes = plt.subplots(n_patterns, n_samples, figsize=(16, 10))
    
    for i, pattern_info in enumerate(top_patterns):
        # Generate pattern
        pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
        series_idx = pattern_info['series_idx']
        histone_name = histone_names[series_idx]
        
        # Get columns for this histone mark
        histone_cols = [col for col in df.columns if col.startswith(f"{histone_name}_")]
        histone_data = df[histone_cols].values
        
        for j, sample_idx in enumerate(selected_samples):
            ax = axes[i, j]
            signal = histone_data[sample_idx]
            
            # Compute sliding window RMSE
            activation = compute_sliding_window_rmse(signal, pattern)
            
            if len(activation) == 0:
                ax.axis('off')
                continue
            
            # Normalize signal
            signal_norm = signal / (signal.max() + 1e-6)
            
            # Invert RMSE for visualization
            activation_inverted = 1 / (activation + 0.1)
            activation_norm = (activation_inverted - activation_inverted.min()) / (activation_inverted.max() - activation_inverted.min())
            
            # Plot signal
            ax.plot(signal_norm, color='black', linewidth=1.5, alpha=0.7)
            
            # Create heatmap overlay
            for k in range(len(activation_norm)):
                color_val = plt.cm.Reds(activation_norm[k])
                ax.axvspan(k, k + pattern_info['width'], alpha=0.3 * activation_norm[k], color=color_val)
            
            # Highlight actual pattern region
            start = pattern_info['start']
            ax.axvspan(start, start + pattern_info['width'], alpha=0.15, color='blue', edgecolor='blue', linewidth=2)
            
            # Mark TSS
            tss_pos = len(histone_cols) // 2
            ax.axvline(tss_pos, color='green', linestyle=':', linewidth=2, alpha=0.6)
            
            ax.set_xlim(0, len(signal))
            ax.set_ylim(-0.1, 1.2)
            
            # Labels
            if i == 0:
                sample_type = 'High Expr' if y.iloc[sample_idx] == 1 else 'Low Expr'
                ax.set_title(f'{sample_type} {j+1}', fontsize=10, fontweight='bold')
            
            if j == 0:
                ax.set_ylabel(f'Pattern {i+1}\n({histone_name})', fontsize=10, fontweight='bold')
            else:
                ax.set_yticks([])
            
            if i == n_patterns - 1:
                ax.set_xlabel('Position (bp)', fontsize=9)
            else:
                ax.set_xticks([])
            
            ax.grid(alpha=0.2)
    
    plt.suptitle('REMC E003: Pattern Activation Heatmaps', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig

def create_combined_activation_figure():
    """Create combined figure with both MITBIH and REMC activation heatmaps."""
    print("Generating combined activation heatmap figure...")
    
    # Load MITBIH data
    mitbih_pattern_file = Path('../json_files/mitbih/pattern_parameters.json')
    mitbih_data_file = Path('../processed_datasets/mitbih_processed.csv')
    
    # Load REMC data
    remc_pattern_file = Path('../json_files/remc/pattern_parameters_E003.json')
    remc_data_file = Path('../processed_datasets/remc/E003.parquet')
    
    # Check files exist
    if not all([mitbih_pattern_file.exists(), mitbih_data_file.exists(), 
                remc_pattern_file.exists(), remc_data_file.exists()]):
        print("  Missing required files")
        return None
    
    # Load MITBIH
    with open(mitbih_pattern_file, 'r') as f:
        mitbih_patterns = json.load(f)['fold_1'][:2]  # Top 2 patterns
    
    mitbih_data = pd.read_csv(mitbih_data_file)
    mitbih_y = (mitbih_data['target'] != 0).astype(int)
    mitbih_X = mitbih_data.drop('target', axis=1).values
    
    # Load REMC
    with open(remc_pattern_file, 'r') as f:
        remc_patterns = json.load(f)['fold_1'][:2]  # Top 2 patterns
    
    remc_df = pd.read_parquet(remc_data_file)
    remc_y = remc_df['target']
    histone_names = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    
    # Create figure: 2 rows (datasets) x 4 columns (samples)
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)
    
    # MITBIH row
    mitbih_samples = [
        np.where(mitbih_y == 0)[0][15],
        np.where(mitbih_y == 0)[0][45],
        np.where(mitbih_y == 1)[0][15],
        np.where(mitbih_y == 1)[0][45]
    ]
    
    pattern = generate_bspline_pattern(mitbih_patterns[0]['control_points'], mitbih_patterns[0]['width'])
    
    for col, sample_idx in enumerate(mitbih_samples):
        ax = fig.add_subplot(gs[0, col])
        signal = mitbih_X[sample_idx]
        
        # Compute activation
        activation = compute_sliding_window_rmse(signal, pattern)
        activation_inverted = 1 / (activation + 0.1)
        activation_norm = (activation_inverted - activation_inverted.min()) / (activation_inverted.max() - activation_inverted.min())
        
        # Normalize signal
        signal_norm = (signal - signal.mean()) / signal.std()
        
        # Plot with heatmap
        ax.plot(signal_norm, color='#2c3e50', linewidth=1.8, alpha=0.8, zorder=10)
        
        for k in range(len(activation_norm)):
            ax.axvspan(k, k + mitbih_patterns[0]['width'], 
                      alpha=0.4 * activation_norm[k], color=plt.cm.Reds(activation_norm[k]), zorder=1)
        
        start = mitbih_patterns[0]['start']
        ax.axvspan(start, start + mitbih_patterns[0]['width'], 
                  alpha=0.2, color='blue', edgecolor='blue', linewidth=2.5, linestyle='--', zorder=5)
        
        ax.set_xlim(0, len(signal))
        ax.set_ylim(signal_norm.min() - 0.3, signal_norm.max() + 0.3)
        
        sample_type = 'Normal' if mitbih_y[sample_idx] == 0 else 'Arrhythmic'
        ax.set_title(f'{sample_type} Beat', fontsize=11, fontweight='bold')
        
        if col == 0:
            ax.set_ylabel('MITBIH\nAmplitude (z-scored)', fontsize=10, fontweight='bold')
        else:
            ax.set_yticks([])
        
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.grid(alpha=0.25, linestyle='--')
    
    # REMC row
    remc_samples = [
        np.where(remc_y == 1)[0][25],
        np.where(remc_y == 1)[0][80],
        np.where(remc_y == 0)[0][25],
        np.where(remc_y == 0)[0][80]
    ]
    
    pattern_info = remc_patterns[0]
    pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
    series_idx = pattern_info['series_idx']
    histone_name = histone_names[series_idx]
    
    histone_cols = [col for col in remc_df.columns if col.startswith(f"{histone_name}_")]
    histone_data = remc_df[histone_cols].values
    
    for col, sample_idx in enumerate(remc_samples):
        ax = fig.add_subplot(gs[1, col])
        signal = histone_data[sample_idx]
        
        # Compute activation
        activation = compute_sliding_window_rmse(signal, pattern)
        activation_inverted = 1 / (activation + 0.1)
        activation_norm = (activation_inverted - activation_inverted.min()) / (activation_inverted.max() - activation_inverted.min())
        
        # Normalize signal
        signal_norm = signal / (signal.max() + 1e-6)
        
        # Plot with heatmap
        ax.plot(signal_norm, color='#2c3e50', linewidth=1.8, alpha=0.8, zorder=10)
        
        for k in range(len(activation_norm)):
            ax.axvspan(k, k + pattern_info['width'], 
                      alpha=0.4 * activation_norm[k], color=plt.cm.Reds(activation_norm[k]), zorder=1)
        
        start = pattern_info['start']
        ax.axvspan(start, start + pattern_info['width'], 
                  alpha=0.2, color='blue', edgecolor='blue', linewidth=2.5, linestyle='--', zorder=5)
        
        # TSS marker
        tss_pos = len(histone_cols) // 2
        ax.axvline(tss_pos, color='green', linestyle=':', linewidth=2.5, alpha=0.7, label='TSS', zorder=8)
        
        ax.set_xlim(0, len(signal))
        ax.set_ylim(-0.05, 1.15)
        
        sample_type = 'High Expr' if remc_y.iloc[sample_idx] == 1 else 'Low Expr'
        ax.set_title(f'{sample_type} Gene', fontsize=11, fontweight='bold')
        
        if col == 0:
            ax.set_ylabel(f'REMC ({histone_name})\nNorm. Signal', fontsize=10, fontweight='bold')
        else:
            ax.set_yticks([])
        
        ax.set_xlabel('Position (bp)', fontsize=9)
        ax.grid(alpha=0.25, linestyle='--')
    
    fig.suptitle('Pattern Activation Heatmaps: MITBIH and REMC', fontsize=15, fontweight='bold', y=0.98)
    
    # Add legend for heatmap interpretation
    fig.text(0.5, 0.01, 'Red intensity: pattern activation strength (high = good match) | Blue box: pattern discovery region', 
             ha='center', fontsize=10, style='italic')
    
    return fig

def main():
    print("="*60)
    print("Activation Heatmap Visualization")
    print("="*60)
    
    output_dir = Path('../manuscript/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate combined figure
    print("\nGenerating combined activation heatmap figure...")
    fig_combined = create_combined_activation_figure()
    if fig_combined:
        output_path = output_dir / 'activation_heatmaps.png'
        fig_combined.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_combined)
        print(f"   Saved to {output_path}")
    else:
        print("   Failed to generate combined figure")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == "__main__":
    main()

