import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def create_cross_fold_pattern_visualization():
    """Create visualization showing first pattern across all 5 folds."""
    print("Generating cross-fold pattern consistency visualization...")
    
    # Load patterns from all folds
    pattern_file = Path('../json_files/mitbih/pattern_parameters.json')
    if not pattern_file.exists():
        print(f"Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        all_data = json.load(f)
    
    # Load MITBIH data
    data_file = Path('../processed_datasets/mitbih_processed.csv')
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return None
    
    data = pd.read_csv(data_file)
    y = data['target']
    X = data.drop('target', axis=1).values
    
    # Get a representative normal ECG beat for display
    normal_idx = np.where(y == 0)[0][15]
    ecg_signal = X[normal_idx]
    ecg_norm = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
    
    # Create figure with 5 subplots (one per fold)
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for fold_idx in range(1, 6):
        fold_key = f'fold_{fold_idx}'
        ax = axes[fold_idx - 1]
        
        if fold_key not in all_data or len(all_data[fold_key]) == 0:
            print(f"  No patterns found for {fold_key}")
            ax.axis('off')
            continue
        
        # Get first pattern from this fold
        pattern_info = all_data[fold_key][0]
        
        # Generate pattern
        pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
        start = pattern_info['start']
        width = pattern_info['width']
        
        # Plot ECG signal
        ax.plot(ecg_norm, color='#2c3e50', linewidth=2, alpha=0.7, label='Normal ECG Beat', zorder=1)
        
        # Normalize pattern to fit on same scale as ECG
        pattern_norm = (pattern - pattern.mean()) / pattern.std()
        
        # Scale pattern to match ECG amplitude range in that region
        ecg_region = ecg_norm[start:start+width]
        if len(ecg_region) > 0:
            scale_factor = (ecg_region.max() - ecg_region.min()) / (pattern_norm.max() - pattern_norm.min() + 1e-6)
            pattern_scaled = pattern_norm * scale_factor
            pattern_scaled = pattern_scaled - pattern_scaled.mean() + ecg_region.mean()
        else:
            pattern_scaled = pattern_norm
        
        # Overlay pattern
        pattern_x = np.arange(start, start + width)
        ax.plot(pattern_x, pattern_scaled, color=colors[fold_idx-1], linewidth=3.5, 
               label=f'Pattern (Fold {fold_idx})', zorder=10, alpha=0.9)
        
        # Highlight pattern region
        ax.axvspan(start, start + width, alpha=0.15, color=colors[fold_idx-1], zorder=0)
        
        # Add vertical lines at boundaries
        ax.axvline(start, color=colors[fold_idx-1], linestyle='--', linewidth=2, alpha=0.7, zorder=5)
        ax.axvline(start + width, color=colors[fold_idx-1], linestyle='--', linewidth=2, alpha=0.7, zorder=5)
        
        # Set limits and labels
        ax.set_xlim(0, len(ecg_signal))
        ax.set_ylim(ecg_norm.min() - 0.5, ecg_norm.max() + 0.5)
        
        # Add fold label
        ax.text(0.02, 0.95, f'Fold {fold_idx}', transform=ax.transAxes, 
               fontsize=13, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=colors[fold_idx-1], 
                        alpha=0.8, edgecolor='white', linewidth=2))
        
        # Add position info
        ax.text(0.98, 0.95, f'Start: {start}  |  Width: {width}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, 
                        edgecolor=colors[fold_idx-1], linewidth=1.5))
        
        # Y-axis label
        if fold_idx == 3:
            ax.set_ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
        
        # X-axis
        if fold_idx == 5:
            ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        else:
            ax.set_xticklabels([])
        
        # Legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Grid
        ax.grid(alpha=0.25, linestyle='--')
    
    plt.suptitle('MITBIH: First Pattern Consistency Across 5 Cross-Validation Folds', 
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig

def create_pattern_comparison_plot():
    """Create overlay showing all fold patterns together."""
    print("Generating pattern overlay comparison...")
    
    # Load patterns
    pattern_file = Path('../json_files/mitbih/pattern_parameters.json')
    if not pattern_file.exists():
        return None
    
    with open(pattern_file, 'r') as f:
        all_data = json.load(f)
    
    # Load data
    data_file = Path('../processed_datasets/mitbih_processed.csv')
    if not data_file.exists():
        return None
    
    data = pd.read_csv(data_file)
    y = data['target']
    X = data.drop('target', axis=1).values
    
    normal_idx = np.where(y == 0)[0][15]
    ecg_signal = X[normal_idx]
    ecg_norm = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Top panel: All patterns on ECG
    ax1.plot(ecg_norm, color='#2c3e50', linewidth=2.5, alpha=0.8, 
            label='Normal ECG Beat', zorder=1)
    
    for fold_idx in range(1, 6):
        fold_key = f'fold_{fold_idx}'
        if fold_key not in all_data or len(all_data[fold_key]) == 0:
            continue
        
        pattern_info = all_data[fold_key][0]
        pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
        start = pattern_info['start']
        width = pattern_info['width']
        
        pattern_norm = (pattern - pattern.mean()) / pattern.std()
        ecg_region = ecg_norm[start:start+width]
        
        if len(ecg_region) > 0:
            scale_factor = (ecg_region.max() - ecg_region.min()) / (pattern_norm.max() - pattern_norm.min() + 1e-6)
            pattern_scaled = pattern_norm * scale_factor
            pattern_scaled = pattern_scaled - pattern_scaled.mean() + ecg_region.mean()
        else:
            pattern_scaled = pattern_norm
        
        pattern_x = np.arange(start, start + width)
        ax1.plot(pattern_x, pattern_scaled, color=colors[fold_idx-1], linewidth=2, 
                alpha=0.7, label=f'Fold {fold_idx}', zorder=5)
    
    ax1.set_xlim(0, len(ecg_signal))
    ax1.set_ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('Pattern Overlay: All 5 Folds', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, ncol=3, framealpha=0.95)
    ax1.grid(alpha=0.25, linestyle='--')
    
    # Bottom panel: Isolated patterns comparison
    for fold_idx in range(1, 6):
        fold_key = f'fold_{fold_idx}'
        if fold_key not in all_data or len(all_data[fold_key]) == 0:
            continue
        
        pattern_info = all_data[fold_key][0]
        pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
        
        # Normalize each pattern individually for comparison
        pattern_norm = (pattern - pattern.mean()) / pattern.std()
        x = np.arange(len(pattern))
        
        ax2.plot(x, pattern_norm, color=colors[fold_idx-1], linewidth=2.5, 
                alpha=0.8, label=f'Fold {fold_idx}', marker='o', markersize=4)
    
    ax2.set_xlabel('Position within Pattern', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('Pattern Shape Comparison (Normalized)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.grid(alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    
    return fig

def main():
    print("="*70)
    print("MITBIH: Cross-Fold Pattern Consistency Analysis")
    print("="*70)
    
    output_dir = Path('../manuscript/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate main cross-fold visualization
    print("\n1. Cross-fold pattern visualization:")
    fig1 = create_cross_fold_pattern_visualization()
    if fig1:
        output_path = output_dir / 'pattern_stability.png'
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"   Saved to {output_path}")
    else:
        print("   Failed to generate visualization")
    
    # Generate comparison plot
    print("\n2. Pattern overlay comparison:")
    fig2 = create_pattern_comparison_plot()
    if fig2:
        output_path = output_dir / 'pattern_comparison.png'
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"   Saved to {output_path}")
    else:
        print("   Failed to generate comparison")
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)

if __name__ == "__main__":
    main()

