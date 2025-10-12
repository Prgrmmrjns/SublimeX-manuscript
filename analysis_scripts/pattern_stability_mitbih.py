import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import BSpline
import seaborn as sns

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

def load_mitbih_patterns():
    """Load MITBIH patterns from fold 1."""
    pattern_file = Path('../json_files/mitbih/pattern_parameters.json')
    if not pattern_file.exists():
        print(f"Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        data = json.load(f)
    
    return data.get('fold_1', [])

def create_intuitive_stability_plot():
    """Create intuitive pattern stability visualization for MITBIH."""
    print("Generating MITBIH pattern stability visualization...")
    
    patterns = load_mitbih_patterns()
    if not patterns or len(patterns) < 3:
        print("Insufficient patterns found")
        return None
    
    # Take top 3 patterns
    top_patterns = patterns[:3]
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, pattern_info in enumerate(top_patterns):
        # Panel 1: Pattern shape
        ax_shape = fig.add_subplot(gs[i, 0])
        
        # Generate pattern
        pattern = generate_bspline_pattern(pattern_info['control_points'], pattern_info['width'])
        x = np.arange(len(pattern))
        
        ax_shape.plot(x, pattern, color=colors[i], linewidth=3, label=f'Pattern {i+1}')
        ax_shape.fill_between(x, pattern, alpha=0.3, color=colors[i])
        
        # Mark control points
        cp_x = np.linspace(0, len(pattern)-1, len(pattern_info['control_points']))
        cp_y = pattern_info['control_points']
        # Need to map control points to actual values
        pattern_at_cp = generate_bspline_pattern(pattern_info['control_points'], len(pattern_info['control_points']))
        ax_shape.scatter(cp_x, cp_y, color='black', s=80, zorder=10, edgecolor='white', linewidth=2)
        
        ax_shape.set_xlabel('Position within Pattern', fontsize=11, fontweight='bold')
        ax_shape.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax_shape.set_title(f'Pattern {i+1} Shape', fontsize=12, fontweight='bold')
        ax_shape.grid(alpha=0.3, linestyle='--')
        ax_shape.legend(loc='upper right', fontsize=10)
        
        # Panel 2: Position on ECG
        ax_position = fig.add_subplot(gs[i, 1])
        
        # Create a mock ECG-like signal to show position
        ecg_length = 100
        mock_ecg = np.zeros(ecg_length)
        # Simple QRS-like shape
        mock_ecg[20:30] = np.sin(np.linspace(0, np.pi, 10)) * 0.3
        mock_ecg[30:45] = -np.sin(np.linspace(0, 2*np.pi, 15)) * 1.5
        mock_ecg[45:55] = np.sin(np.linspace(0, np.pi, 10)) * 0.5
        mock_ecg[60:75] = np.sin(np.linspace(0, 2*np.pi, 15)) * 0.4
        
        ax_position.plot(mock_ecg, color='gray', linewidth=2, alpha=0.6, label='ECG Beat')
        
        # Highlight pattern region
        start = pattern_info['start']
        width = pattern_info['width']
        ax_position.axvspan(start, start + width, alpha=0.4, color=colors[i], 
                           label=f'Pattern Region')
        ax_position.axvline(start, color=colors[i], linestyle='--', linewidth=2, alpha=0.8)
        ax_position.axvline(start + width, color=colors[i], linestyle='--', linewidth=2, alpha=0.8)
        
        ax_position.set_xlabel('ECG Sample Index', fontsize=11, fontweight='bold')
        ax_position.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax_position.set_title(f'Pattern {i+1} Location', fontsize=12, fontweight='bold')
        ax_position.set_xlim(0, ecg_length)
        ax_position.grid(alpha=0.3, linestyle='--')
        ax_position.legend(loc='upper right', fontsize=9)
        
        # Add text annotation
        ax_position.text(start + width/2, ax_position.get_ylim()[1] * 0.85, 
                        f'Start: {start}\nWidth: {width}',
                        ha='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors[i], linewidth=2))
        
        # Panel 3: Pattern characteristics summary
        ax_summary = fig.add_subplot(gs[i, 2])
        ax_summary.axis('off')
        
        # Create summary statistics
        summary_text = f"""
Pattern {i+1} Characteristics:

Control Points: {len(pattern_info['control_points'])}
Start Position: {pattern_info['start']}
Width: {pattern_info['width']}
End Position: {pattern_info['start'] + pattern_info['width']}

Control Point Values:
"""
        for j, cp in enumerate(pattern_info['control_points']):
            summary_text += f"  CP{j+1}: {cp:.3f}\n"
        
        # Pattern stats
        summary_text += f"\nPattern Statistics:\n"
        summary_text += f"  Min: {pattern.min():.3f}\n"
        summary_text += f"  Max: {pattern.max():.3f}\n"
        summary_text += f"  Mean: {pattern.mean():.3f}\n"
        summary_text += f"  Std: {pattern.std():.3f}\n"
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.2, 
                                edgecolor=colors[i], linewidth=2))
    
    plt.suptitle('MITBIH: Discovered Pattern Characteristics and Localization', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def create_pattern_overlay_plot():
    """Create a simpler overlay plot showing all patterns on a representative ECG."""
    print("Generating MITBIH pattern overlay visualization...")
    
    patterns = load_mitbih_patterns()
    if not patterns:
        return None
    
    # Load actual MITBIH data
    data_file = Path('../processed_datasets/mitbih_processed.csv')
    if data_file.exists():
        data = pd.read_csv(data_file)
        y = data['target']
        X = data.drop('target', axis=1).values
        
        # Get a normal beat
        normal_idx = np.where(y == 0)[0][10]
        ecg_signal = X[normal_idx]
    else:
        # Fallback to mock ECG
        ecg_length = 100
        ecg_signal = np.zeros(ecg_length)
        ecg_signal[20:30] = np.sin(np.linspace(0, np.pi, 10)) * 0.3
        ecg_signal[30:45] = -np.sin(np.linspace(0, 2*np.pi, 15)) * 1.5
        ecg_signal[45:55] = np.sin(np.linspace(0, np.pi, 10)) * 0.5
        ecg_signal[60:75] = np.sin(np.linspace(0, 2*np.pi, 15)) * 0.4
    
    # Normalize for display
    ecg_norm = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot ECG
    ax.plot(ecg_norm, color='black', linewidth=2.5, label='Normal ECG Beat', zorder=1)
    
    # Overlay patterns
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    top_patterns = patterns[:min(5, len(patterns))]
    
    for i, pattern_info in enumerate(top_patterns):
        start = pattern_info['start']
        width = pattern_info['width']
        
        # Highlight region
        ax.axvspan(start, start + width, alpha=0.2, color=colors[i], zorder=2)
        
        # Add boundary lines
        ax.axvline(start, color=colors[i], linestyle='--', linewidth=2, alpha=0.7, zorder=3)
        ax.axvline(start + width, color=colors[i], linestyle='--', linewidth=2, alpha=0.7, zorder=3)
        
        # Add label
        mid_point = start + width / 2
        y_pos = 3.5 - i * 0.5
        ax.text(mid_point, y_pos, f'P{i+1}', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white',
               bbox=dict(boxstyle='circle', facecolor=colors[i], edgecolor='white', linewidth=2))
    
    ax.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Amplitude', fontsize=13, fontweight='bold')
    ax.set_title('MITBIH: All Discovered Patterns Overlaid on Normal ECG Beat', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, len(ecg_signal))
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add annotation
    legend_text = '\n'.join([f'P{i+1}: Start={p["start"]}, Width={p["width"]}' 
                            for i, p in enumerate(top_patterns)])
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1))
    
    plt.tight_layout()
    return fig

def main():
    print("="*70)
    print("MITBIH Pattern Stability and Characteristics Visualization")
    print("="*70)
    
    output_dir = Path('../manuscript/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate main stability plot
    print("\n1. Pattern characteristics plot:")
    fig1 = create_intuitive_stability_plot()
    if fig1:
        output_path = output_dir / 'pattern_stability.png'
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"   Saved to {output_path}")
    
    # Generate overlay plot
    print("\n2. Pattern overlay plot:")
    fig2 = create_pattern_overlay_plot()
    if fig2:
        output_path = output_dir / 'mitbih_pattern_overlay.png'
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"   Saved to {output_path}")
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)

if __name__ == "__main__":
    main()

