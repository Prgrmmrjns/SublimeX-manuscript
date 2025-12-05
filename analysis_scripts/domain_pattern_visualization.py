import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import BSpline
import matplotlib.gridspec as gridspec

# --- Core Logic ---

def generate_bspline_pattern(control_points, width):
    degree = 3
    n_cp = len(control_points)
    knots = np.concatenate([np.zeros(degree + 1), np.linspace(0, 1, n_cp - degree + 1)[1:-1], np.ones(degree + 1)])
    t = np.linspace(0, 1, int(width))
    return BSpline(knots, np.asarray(control_points), degree)(t)

def apply_transformation(series, transform_type):
    if transform_type == 'raw':
        return series
    elif transform_type == 'derivative':
        return np.gradient(series)
    elif transform_type == 'cumsum':
        return np.cumsum(series - np.mean(series))
    elif transform_type == 'abs':
        return np.abs(series)
    elif transform_type == 'sorted':
        return np.sort(series)
    return series

def compute_rmse(signal, pattern, start, width):
    w = int(round(width))
    if start + w > len(signal) or len(pattern) != w:
        return np.inf
    return np.sqrt(((signal[start:start + w] - pattern) ** 2).mean())

def find_best_pattern(patterns, remc_y, histone_names, remc_df):
    # Prefer interpretable transforms for visualization
    prefer = ['raw', 'cumsum', 'derivative', 'abs']
    best_score, best_idx = -np.inf, 0
    
    for i, p in enumerate(patterns):
        if p.get('transform_type', 'raw') not in prefer:
            continue
            
        series_idx = p['series_idx']
        histone_name = histone_names[series_idx]
        histone_cols = [col for col in remc_df.columns if col.startswith(histone_name + "_")]
        histone_data = remc_df[histone_cols].values
        
        transform_type = p.get('transform_type', 'raw')
        X_t = np.array([apply_transformation(row, transform_type) for row in histone_data])
        
        start = int(p['start'])
        width = int(round(p['width']))
        pattern = generate_bspline_pattern(p['control_points'], width)
        
        rmse_all = np.array([compute_rmse(x, pattern, start, width) for x in X_t])
        valid = np.isfinite(rmse_all)
        
        if valid.sum() < 10:
            continue
            
        sep = abs(rmse_all[(remc_y==1) & valid].mean() - rmse_all[(remc_y==0) & valid].mean())
        if sep > best_score:
            best_score, best_idx = sep, i
            
    return best_idx

# --- Visualization ---

def visualize_pattern_story(cell_line='E004'):
    """
    Creates a 3-panel storytelling visualization:
    1. Biological Context (Raw Signal)
    2. The Pattern Match Mechanism (Transformed Space)
    3. Discriminative Power (Resulting Feature Separation)
    """
    print(f"Generating visualization for {cell_line}...")
    
    # Load Data
    with open(Path(f'../json_files/remc/pattern_parameters_{cell_line}.json'), 'r') as f:
        data = json.load(f)
    patterns = data[list(data.keys())[0]]
    
    df = pd.read_parquet(Path(f'../processed_datasets/remc/{cell_line}.parquet'))
    y = df['target'].values
    histone_names = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    
    # Find discriminative pattern
    idx = find_best_pattern(patterns, y, histone_names, df)
    p = patterns[idx]
    
    # Extract info
    h_name = histone_names[p['series_idx']]
    transform = p.get('transform_type', 'raw')
    start = int(p['start'])
    width = int(round(p['width']))
    
    # Prepare signals
    cols = [c for c in df.columns if c.startswith(h_name + "_")]
    X_raw = df[cols].values
    X_trans = np.array([apply_transformation(x, transform) for x in X_raw])
    
    # Generate pattern curve
    pattern_curve = generate_bspline_pattern(p['control_points'], width)
    
    # Calc RMSEs to pick representatives
    rmses = np.array([compute_rmse(x, pattern_curve, start, width) for x in X_trans])
    
    # Select representative samples (close to median of each class)
    def get_rep_idx(target_class):
        idxs = np.where(y == target_class)[0]
        cls_rmses = rmses[idxs]
        median_val = np.median(cls_rmses)
        # Find index closest to median
        return idxs[np.argmin(np.abs(cls_rmses - median_val))]

    idx_high = get_rep_idx(1)
    idx_low = get_rep_idx(0)
    
    # Setup Figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1.2, 0.8], hspace=0.4)
    
    # Colors
    c_high = '#2980b9' # Strong Blue
    c_low = '#c0392b'  # Strong Red
    c_pat = '#27ae60'  # Green
    c_fill_high = '#d4e6f1'
    c_fill_low = '#fadbd8'
    
    # X-axis (genomic position)
    tss_offset = X_raw.shape[1] // 2
    x_genome = np.arange(X_raw.shape[1]) - tss_offset
    pat_x_genome = np.arange(start, start+width) - tss_offset
    
    # --- Panel 1: Biological Context ---
    ax1 = fig.add_subplot(gs[0])
    
    # Plot Mean +/- Std for context
    mean_high = X_raw[y==1].mean(axis=0)
    mean_low = X_raw[y==0].mean(axis=0)
    
    ax1.plot(x_genome, mean_high, color=c_high, label='High Expression (Avg)', linewidth=2)
    ax1.plot(x_genome, mean_low, color=c_low, label='Low Expression (Avg)', linewidth=2)
    ax1.fill_between(x_genome, mean_high, 0, color=c_high, alpha=0.1)
    ax1.fill_between(x_genome, mean_low, 0, color=c_low, alpha=0.1)
    
    # Highlight pattern region
    ax1.axvspan(pat_x_genome[0], pat_x_genome[-1], color='gray', alpha=0.15, label='Pattern Region')
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5, label='TSS')
    
    # Annotation
    ax1.set_title(f'A. Biological Context: {h_name} Profile', loc='left', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ChIP-seq Signal', fontsize=12)
    ax1.set_xlabel('Distance from TSS (bp)', fontsize=10)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.2, linestyle=':')
    
    # --- Panel 2: Mechanism (Transform Space) ---
    ax2 = fig.add_subplot(gs[1])
    
    # Get signal slices for representatives
    sig_high = X_trans[idx_high]
    sig_low = X_trans[idx_low]
    
    # Plot full transformed signals
    ax2.plot(x_genome, sig_high, color=c_high, alpha=0.4, linestyle='-', linewidth=1)
    ax2.plot(x_genome, sig_low, color=c_low, alpha=0.4, linestyle='-', linewidth=1)
    
    # Plot zoomed segments where pattern matches
    seg_high = sig_high[start:start+width]
    seg_low = sig_low[start:start+width]
    
    # Scale pattern to match the "High" sample (assuming it's the target) for visualization
    # We align the pattern to the sample that matches it best (lowest RMSE)
    target_seg = seg_high if rmses[idx_high] < rmses[idx_low] else seg_low
    
    # Simple Min-Max scaling of pattern to target segment range
    p_min, p_max = pattern_curve.min(), pattern_curve.max()
    t_min, t_max = target_seg.min(), target_seg.max()
    
    if p_max - p_min > 1e-5:
        pat_scaled = (pattern_curve - p_min) / (p_max - p_min) * (t_max - t_min) + t_min
    else:
        pat_scaled = pattern_curve + t_min
        
    # Plot the Pattern itself
    ax2.plot(pat_x_genome, pat_scaled, color=c_pat, linewidth=5, label='Learned Pattern (B-spline)')
    
    # Plot the representative segments thicker
    ax2.plot(pat_x_genome, seg_high, color=c_high, linewidth=2.5, label='High Expr Sample')
    ax2.plot(pat_x_genome, seg_low, color=c_low, linewidth=2.5, label='Low Expr Sample')
    
    # Visualize "Error" (RMSE area)
    ax2.fill_between(pat_x_genome, pat_scaled, seg_high, color=c_high, alpha=0.2, hatch='///')
    ax2.fill_between(pat_x_genome, pat_scaled, seg_low, color=c_low, alpha=0.2, hatch='\\\\')
    
    # Annotations
    ax2.set_title(f'B. Pattern Matching Mechanism: {transform.capitalize()} Space', loc='left', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'{transform.capitalize()} Signal Value', fontsize=12)
    ax2.set_xlabel('Distance from TSS (bp)', fontsize=10)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, alpha=0.2, linestyle=':')
    
    # Add text box explaining the match
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    txt = (f"Pattern detects shape similarity.\n"
           f"High Expr Error (RMSE): {rmses[idx_high]:.1f}\n"
           f"Low Expr Error (RMSE): {rmses[idx_low]:.1f}")
    ax2.text(0.02, 0.95, txt, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=bbox_props)

    # --- Panel 3: Separation (Distribution) ---
    ax3 = fig.add_subplot(gs[2])
    
    # Density plots
    sns.kdeplot(rmses[y==1], color=c_high, fill=True, alpha=0.3, label='High Expression', ax=ax3, linewidth=2)
    sns.kdeplot(rmses[y==0], color=c_low, fill=True, alpha=0.3, label='Low Expression', ax=ax3, linewidth=2)
    
    ax3.set_title('C. Feature Discrimination: RMSE Distribution', loc='left', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Pattern Dissimilarity (RMSE Feature)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.2, axis='x')
    return fig

if __name__ == "__main__":
    output_dir = Path('../manuscript/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig = visualize_pattern_story('E004')
    if fig:
        fig.savefig(output_dir / 'domain_pattern_interpretation.png', dpi=300, bbox_inches='tight')
        print("Saved domain_pattern_interpretation.png")
