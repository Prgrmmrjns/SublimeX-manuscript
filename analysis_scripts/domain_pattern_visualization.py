import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import BSpline
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.gridspec as gridspec

# --- Core Logic ---

def generate_bspline_pattern(control_points, width):
    """Generate B-spline pattern from control points (matching core.py)."""
    cps = np.asarray(control_points, dtype=np.float32)
    n_cp = len(cps)
    if n_cp == 1:
        return np.full(int(round(width)), float(cps[0]), dtype=np.float32)
    degree = min(3, n_cp - 1)
    knots = np.concatenate([
        np.zeros(degree + 1),
        np.linspace(0, 1, n_cp - degree + 1)[1:-1],
        np.ones(degree + 1),
    ])
    t = np.linspace(0, 1, width)
    return BSpline(knots, cps, degree)(t).astype(np.float32)

def _fft_power(data):
    """Return FFT power spectrum (matching core.py)."""
    return np.abs(np.fft.rfft(data, axis=-1)) ** 2

def _zscore(data):
    """Z-score normalization (matching core.py)."""
    mean = np.mean(data, axis=-1, keepdims=True) if data.ndim > 1 else np.mean(data)
    std = np.std(data, axis=-1, keepdims=True) if data.ndim > 1 else np.std(data)
    std = np.where(std == 0, 1.0, std)
    return (data - mean) / std

def apply_transformation(series, transform_name):
    """Apply transformation matching core.py transforms."""
    if transform_name == 'raw':
        return series
    elif transform_name == 'fft_power':
        return _fft_power(series)
    elif transform_name == 'zscore':
        return _zscore(series)
    else:
        return series

def compute_rmse_sliding(signal, pattern, start_frac, end_frac, width):
    """Compute minimum RMSE using sliding window (matching core.py pattern mapping)."""
    w = int(round(width))
    sig_len = len(signal)
    max_start = sig_len - w
    start = int(min(start_frac, end_frac) * max_start)
    end = int(max(start_frac, end_frac) * max_start)
    
    if end + w > sig_len or len(pattern) != w:
        return np.inf
    
    # Use sliding window to find best match
    signal_subset = signal[start:end + w]
    if len(signal_subset) < w:
        return np.inf
    
    windows = sliding_window_view(signal_subset, w)
    if len(windows) == 0:
        return np.inf
    
    # MSE distance
    diff = windows - pattern
    distances = np.einsum('...i,...i->...', diff, diff) / diff.shape[-1]
    return np.sqrt(np.min(distances))

def find_best_pattern_remc(patterns, y, histone_names, df):
    """Find the most discriminative pattern for REMC visualization."""
    # Prefer interpretable transforms for visualization
    prefer = ['raw', 'zscore']  # fft_power is less interpretable
    best_score, best_idx = -np.inf, 0
    
    for i, p in enumerate(patterns):
        transform_name = p.get('transform', 'raw')
        if transform_name not in prefer:
            continue
            
        channel = p['channel']
        histone_name = histone_names[channel]
        histone_cols = [col for col in df.columns if col.startswith(histone_name + "_")]
        histone_data = df[histone_cols].values
        
        # Apply transformation
        X_t = np.array([apply_transformation(row, transform_name) for row in histone_data])
        
        width = int(round(p['width']))
        pattern = generate_bspline_pattern(p['control_points'], width)
        
        # Compute RMSEs using sliding window
        rmse_all = np.array([
            compute_rmse_sliding(x, pattern, p['start_frac'], p['end_frac'], width) 
            for x in X_t
        ])
        valid = np.isfinite(rmse_all)
        
        if valid.sum() < 10:
            continue
            
        sep = abs(rmse_all[(y==1) & valid].mean() - rmse_all[(y==0) & valid].mean())
        if sep > best_score:
            best_score, best_idx = sep, i
            
    return best_idx

def find_best_pattern_mitbih(patterns, y, df):
    """Find the most discriminative pattern for MITBIH visualization."""
    # Prefer interpretable transforms for visualization
    prefer = ['raw', 'zscore']
    best_score, best_idx = -np.inf, 0
    
    # Get time series columns (exclude target)
    time_cols = [c for c in df.columns if c != 'target' and str(c).isdigit()]
    time_cols = sorted(time_cols, key=lambda x: int(x))
    X_data = df[time_cols].values
    
    for i, p in enumerate(patterns):
        transform_name = p.get('transform', 'raw')
        if transform_name not in prefer:
            continue
            
        # MITBIH is single channel, so channel should be 0
        if p.get('channel', 0) != 0:
            continue
        
        # Apply transformation to each sample
        X_t = np.array([apply_transformation(row, transform_name) for row in X_data])
        
        width = int(round(p['width']))
        pattern = generate_bspline_pattern(p['control_points'], width)
        
        # Compute RMSEs using sliding window
        rmse_all = np.array([
            compute_rmse_sliding(x, pattern, p['start_frac'], p['end_frac'], width) 
            for x in X_t
        ])
        valid = np.isfinite(rmse_all)
        
        if valid.sum() < 10:
            continue
        
        # For MITBIH, we need to check class separation (assuming binary classification)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            continue
        
        class_0, class_1 = unique_classes[0], unique_classes[1]
        sep = abs(rmse_all[(y==class_1) & valid].mean() - rmse_all[(y==class_0) & valid].mean())
        if sep > best_score:
            best_score, best_idx = sep, i
            
    return best_idx

# --- Visualization ---

def create_remc_panels(fig, gs_remc, cell_line):
    """Create REMC visualization panels."""
    """Create REMC visualization panels in the left column."""
    # Load Data
    with open(Path(f'../json_files/remc/pattern_parameters_{cell_line}.json'), 'r') as f:
        data = json.load(f)
    patterns = data[list(data.keys())[0]]
    
    df = pd.read_parquet(Path(f'../processed_datasets/remc/{cell_line}.parquet'))
    y = df['target'].values
    histone_names = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    
    # Find discriminative pattern
    best_idx = find_best_pattern_remc(patterns, y, histone_names, df)
    p = patterns[best_idx]
    
    # Extract info
    h_name = histone_names[p['channel']]
    transform = p.get('transform', 'raw')
    width = int(round(p['width']))
    start_frac = p['start_frac']
    end_frac = p['end_frac']
    
    # Prepare signals
    cols = [c for c in df.columns if c.startswith(h_name + "_")]
    X_raw = df[cols].values
    X_trans = np.array([apply_transformation(x, transform) for x in X_raw])
    
    # Generate pattern curve
    pattern_curve = generate_bspline_pattern(p['control_points'], width)
    
    # Calc RMSEs to pick representatives
    rmses = np.array([
        compute_rmse_sliding(x, pattern_curve, start_frac, end_frac, width) 
        for x in X_trans
    ])
    valid = np.isfinite(rmses)
    
    # Select representative samples
    def get_rep_idx(target_class):
        idxs = np.where((y == target_class) & valid)[0]
        if len(idxs) == 0:
            idxs = np.where(y == target_class)[0]
        cls_rmses = rmses[idxs]
        median_val = np.median(cls_rmses)
        return idxs[np.argmin(np.abs(cls_rmses - median_val))]

    idx_high = get_rep_idx(1)
    idx_low = get_rep_idx(0)
    
    # Colors
    c_high = '#2980b9'
    c_low = '#c0392b'
    c_pat = '#27ae60'
    
    # X-axis (genomic position)
    tss_offset = X_raw.shape[1] // 2
    x_genome = np.arange(X_raw.shape[1]) - tss_offset
    
    # Find best match position for visualization
    rep_sig = X_trans[valid][0] if valid.any() else X_trans[0]
    sig_len = len(rep_sig)
    max_start = sig_len - width
    search_start = int(min(start_frac, end_frac) * max_start)
    search_end = int(max(start_frac, end_frac) * max_start)
    
    signal_subset = rep_sig[search_start:search_end + width]
    if len(signal_subset) >= width:
        windows = sliding_window_view(signal_subset, width)
        diff = windows - pattern_curve
        distances = np.einsum('...i,...i->...', diff, diff) / diff.shape[-1]
        actual_start = search_start + np.argmin(distances)
    else:
        actual_start = search_start
    
    # --- Panel 1: Biological Context ---
    ax1 = fig.add_subplot(gs_remc[0])
    
    # Plot Mean +/- Std for context
    mean_high = X_raw[y==1].mean(axis=0)
    mean_low = X_raw[y==0].mean(axis=0)
    
    ax1.plot(x_genome, mean_high, color=c_high, label='High Expression (Avg)', linewidth=2)
    ax1.plot(x_genome, mean_low, color=c_low, label='Low Expression (Avg)', linewidth=2)
    ax1.fill_between(x_genome, mean_high, 0, color=c_high, alpha=0.1)
    ax1.fill_between(x_genome, mean_low, 0, color=c_low, alpha=0.1)
    
    # Highlight pattern region (use the actual match position from representative)
    pat_region_start = actual_start - tss_offset
    pat_region_end = (actual_start + width) - tss_offset
    ax1.axvspan(pat_region_start, pat_region_end, color='gray', alpha=0.15, label='Pattern Region')
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5, label='TSS')
    
    # Annotation
    ax1.set_title(f'A. Biological Context: {h_name} Profile', loc='left', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ChIP-seq Signal', fontsize=10)
    ax1.set_xlabel('Distance from TSS (bp)', fontsize=9)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.2, linestyle=':')
    
    # --- Panel 2: Mechanism (Transform Space) ---
    ax2 = fig.add_subplot(gs_remc[1])
    
    # Get signal slices for representatives
    sig_high = X_trans[idx_high]
    sig_low = X_trans[idx_low]
    
    # Find best match positions for each representative sample
    def find_best_match_pos(signal, pattern, start_frac, end_frac, width):
        sig_len = len(signal)
        max_start = sig_len - width
        search_start = int(min(start_frac, end_frac) * max_start)
        search_end = int(max(start_frac, end_frac) * max_start)
        signal_subset = signal[search_start:search_end + width]
        if len(signal_subset) < width:
            return search_start
        windows = sliding_window_view(signal_subset, width)
        diff = windows - pattern
        distances = np.einsum('...i,...i->...', diff, diff) / diff.shape[-1]
        best_pos = np.argmin(distances)
        return search_start + best_pos
    
    pos_high = find_best_match_pos(sig_high, pattern_curve, start_frac, end_frac, width)
    pos_low = find_best_match_pos(sig_low, pattern_curve, start_frac, end_frac, width)
    
    # Highlight pattern regions with background (using search range for context)
    sig_len_panel2 = len(sig_high)  # Use signal length from transformed data
    search_start = int(min(start_frac, end_frac) * (sig_len_panel2 - width))
    search_end = int(max(start_frac, end_frac) * (sig_len_panel2 - width))
    pat_region_start = search_start - tss_offset
    pat_region_end = (search_end + width) - tss_offset
    ax2.axvspan(pat_region_start, pat_region_end, color='gray', alpha=0.15, zorder=0)
    
    # Plot full transformed signals (on top of background)
    ax2.plot(x_genome, sig_high, color=c_high, alpha=0.4, linestyle='-', linewidth=1, zorder=1)
    ax2.plot(x_genome, sig_low, color=c_low, alpha=0.4, linestyle='-', linewidth=1, zorder=1)
    
    # Plot zoomed segments where pattern matches
    seg_high = sig_high[pos_high:pos_high+width]
    seg_low = sig_low[pos_low:pos_low+width]
    
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
        
    # Plot the Pattern itself (use high sample's position for consistency)
    pat_x_high = np.arange(pos_high, pos_high + width) - tss_offset
    pat_x_low = np.arange(pos_low, pos_low + width) - tss_offset
    
    ax2.plot(pat_x_high, pat_scaled, color=c_pat, linewidth=5, label='Learned Pattern (B-spline)', zorder=4)
    
    # Plot the representative segments thicker
    ax2.plot(pat_x_high, seg_high, color=c_high, linewidth=2.5, label='High Expr Sample', zorder=3)
    ax2.plot(pat_x_low, seg_low, color=c_low, linewidth=2.5, label='Low Expr Sample', zorder=3)
    
    # Visualize "Error" (RMSE area) - scale pattern for low sample too
    p_min_low, p_max_low = pattern_curve.min(), pattern_curve.max()
    t_min_low, t_max_low = seg_low.min(), seg_low.max()
    if p_max_low - p_min_low > 1e-5:
        pat_scaled_low = (pattern_curve - p_min_low) / (p_max_low - p_min_low) * (t_max_low - t_min_low) + t_min_low
    else:
        pat_scaled_low = pattern_curve + t_min_low
    
    ax2.fill_between(pat_x_high, pat_scaled, seg_high, color=c_high, alpha=0.2, hatch='///')
    ax2.fill_between(pat_x_low, pat_scaled_low, seg_low, color=c_low, alpha=0.2, hatch='\\\\')
    
    # Annotations
    transform_label = transform.replace('_', ' ').title()
    ax2.set_title(f'B. Pattern Matching: {transform_label} Space', loc='left', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{transform_label} Signal Value', fontsize=10)
    ax2.set_xlabel('Distance from TSS (bp)', fontsize=9)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle=':')
    
    # Add text box explaining the match
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    txt = (f"High Expr Error: {rmses[idx_high]:.1f}\n"
           f"Low Expr Error: {rmses[idx_low]:.1f}")
    ax2.text(0.02, 0.95, txt, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=bbox_props)

    # --- Panel 3: Separation (Distribution) ---
    ax3 = fig.add_subplot(gs_remc[2])
    
    # Density plots
    sns.kdeplot(rmses[(y==1) & valid], color=c_high, fill=True, alpha=0.3, label='High Expression', ax=ax3, linewidth=2)
    sns.kdeplot(rmses[(y==0) & valid], color=c_low, fill=True, alpha=0.3, label='Low Expression', ax=ax3, linewidth=2)
    
    ax3.set_title('C. Feature Discrimination: RMSE Distribution', loc='left', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Pattern Dissimilarity (RMSE)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.2, axis='x')

def create_mitbih_panels(fig, gs_mitbih):
    """Create MITBIH visualization panels in the right column."""
    # Load Data
    with open(Path(f'../json_files/mitbih/pattern_parameters.json'), 'r') as f:
        data = json.load(f)
    patterns = data[list(data.keys())[0]]
    
    df = pd.read_csv(Path(f'../processed_datasets/mitbih/mitbih_processed.csv'))
    y = df['target'].values
    
    # Get time series columns
    time_cols = [c for c in df.columns if c != 'target' and str(c).isdigit()]
    time_cols = sorted(time_cols, key=lambda x: int(x))
    X_data = df[time_cols].values
    
    # Find discriminative pattern
    best_idx = find_best_pattern_mitbih(patterns, y, df)
    p = patterns[best_idx]
    
    # Extract info
    transform = p.get('transform', 'raw')
    width = int(round(p['width']))
    start_frac = p['start_frac']
    end_frac = p['end_frac']
    
    # Prepare signals
    X_raw = X_data
    X_trans = np.array([apply_transformation(x, transform) for x in X_raw])
    
    # Generate pattern curve
    pattern_curve = generate_bspline_pattern(p['control_points'], width)
    
    # Calc RMSEs
    rmses = np.array([
        compute_rmse_sliding(x, pattern_curve, start_frac, end_frac, width) 
        for x in X_trans
    ])
    valid = np.isfinite(rmses)
    
    # Select representative samples (use first two classes)
    unique_classes = np.unique(y)
    class_0, class_1 = unique_classes[0], unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
    
    def get_rep_idx(target_class):
        idxs = np.where((y == target_class) & valid)[0]
        if len(idxs) == 0:
            idxs = np.where(y == target_class)[0]
        cls_rmses = rmses[idxs]
        median_val = np.median(cls_rmses)
        return idxs[np.argmin(np.abs(cls_rmses - median_val))]
    
    idx_1 = get_rep_idx(class_1)
    idx_0 = get_rep_idx(class_0)
    
    # Colors
    c_1 = '#2980b9'
    c_0 = '#c0392b'
    c_pat = '#27ae60'
    
    # X-axis (sample points, centered)
    x_samples = np.arange(X_raw.shape[1]) - X_raw.shape[1] // 2
    
    # Find best match position
    rep_sig = X_trans[valid][0] if valid.any() else X_trans[0]
    sig_len = len(rep_sig)
    max_start = sig_len - width
    search_start = int(min(start_frac, end_frac) * max_start)
    search_end = int(max(start_frac, end_frac) * max_start)
    
    signal_subset = rep_sig[search_start:search_end + width]
    if len(signal_subset) >= width:
        windows = sliding_window_view(signal_subset, width)
        diff = windows - pattern_curve
        distances = np.einsum('...i,...i->...', diff, diff) / diff.shape[-1]
        actual_start = search_start + np.argmin(distances)
    else:
        actual_start = search_start
    
    # --- Panel 1: Biological Context ---
    ax1 = fig.add_subplot(gs_mitbih[0])
    
    # Plot mean signals for each class
    mean_1 = X_raw[y==class_1].mean(axis=0)
    mean_0 = X_raw[y==class_0].mean(axis=0)
    
    ax1.plot(x_samples, mean_1, color=c_1, label=f'Class {class_1} (Avg)', linewidth=2)
    ax1.plot(x_samples, mean_0, color=c_0, label=f'Class {class_0} (Avg)', linewidth=2)
    ax1.fill_between(x_samples, mean_1, 0, color=c_1, alpha=0.1)
    ax1.fill_between(x_samples, mean_0, 0, color=c_0, alpha=0.1)
    
    # Highlight pattern region
    pat_region_start = actual_start - X_raw.shape[1] // 2
    pat_region_end = (actual_start + width) - X_raw.shape[1] // 2
    ax1.axvspan(pat_region_start, pat_region_end, color='gray', alpha=0.15, label='Pattern Region')
    
    ax1.set_title(f'A. Biological Context: ECG Signal', loc='left', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ECG Amplitude', fontsize=10)
    ax1.set_xlabel('Sample (relative to center)', fontsize=9)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.2, linestyle=':')
    
    # --- Panel 2: Mechanism (Transform Space) ---
    ax2 = fig.add_subplot(gs_mitbih[1])
    
    sig_1 = X_trans[idx_1]
    sig_0 = X_trans[idx_0]
    
    # Find best match positions
    def find_best_match_pos(signal, pattern, start_frac, end_frac, width):
        sig_len = len(signal)
        max_start = sig_len - width
        search_start = int(min(start_frac, end_frac) * max_start)
        search_end = int(max(start_frac, end_frac) * max_start)
        signal_subset = signal[search_start:search_end + width]
        if len(signal_subset) < width:
            return search_start
        windows = sliding_window_view(signal_subset, width)
        diff = windows - pattern
        distances = np.einsum('...i,...i->...', diff, diff) / diff.shape[-1]
        return search_start + np.argmin(distances)
    
    pos_1 = find_best_match_pos(sig_1, pattern_curve, start_frac, end_frac, width)
    pos_0 = find_best_match_pos(sig_0, pattern_curve, start_frac, end_frac, width)
    
    # Highlight pattern regions with background (using search range for context)
    search_start = int(min(start_frac, end_frac) * (sig_len - width))
    search_end = int(max(start_frac, end_frac) * (sig_len - width))
    pat_region_start = search_start - X_raw.shape[1] // 2
    pat_region_end = (search_end + width) - X_raw.shape[1] // 2
    ax2.axvspan(pat_region_start, pat_region_end, color='gray', alpha=0.15, zorder=0)
    
    # Plot full transformed signals (on top of background)
    ax2.plot(x_samples, sig_1, color=c_1, alpha=0.4, linestyle='-', linewidth=1, zorder=1)
    ax2.plot(x_samples, sig_0, color=c_0, alpha=0.4, linestyle='-', linewidth=1, zorder=1)
    
    seg_1 = sig_1[pos_1:pos_1+width]
    seg_0 = sig_0[pos_0:pos_0+width]
    
    target_seg = seg_1 if rmses[idx_1] < rmses[idx_0] else seg_0
    p_min, p_max = pattern_curve.min(), pattern_curve.max()
    t_min, t_max = target_seg.min(), target_seg.max()
    
    if p_max - p_min > 1e-5:
        pat_scaled = (pattern_curve - p_min) / (p_max - p_min) * (t_max - t_min) + t_min
    else:
        pat_scaled = pattern_curve + t_min
    
    pat_x_1 = np.arange(pos_1, pos_1 + width) - X_raw.shape[1] // 2
    pat_x_0 = np.arange(pos_0, pos_0 + width) - X_raw.shape[1] // 2
    
    ax2.plot(pat_x_1, pat_scaled, color=c_pat, linewidth=5, label='Learned Pattern', zorder=4)
    ax2.plot(pat_x_1, seg_1, color=c_1, linewidth=2.5, label=f'Class {class_1} Sample', zorder=3)
    ax2.plot(pat_x_0, seg_0, color=c_0, linewidth=2.5, label=f'Class {class_0} Sample', zorder=3)
    
    # Error visualization
    p_min_0, p_max_0 = pattern_curve.min(), pattern_curve.max()
    t_min_0, t_max_0 = seg_0.min(), seg_0.max()
    if p_max_0 - p_min_0 > 1e-5:
        pat_scaled_0 = (pattern_curve - p_min_0) / (p_max_0 - p_min_0) * (t_max_0 - t_min_0) + t_min_0
    else:
        pat_scaled_0 = pattern_curve + t_min_0
    
    ax2.fill_between(pat_x_1, pat_scaled, seg_1, color=c_1, alpha=0.2, hatch='///')
    ax2.fill_between(pat_x_0, pat_scaled_0, seg_0, color=c_0, alpha=0.2, hatch='\\\\')
    
    transform_label = transform.replace('_', ' ').title()
    ax2.set_title(f'B. Pattern Matching: {transform_label} Space', loc='left', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{transform_label} Value', fontsize=10)
    ax2.set_xlabel('Sample (relative to center)', fontsize=9)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle=':')
    
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    txt = (f"Class {class_1} Error: {rmses[idx_1]:.1f}\n"
           f"Class {class_0} Error: {rmses[idx_0]:.1f}")
    ax2.text(0.02, 0.95, txt, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=bbox_props)
    
    # --- Panel 3: Separation (Distribution) ---
    ax3 = fig.add_subplot(gs_mitbih[2])
    
    sns.kdeplot(rmses[(y==class_1) & valid], color=c_1, fill=True, alpha=0.3, 
                label=f'Class {class_1}', ax=ax3, linewidth=2)
    sns.kdeplot(rmses[(y==class_0) & valid], color=c_0, fill=True, alpha=0.3, 
                label=f'Class {class_0}', ax=ax3, linewidth=2)
    
    ax3.set_title('C. Feature Discrimination: RMSE Distribution', loc='left', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Pattern Dissimilarity (RMSE)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.2, axis='x')

def visualize_dual_pattern_story(cell_line='E003'):
    """Create a 2-column, 3-row visualization with REMC on left and MITBIH on right."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1.2, 0.8], 
                           width_ratios=[1, 1], hspace=0.4, wspace=0.3)
    
    # Left column: REMC
    gs_remc = [gs[0, 0], gs[1, 0], gs[2, 0]]
    create_remc_panels(fig, gs_remc, cell_line)
    
    # Right column: MITBIH
    gs_mitbih = [gs[0, 1], gs[1, 1], gs[2, 1]]
    create_mitbih_panels(fig, gs_mitbih)
    
    return fig

if __name__ == "__main__":
    output_dir = Path('../manuscript/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig = visualize_dual_pattern_story('E003')
    fig.savefig(output_dir / 'domain_pattern_interpretation.png', dpi=300, bbox_inches='tight')
    print("Saved domain_pattern_interpretation.png")
