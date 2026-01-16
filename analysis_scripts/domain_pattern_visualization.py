import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import BSpline
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.gridspec as gridspec
import shap
import sys
from pathlib import Path as _Path

# Suppress SHAP LightGBM warning
warnings.filterwarnings("ignore", message="LightGBM binary classifier.*TreeExplainer")

_BASE_DIR = _Path(__file__).resolve().parents[1]
sys.path.append(str(_BASE_DIR / "eval_scripts"))
from models import LightGBMWrapper

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

def compute_cosine_distance_sliding(signal, pattern, start_frac, end_frac, width):
    """Compute minimum cosine distance using sliding window (matching core.py)."""
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
    
    # Cosine distance = 1 - cosine_similarity
    pat_norm = np.sqrt(np.dot(pattern, pattern))
    if pat_norm < 1e-8:
        pat_norm = 1.0
    dot_product = np.einsum('...i,i->...', windows, pattern)
    window_norm = np.sqrt(np.einsum('...i,...i->...', windows, windows))
    window_norm = np.where(window_norm < 1e-8, 1.0, window_norm)
    cosine_sim = dot_product / (window_norm * pat_norm)
    distances = 1.0 - cosine_sim
    return float(np.min(distances))


def find_best_match_pos_core(signal, pattern, start_frac, end_frac, width):
    """Find best match position using core.py sliding-window cosine distance."""
    w = int(round(width))
    sig_len = len(signal)
    max_start = sig_len - w
    start = int(min(start_frac, end_frac) * max_start)
    end = int(max(start_frac, end_frac) * max_start)
    if end + w > sig_len or len(pattern) != w:
        return start
    signal_subset = signal[start:end + w]
    if len(signal_subset) < w:
        return start
    windows = sliding_window_view(signal_subset, w)
    if len(windows) == 0:
        return start
    # Cosine distance = 1 - cosine_similarity
    pat_norm = np.sqrt(np.dot(pattern, pattern))
    if pat_norm < 1e-8:
        pat_norm = 1.0
    dot_product = np.einsum('...i,i->...', windows, pattern)
    window_norm = np.sqrt(np.einsum('...i,...i->...', windows, windows))
    window_norm = np.where(window_norm < 1e-8, 1.0, window_norm)
    cosine_sim = dot_product / (window_norm * pat_norm)
    distances = 1.0 - cosine_sim
    best_pos = int(np.argmin(distances))
    return start + best_pos

def find_top_patterns_remc(patterns, y, histone_names, df, top_k=5):
    """Find the top-k discriminative patterns for REMC visualization."""
    # Prefer interpretable transforms for visualization
    prefer = ['raw', 'zscore']  # fft_power is less interpretable
    scores = []
    
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
        mse_all = np.array([
            compute_cosine_distance_sliding(x, pattern, p['start_frac'], p['end_frac'], width) 
            for x in X_t
        ])
        valid = np.isfinite(mse_all)
        
        if valid.sum() < 10:
            continue
            
        sep = abs(mse_all[(y==1) & valid].mean() - mse_all[(y==0) & valid].mean())
        scores.append((sep, i))
    if not scores:
        return [0]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in scores[:top_k]]

def compute_feature_matrix(patterns, X_list, y):
    """Compute patX features for all patterns on REMC data."""
    features = []
    for p in patterns:
        transform_name = p.get("transform", "raw")
        channel = p["channel"]
        width = int(round(p["width"]))
        pattern = generate_bspline_pattern(p["control_points"], width)

        X_raw = X_list[channel].values
        X_t = np.array([apply_transformation(x, transform_name) for x in X_raw])
        mse_all = np.array([
            compute_cosine_distance_sliding(x, pattern, p["start_frac"], p["end_frac"], width)
            for x in X_t
        ])
        features.append(mse_all.reshape(-1, 1))
    return np.hstack(features)

# --- Visualization ---

def create_remc_row(fig, gs_row, p, df, y, histone_names, row_label):
    """Create one REMC row: pattern matching (means) + RMSE distribution."""
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
    
    # Calc RMSEs for all samples
    mses = np.array([
        compute_cosine_distance_sliding(x, pattern_curve, start_frac, end_frac, width) 
        for x in X_trans
    ])
    valid = np.isfinite(mses)
    
    # Select representative samples (median RMSE per class)
    def get_rep_idx(target_class):
        idxs = np.where((y == target_class) & valid)[0]
        if len(idxs) == 0:
            idxs = np.where(y == target_class)[0]
        cls_mses = mses[idxs]
        median_val = np.median(cls_mses)
        return idxs[np.argmin(np.abs(cls_mses - median_val))]

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
    rep_sig = X_trans[idx_high]
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
    
    # --- Panel: Pattern Matching (Transform Space) ---
    ax2 = fig.add_subplot(gs_row)
    
    # Get signal slices for representatives
    sig_high = X_trans[idx_high]
    sig_low = X_trans[idx_low]
    
    # Find best match position for each sample independently using core.py logic
    pos_high = find_best_match_pos_core(sig_high, pattern_curve, start_frac, end_frac, width)
    pos_low = find_best_match_pos_core(sig_low, pattern_curve, start_frac, end_frac, width)
    
    # Highlight search region with background
    sig_len_panel2 = len(sig_high)
    search_start = int(min(start_frac, end_frac) * (sig_len_panel2 - width))
    search_end = int(max(start_frac, end_frac) * (sig_len_panel2 - width))
    pat_region_start = search_start - tss_offset
    pat_region_end = (search_end + width) - tss_offset
    ax2.axvspan(pat_region_start, pat_region_end, color='gray', alpha=0.15, zorder=0)
    
    # Plot full transformed signals (faint background)
    ax2.plot(x_genome, sig_high, color=c_high, alpha=0.3, linestyle='-', linewidth=1, zorder=1)
    ax2.plot(x_genome, sig_low, color=c_low, alpha=0.3, linestyle='-', linewidth=1, zorder=1)
    
    # Extract matched segments
    seg_high = sig_high[pos_high:pos_high+width]
    seg_low = sig_low[pos_low:pos_low+width]
    
    # X positions for each matched segment (DIFFERENT positions for high vs low)
    pat_x_high = np.arange(pos_high, pos_high + width) - tss_offset
    pat_x_low = np.arange(pos_low, pos_low + width) - tss_offset
    
    # Plot matched sample segments (solid lines)
    ax2.plot(pat_x_high, seg_high, color=c_high, linewidth=2.5, label='High Expr Sample', zorder=3)
    ax2.plot(pat_x_low, seg_low, color=c_low, linewidth=2.5, label='Low Expr Sample', zorder=3)
    
    # Plot learned pattern (ONE pattern, same shape for both, just shifted to match position)
    # Scale pattern once to the overall signal range for visualization
    sig_all = np.concatenate([seg_high, seg_low])
    p_min, p_max = pattern_curve.min(), pattern_curve.max()
    s_min, s_max = sig_all.min(), sig_all.max()
    if p_max - p_min > 1e-5:
        pat_scaled = (pattern_curve - p_min) / (p_max - p_min) * (s_max - s_min) + s_min
    else:
        pat_scaled = pattern_curve + s_min
    
    # Plot the SAME pattern at both positions (use matching colors for clarity)
    ax2.plot(pat_x_high, pat_scaled, color=c_high, linewidth=3, linestyle='--', 
             label='Learned Pattern (High)', zorder=4, alpha=0.8)
    ax2.plot(pat_x_low, pat_scaled, color=c_low, linewidth=3, linestyle='--', 
             label='Learned Pattern (Low)', zorder=4, alpha=0.8)
    
    # Annotations
    transform_label = transform.replace('_', ' ').title()
    ax2.set_title(
        f"{row_label}. Pattern Matching: {transform_label} Space",
        loc='left',
        fontsize=10,
        fontweight='bold'
    )
    ax2.set_ylabel(f'{transform_label} Signal Value', fontsize=10)
    ax2.set_xlabel('Distance from TSS (bp)', fontsize=9)
    ax2.legend().remove()
    ax2.grid(True, alpha=0.2, linestyle=':')
    
    # Add cosine distance and position annotations inside the plot
    dist_high = mses[idx_high]
    dist_low = mses[idx_low]
    # Convert positions to TSS-relative for display
    pos_high_tss = pos_high - tss_offset
    pos_low_tss = pos_low - tss_offset
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    ax2.text(
        0.02, 0.95,
        f"High Expr: CosDist={dist_high:.4f}, pos={pos_high_tss:+d}bp",
        color=c_high,
        fontsize=9,
        fontweight="bold",
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=bbox_props,
    )
    ax2.text(
        0.02, 0.82,
        f"Low Expr: CosDist={dist_low:.4f}, pos={pos_low_tss:+d}bp",
        color=c_low,
        fontsize=9,
        fontweight="bold",
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=bbox_props,
    )
    return ax2

def visualize_remc_story(cell_line="E003", top_k=5):
    """Create a multi-row visualization with top-k REMC patterns."""
    base_dir = Path(__file__).resolve().parents[1]
    with open(base_dir / "json_files" / "remc" / f"pattern_parameters_{cell_line}.json", "r") as f:
        data = json.load(f)
    patterns = data[list(data.keys())[0]]
    
    df = pd.read_parquet(base_dir / "processed_datasets" / "remc" / f"{cell_line}.parquet")
    y = df['target'].values
    histone_names = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

    top_indices = find_top_patterns_remc(patterns, y, histone_names, df, top_k=top_k)

    fig = plt.figure(figsize=(14, 2.6 * top_k))
    gs = gridspec.GridSpec(top_k, 1, hspace=0.4, top=0.95)
    legend_handles = None
    legend_labels = None
    for row, idx in enumerate(top_indices):
        p = patterns[idx]
        row_label = chr(ord("A") + row)
        ax_pat = create_remc_row(fig, gs[row, 0], p, df, y, histone_names, row_label)
        if legend_handles is None:
            legend_handles, legend_labels = ax_pat.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=4,
            frameon=True,
            fontsize=9,
            bbox_to_anchor=(0.5, 1.0),
        )
    return fig


def save_remc_shap_summary(cell_line="E003"):
    """Save a SHAP summary plot (global, all samples) for REMC."""
    base_dir = Path(__file__).resolve().parents[1]
    with open(base_dir / "json_files" / "remc" / f"pattern_parameters_{cell_line}.json", "r") as f:
        data = json.load(f)
    patterns = data[list(data.keys())[0]]

    df = pd.read_parquet(base_dir / "processed_datasets" / "remc" / f"{cell_line}.parquet")
    y = df["target"].values
    histone_names = ["H3K4me3", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]
    X_list = []
    for h in histone_names:
        cols = [c for c in df.columns if c.startswith(h + "_")]
        X_list.append(df[cols])

    X_feat = compute_feature_matrix(patterns, X_list, y)
    model = LightGBMWrapper(task_type="classification", n_classes=2, inner_cv=1)
    model.fit(X_feat, y)

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_feat)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_names = [
        f"P{i+1}: {histone_names[p['channel']]} ({p.get('transform', 'raw')})"
        for i, p in enumerate(patterns)
    ]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_feat,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=10,
    )
    out_path = base_dir / "elsarticle" / "images" / "remc_shap_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    output_dir = Path('../elsarticle/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig = visualize_remc_story("E003", top_k=5)
    fig.savefig(output_dir / "domain_pattern_interpretation.png", dpi=300, bbox_inches="tight")
    save_remc_shap_summary("E003")
