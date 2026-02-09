#!/usr/bin/env python3
"""
Performance Analysis: Why SublimeX Succeeds or Fails

This script investigates why SublimeX performs well on some
datasets but not others, through two complementary analyses:
1. Transform composition per dataset (which transforms are
   selected)
2. Pattern stability across folds (segment positions)

Results are saved to a single CSV in results/.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter

# Paths
BASE_DIR = Path(__file__).parent.parent
PARAMETERS_DIR = BASE_DIR / 'parameters'
OUTPUT_DIR = BASE_DIR / 'elsarticle' / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Transform name mapping
TRANSFORM_NAMES = ['raw', 'zscore', 'derivative', 'fft_power']
TRANSFORM_LABELS = {
    'raw': 'Raw', 'zscore': 'Z-score',
    'derivative': 'Derivative', 'fft_power': 'FFT Power',
}
TRANSFORM_COLORS = {
    'raw': '#1f77b4', 'zscore': '#ff7f0e',
    'derivative': '#2ca02c', 'fft_power': '#d62728',
}

# Dataset configuration
DATASET_INFO = {
    'pamap2': {
        'display': 'PAMAP2',
        'n_folds': 5,
        'n_timepoints': 100,
        'n_channels': 36,
    },
    'emotions': {
        'display': 'Emotions',
        'n_folds': 5,
        'n_timepoints': 254,
        'n_channels': 2,
    },
    'mitbih': {
        'display': 'MITBIH',
        'n_folds': 5,
        'n_timepoints': 100,
        'n_channels': 1,
    },
    'remc': {
        'display': 'REMC',
        'n_folds': 5,
        'n_timepoints': 200,
        'n_channels': 5,
    },
    'mimic': {
        'display': 'MIMIC',
        'n_folds': 5,
        'n_timepoints': 24,
        'n_channels': 8,
    },
    'svd': {
        'display': 'SVD',
        'n_folds': 5,
        'n_timepoints': 1000,
        'n_channels': 1,
    },
    'azt1d': {
        'display': 'AZT1D',
        'n_folds': 1,
        'n_timepoints': 24,
        'n_channels': 3,
        'task': 'regression',
    },
}


# =============================================================
# Data Loading Utilities
# =============================================================

def load_fold_parameters(dataset_name, fold):
    """Load parameters for a specific fold."""
    paths_to_try = [
        PARAMETERS_DIR / dataset_name
        / f'mean_only_fold{fold}.json',
        PARAMETERS_DIR / dataset_name / f'fold{fold}.json',
    ]
    for json_path in paths_to_try:
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
    return None


def load_all_parameters(dataset_name):
    """Load parameters across all folds for a dataset."""
    info = DATASET_INFO.get(dataset_name, {})
    n_folds = info.get('n_folds', 5)
    all_params = {}
    for fold in range(1, n_folds + 1):
        params = load_fold_parameters(dataset_name, fold)
        if params:
            all_params[fold] = params
    return all_params


def get_remc_cell_lines():
    """Get list of REMC cell lines with parameters."""
    cell_lines = []
    for path in PARAMETERS_DIR.iterdir():
        if path.is_dir() and path.name.startswith('remc_E'):
            cell_lines.append(
                path.name.replace('remc_', '')
            )
    return sorted(cell_lines)


def load_cell_line_parameters(cell_line):
    """Load parameters for a specific REMC cell line."""
    dataset_name = f'remc_{cell_line}'
    all_params = {}
    for fold in range(1, 6):
        params = load_fold_parameters(dataset_name, fold)
        if params:
            all_params[fold] = params
    return all_params


# =============================================================
# Analysis Functions
# =============================================================

def compute_segment_bounds(c, r, n_timepoints):
    """Compute segment start/end from center and range."""
    center_idx = c * (n_timepoints - 1)
    half_width = (r * (n_timepoints - 1)) * 0.5
    start = max(0, int(center_idx - half_width))
    end = min(n_timepoints - 1, int(center_idx + half_width))
    return start, end


def analyze_transform_composition(datasets):
    """Analyze transform composition across datasets."""
    results = {}
    for ds in datasets:
        all_params = load_all_parameters(ds)
        if not all_params:
            if ds == 'remc':
                cell_lines = get_remc_cell_lines()[:10]
                tc = defaultdict(int)
                total = 0
                for cl in cell_lines:
                    params = load_cell_line_parameters(cl)
                    for fp in params.values():
                        for p in fp:
                            ti = int(p.get('t', 0))
                            tn = (TRANSFORM_NAMES[ti]
                                  if ti < len(TRANSFORM_NAMES)
                                  else 'unknown')
                            tc[tn] += 1
                            total += 1
                if total > 0:
                    results[ds] = {
                        t: tc[t] / total
                        for t in TRANSFORM_NAMES
                    }
            continue

        tc = defaultdict(int)
        total = 0
        for params in all_params.values():
            for p in params:
                ti = int(p.get('t', 0))
                tn = (TRANSFORM_NAMES[ti]
                      if ti < len(TRANSFORM_NAMES)
                      else 'unknown')
                tc[tn] += 1
                total += 1
        if total > 0:
            results[ds] = {
                t: tc[t] / total for t in TRANSFORM_NAMES
            }
    return results


def analyze_pattern_stability(datasets):
    """Analyze first pattern stability across folds."""
    results = {}
    for ds in datasets:
        # Skip AZT1D: single fold, stability undefined
        if DATASET_INFO.get(ds, {}).get('n_folds', 5) < 2:
            continue

        info = DATASET_INFO.get(ds, {})
        n_tp = info.get('n_timepoints', 100)

        all_params = load_all_parameters(ds)
        if not all_params:
            if ds == 'remc':
                cls = get_remc_cell_lines()
                if cls:
                    all_params = load_cell_line_parameters(
                        cls[0]
                    )
                    n_tp = 200
        if not all_params:
            continue

        first_patterns = []
        for fold, params in sorted(all_params.items()):
            if params:
                p = params[0]
                c = p.get('c', 0.5)
                r = p.get('r', 0.5)
                ti = int(p.get('t', 0))
                ch = int(p.get('ch', 0))
                s, e = compute_segment_bounds(c, r, n_tp)
                first_patterns.append({
                    'fold': fold,
                    'center': c,
                    'range': r,
                    'start': s / n_tp,
                    'end': e / n_tp,
                    'transform': (
                        TRANSFORM_NAMES[ti]
                        if ti < len(TRANSFORM_NAMES)
                        else 'unknown'
                    ),
                    'channel': ch,
                })
        results[ds] = first_patterns
    return results


# =============================================================
# Figure Generation
# =============================================================

def generate_figure(transform_results, stability_results):
    """Generate 2-panel figure (A: transforms, B: stability)."""

    # Panel A includes all datasets; Panel B excludes AZT1D
    datasets_panel_a = [
        'pamap2', 'emotions', 'mitbih', 'remc',
        'azt1d', 'mimic', 'svd',
    ]
    datasets_panel_b = [
        'pamap2', 'emotions', 'mitbih', 'remc',
        'mimic', 'svd',
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ===== Panel A: Transform Composition =====
    ax1 = axes[0]
    x = np.arange(len(datasets_panel_a))
    width = 0.6
    bottom = np.zeros(len(datasets_panel_a))
    for transform in TRANSFORM_NAMES:
        heights = []
        for ds in datasets_panel_a:
            if ds in transform_results:
                heights.append(
                    transform_results[ds].get(transform, 0)
                )
            else:
                heights.append(0)
        ax1.bar(
            x, heights, width, bottom=bottom,
            label=TRANSFORM_LABELS[transform],
            color=TRANSFORM_COLORS[transform], alpha=0.85,
        )
        bottom += heights

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [DATASET_INFO[ds]['display']
         for ds in datasets_panel_a],
        fontsize=8, rotation=30, ha='right',
    )
    ax1.set_ylabel('Proportion of Features', fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.text(
        -0.08, 1.02, 'A', transform=ax1.transAxes,
        fontsize=14, fontweight='bold', va='bottom',
        ha='left',
    )

    # ===== Panel B: Pattern Stability =====
    ax2 = axes[1]
    y_pos = 0
    y_positions = []
    y_labels = []

    for ds in datasets_panel_b:
        if ds not in stability_results:
            continue
        patterns = stability_results[ds]
        n_folds = len(patterns)

        starts = np.array([p['start'] for p in patterns])
        ends = np.array([p['end'] for p in patterns])
        mean_start = np.mean(starts)
        mean_end = np.mean(ends)
        std_start = np.std(starts)
        std_end = np.std(ends)

        transforms = [p['transform'] for p in patterns]
        channels = [p.get('channel', 0) for p in patterns]
        tc = Counter(transforms)
        cc = Counter(channels)
        dom_t = tc.most_common(1)[0][0]
        t_cons = tc[dom_t] / n_folds
        dom_c = cc.most_common(1)[0][0]
        c_cons = cc[dom_c] / n_folds

        ax2.barh(
            y_pos, mean_end - mean_start,
            left=mean_start, height=0.5,
            color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=1,
        )

        cap_h = 0.2
        left_err = min(std_start, mean_start)
        if left_err > 0.001:
            le = mean_start - left_err
            ax2.hlines(
                y_pos, le, mean_start,
                colors='black', linewidth=1.5,
            )
            ax2.vlines(
                le, y_pos - cap_h, y_pos + cap_h,
                colors='black', linewidth=1.5,
            )
        right_err = min(std_end, 1 - mean_end)
        if right_err > 0.001:
            re = mean_end + right_err
            ax2.hlines(
                y_pos, mean_end, re,
                colors='black', linewidth=1.5,
            )
            ax2.vlines(
                re, y_pos - cap_h, y_pos + cap_h,
                colors='black', linewidth=1.5,
            )

        label = DATASET_INFO[ds]['display']
        n_ch = DATASET_INFO[ds].get('n_channels', 1)
        if n_ch > 1:
            cs = (f"[T:{t_cons:.1f},"
                  f" C:{c_cons:.1f}]")
        else:
            cs = f"[T:{t_cons:.1f}]"
        y_positions.append(y_pos)
        y_labels.append(f"{label} {cs}")
        y_pos += 1

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(y_labels, fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Normalized Position', fontsize=11)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    ax2.text(
        -0.08, 1.02, 'B', transform=ax2.transAxes,
        fontsize=14, fontweight='bold', va='bottom',
        ha='left',
    )

    plt.tight_layout()
    out = OUTPUT_DIR / 'performance_stability_analysis.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out}")
    return fig


# =============================================================
# Summary and CSV
# =============================================================

def print_summary(transform_results, stability_results):
    """Print summary of analysis."""
    all_ds = [
        'pamap2', 'emotions', 'mitbih', 'remc',
        'azt1d', 'mimic', 'svd',
    ]
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n--- Transform Composition ---")
    print(f"{'Dataset':<10} {'Raw':<8} {'Z-score':<10} "
          f"{'Deriv.':<10} {'FFT':<8}")
    print("-" * 50)
    for ds in all_ds:
        if ds in transform_results:
            tr = transform_results[ds]
            print(
                f"{ds.upper():<10} "
                f"{tr.get('raw', 0):.0%}     "
                f"{tr.get('zscore', 0):.0%}       "
                f"{tr.get('derivative', 0):.0%}       "
                f"{tr.get('fft_power', 0):.0%}"
            )

    print("\n--- Pattern Stability ---")
    for ds in all_ds:
        if ds in stability_results:
            pats = stability_results[ds]
            positions = [
                (p['start'] + p['end']) / 2 for p in pats
            ]
            transforms = [p['transform'] for p in pats]
            ps = np.std(positions)
            tm = max(set(transforms), key=transforms.count)
            tc = transforms.count(tm) / len(transforms)
            print(f"{ds.upper()}: pos_std={ps:.3f}, "
                  f"T_cons={tc:.0%} ({tm})")


def save_results_to_csv(transform_results,
                        stability_results):
    """Save all analysis results to a single CSV."""
    all_ds = [
        'pamap2', 'emotions', 'mitbih', 'remc',
        'azt1d', 'mimic', 'svd',
    ]
    rows = []
    for ds in all_ds:
        info = DATASET_INFO.get(ds, {})
        row = {
            'dataset': ds,
            'display_name': info.get('display', ds.upper()),
            'task': info.get('task', 'classification'),
            'n_channels': info.get('n_channels', 1),
            'n_timepoints': info.get('n_timepoints', 0),
        }
        if ds in transform_results:
            for t in TRANSFORM_NAMES:
                row[f'transform_prop_{t}'] = (
                    transform_results[ds].get(t, 0)
                )
        if ds in stability_results:
            pats = stability_results[ds]
            nf = len(pats)
            starts = np.array([p['start'] for p in pats])
            ends = np.array([p['end'] for p in pats])
            transforms = [p['transform'] for p in pats]
            channels = [p.get('channel', 0) for p in pats]
            tc = Counter(transforms)
            cc = Counter(channels)
            dt = tc.most_common(1)[0][0]
            dc = cc.most_common(1)[0][0]
            row.update({
                'n_folds': nf,
                'mean_start': np.mean(starts),
                'mean_end': np.mean(ends),
                'std_start': np.std(starts),
                'std_end': np.std(ends),
                'position_std': np.std(
                    (starts + ends) / 2
                ),
                'dominant_transform': dt,
                'transform_consistency': tc[dt] / nf,
                'dominant_channel': dc,
                'channel_consistency': cc[dc] / nf,
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'performance_analysis.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved results to {out}")


# =============================================================
# Main
# =============================================================

datasets = [
    'pamap2', 'emotions', 'mitbih', 'remc',
    'azt1d', 'mimic', 'svd',
]
transform_results = analyze_transform_composition(datasets)
stability_results = analyze_pattern_stability(datasets)

print_summary(transform_results, stability_results)
save_results_to_csv(transform_results, stability_results)
generate_figure(transform_results, stability_results)
