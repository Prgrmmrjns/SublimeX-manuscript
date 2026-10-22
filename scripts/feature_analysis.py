import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import apply_manuscript_figure_style, mm_size, save_png_pdf_eps
from config import RESULTS, PARAMETERS, ELARTICLE

RESULTS.mkdir(parents=True, exist_ok=True)
apply_manuscript_figure_style(13)

TRANSFORM_NAMES = ['raw', 'zscore', 'derivative', 'fft_power']
TRANSFORM_LABELS = {'raw': 'Raw', 'zscore': 'Z-score', 'derivative': 'Derivative', 'fft_power': 'FFT Power'}
TRANSFORM_COLORS = {'raw': '#1f77b4', 'zscore': '#ff7f0e', 'derivative': '#2ca02c', 'fft_power': '#d62728'}

DATASET_INFO = {
    'pamap2': {'display': 'PAMAP2', 'n_folds': 5, 'n_timepoints': 100, 'n_channels': 36},
    'emotions': {'display': 'Emotions', 'n_folds': 5, 'n_timepoints': 254, 'n_channels': 2},
    'mitbih': {'display': 'MIT-BIH', 'n_folds': 5, 'n_timepoints': 100, 'n_channels': 1},
    'remc': {'display': 'REMC', 'n_folds': 5, 'n_timepoints': 200, 'n_channels': 5},
    'mimic': {'display': 'MIMIC-IV', 'n_folds': 5, 'n_timepoints': 24, 'n_channels': 8},
    'svd': {'display': 'SVD', 'n_folds': 5, 'n_timepoints': 1000, 'n_channels': 1},
    'azt1d': {'display': 'AZT1D', 'n_folds': 1, 'n_timepoints': 24, 'n_channels': 3, 'task': 'regression'},
}


def load_fold_parameters(dataset_name, fold):
    for name in (f'mean_only_fold{fold}.json', f'fold{fold}.json'):
        p = PARAMETERS / dataset_name / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def load_all_parameters(dataset_name):
    n_folds = DATASET_INFO.get(dataset_name, {}).get('n_folds', 5)
    out = {}
    for fold in range(1, n_folds + 1):
        params = load_fold_parameters(dataset_name, fold)
        if params:
            out[fold] = params
    return out


def get_remc_cell_lines():
    return sorted(p.name.replace('remc_', '') for p in PARAMETERS.iterdir()
                  if p.is_dir() and p.name.startswith('remc_E'))


def load_cell_line_parameters(cell_line):
    return {f: p for f in range(1, 6) if (p := load_fold_parameters(f'remc_{cell_line}', f))}


def compute_segment_bounds(c, r, n_timepoints):
    center_idx = c * (n_timepoints - 1)
    half_width = (r * (n_timepoints - 1)) * 0.5
    return max(0, int(center_idx - half_width)), min(n_timepoints - 1, int(center_idx + half_width))


def analyze_transform_composition(datasets):
    results = {}
    for ds in datasets:
        all_params = load_all_parameters(ds)
        if not all_params and ds == 'remc':
            tc, total = defaultdict(int), 0
            for cl in get_remc_cell_lines()[:10]:
                for fp in load_cell_line_parameters(cl).values():
                    for p in fp:
                        ti = int(p.get('t', 0))
                        tc[TRANSFORM_NAMES[ti] if ti < len(TRANSFORM_NAMES) else 'unknown'] += 1
                        total += 1
            if total > 0:
                results[ds] = {t: tc[t] / total for t in TRANSFORM_NAMES}
            continue
        tc, total = defaultdict(int), 0
        for params in all_params.values():
            for p in params:
                ti = int(p.get('t', 0))
                tc[TRANSFORM_NAMES[ti] if ti < len(TRANSFORM_NAMES) else 'unknown'] += 1
                total += 1
        if total > 0:
            results[ds] = {t: tc[t] / total for t in TRANSFORM_NAMES}
    return results


def analyze_pattern_stability(datasets):
    results = {}
    for ds in datasets:
        info = DATASET_INFO.get(ds, {})
        if info.get('n_folds', 5) < 2:
            continue
        n_tp = info.get('n_timepoints', 100)
        all_params = load_all_parameters(ds)
        if not all_params and ds == 'remc':
            cls = get_remc_cell_lines()
            if cls:
                all_params = load_cell_line_parameters(cls[0])
                n_tp = 200
        if not all_params:
            continue
        first_patterns = []
        for fold, params in sorted(all_params.items()):
            p = params[0]
            c, r = p.get('c', 0.5), p.get('r', 0.5)
            ti, ch = int(p.get('t', 0)), int(p.get('ch', 0))
            s, e = compute_segment_bounds(c, r, n_tp)
            first_patterns.append({
                'fold': fold, 'center': c, 'range': r,
                'start': s / n_tp, 'end': e / n_tp,
                'transform': TRANSFORM_NAMES[ti] if ti < len(TRANSFORM_NAMES) else 'unknown',
                'channel': ch,
            })
        results[ds] = first_patterns
    return results


def generate_figure(transform_results, stability_results):
    datasets_a = ['pamap2', 'emotions', 'mitbih', 'remc', 'azt1d', 'mimic', 'svd']
    datasets_b = ['pamap2', 'emotions', 'mitbih', 'remc', 'mimic', 'svd']
    fig, axes = plt.subplots(1, 2, figsize=mm_size(220, 82), layout='constrained',
                             gridspec_kw={'width_ratios': [1.2, 1]})
    ax1, ax2 = axes
    x = np.arange(len(datasets_a))
    bottom = np.zeros(len(datasets_a))
    for transform in TRANSFORM_NAMES:
        heights = [transform_results.get(ds, {}).get(transform, 0) for ds in datasets_a]
        ax1.bar(x, heights, 0.6, bottom=bottom, label=TRANSFORM_LABELS[transform],
                color=TRANSFORM_COLORS[transform], alpha=0.85)
        bottom += heights
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_INFO[ds]['display'] for ds in datasets_a],
                        fontsize=10, rotation=35, ha='right')
    ax1.tick_params(axis='x', pad=4)
    ax1.margins(x=0.08)
    ax1.set_ylabel('Proportion of Features', fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
               fontsize=7.5, ncol=4, handlelength=1.2, columnspacing=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.text(-0.08, 1.02, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='left')

    y_positions, y_labels = [], []
    for y_pos, ds in enumerate([d for d in datasets_b if d in stability_results]):
        patterns = stability_results[ds]
        n_folds = len(patterns)
        starts = np.array([p['start'] for p in patterns])
        ends = np.array([p['end'] for p in patterns])
        mean_start, mean_end = np.mean(starts), np.mean(ends)
        std_start, std_end = np.std(starts), np.std(ends)
        tc = Counter(p['transform'] for p in patterns)
        cc = Counter(p.get('channel', 0) for p in patterns)
        t_cons = tc.most_common(1)[0][1] / n_folds
        c_cons = cc.most_common(1)[0][1] / n_folds
        ax2.barh(y_pos, mean_end - mean_start, left=mean_start, height=0.5,
                 color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
        left_err = min(std_start, mean_start)
        if left_err > 0.001:
            le = mean_start - left_err
            ax2.hlines(y_pos, le, mean_start, colors='black', linewidth=1.5)
            ax2.vlines(le, y_pos - 0.2, y_pos + 0.2, colors='black', linewidth=1.5)
        right_err = min(std_end, 1 - mean_end)
        if right_err > 0.001:
            re = mean_end + right_err
            ax2.hlines(y_pos, mean_end, re, colors='black', linewidth=1.5)
            ax2.vlines(re, y_pos - 0.2, y_pos + 0.2, colors='black', linewidth=1.5)
        lbl = DATASET_INFO[ds]['display']
        cs = f"[T:{t_cons:.1f}, C:{c_cons:.1f}]" if DATASET_INFO[ds].get('n_channels', 1) > 1 else f"[T:{t_cons:.1f}]"
        y_positions.append(y_pos)
        y_labels.append(f"{lbl} {cs}")

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(y_labels, fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Normalized Position', fontsize=13)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    ax2.text(-0.08, 1.02, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    save_png_pdf_eps(fig, ELARTICLE / 'feature_analysis', dpi=300)
    return fig


def save_results_to_csv(transform_results, stability_results):
    rows = []
    for ds in ['pamap2', 'emotions', 'mitbih', 'remc', 'azt1d', 'mimic', 'svd']:
        info = DATASET_INFO[ds]
        row = {'dataset': ds, 'display_name': info['display'], 'task': info.get('task', 'classification'),
               'n_channels': info['n_channels'], 'n_timepoints': info['n_timepoints']}
        if ds in transform_results:
            for t in TRANSFORM_NAMES:
                row[f'transform_prop_{t}'] = transform_results[ds].get(t, 0)
        if ds in stability_results:
            pats = stability_results[ds]
            nf = len(pats)
            starts = np.array([p['start'] for p in pats])
            ends = np.array([p['end'] for p in pats])
            tc = Counter(p['transform'] for p in pats)
            cc = Counter(p.get('channel', 0) for p in pats)
            dt, dc = tc.most_common(1)[0][0], cc.most_common(1)[0][0]
            row.update({
                'n_folds': nf, 'mean_start': np.mean(starts), 'mean_end': np.mean(ends),
                'std_start': np.std(starts), 'std_end': np.std(ends),
                'position_std': np.std((starts + ends) / 2),
                'dominant_transform': dt, 'transform_consistency': tc[dt] / nf,
                'dominant_channel': dc, 'channel_consistency': cc[dc] / nf,
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / 'performance_analysis.csv', index=False)


def main():
    datasets = ['pamap2', 'emotions', 'mitbih', 'remc', 'azt1d', 'mimic', 'svd']
    tr = analyze_transform_composition(datasets)
    st = analyze_pattern_stability(datasets)
    save_results_to_csv(tr, st)
    generate_figure(tr, st)


if __name__ == '__main__':
    main()
