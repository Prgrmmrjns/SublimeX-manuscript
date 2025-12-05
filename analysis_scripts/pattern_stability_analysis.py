import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import BSpline
from itertools import combinations
import sys
sys.path.append('../eval_scripts')
from core import apply_transformation, generate_bspline_pattern
from models import LightGBMModelWrapper
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'folds': ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
}
HISTONE_NAMES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

def compute_rmse(signal, pattern, start, width):
    w = int(round(width))
    if start + w > len(signal) or len(pattern) != w:
        return np.inf
    return np.sqrt(((signal[start:start + w] - pattern) ** 2).mean())

def get_pattern_importance_remc(patterns, X_list, y):
    n_samples = len(y)
    n_patterns = len(patterns)
    X_stacked = np.stack([x.values for x in X_list], axis=1)
    
    features = np.zeros((n_samples, n_patterns))
    for i, p in enumerate(patterns):
        transform = p.get('transform_type', 'raw')
        X_trans = apply_transformation(X_stacked, transform)
        series = X_trans[:, p['series_idx'], :]
        width = int(round(p['width']))
        pattern_curve = generate_bspline_pattern(p['control_points'], width)
        start = int(p['start'])
        features[:, i] = [compute_rmse(s, pattern_curve, start, width) for s in series]
    
    tr_idx, val_idx = train_test_split(np.arange(n_samples), test_size=0.2, random_state=42, stratify=y)
    model = LightGBMModelWrapper('classification', n_classes=2)
    model.fit(features[tr_idx], y[tr_idx], features[val_idx], y[val_idx])
    importances = model.model.feature_importance(importance_type='gain')
    return importances, np.argmax(importances)

def analyze_fold_dataset(dataset):
    json_path = f'../json_files/{dataset}/pattern_parameters.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fold_names = sorted(data.keys())
    n_folds = len(fold_names)
    
    # Get first pattern from each fold (highest importance proxy)
    first_patterns = [data[f][0] for f in fold_names]
    
    centers = [p['center'] for p in first_patterns]
    widths = [p['width'] for p in first_patterns]
    transforms = [p['transform_type'] for p in first_patterns]
    
    # Compute correlations
    correlations = []
    for i, j in combinations(range(n_folds), 2):
        p1 = generate_bspline_pattern(first_patterns[i]['control_points'], 50)
        p2 = generate_bspline_pattern(first_patterns[j]['control_points'], 50)
        rho, _ = spearmanr(p1, p2)
        correlations.append(rho)
    
    # Transform consistency
    unique_transforms = list(set(transforms))
    dominant_transform = max(set(transforms), key=transforms.count)
    transform_consistency = transforms.count(dominant_transform) / n_folds
    
    return {
        'dataset': dataset,
        'n_folds': n_folds,
        'center_mean': np.mean(centers),
        'center_std': np.std(centers),
        'center_cv': np.std(centers) / np.mean(centers) if np.mean(centers) > 0 else 0,
        'width_mean': np.mean(widths),
        'width_std': np.std(widths),
        'width_cv': np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0,
        'corr_mean': np.mean(correlations) if correlations else 0,
        'corr_std': np.std(correlations) if correlations else 0,
        'dominant_transform': dominant_transform,
        'transform_consistency': transform_consistency,
        'transforms': transforms,
        'first_patterns': first_patterns
    }

def analyze_remc(cell_line='E004'):
    with open(f'../json_files/remc/pattern_parameters_{cell_line}.json', 'r') as f:
        data = json.load(f)
    
    df = pd.read_parquet(f'../processed_datasets/remc/{cell_line}.parquet')
    y = df['target'].values
    X_list = [df[[c for c in df.columns if c.startswith(h + "_")]] for h in HISTONE_NAMES]
    
    fold_names = sorted(data.keys())
    n_folds = len(fold_names)
    
    best_patterns = []
    for fold_name in fold_names:
        patterns = data[fold_name]
        importances, best_idx = get_pattern_importance_remc(patterns, X_list, y)
        best_patterns.append(patterns[best_idx])
    
    centers = [p['center'] for p in best_patterns]
    widths = [p['width'] for p in best_patterns]
    transforms = [p['transform_type'] for p in best_patterns]
    histones = [HISTONE_NAMES[p['series_idx']] for p in best_patterns]
    
    correlations = []
    for i, j in combinations(range(n_folds), 2):
        p1 = generate_bspline_pattern(best_patterns[i]['control_points'], 50)
        p2 = generate_bspline_pattern(best_patterns[j]['control_points'], 50)
        rho, _ = spearmanr(p1, p2)
        correlations.append(rho)
    
    dominant_transform = max(set(transforms), key=transforms.count)
    dominant_histone = max(set(histones), key=histones.count)
    
    return {
        'dataset': f'REMC ({cell_line})',
        'n_folds': n_folds,
        'center_mean': np.mean(centers),
        'center_std': np.std(centers),
        'center_cv': np.std(centers) / np.mean(centers) if np.mean(centers) > 0 else 0,
        'width_mean': np.mean(widths),
        'width_std': np.std(widths),
        'width_cv': np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0,
        'corr_mean': np.mean(correlations) if correlations else 0,
        'corr_std': np.std(correlations) if correlations else 0,
        'dominant_transform': dominant_transform,
        'transform_consistency': transforms.count(dominant_transform) / n_folds,
        'dominant_histone': dominant_histone,
        'histone_consistency': histones.count(dominant_histone) / n_folds,
        'transforms': transforms,
        'histones': histones,
        'best_patterns': best_patterns
    }

def visualize_stability(results, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    patterns = results.get('best_patterns', results.get('first_patterns'))
    n_folds = len(patterns)
    for i, p in enumerate(patterns):
        width = int(round(p['width']))
        pattern_curve = generate_bspline_pattern(p['control_points'], width)
        x = np.linspace(0, 1, len(pattern_curve))
        label = f"Fold {i+1}: {p['transform_type']}"
        if 'series_idx' in p and 'REMC' in results['dataset']:
            label = f"Fold {i+1}: {HISTONE_NAMES[p['series_idx']][:6]} ({p['transform_type']})"
        ax1.plot(x, pattern_curve, color=COLORS['folds'][i % 5], linewidth=2.5, label=label)
    ax1.set_xlabel('Normalized Position', fontsize=12)
    ax1.set_ylabel('Pattern Value', fontsize=12)
    ax1.set_title('A. Most Important Pattern Shape per Fold', loc='left', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, frameon=True)
    ax1.grid(True, alpha=0.2, linestyle=':')
    centers = [p['center'] for p in patterns]
    widths = [p['width'] for p in patterns]
    folds = [f"Fold {i+1}" for i in range(n_folds)]
    ax2.barh(folds, widths, left=[c - w/2 for c, w in zip(centers, widths)], 
             color=[COLORS['folds'][i % 5] for i in range(n_folds)], alpha=0.7, height=0.6)
    ax2.set_xlabel('Position (samples/bins)', fontsize=12)
    ax2.set_title('B. Pattern Position & Width per Fold', loc='left', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='x', linestyle=':')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_latex_table(all_results):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Pattern Stability Analysis Across Datasets. For each dataset, we identify the most",
        r"important pattern per fold using LightGBM feature importance and analyze positional consistency",
        r"(center, width), transformation selection, and shape correlation across folds.",
        r"For REMC, we report mean $\pm$ std across all cell lines analyzed.}",
        r"\label{tab:pattern_stability}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Folds} & \textbf{Center CV} & \textbf{Width CV} & ",
        r"\textbf{Shape $\rho$} & \textbf{Transform} & \textbf{Consistency} \\",
        r"\midrule"
    ]
    
    for r in all_results:
        transform = r['dominant_transform'].replace('_', r'\_')
        consistency = f"{r['transform_consistency']*100:.0f}\\%"
        corr = f"{r['corr_mean']:.2f} $\\pm$ {r['corr_std']:.2f}" if r['corr_std'] > 0 else f"{r['corr_mean']:.2f}"
        
        # Format CVs with std if available
        center_cv_str = f"{r['center_cv']:.3f}"
        if 'center_cv_std' in r and r['center_cv_std'] > 0:
            center_cv_str = f"{r['center_cv']:.3f} $\\pm$ {r['center_cv_std']:.3f}"
        
        width_cv_str = f"{r['width_cv']:.3f}"
        if 'width_cv_std' in r and r['width_cv_std'] > 0:
            width_cv_str = f"{r['width_cv']:.3f} $\\pm$ {r['width_cv_std']:.3f}"
        
        line = f"{r['dataset']} & {r['n_folds']} & {center_cv_str} & {width_cv_str} & {corr} & {transform} & {consistency} \\\\"
        lines.append(line)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}"
    ])
    
    return '\n'.join(lines)

def main():
    import glob
    import os
    
    print("="*70)
    print("PATTERN STABILITY ANALYSIS")
    print("="*70)
    
    all_results = []
    
    # Analyze fold-based datasets
    for dataset in ['mitbih', 'emotions', 'mimic', 'pamap2', 'pancancer']:
        print(f"\n--- {dataset.upper()} ---")
        try:
            results = analyze_fold_dataset(dataset)
            all_results.append(results)
            print(f"Center: {results['center_mean']:.1f} ± {results['center_std']:.1f} (CV={results['center_cv']:.3f})")
            print(f"Width: {results['width_mean']:.1f} ± {results['width_std']:.1f} (CV={results['width_cv']:.3f})")
            print(f"Shape correlation: {results['corr_mean']:.3f} ± {results['corr_std']:.3f}")
            print(f"Dominant transform: {results['dominant_transform']} ({results['transform_consistency']*100:.0f}%)")
        except Exception as e:
            print(f"Error: {e}")
    
    # Analyze all REMC cell lines
    print(f"\n--- REMC (All Cell Lines) ---")
    remc_files = sorted(glob.glob('../json_files/remc/pattern_parameters_*.json'))
    remc_cell_lines = [os.path.basename(f).replace('pattern_parameters_', '').replace('.json', '') for f in remc_files]
    
    remc_all_cvs_center = []
    remc_all_cvs_width = []
    remc_all_corrs = []
    remc_all_transforms = []
    remc_all_histones = []
    
    for cell_line in remc_cell_lines:
        try:
            r = analyze_remc(cell_line)
            remc_all_cvs_center.append(r['center_cv'])
            remc_all_cvs_width.append(r['width_cv'])
            remc_all_corrs.append(r['corr_mean'])
            remc_all_transforms.extend(r['transforms'])
            remc_all_histones.extend(r['histones'])
        except Exception as e:
            print(f"Error with {cell_line}: {e}")
    
    # Aggregate statistics
    dominant_transform_remc = max(set(remc_all_transforms), key=remc_all_transforms.count)
    dominant_histone = max(set(remc_all_histones), key=remc_all_histones.count)
    
    remc_aggregate = {
        'dataset': 'REMC (mean)',
        'n_folds': 5,
        'center_mean': 0,
        'center_std': 0,
        'center_cv': np.mean(remc_all_cvs_center),
        'center_cv_std': np.std(remc_all_cvs_center),
        'width_mean': 0,
        'width_std': 0,
        'width_cv': np.mean(remc_all_cvs_width),
        'width_cv_std': np.std(remc_all_cvs_width),
        'corr_mean': np.mean(remc_all_corrs),
        'corr_std': np.std(remc_all_corrs),
        'dominant_transform': dominant_transform_remc,
        'transform_consistency': remc_all_transforms.count(dominant_transform_remc) / len(remc_all_transforms),
        'dominant_histone': dominant_histone,
        'histone_consistency': remc_all_histones.count(dominant_histone) / len(remc_all_histones)
    }
    all_results.append(remc_aggregate)
    
    print(f"Analyzed {len(remc_cell_lines)} cell lines")
    print(f"Mean Center CV: {remc_aggregate['center_cv']:.3f} ± {remc_aggregate['center_cv_std']:.3f}")
    print(f"Mean Width CV: {remc_aggregate['width_cv']:.3f} ± {remc_aggregate['width_cv_std']:.3f}")
    print(f"Mean Shape correlation: {remc_aggregate['corr_mean']:.3f} ± {remc_aggregate['corr_std']:.3f}")
    print(f"Dominant transform: {dominant_transform_remc} ({remc_aggregate['transform_consistency']*100:.0f}%)")
    print(f"Dominant histone: {dominant_histone} ({remc_aggregate['histone_consistency']*100:.0f}%)")
    
    # Analyze all AZT1D subjects (limited analysis as no fold structure)
    print(f"\n--- AZT1D (All Subjects) ---")
    azt1d_files = sorted(glob.glob('../json_files/azt1d/pattern_parameters_*.json'))
    azt1d_subjects = [os.path.basename(f).replace('pattern_parameters_', '').replace('.json', '') for f in azt1d_files]
    
    azt1d_n_patterns = []
    azt1d_transforms = []
    
    for subject in azt1d_subjects:
        try:
            with open(f'../json_files/azt1d/pattern_parameters_{subject}.json', 'r') as f:
                data = json.load(f)
            patterns = data if isinstance(data, list) else data.get('patterns', [])
            azt1d_n_patterns.append(len(patterns))
            azt1d_transforms.extend([p.get('transform_type', 'raw') for p in patterns])
        except Exception as e:
            print(f"Error with subject {subject}: {e}")
    
    dominant_transform_azt1d = max(set(azt1d_transforms), key=azt1d_transforms.count)
    
    print(f"Analyzed {len(azt1d_subjects)} subjects")
    print(f"Mean patterns per subject: {np.mean(azt1d_n_patterns):.1f} ± {np.std(azt1d_n_patterns):.1f}")
    print(f"Dominant transform: {dominant_transform_azt1d} ({azt1d_transforms.count(dominant_transform_azt1d)/len(azt1d_transforms)*100:.0f}%)")
    
    # Generate visualization for REMC E004 only
    remc_e004_results = analyze_remc('E004')
    visualize_stability(remc_e004_results, '../manuscript/images/pattern_stability.png')
    print("\nSaved pattern_stability.png (REMC E004)")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(all_results)
    with open('../manuscript/tables/pattern_stability_table.tex', 'w') as f:
        f.write(latex_table)
    print("Saved pattern_stability_table.tex")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in all_results:
        cv_str = f"{r['center_cv']:.3f}"
        if 'center_cv_std' in r and r['center_cv_std'] > 0:
            cv_str += f"±{r['center_cv_std']:.3f}"
        print(f"{r['dataset']:15} | Center CV: {cv_str} | Width CV: {r['width_cv']:.3f} | "
              f"ρ: {r['corr_mean']:+.2f} | {r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)")

if __name__ == "__main__":
    main()
