import json
import numpy as np
import glob
import os
import sys
sys.path.append('../eval_scripts')
from core import generate_bspline_pattern
from scipy.stats import spearmanr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

HISTONE_NAMES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

def analyze_fold_dataset(dataset):
    json_path = f'../json_files/{dataset}/pattern_parameters.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fold_names = sorted(data.keys())
    n_folds = len(fold_names)
    first_patterns = [data[f][0] for f in fold_names]
    
    transforms = [p['transform_type'] for p in first_patterns]
    dominant_transform = max(set(transforms), key=transforms.count)
    
    correlations = []
    for i, j in combinations(range(n_folds), 2):
        p1 = generate_bspline_pattern(first_patterns[i]['control_points'], 50)
        p2 = generate_bspline_pattern(first_patterns[j]['control_points'], 50)
        rho, _ = spearmanr(p1, p2)
        correlations.append(rho)
    
    return {
        'dataset': dataset.upper(),
        'n_units': n_folds,
        'unit_type': 'folds',
        'n_patterns_mean': np.mean([len(data[f]) for f in fold_names]),
        'n_patterns_std': np.std([len(data[f]) for f in fold_names]),
        'dominant_transform': dominant_transform,
        'transform_consistency': transforms.count(dominant_transform) / n_folds,
        'shape_corr_mean': np.mean(correlations) if correlations else np.nan,
        'shape_corr_std': np.std(correlations) if correlations else np.nan
    }

def analyze_azt1d():
    files = sorted(glob.glob('../json_files/azt1d/pattern_parameters_*.json'))
    all_transforms = []
    n_patterns = []
    
    for f in files:
        with open(f, 'r') as fp:
            patterns = json.load(fp)
        n_patterns.append(len(patterns))
        all_transforms.extend([p['transform_type'] for p in patterns])
    
    dominant = max(set(all_transforms), key=all_transforms.count)
    
    return {
        'dataset': 'AZT1D',
        'n_units': len(files),
        'unit_type': 'subjects',
        'n_patterns_mean': np.mean(n_patterns),
        'n_patterns_std': np.std(n_patterns),
        'dominant_transform': dominant,
        'transform_consistency': all_transforms.count(dominant) / len(all_transforms),
        'shape_corr_mean': np.nan,
        'shape_corr_std': np.nan
    }

def analyze_remc():
    files = sorted(glob.glob('../json_files/remc/pattern_parameters_*.json'))
    all_transforms = []
    all_histones = []
    n_patterns = []
    
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        for fold_name in data.keys():
            patterns = data[fold_name]
            n_patterns.append(len(patterns))
            all_transforms.extend([p['transform_type'] for p in patterns])
            all_histones.extend([HISTONE_NAMES[p['series_idx']] for p in patterns])
    
    dominant_transform = max(set(all_transforms), key=all_transforms.count)
    dominant_histone = max(set(all_histones), key=all_histones.count)
    
    return {
        'dataset': 'REMC',
        'n_units': len(files),
        'unit_type': 'cell lines',
        'n_patterns_mean': np.mean(n_patterns),
        'n_patterns_std': np.std(n_patterns),
        'dominant_transform': dominant_transform,
        'transform_consistency': all_transforms.count(dominant_transform) / len(all_transforms),
        'dominant_histone': dominant_histone,
        'histone_consistency': all_histones.count(dominant_histone) / len(all_histones),
        'shape_corr_mean': np.nan,
        'shape_corr_std': np.nan
    }

def generate_latex_table(results):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Pattern discovery characteristics across datasets. We report the number of",
        r"patterns discovered, dominant signal transformation, and transformation consistency",
        r"(fraction of patterns using the dominant transform). Shape correlation measures",
        r"similarity of primary patterns across cross-validation folds.}",
        r"\label{tab:pattern_stability}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Units} & \textbf{Patterns} & \textbf{Transform} & ",
        r"\textbf{Consistency} & \textbf{Shape $\rho$} \\",
        r"\midrule"
    ]
    
    for r in results:
        transform = r['dominant_transform'].replace('_', r'\_')
        consistency = f"{r['transform_consistency']*100:.0f}\\%"
        patterns = f"{r['n_patterns_mean']:.1f} $\\pm$ {r['n_patterns_std']:.1f}"
        units = f"{r['n_units']} {r['unit_type']}"
        
        if np.isnan(r['shape_corr_mean']):
            corr = "---"
        else:
            corr = f"{r['shape_corr_mean']:.2f} $\\pm$ {r['shape_corr_std']:.2f}"
        
        line = f"{r['dataset']} & {units} & {patterns} & {transform} & {consistency} & {corr} \\\\"
        lines.append(line)
    
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    return '\n'.join(lines)

def main():
    print("=" * 70)
    print("PATTERN STABILITY ANALYSIS")
    print("=" * 70)
    
    results = []
    
    for dataset in ['mitbih', 'emotions', 'mimic', 'pamap2']:
        print(f"\n{dataset.upper()}")
        try:
            r = analyze_fold_dataset(dataset)
            results.append(r)
            print(f"  {r['n_units']} {r['unit_type']}, {r['n_patterns_mean']:.1f}±{r['n_patterns_std']:.1f} patterns")
            print(f"  Transform: {r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)")
            if not np.isnan(r['shape_corr_mean']):
                print(f"  Shape ρ: {r['shape_corr_mean']:.2f}±{r['shape_corr_std']:.2f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nAZT1D")
    r = analyze_azt1d()
    results.append(r)
    print(f"  {r['n_units']} {r['unit_type']}, {r['n_patterns_mean']:.1f}±{r['n_patterns_std']:.1f} patterns")
    print(f"  Transform: {r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)")
    
    print(f"\nREMC")
    r = analyze_remc()
    results.append(r)
    print(f"  {r['n_units']} {r['unit_type']}, {r['n_patterns_mean']:.1f}±{r['n_patterns_std']:.1f} patterns")
    print(f"  Transform: {r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)")
    print(f"  Histone: {r['dominant_histone']} ({r['histone_consistency']*100:.0f}%)")
    
    latex = generate_latex_table(results)
    with open('../manuscript/tables/pattern_stability_table.tex', 'w') as f:
        f.write(latex)
    print(f"\nSaved pattern_stability_table.tex")
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Dataset':<10} {'Units':<15} {'Patterns':<15} {'Transform':<12} {'Consist.':<10}")
    print("-" * 70)
    for r in results:
        pat_str = f"{r['n_patterns_mean']:.1f}±{r['n_patterns_std']:.1f}"
        unit_str = f"{r['n_units']} {r['unit_type']}"
        print(f"{r['dataset']:<10} {unit_str:<15} {pat_str:<15} {r['dominant_transform']:<12} {r['transform_consistency']*100:.0f}%")

if __name__ == "__main__":
    main()
