import json
import numpy as np
import glob
import os
import sys
sys.path.append('../eval_scripts')
from core import generate_bspline_pattern
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

HISTONE_NAMES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

def compute_overlap(patterns):
    if len(patterns) < 2:
        return np.nan, np.nan
    overlaps = []
    for i, j in combinations(range(len(patterns)), 2):
        p1, p2 = patterns[i], patterns[j]
        s1, e1 = p1['start'], p1['start'] + p1['width']
        s2, e2 = p2['start'], p2['start'] + p2['width']
        overlap = max(0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        overlaps.append(overlap / union if union > 0 else 0)
    return np.mean(overlaps), np.std(overlaps)

def analyze_fold_dataset(dataset, channel_names=None):
    json_path = f'../json_files/{dataset}/pattern_parameters.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fold_names = sorted(data.keys())
    n_folds = len(fold_names)
    first_patterns = [data[f][0] for f in fold_names]
    all_patterns = [p for f in fold_names for p in data[f]]
    
    transforms = [p['transform_type'] for p in first_patterns]
    dominant_transform = max(set(transforms), key=transforms.count)
    
    overlap_mean, overlap_std = compute_overlap(first_patterns)
    
    result = {
        'dataset': dataset.upper(),
        'n_folds': n_folds,
        'n_patterns_mean': np.mean([len(data[f]) for f in fold_names]),
        'n_patterns_std': np.std([len(data[f]) for f in fold_names]),
        'dominant_transform': dominant_transform,
        'transform_consistency': transforms.count(dominant_transform) / n_folds,
        'overlap_mean': overlap_mean,
        'overlap_std': overlap_std
    }
    
    if channel_names:
        channels = [channel_names[p['series_idx']] for p in first_patterns]
        dominant_channel = max(set(channels), key=channels.count)
        result['dominant_channel'] = dominant_channel
        result['channel_consistency'] = channels.count(dominant_channel) / n_folds
    
    return result

def analyze_remc():
    files = sorted(glob.glob('../json_files/remc/pattern_parameters_*.json'))
    first_cell = files[0]
    with open(first_cell, 'r') as f:
        data = json.load(f)
    
    fold_names = sorted(data.keys())
    n_folds = len(fold_names)
    first_patterns = [data[f][0] for f in fold_names]
    
    transforms = [p['transform_type'] for p in first_patterns]
    histones = [HISTONE_NAMES[p['series_idx']] for p in first_patterns]
    
    dominant_transform = max(set(transforms), key=transforms.count)
    dominant_histone = max(set(histones), key=histones.count)
    
    overlap_mean, overlap_std = compute_overlap(first_patterns)
    
    return {
        'dataset': 'REMC',
        'n_folds': n_folds,
        'n_patterns_mean': np.mean([len(data[f]) for f in fold_names]),
        'n_patterns_std': np.std([len(data[f]) for f in fold_names]),
        'dominant_transform': dominant_transform,
        'transform_consistency': transforms.count(dominant_transform) / n_folds,
        'dominant_channel': dominant_histone,
        'channel_consistency': histones.count(dominant_histone) / n_folds,
        'overlap_mean': overlap_mean,
        'overlap_std': overlap_std
    }

def analyze_azt1d():
    files = sorted(glob.glob('../json_files/azt1d/pattern_parameters_*.json'))
    first_subject = files[0]
    with open(first_subject, 'r') as f:
        patterns = json.load(f)
    
    transforms = [p['transform_type'] for p in patterns]
    dominant = max(set(transforms), key=transforms.count)
    
    overlap_mean, overlap_std = compute_overlap(patterns)
    
    return {
        'dataset': 'AZT1D',
        'n_folds': 1,
        'n_patterns_mean': len(patterns),
        'n_patterns_std': 0,
        'dominant_transform': dominant,
        'transform_consistency': transforms.count(dominant) / len(transforms),
        'overlap_mean': overlap_mean,
        'overlap_std': overlap_std
    }

def generate_latex_table(results):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Pattern discovery characteristics across datasets. We report the dominant",
        r"signal transformation and its consistency across folds, channel consistency for",
        r"multi-channel datasets, and pattern area overlap (IoU) between primary patterns.}",
        r"\label{tab:pattern_stability}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Patterns} & \textbf{Transform} & \textbf{Trans. \%} & ",
        r"\textbf{Channel} & \textbf{Chan. \%} & \textbf{Overlap} \\",
        r"\midrule"
    ]
    
    for r in results:
        transform = r['dominant_transform'].replace('_', r'\_')
        trans_pct = f"{r['transform_consistency']*100:.0f}\\%"
        patterns = f"{r['n_patterns_mean']:.1f}"
        if r['n_patterns_std'] > 0:
            patterns += f" $\\pm$ {r['n_patterns_std']:.1f}"
        
        if 'dominant_channel' in r:
            channel = r['dominant_channel'].replace('_', r'\_')
            chan_pct = f"{r['channel_consistency']*100:.0f}\\%"
        else:
            channel = "---"
            chan_pct = "---"
        
        if np.isnan(r['overlap_mean']):
            overlap = "---"
        else:
            overlap = f"{r['overlap_mean']:.2f}"
            if r['overlap_std'] > 0:
                overlap += f" $\\pm$ {r['overlap_std']:.2f}"
        
        line = f"{r['dataset']} & {patterns} & {transform} & {trans_pct} & {channel} & {chan_pct} & {overlap} \\\\"
        lines.append(line)
    
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    return '\n'.join(lines)

def main():
    print("=" * 80)
    print("PATTERN STABILITY ANALYSIS")
    print("=" * 80)
    
    results = []
    
    for dataset in ['mitbih', 'emotions', 'mimic', 'pamap2']:
        print(f"\n{dataset.upper()}")
        r = analyze_fold_dataset(dataset)
        results.append(r)
        print(f"  Patterns: {r['n_patterns_mean']:.1f}±{r['n_patterns_std']:.1f}")
        print(f"  Transform: {r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)")
        print(f"  Overlap: {r['overlap_mean']:.2f}±{r['overlap_std']:.2f}")
    
    print(f"\nREMC")
    r = analyze_remc()
    results.append(r)
    print(f"  Patterns: {r['n_patterns_mean']:.1f}±{r['n_patterns_std']:.1f}")
    print(f"  Transform: {r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)")
    print(f"  Channel: {r['dominant_channel']} ({r['channel_consistency']*100:.0f}%)")
    print(f"  Overlap: {r['overlap_mean']:.2f}±{r['overlap_std']:.2f}")
    
    print(f"\nAZT1D")
    r = analyze_azt1d()
    results.append(r)
    print(f"  Patterns: {r['n_patterns_mean']:.0f}")
    print(f"  Transform: {r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)")
    print(f"  Overlap: {r['overlap_mean']:.2f}±{r['overlap_std']:.2f}")
    
    latex = generate_latex_table(results)
    with open('../manuscript/tables/pattern_stability_table.tex', 'w') as f:
        f.write(latex)
    print(f"\nSaved pattern_stability_table.tex")
    
    print("\n" + "=" * 80)
    print(f"{'Dataset':<10} {'Patterns':<12} {'Transform':<12} {'T%':<6} {'Channel':<10} {'C%':<6} {'Overlap':<10}")
    print("-" * 80)
    for r in results:
        pat = f"{r['n_patterns_mean']:.1f}"
        ch = r.get('dominant_channel', '---')
        ch_pct = f"{r.get('channel_consistency', 0)*100:.0f}%" if 'dominant_channel' in r else "---"
        ovl = f"{r['overlap_mean']:.2f}" if not np.isnan(r['overlap_mean']) else "---"
        print(f"{r['dataset']:<10} {pat:<12} {r['dominant_transform']:<12} {r['transform_consistency']*100:.0f}%   {ch:<10} {ch_pct:<6} {ovl:<10}")

if __name__ == "__main__":
    main()
