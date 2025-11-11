"""
Pattern Stability Analysis across Cross-Validation Folds

Analyzes and visualizes the first pattern discovered across different CV folds
to assess pattern reproducibility and positional consistency.

Usage:
    python pattern_stability_analysis.py                              # default: mitbih
    python pattern_stability_analysis.py --dataset=mitbih
    python pattern_stability_analysis.py --dataset=emotions
    python pattern_stability_analysis.py --dataset=mimic
    python pattern_stability_analysis.py --dataset=azt1d              # picks first subject
    python pattern_stability_analysis.py --dataset=azt1d --subject=540
    python pattern_stability_analysis.py --dataset=remc               # picks first cell line
    python pattern_stability_analysis.py --dataset=remc --cell_line=E003
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from itertools import combinations
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mitbih', 
                    choices=['mitbih', 'emotions', 'mimic', 'azt1d', 'remc'])
parser.add_argument('--subject', type=str, default=None, 
                    help='Subject ID for azt1d dataset')
parser.add_argument('--cell_line', type=str, default=None, 
                    help='Cell line for remc dataset')
args = parser.parse_args()

dataset = args.dataset
fold_configs = {
    'mitbih': 5,
    'emotions': 5,
    'mimic': 3
}

if dataset in ['mitbih', 'emotions', 'mimic']:
    json_path = f'../json_files/{dataset}/pattern_parameters.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    n_folds = fold_configs[dataset]
    first_patterns = {}
    for i in range(1, n_folds + 1):
        fold_name = f'fold_{i}'
        first_patterns[fold_name] = data[fold_name][0]
    
elif dataset == 'azt1d':
    if args.subject is None:
        files = glob.glob('../json_files/azt1d/pattern_parameters_*.json')
        if not files:
            raise ValueError("No azt1d files found")
        args.subject = files[0].split('_')[-1].replace('.json', '')
    json_path = f'../json_files/azt1d/pattern_parameters_{args.subject}.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    first_patterns = {'pattern_1': data[0]} if isinstance(data, list) else data
    
elif dataset == 'remc':
    if args.cell_line is None:
        files = glob.glob('../json_files/remc/pattern_parameters_*.json')
        if not files:
            raise ValueError("No remc files found")
        args.cell_line = files[0].split('_')[-1].replace('.json', '')
    json_path = f'../json_files/remc/pattern_parameters_{args.cell_line}.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    n_folds = 5
    first_patterns = {}
    for i in range(1, n_folds + 1):
        fold_name = f'fold_{i}'
        first_patterns[fold_name] = data[fold_name][0]

centers = [first_patterns[f]['center'] for f in first_patterns]
widths = [first_patterns[f]['width'] for f in first_patterns]
transforms = [first_patterns[f]['transform_type'] for f in first_patterns]
use_relatives = [first_patterns[f]['use_relative'] for f in first_patterns]
scores = [first_patterns[f]['score'] for f in first_patterns]
starts = [first_patterns[f]['start'] for f in first_patterns]

center_mean, center_std = np.mean(centers), np.std(centers)
width_mean, width_std = np.mean(widths), np.std(widths)
cv_center = center_std / center_mean if center_mean > 0 else 0
cv_width = width_std / width_mean if width_mean > 0 else 0

print(f"\n{'='*60}")
print(f"Dataset: {dataset.upper()}")
if dataset == 'azt1d':
    print(f"Subject: {args.subject}")
elif dataset == 'remc':
    print(f"Cell line: {args.cell_line}")
print(f"{'='*60}")
print("\nPattern Stability Statistics:")
print(f"Center: {center_mean:.1f} ± {center_std:.1f} samples (CV={cv_center:.3f})")
print(f"Width: {width_mean:.1f} ± {width_std:.1f} samples (CV={cv_width:.3f})")
print(f"\nTransformations: {transforms}")
print(f"Use relative: {use_relatives}")
print(f"Scores: {[f'{s:.4f}' for s in scores]}")

fold_names = list(first_patterns.keys())
if len(fold_names) > 1:
    correlations = []
    for i, j in combinations(range(len(fold_names)), 2):
        p1 = np.array(first_patterns[fold_names[i]]['pattern'])
        p2 = np.array(first_patterns[fold_names[j]]['pattern'])
        if len(p1) >= len(p2):
            p1 = p1[:len(p2)]
        else:
            p2 = p2[:len(p1)]
        rho, _ = spearmanr(p1, p2)
        correlations.append(rho)
        print(f"\n{fold_names[i]} vs {fold_names[j]}: ρ = {rho:.3f}")
    
    print(f"\nSpearman correlation: mean={np.mean(correlations):.3f}, "
          f"median={np.median(correlations):.3f}, "
          f"range=[{np.min(correlations):.3f}, {np.max(correlations):.3f}]")
else:
    print("\nSingle pattern - no correlation analysis")

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
fold_labels = [f'Fold {i+1}' if 'fold' in k else f'Pattern {i+1}' 
               for i, k in enumerate(fold_names)]

max_x = 0
for idx, (fold_name, label, color) in enumerate(zip(fold_names, fold_labels, colors)):
    pattern_data = first_patterns[fold_name]
    pattern = np.array(pattern_data['pattern'])
    start = pattern_data['start']
    center = pattern_data['center']
    width = pattern_data['width']
    transform = pattern_data['transform_type']
    use_rel = pattern_data['use_relative']
    
    x = np.arange(len(pattern)) + start
    search_start = center - width
    search_end = center + width
    max_x = max(max_x, x[-1], search_end)
    
    ax.plot(x, pattern, color=color, linewidth=2.5,
            label=f'{label}: {transform} (rel={use_rel})')
    ax.axvline(search_start, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(search_end, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

xlim_max = int(np.ceil(max_x / 10) * 10)
ax.set_xlim(0, xlim_max)
ax.set_xlabel('Sample Index', fontsize=13)
ax.set_ylabel('Pattern Value', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='best', framealpha=0.95)

plt.tight_layout()
if dataset == 'mitbih':
    output_path = '../manuscript/images/pattern_stability.png'
else:
    output_name = f'pattern_stability_{dataset}'
    if dataset == 'azt1d':
        output_name += f'_{args.subject}'
    elif dataset == 'remc':
        output_name += f'_{args.cell_line}'
    output_path = f'../manuscript/images/{output_name}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved: {output_path}")

