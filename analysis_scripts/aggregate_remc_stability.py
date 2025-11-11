import json
import numpy as np
import glob
from scipy.stats import spearmanr
from itertools import combinations

remc_files = glob.glob('../json_files/remc/pattern_parameters_*.json')
cell_lines = sorted([f.split('_')[-1].replace('.json', '') for f in remc_files])

all_centers_cv = []
all_widths_cv = []
all_correlations = []
all_scores = []

for cell_line in cell_lines:
    json_path = f'../json_files/remc/pattern_parameters_{cell_line}.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    first_patterns = {}
    for i in range(1, 6):
        fold_name = f'fold_{i}'
        if fold_name in data and len(data[fold_name]) > 0:
            first_patterns[fold_name] = data[fold_name][0]
    
    if len(first_patterns) < 2:
        continue
    
    centers = [first_patterns[f]['center'] for f in first_patterns]
    widths = [first_patterns[f]['width'] for f in first_patterns]
    scores = [first_patterns[f]['score'] for f in first_patterns]
    
    center_std = np.std(centers)
    center_mean = np.mean(centers)
    cv_center = center_std / center_mean if center_mean > 0 else 0
    
    width_std = np.std(widths)
    width_mean = np.mean(widths)
    cv_width = width_std / width_mean if width_mean > 0 else 0
    
    all_centers_cv.append(cv_center)
    all_widths_cv.append(cv_width)
    all_scores.extend(scores)
    
    fold_names = list(first_patterns.keys())
    for i, j in combinations(range(len(fold_names)), 2):
        p1 = np.array(first_patterns[fold_names[i]]['pattern'])
        p2 = np.array(first_patterns[fold_names[j]]['pattern'])
        if len(p1) >= len(p2):
            p1 = p1[:len(p2)]
        else:
            p2 = p2[:len(p1)]
        rho, _ = spearmanr(p1, p2)
        all_correlations.append(rho)

print("REMC Aggregate Statistics (across 46 cell lines):")
print(f"Center CV: median={np.median(all_centers_cv):.3f}, range=[{np.min(all_centers_cv):.3f}, {np.max(all_centers_cv):.3f}]")
print(f"Width CV: median={np.median(all_widths_cv):.3f}, range=[{np.min(all_widths_cv):.3f}, {np.max(all_widths_cv):.3f}]")
print(f"Spearman correlation: mean={np.mean(all_correlations):.3f}, median={np.median(all_correlations):.3f}, range=[{np.min(all_correlations):.3f}, {np.max(all_correlations):.3f}]")
print(f"Scores: mean={np.mean(all_scores):.3f}, std={np.std(all_scores):.3f}")

