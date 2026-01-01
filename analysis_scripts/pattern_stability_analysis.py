import json
import numpy as np
import glob
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

HISTONE_NAMES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

AZT1D_CHANNELS = ['CGM', 'Insulin', 'Carbs']

MIMIC_CHANNELS = ['resp_rate', 'heart_rate', 'o2_sat', 'bp_dias', 'bp_sys', 'foley',
                  'bp_art_mean', 'temp', 'gcs_eye', 'gcs_verbal', 'gcs_motor', 'fio2',
                  'peep_set', 'norepinephrine', 'creatinine', 'platelets', 'wbc',
                  'ph_arterial', 'phenylephrine', 'pao2', 'inr', 'lactate', 'peep_total',
                  'bilirubin', 'dobutamine', 'epinephrine', 'dopamine', 'pao2_fio2', 'pulse_press']

PAMAP2_CHANNELS = ['heart_rate',
    'hand_temp', 'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z', 'hand_acc6_x', 'hand_acc6_y', 'hand_acc6_z',
    'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z', 'hand_mag_x', 'hand_mag_y', 'hand_mag_z',
    'hand_ori_x', 'hand_ori_y', 'hand_ori_z', 'hand_ori_w',
    'chest_temp', 'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z', 'chest_acc6_x', 'chest_acc6_y', 'chest_acc6_z',
    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z', 'chest_mag_x', 'chest_mag_y', 'chest_mag_z',
    'chest_ori_x', 'chest_ori_y', 'chest_ori_z', 'chest_ori_w',
    'ankle_temp', 'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z', 'ankle_acc6_x', 'ankle_acc6_y', 'ankle_acc6_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z', 'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'ankle_ori_x', 'ankle_ori_y', 'ankle_ori_z', 'ankle_ori_w']

VOWELS = ["a_n", "i_n", "u_n"]
SVD_CHANNELS = [f"{v}_mfcc{i}" for v in VOWELS for i in range(13)]

def generate_bspline_pattern(control_points, width, x_positions=None):
    cps = np.asarray(control_points)
    if len(cps) == 1:
        return np.full(int(round(width)), float(cps[0]))
    xs = np.asarray(x_positions) if x_positions is not None else np.linspace(0, 1, len(cps))
    k = min(3, len(cps) - 1)
    t = np.linspace(xs.min(), xs.max(), int(round(width)))
    return make_interp_spline(xs, cps, k=k)(t)

def plot_patterns(dataset_name, patterns_dict, top_only=False, channel_names=None):
    plt.figure(figsize=(10, 6))
    
    # Get all patterns
    all_patterns = []
    if isinstance(patterns_dict, dict):
        # Handle nested dict structure from json (fold_name -> patterns)
        for fold in patterns_dict:
            if isinstance(patterns_dict[fold], list):
                if top_only and len(patterns_dict[fold]) > 0:
                    all_patterns.append(patterns_dict[fold][0])
                else:
                    all_patterns.extend(patterns_dict[fold])
    elif isinstance(patterns_dict, list):
        # Handle list structure
        if top_only and len(patterns_dict) > 0:
            all_patterns = [patterns_dict[0]]
        else:
            all_patterns = patterns_dict
        
    annotations = []
    
    # Define a color cycle manually to ensure consistency
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(all_patterns))))
    
    for i, p in enumerate(all_patterns):
        width = p['width']
        start = p['start']
        cps = p['control_points']
        x_positions = p.get('x_positions')
        
        # Generate shape
        shape = generate_bspline_pattern(cps, width, x_positions=x_positions)
        
        # Create x-axis (position)
        x = np.linspace(start, start + width, len(shape))
        
        color = colors[i % len(colors)]
        plt.plot(x, shape, alpha=0.8, linewidth=2, color=color)
        
        # Collect annotation info
        transform = p['transform_type']
        s_idx = p['series_idx']
        if channel_names and s_idx < len(channel_names):
            c_name = channel_names[s_idx]
        else:
            c_name = f"Ch{s_idx}"
            
        annot_text = f"{c_name} ({transform})"
        
        # Place text near the start of the pattern with matching color
        # Adding a small random y-offset to avoid overlap if starts are identical
        y_pos = shape[0] + np.random.uniform(-0.1, 0.1) 
        plt.text(start, y_pos, annot_text, color=color, fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=color))

    # plt.title(f'Pattern Stability: {dataset_name.upper()}')
    plt.xlabel('Time')
    plt.ylabel('Pattern Value')
    plt.grid(True, alpha=0.3)
    
    # Save
    import os
    os.makedirs('../manuscript/images', exist_ok=True)
    plt.savefig(f'../manuscript/images/stability_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    
    transforms = [p['transform_type'] for p in first_patterns]
    dominant_transform = max(set(transforms), key=transforms.count)
    
    channels = [p['series_idx'] for p in first_patterns]
    dominant_channel_idx = max(set(channels), key=channels.count)
    if channel_names:
        dominant_channel = channel_names[dominant_channel_idx] if dominant_channel_idx < len(channel_names) else str(dominant_channel_idx)
    else:
        dominant_channel = str(dominant_channel_idx)
    
    n_patterns_list = [len(data[f]) for f in fold_names]
    
    # Single channel datasets: mark as N/A
    n_channels = len(set(channels))
    if n_channels == 1:
        channel_consistency = None
    else:
        channel_consistency = channels.count(dominant_channel_idx) / n_folds
    
    return {
        'dataset': dataset.upper(),
        'n_folds': n_folds,
        'n_patterns_mean': np.mean(n_patterns_list),
        'n_patterns_std': np.std(n_patterns_list),
        'dominant_transform': dominant_transform,
        'transform_consistency': transforms.count(dominant_transform) / n_folds,
        'dominant_channel': dominant_channel,
        'channel_consistency': channel_consistency,
    }

def analyze_remc(cell_line='E003'):
    with open(f'../json_files/remc/pattern_parameters_{cell_line}.json', 'r') as f:
        data = json.load(f)
    
    plot_patterns('remc', data, top_only=True, channel_names=HISTONE_NAMES)
    
    fold_names = sorted(data.keys())
    n_folds = len(fold_names)
    first_patterns = [data[f][0] for f in fold_names]
    
    transforms = [p['transform_type'] for p in first_patterns]
    histones = [HISTONE_NAMES[p['series_idx']] for p in first_patterns]
    
    dominant_transform = max(set(transforms), key=transforms.count)
    dominant_histone = max(set(histones), key=histones.count)
    
    n_patterns_list = [len(data[f]) for f in fold_names]
    
    return {
        'dataset': f'REMC ({cell_line})',
        'n_folds': n_folds,
        'n_patterns_mean': np.mean(n_patterns_list),
        'n_patterns_std': np.std(n_patterns_list),
        'dominant_transform': dominant_transform,
        'transform_consistency': transforms.count(dominant_transform) / n_folds,
        'dominant_channel': dominant_histone,
        'channel_consistency': histones.count(dominant_histone) / n_folds,
    }

def analyze_azt1d():
    files = sorted(glob.glob('../json_files/azt1d/pattern_parameters_*.json'))
    
    all_n_patterns = []
    all_transforms = []
    all_channels = []
    
    for f in files:
        with open(f, 'r') as fp:
            patterns = json.load(fp)
        all_n_patterns.append(len(patterns))
        if patterns:
            all_transforms.append(patterns[0]['transform_type'])
            all_channels.append(patterns[0]['series_idx'])
    
    dominant_transform = max(set(all_transforms), key=all_transforms.count) if all_transforms else 'raw'
    dominant_channel_idx = max(set(all_channels), key=all_channels.count) if all_channels else 0
    dominant_channel = AZT1D_CHANNELS[dominant_channel_idx]
    
    return {
        'dataset': 'AZT1D',
        'n_folds': len(files),
        'n_patterns_mean': np.mean(all_n_patterns),
        'n_patterns_std': np.std(all_n_patterns),
        'dominant_transform': dominant_transform,
        'transform_consistency': all_transforms.count(dominant_transform) / len(all_transforms) if all_transforms else 0,
        'dominant_channel': dominant_channel,
        'channel_consistency': all_channels.count(dominant_channel_idx) / len(all_channels) if all_channels else 0,
    }

def generate_latex_table(results):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Pattern discovery characteristics across datasets. For each dataset, we report",
        r"the number of selected patterns (mean $\pm$ std across folds/subjects), and the dominant",
        r"transformation and channel of the most important pattern with consistency percentages.",
        r"Single-channel datasets show ``---'' for channel consistency.}",
        r"\label{tab:pattern_stability}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Patterns} & \textbf{Transform (consistency)} & \textbf{Channel (consistency)} \\",
        r"\midrule"
    ]
    
    for r in results:
        transform = r['dominant_transform'].replace('_', r'\_')
        trans_combined = f"{transform} ({r['transform_consistency']*100:.0f}\\%)"
        
        patterns = f"{r['n_patterns_mean']:.1f} $\\pm$ {r['n_patterns_std']:.1f}"
        
        channel = str(r['dominant_channel']).replace('_', r'\_')
        if r['channel_consistency'] is None:
            chan_combined = f"{channel} (---)"
        else:
            chan_combined = f"{channel} ({r['channel_consistency']*100:.0f}\\%)"
        
        line = f"{r['dataset']} & {patterns} & {trans_combined} & {chan_combined} \\\\"
        lines.append(line)
    
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    return '\n'.join(lines)

def main():
    print("=" * 80)
    print("PATTERN STABILITY ANALYSIS")
    print("=" * 80)
    
    results = []
    
    def print_result(name, r):
        print(f"\n{name}")
        print(f"  Patterns: {r['n_patterns_mean']:.1f}±{r['n_patterns_std']:.1f}")
        print(f"  Transform: {r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)")
        ch_pct = "---" if r['channel_consistency'] is None else f"{r['channel_consistency']*100:.0f}%"
        print(f"  Channel: {r['dominant_channel']} ({ch_pct})")
    
    r = analyze_fold_dataset('mitbih', channel_names=['ECG'])
    results.append(r)
    print_result("MITBIH", r)
    
    r = analyze_fold_dataset('emotions', channel_names=['EEG'])
    results.append(r)
    print_result("EMOTIONS", r)
    
    r = analyze_fold_dataset('mimic', channel_names=MIMIC_CHANNELS)
    results.append(r)
    print_result("MIMIC", r)
    
    r = analyze_fold_dataset('pamap2', channel_names=PAMAP2_CHANNELS)
    results.append(r)
    print_result("PAMAP2", r)
    
    r = analyze_fold_dataset('svd', channel_names=SVD_CHANNELS)
    results.append(r)
    print_result("SVD", r)
    
    r = analyze_remc(cell_line='E003')
    results.append(r)
    print_result("REMC (E003)", r)
    
    r = analyze_azt1d()
    results.append(r)
    print_result("AZT1D", r)
    
    latex = generate_latex_table(results)
    with open('../manuscript/tables/pattern_stability_table.tex', 'w') as f:
        f.write(latex)
    print(f"\nSaved pattern_stability_table.tex")
    
    print("\n" + "=" * 80)
    print(f"{'Dataset':<15} {'Patterns':<15} {'Transform':<20} {'Channel':<20}")
    print("-" * 80)
    for r in results:
        pat = f"{r['n_patterns_mean']:.1f}±{r['n_patterns_std']:.1f}"
        trans = f"{r['dominant_transform']} ({r['transform_consistency']*100:.0f}%)"
        if r['channel_consistency'] is None:
            ch = f"{r['dominant_channel']} (---)"
        else:
            ch = f"{r['dominant_channel']} ({r['channel_consistency']*100:.0f}%)"
        print(f"{r['dataset']:<15} {pat:<15} {trans:<20} {ch:<20}")

if __name__ == "__main__":
    main()
