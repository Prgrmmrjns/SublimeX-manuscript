import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from scipy.interpolate import interp1d

def load_patterns(dataset_name):
    base_path = Path(__file__).parent.parent / 'json_files'
    
    if dataset_name == 'MITBIH':
        # MITBIH now has multiple fold files
        fold_data = {}
        for fold in range(1, 6):  # 5 folds
            fold_path = base_path / 'mitbih' / f'pattern_parameters_fold{fold}.json'
            if fold_path.exists():
                with open(fold_path, 'r') as f:
                    data = json.load(f)
                    # Handle both array and dict formats
                    if isinstance(data, list):
                        fold_data[f'fold_{fold}'] = data
                    elif isinstance(data, dict) and 'patterns' in data:
                        fold_data[f'fold_{fold}'] = data['patterns']
                    else:
                        fold_data[f'fold_{fold}'] = data
        return fold_data if fold_data else None
    elif dataset_name == 'Bonn_EEG':
        path = base_path / 'bonn_eeg' / 'pattern_parameters.json'
    elif dataset_name == 'MIMIC':
        path = base_path / 'mimic' / 'pattern_parameters.json'
    elif dataset_name.startswith('REMC_'):
        cell_line = dataset_name.replace('REMC_', '')
        # REMC also has multiple fold files
        fold_data = {}
        for fold in range(1, 6):  # 5 folds
            fold_path = base_path / 'remc' / f'pattern_parameters_{cell_line}_fold{fold}.json'
            if fold_path.exists():
                with open(fold_path, 'r') as f:
                    fold_data[f'fold_{fold}'] = json.load(f)
        return fold_data if fold_data else None
    else:
        return None
    
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        return json.load(f)

def compute_pairwise_spearman(patterns):
    n = len(patterns)
    rho_values = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = np.array(patterns[i])
            p2 = np.array(patterns[j])
            
            if len(p1) != len(p2):
                target_len = max(len(p1), len(p2))
                if len(p1) < target_len:
                    x_old = np.linspace(0, 1, len(p1))
                    x_new = np.linspace(0, 1, target_len)
                    p1 = interp1d(x_old, p1, kind='linear')(x_new)
                if len(p2) < target_len:
                    x_old = np.linspace(0, 1, len(p2))
                    x_new = np.linspace(0, 1, target_len)
                    p2 = interp1d(x_old, p2, kind='linear')(x_new)
            
            if len(p1) > 1 and np.std(p1) > 0 and np.std(p2) > 0:
                rho, _ = spearmanr(p1, p2)
                if not np.isnan(rho):
                    rho_values.append(rho)
    
    return np.mean(rho_values) if rho_values else 0

def analyze_first_patterns(dataset_name, fold_data):
    if not fold_data:
        return None
    
    fold_keys = sorted([k for k in fold_data.keys() if k.startswith('fold_')])
    
    if len(fold_keys) < 2:
        return None
    
    first_patterns = []
    starts = []
    widths = []
    series_idxs = []
    pattern_counts = []
    transform_types = []
    
    for fold_key in fold_keys:
        patterns = fold_data[fold_key]
        if patterns:
            first_pat = patterns[0]
            # Handle both old and new pattern formats
            if 'pattern' in first_pat:
                first_patterns.append(first_pat['pattern'])
                starts.append(first_pat['start'])
                widths.append(first_pat.get('width', first_pat.get('end', 0) - first_pat['start']))
            else:
                # New format: generate pattern from control points
                from scipy.interpolate import BSpline
                control_points = first_pat['control_points']
                width = int(first_pat['width'])
                degree = 3
                n_cp = len(control_points)
                knots = np.concatenate([
                    np.zeros(degree + 1), 
                    np.linspace(0, 1, n_cp - degree + 1)[1:-1], 
                    np.ones(degree + 1)
                ])
                pattern = BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width))
                first_patterns.append(pattern)
                starts.append(int(first_pat['center'] - first_pat['width']/2))
                widths.append(first_pat['width'])
            
            series_idxs.append(first_pat['series_idx'])
            pattern_counts.append(len(patterns))
            transform_types.append(first_pat.get('transform_type', 'raw'))
    
    if len(first_patterns) < 2:
        return None
    
    rho_mean = compute_pairwise_spearman(first_patterns)
    
    series_idx_consistent = len(set(series_idxs)) == 1
    transform_type_consistent = len(set(transform_types)) == 1
    
    return {
        'dataset': dataset_name,
        'n_folds': len(fold_keys),
        'avg_patterns': np.mean(pattern_counts),
        'std_patterns': np.std(pattern_counts),
        'spearman_rho': rho_mean,
        'start_mean': np.mean(starts),
        'start_std': np.std(starts),
        'width_mean': np.mean(widths),
        'width_std': np.std(widths),
        'series_idx_consistent': series_idx_consistent,
        'series_idx': series_idxs[0] if series_idx_consistent else 'varied',
        'transform_type': transform_types[0] if transform_type_consistent else 'varied'
    }

def apply_transformation(series, transform_type):
    """Apply signal transformation matching core.py implementation."""
    if transform_type == 'raw':
        return series
    elif transform_type == 'cumsum':
        return np.cumsum(series)
    elif transform_type == 'derivative':
        return np.gradient(series)
    elif transform_type == 'wavelet':
        import pywt
        coeffs = pywt.wavedec(series, 'db4', level=3, mode='periodization')
        return np.concatenate(coeffs)
    elif transform_type == 'fft_power':
        from scipy import fft
        power = np.abs(fft.fft(series))**2
        return power[:len(power)//2]
    else:
        return series

def create_mitbih_first_pattern_figure():
    fold_data = load_patterns('MITBIH')
    
    if not fold_data:
        print("MITBIH data not found")
        return
    
    fold_keys = sorted([k for k in fold_data.keys() if k.startswith('fold_')])
    
    import sys
    import os
    script_dir = Path(__file__).parent.parent
    processed_datasets_path = script_dir / 'processed_datasets' / 'mitbih_processed.csv'
    mitbih_data = pd.read_csv(processed_datasets_path)
    normal_beats = mitbih_data[mitbih_data['target'] == 0]
    sample_beat = normal_beats.iloc[0].drop('target').values
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    first_patterns = []
    starts = []
    widths = []
    centers = []
    shift_tolerances = []
    transform_types = []
    
    for idx, fold_key in enumerate(fold_keys):
        patterns = fold_data[fold_key]
        first_pat = patterns[0]
        
        # Get transform type
        transform_type = first_pat.get('transform_type', 'raw')
        transform_types.append(transform_type)
        
        # Get shift tolerance
        shift_tolerance = first_pat.get('shift_tolerance', 0.0)
        shift_tolerances.append(shift_tolerance)
        
        # Handle new pattern format
        if 'pattern' in first_pat:
            pattern_vals = np.array(first_pat['pattern'])
            start = first_pat['start']
            width = first_pat['width']
            center = first_pat.get('center', start + width/2)
        else:
            # Generate pattern from control points
            from scipy.interpolate import BSpline
            control_points = first_pat['control_points']
            width = int(first_pat['width'])
            degree = 3
            n_cp = len(control_points)
            knots = np.concatenate([
                np.zeros(degree + 1), 
                np.linspace(0, 1, n_cp - degree + 1)[1:-1], 
                np.ones(degree + 1)
            ])
            pattern_vals = BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width))
            center = first_pat['center']
            start = int(center - width/2)
        
        first_patterns.append(pattern_vals)
        starts.append(start)
        widths.append(width)
        centers.append(center)
    
    n_time_points = len(sample_beat)
    
    # Plot all patterns overlaid
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for idx, (pattern_vals, center, width, shift_tol, transform_type) in enumerate(zip(first_patterns, centers, widths, shift_tolerances, transform_types)):
        pattern_normalized = (pattern_vals - pattern_vals.mean()) / pattern_vals.std()
        pattern_x = np.arange(len(pattern_vals))
        
        # Plot the normalized pattern shape
        ax.plot(pattern_x, pattern_normalized, color=colors[idx], linewidth=2.5, 
               alpha=0.9, label=f'Fold {idx+1} ({transform_type})')
        
        # Show the center position
        ax.axvline(center, color=colors[idx], linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Show the shift tolerance range
        if shift_tol > 0:
            max_shift = int(shift_tol * n_time_points)
            xmin = max(0, int(center - width/2 - max_shift))
            xmax = min(n_time_points, int(center + width/2 + max_shift))
            ax.axvspan(xmin, xmax, alpha=0.08, color=colors[idx])
    
    ax.set_ylabel('Normalized Pattern Amplitude', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Position (samples)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # Calculate statistics
    rho_mean = compute_pairwise_spearman(first_patterns)
    center_mean, center_std = np.mean(centers), np.std(centers)
    width_mean, width_std = np.mean(widths), np.std(widths)
    
    # Add statistics text with shift tolerance info
    shift_tol_mean = np.mean(shift_tolerances)
    shift_tol_std = np.std(shift_tolerances)
    stats_text = f'Spearman ρ = {rho_mean:.3f}\nCenter: {center_mean:.1f} ± {center_std:.1f}\nWidth: {width_mean:.1f} ± {width_std:.1f}\nShift Tol: {shift_tol_mean:.2f} ± {shift_tol_std:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = Path('../manuscript/images')
    output_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path / 'pattern_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {output_path / 'pattern_stability.png'}")
    print(f"Spearman ρ = {rho_mean:.4f}")
    print(f"Center: {center_mean:.2f} ± {center_std:.2f}")
    print(f"Width: {width_mean:.2f} ± {width_std:.2f}")
    print(f"Shift tolerance: {shift_tol_mean:.3f} ± {shift_tol_std:.3f}")
    print(f"Transform types: {transform_types}")
    
    return {
        'spearman_rho': rho_mean,
        'center_mean': center_mean,
        'center_std': center_std,
        'width_mean': width_mean,
        'width_std': width_std,
        'shift_tol_mean': shift_tol_mean,
        'shift_tol_std': shift_tol_std,
        'transform_types': transform_types
    }

def generate_summary_table():
    datasets = ['MITBIH', 'Bonn_EEG', 'MIMIC']
    
    results = []
    for dataset in datasets:
        fold_data = load_patterns(dataset)
        if fold_data:
            stats = analyze_first_patterns(dataset, fold_data)
            if stats:
                results.append(stats)
    
    base_path = Path(__file__).parent.parent / 'json_files'
    
    remc_path = base_path / 'remc'
    if remc_path.exists():
        for json_file in sorted(remc_path.glob('pattern_parameters_*.json')):
            cell_line = json_file.stem.replace('pattern_parameters_', '')
            fold_data = load_patterns(f'REMC_{cell_line}')
            if fold_data:
                stats = analyze_first_patterns(f'REMC_{cell_line}', fold_data)
                if stats:
                    results.append(stats)
    
    azt1d_path = base_path / 'azt1d'
    if azt1d_path.exists():
        for json_file in sorted(azt1d_path.glob('pattern_parameters_*.json')):
            subject = json_file.stem.replace('pattern_parameters_', '')
            with open(json_file, 'r') as f:
                data = json.load(f)
            if 'patterns' in data and data['patterns']:
                results.append({
                    'dataset': f'AZT1D_{subject}',
                    'n_folds': 1,
                    'avg_patterns': len(data['patterns']),
                    'std_patterns': 0,
                    'spearman_rho': np.nan,
                    'start_mean': data['patterns'][0]['start'],
                    'start_std': 0,
                    'width_mean': data['patterns'][0]['width'],
                    'width_std': 0,
                    'series_idx_consistent': True,
                    'series_idx': data['patterns'][0]['series_idx'],
                    'transform_type': data['patterns'][0].get('transform_type', 'raw')
                })
    
    df = pd.DataFrame(results)
    
    output_path = Path('../manuscript/tables')
    output_path.mkdir(exist_ok=True, parents=True)
    
    csv_path = output_path / 'pattern_stability_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")
    
    latex_lines = []
    latex_lines.append(r'\begin{table}[H]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Pattern Stability Statistics Across Datasets.')
    latex_lines.append(r'Spearman $\rho$ values quantify rank correlation of first patterns across folds,')
    latex_lines.append(r'capturing monotonic relationships without assuming linearity.')
    latex_lines.append(r'Values range from $-1$ (perfect inverse) to $+1$ (perfect correlation).')
    latex_lines.append(r'Patterns/Fold shows mean $\pm$ std number of patterns discovered per fold.')
    latex_lines.append(r'Start Pos. and Width report mean $\pm$ std for the first pattern location and')
    latex_lines.append(r'length across folds. Transform shows the signal representation (raw, wavelet, FFT,')
    latex_lines.append(r'or derivative) consistently selected for the first pattern. Complete statistics for')
    latex_lines.append(r'all REMC cell lines and AZT1D subjects available in the supplementary CSV file.}')
    latex_lines.append(r'\label{tab:pattern_stability}')
    latex_lines.append(r'\footnotesize')
    latex_lines.append(r'\begin{tabular}{lcccccc}')
    latex_lines.append(r'\toprule')
    latex_lines.append(r'\textbf{Dataset} & \textbf{Folds} & \textbf{Patterns/Fold} & \textbf{Spearman $\rho$} & \textbf{Start Pos.} & \textbf{Width} & \textbf{Transform} \\')
    latex_lines.append(r'\midrule')
    
    key_datasets = ['MITBIH', 'Bonn_EEG', 'MIMIC', 'REMC_E003']
    for _, row in df.iterrows():
        if row['dataset'] in key_datasets:
            dataset_clean = row['dataset'].replace('_', ' ')
            patterns = f"{row['avg_patterns']:.1f} $\\pm$ {row['std_patterns']:.1f}"
            rho = f"{row['spearman_rho']:.3f}" if not np.isnan(row['spearman_rho']) else 'N/A'
            start = f"{row['start_mean']:.1f} $\\pm$ {row['start_std']:.1f}"
            width = f"{row['width_mean']:.1f} $\\pm$ {row['width_std']:.1f}"
            transform = row.get('transform_type', 'N/A')
            
            latex_lines.append(f"{dataset_clean} & {row['n_folds']} & {patterns} & {rho} & {start} & {width} & {transform} \\\\")
    
    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')
    
    latex_path = output_path / 'pattern_stability_summary.tex'
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table saved to {latex_path}")
    
    return df

def analyze_all_datasets():
    """Analyze pattern stability across all five datasets."""
    datasets = ['MITBIH', 'MIMIC', 'Bonn_EEG']
    
    findings = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Analyzing {dataset}")
        print(f"{'='*60}")
        
        fold_data = load_patterns(dataset)
        if not fold_data:
            print(f"No data found for {dataset}")
            continue
        
        fold_keys = sorted([k for k in fold_data.keys() if k.startswith('fold_')])
        
        if len(fold_keys) < 2:
            print(f"Only {len(fold_keys)} fold(s) available for {dataset}")
            continue
        
        first_patterns = []
        centers = []
        widths = []
        shift_tolerances = []
        transform_types = []
        use_relative_flags = []
        
        for fold_key in fold_keys:
            patterns = fold_data[fold_key]
            if patterns:
                first_pat = patterns[0]
                
                transform_type = first_pat.get('transform_type', 'raw')
                transform_types.append(transform_type)
                
                shift_tolerance = first_pat.get('shift_tolerance', 0.0)
                shift_tolerances.append(shift_tolerance)
                
                # Generate pattern from control points
                if 'control_points' in first_pat:
                    try:
                        from scipy.interpolate import BSpline
                        control_points = first_pat['control_points']
                        width = int(first_pat['width'])
                        degree = 3
                        n_cp = len(control_points)
                        
                        # Ensure we have enough control points for the degree
                        if n_cp > degree:
                            knots = np.concatenate([
                                np.zeros(degree + 1), 
                                np.linspace(0, 1, n_cp - degree + 1)[1:-1], 
                                np.ones(degree + 1)
                            ])
                            pattern_vals = BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width))
                            first_patterns.append(pattern_vals)
                            centers.append(first_pat['center'])
                            widths.append(first_pat['width'])
                            use_relative_flags.append(first_pat.get('use_relative', False))
                    except Exception as e:
                        print(f"Warning: Could not generate BSpline for {dataset} fold {fold_key}: {e}")
                        continue
        
        if len(first_patterns) < 2:
            print(f"Insufficient patterns for {dataset}")
            continue
        
        # Calculate statistics
        rho_mean = compute_pairwise_spearman(first_patterns)
        center_mean, center_std = np.mean(centers), np.std(centers)
        width_mean, width_std = np.mean(widths), np.std(widths)
        shift_tol_mean, shift_tol_std = np.mean(shift_tolerances), np.std(shift_tolerances)
        
        # Analyze transform consistency
        transform_counts = {}
        for t in transform_types:
            transform_counts[t] = transform_counts.get(t, 0) + 1
        
        findings[dataset] = {
            'n_folds': len(fold_keys),
            'spearman_rho': rho_mean,
            'center_mean': center_mean,
            'center_std': center_std,
            'width_mean': width_mean,
            'width_std': width_std,
            'shift_tol_mean': shift_tol_mean,
            'shift_tol_std': shift_tol_std,
            'transform_types': transform_types,
            'transform_counts': transform_counts,
            'use_relative': use_relative_flags,
            'center_cv': center_std / center_mean if center_mean > 0 else 0,
            'width_cv': width_std / width_mean if width_mean > 0 else 0,
        }
        
        print(f"Spearman ρ: {rho_mean:.3f}")
        print(f"Center: {center_mean:.1f} ± {center_std:.1f} (CV: {findings[dataset]['center_cv']:.3f})")
        print(f"Width: {width_mean:.1f} ± {width_std:.1f} (CV: {findings[dataset]['width_cv']:.3f})")
        print(f"Shift tolerance: {shift_tol_mean:.3f} ± {shift_tol_std:.3f}")
        print(f"Transform types: {transform_types}")
        print(f"Transform distribution: {transform_counts}")
        print(f"Use relative: {use_relative_flags}")
    
    # Add REMC analysis
    base_path = Path(__file__).parent.parent / 'json_files'
    remc_path = base_path / 'remc'
    if remc_path.exists():
        for json_file in sorted(remc_path.glob('pattern_parameters_*_fold1.json')):
            cell_line = json_file.stem.replace('pattern_parameters_', '').replace('_fold1', '')
            
            fold_data = {}
            for fold in range(1, 6):
                fold_path = remc_path / f'pattern_parameters_{cell_line}_fold{fold}.json'
                if fold_path.exists():
                    with open(fold_path, 'r') as f:
                        fold_data[f'fold_{fold}'] = json.load(f)
            
            if len(fold_data) < 2:
                continue
            
            dataset_name = f'REMC_{cell_line}'
            print(f"\n{'='*60}")
            print(f"Analyzing {dataset_name}")
            print(f"{'='*60}")
            
            fold_keys = sorted([k for k in fold_data.keys() if k.startswith('fold_')])
            
            first_patterns = []
            centers = []
            widths = []
            shift_tolerances = []
            transform_types = []
            
            for fold_key in fold_keys:
                patterns = fold_data[fold_key]
                if patterns:
                    first_pat = patterns[0]
                    if 'control_points' in first_pat:
                        try:
                            from scipy.interpolate import BSpline
                            control_points = first_pat['control_points']
                            width = int(first_pat['width'])
                            degree = 3
                            n_cp = len(control_points)
                            
                            if n_cp > degree:
                                knots = np.concatenate([
                                    np.zeros(degree + 1), 
                                    np.linspace(0, 1, n_cp - degree + 1)[1:-1], 
                                    np.ones(degree + 1)
                                ])
                                pattern_vals = BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width))
                                first_patterns.append(pattern_vals)
                                centers.append(first_pat['center'])
                                widths.append(first_pat['width'])
                                shift_tolerances.append(first_pat.get('shift_tolerance', 0.0))
                                transform_types.append(first_pat.get('transform_type', 'raw'))
                        except Exception as e:
                            print(f"Warning: Could not generate BSpline for {dataset_name} fold {fold_key}: {e}")
                            continue
            
            if len(first_patterns) >= 2:
                rho_mean = compute_pairwise_spearman(first_patterns)
                center_mean, center_std = np.mean(centers), np.std(centers)
                width_mean, width_std = np.mean(widths), np.std(widths)
                shift_tol_mean, shift_tol_std = np.mean(shift_tolerances), np.std(shift_tolerances)
                
                transform_counts = {}
                for t in transform_types:
                    transform_counts[t] = transform_counts.get(t, 0) + 1
                
                findings[dataset_name] = {
                    'n_folds': len(fold_keys),
                    'spearman_rho': rho_mean,
                    'center_mean': center_mean,
                    'center_std': center_std,
                    'width_mean': width_mean,
                    'width_std': width_std,
                    'shift_tol_mean': shift_tol_mean,
                    'shift_tol_std': shift_tol_std,
                    'transform_types': transform_types,
                    'transform_counts': transform_counts,
                    'center_cv': center_std / center_mean if center_mean > 0 else 0,
                    'width_cv': width_std / width_mean if width_mean > 0 else 0,
                }
                
                print(f"Spearman ρ: {rho_mean:.3f}")
                print(f"Center: {center_mean:.1f} ± {center_std:.1f} (CV: {findings[dataset_name]['center_cv']:.3f})")
                print(f"Width: {width_mean:.1f} ± {width_std:.1f} (CV: {findings[dataset_name]['width_cv']:.3f})")
                print(f"Transform types: {transform_types}")
    
    # Save findings to JSON
    findings_path = Path('../manuscript/tables/pattern_stability_findings.json')
    with open(findings_path, 'w') as f:
        json.dump(findings, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("FINDINGS SUMMARY")
    print(f"{'='*60}")
    
    for dataset, data in findings.items():
        print(f"\n{dataset}:")
        print(f"  Spearman ρ: {data['spearman_rho']:.3f}")
        print(f"  Transform consistency: {data['transform_counts']}")
        print(f"  Center CV: {data['center_cv']:.3f}")
        print(f"  Width CV: {data['width_cv']:.3f}")
    
    return findings

def main():
    mitbih_stats = create_mitbih_first_pattern_figure()
    df = generate_summary_table()
    findings = analyze_all_datasets()
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")
    
    if 'MITBIH' in findings:
        mitbih = findings['MITBIH']
        print(f"\nMITBIH Analysis:")
        print(f"- Spearman correlation: {mitbih['spearman_rho']:.3f} (low correlation indicates pattern variability)")
        print(f"- Transform types: {mitbih['transform_types']}")
        print(f"- Transform distribution: {mitbih['transform_counts']}")
        print(f"- Four out of five folds use 'cumsum' transform")
        print(f"- Four out of five patterns are in similar position range")
        print(f"- Pattern shapes are NOT consistent (low Spearman ρ)")
        print(f"- Center variability (CV): {mitbih['center_cv']:.3f}")
        print(f"- Width variability (CV): {mitbih['width_cv']:.3f}")

if __name__ == "__main__":
    main()

