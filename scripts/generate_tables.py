"""Generate LaTeX tables from evaluation results.

Handles:
- main_eval.csv: Main results with all approaches (SublimeX + baselines)
- ablation_study.csv: Ablation study results (excludes mean_only/SublimeX baseline)

Aggregation:
- Standard datasets: mean±std across 5 folds
- REMC: mean±std across ALL cell lines and folds
- AZT1D: mean±std across ALL subjects
"""
import pandas as pd
import numpy as np
import os

# Configuration
RESULTS_DIR = '../results'
OUTPUT_DIR = '../elsarticle/tables'

# Dataset information
DATASET_INFO = {
    'azt1d': {'name': 'AZT1D', 'metric': 'RMSE', 'direction': 'minimize'},
    'emotions': {'name': 'Emotions', 'metric': 'Accuracy', 'direction': 'maximize'},
    'mimic': {'name': 'MIMIC-IV', 'metric': 'AUC', 'direction': 'maximize'},
    'mitbih': {'name': 'MITBIH', 'metric': 'Accuracy', 'direction': 'maximize'},
    'remc': {'name': 'REMC', 'metric': 'AUC', 'direction': 'maximize'},
    'pamap2': {'name': 'PAMAP2', 'metric': 'Accuracy', 'direction': 'maximize'},
    'svd': {'name': 'SVD', 'metric': 'Accuracy', 'direction': 'maximize'}
}

# Initial features for datasets (to compute extracted features)
# MIMIC has 38 initial static features (demographics, comorbidities, aggregated vitals)
INITIAL_FEATURES = {
    'mimic': 34,
}

# Ablation variants (actual names from CSV)
ABLATION_VARIANTS = [
    'aggregate',
    'pattern',
    'decision_tree',
    'n_trials_1000',
    'raw_only',
    'nsga2',
    'parallel',
]

VARIANT_LABELS = {
    'aggregate': ('Optimize', 'Aggregates'),
    'pattern': ('Pattern', 'Search'),
    'decision_tree': ('Decision', 'Tree'),
    'n_trials_1000': ('1000', 'Trials'),
    'raw_only': ('Raw', 'Only'),
    'nsga2': ('NSGA-II', ''),
    'parallel': ('Parallel', ''),
}

# Approaches for main results table
APPROACHES = ['SublimeX', 'TSFRESH', 'CATCH22', 'MiniRocket', 'RDST', 'CNN']
APPROACH_NORMALIZE = {
    'sublimex': 'SublimeX',
    'tsfresh': 'TSFRESH',
    'catch22': 'CATCH22',
    'cnn': 'CNN',
    'minirocket': 'MiniRocket',
    'rdst': 'RDST'
}


def normalize_approach(name):
    return APPROACH_NORMALIZE.get(name.lower(), name)


def format_score(score, score_std, decimals=3):
    """Format score with optional std. Returns '-' if None."""
    if score is None:
        return "-"
    if score_std is not None and score_std > 0:
        return f"{score:.{decimals}f}±{score_std:.{decimals}f}"
    return f"{score:.{decimals}f}"


def format_features(n_features, n_features_std):
    """Format feature count with optional std. Returns '-' if None."""
    if n_features is None:
        return "-"
    if n_features_std is not None and n_features_std > 0.5:
        return f"{n_features:.1f}±{n_features_std:.1f}"
    return f"{n_features:.0f}"


def format_time(time_seconds, time_std=None):
    """Format time in seconds."""
    if time_seconds is None:
        return "-"
    if time_std is not None and time_std > 0:
        return f"{time_seconds:.0f}±{time_std:.0f}"
    return f"{time_seconds:.0f}"


def get_dataset_key(dataset_name):
    """Extract base dataset key from dataset name (handles remc_E003, azt1d_s1, etc.)."""
    if dataset_name.startswith('remc_'):
        return 'remc'
    if dataset_name.startswith('azt1d_'):
        return 'azt1d'
    return dataset_name


# =============================================================================
# MAIN RESULTS TABLE
# =============================================================================

def load_main_eval_data():
    """Load and process main_eval.csv results."""
    filepath = os.path.join(RESULTS_DIR, 'main_eval.csv')
    
    df = pd.read_csv(filepath)
    df['approach'] = df['approach'].apply(normalize_approach)
    df['base_dataset'] = df['dataset'].apply(get_dataset_key)
    
    results = {}
    for base_ds in ['emotions', 'mimic', 'mitbih', 'pamap2', 'svd', 'remc', 'azt1d']:
        ds_data = df[df['base_dataset'] == base_ds]
        if ds_data.empty:
            results[base_ds] = {}
            continue
        
        ds_results = {}
        for approach in ds_data['approach'].unique():
            app_data = ds_data[ds_data['approach'] == approach]
            if app_data.empty:
                continue
            
            # Aggregate across all folds/cell_lines/subjects
            n_features_mean = app_data['n_features'].mean()
            n_features_std = app_data['n_features'].std() if len(app_data) > 1 else 0.0
            
            # Subtract initial features to get extracted features (only for SublimeX)
            if approach == 'SublimeX':
                initial_feat = INITIAL_FEATURES.get(base_ds, 0)
                n_features_mean = max(0, n_features_mean - initial_feat)
            
            ds_results[approach] = {
                'score': app_data['score'].mean(),
                'score_std': app_data['score'].std() if len(app_data) > 1 else 0.0,
                'n_features': n_features_mean,
                'n_features_std': n_features_std,
                'n_samples': len(app_data),
            }
        results[base_ds] = ds_results
    
    return results


def generate_results_table(results):
    """Generate main results comparison table."""
    output_file = os.path.join(OUTPUT_DIR, 'results_table.tex')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    dataset_order = ['azt1d', 'mitbih', 'emotions', 'remc', 'mimic', 'pamap2', 'svd']
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance comparison across biomedical datasets. "
                "Results show mean $\\pm$ standard deviation across subjects/folds. "
                "Best performance per dataset highlighted in bold.}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\tiny\n")
        f.write("\\setlength{\\tabcolsep}{1pt}\n")
        f.write("\\renewcommand{\\arraystretch}{0.85}\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll" + 
                "c" * len(APPROACHES) + "@{}}\n")
        f.write("\\toprule\n")
        
        # Header (two empty columns for dataset name + metric label columns)
        header = " &  & " + " & ".join([f"\\textbf{{{a}}}" for a in APPROACHES])
        f.write(header + " \\\\\n")
        f.write("\\midrule\n")
        
        for ds_idx, dataset in enumerate(dataset_order):
            info = DATASET_INFO[dataset]
            ds_results = results.get(dataset, {})
            
            # Find best score rounded to 3 decimals
            decimals = 2 if info['metric'] == 'RMSE' else 3
            best_rounded_score = None
            for approach in APPROACHES:
                if approach in ds_results:
                    score = ds_results[approach]['score']
                    rounded_score = round(score, decimals)
                    if best_rounded_score is None:
                        best_rounded_score = rounded_score
                    elif info['direction'] == 'maximize':
                        best_rounded_score = max(best_rounded_score, rounded_score)
                    else:
                        best_rounded_score = min(best_rounded_score, rounded_score)
            
            # Format dataset label
            sign = '↑' if info['direction'] == 'maximize' else '↓'
            metric_abbrev = {'Accuracy': 'Acc', 'AUC': 'AUC', 'RMSE': 'RMSE'}[info['metric']]
            
            # Collect scores only (no # Features in main table)
            scores = []
            for approach in APPROACHES:
                if approach in ds_results:
                    data = ds_results[approach]
                    score_str = format_score(data['score'], data['score_std'], decimals)
                    
                    # Bold if rounded score matches best rounded score
                    rounded_score = round(data['score'], decimals)
                    is_best = (best_rounded_score is not None and 
                               abs(rounded_score - best_rounded_score) < 1e-6)
                    if is_best:
                        score_str = f"\\textbf{{{score_str}}}"
                    
                    scores.append(score_str)
                else:
                    scores.append("-")
            
            # Single row: Dataset name + metric label + scores
            f.write(f"\\textbf{{{info['name']}}} & {metric_abbrev} ({sign}) & " + " & ".join(scores) + " \\\\\n")
            
            if ds_idx < len(dataset_order) - 1:
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\end{table}\n")


# =============================================================================
# ABLATION TABLE
# =============================================================================

def load_ablation_data():
    """Load and process ablation_study.csv results.
    
    For REMC and AZT1D, ablation study only runs on first cell line / first subject.
    So for fair comparison, baseline should also use only the first cell line / subject.
    """
    filepath = os.path.join(RESULTS_DIR, 'ablation_study.csv')
    
    df = pd.read_csv(filepath)
    # If all-patients 'azt1d' rows exist, drop old per-subject 'azt1d_s*' rows
    if 'azt1d' in df['dataset'].values:
        df = df[~df['dataset'].str.startswith('azt1d_s')]
    df['base_dataset'] = df['dataset'].apply(get_dataset_key)
    
    # Also load main_eval to get SublimeX baseline for comparison
    main_eval_path = os.path.join(RESULTS_DIR, 'main_eval.csv')
    baseline_df = None
    if os.path.exists(main_eval_path):
        baseline_df = pd.read_csv(main_eval_path)
        baseline_df['approach'] = baseline_df['approach'].apply(normalize_approach)
        baseline_df = baseline_df[baseline_df['approach'] == 'SublimeX']
        baseline_df['base_dataset'] = baseline_df['dataset'].apply(get_dataset_key)
    
    results = {}
    for base_ds in ['mitbih', 'emotions', 'mimic', 'svd', 'pamap2', 'remc', 'azt1d']:
        ds_data = df[df['base_dataset'] == base_ds]
        
        ds_results = {}
        
        # Add baseline (SublimeX from main_eval, fold 1 only like ablation study)
        if baseline_df is not None:
            bl_data = baseline_df[baseline_df['base_dataset'] == base_ds]
            
            # For REMC: filter to first cell line only (matches ablation study)
            if base_ds == 'remc' and not bl_data.empty:
                first_cell_line = bl_data['dataset'].iloc[0]  # e.g., remc_E003
                bl_data = bl_data[bl_data['dataset'] == first_cell_line]
            
            # For AZT1D: use 'azt1d' (all patients) if available,
            # else fall back to first subject
            if base_ds == 'azt1d' and not bl_data.empty:
                if 'azt1d' in bl_data['dataset'].values:
                    bl_data = bl_data[bl_data['dataset'] == 'azt1d']
                else:
                    first_subject = bl_data['dataset'].iloc[0]
                    bl_data = bl_data[bl_data['dataset'] == first_subject]
            
            # For PAMAP2: filter to first subject only (matches ablation study)
            if base_ds == 'pamap2' and not bl_data.empty:
                # First LOSO fold = first subject as test set
                bl_data = bl_data[bl_data['fold'] == 1]
            
            # Filter to fold 1 only for standard datasets (matches ablation study)
            if base_ds in ['emotions', 'mimic', 'mitbih', 'svd'] and not bl_data.empty:
                bl_data = bl_data[bl_data['fold'] == 1]
            
            if not bl_data.empty:
                n_features_total = bl_data['n_features'].iloc[0]
                
                # Subtract initial features to get extracted features
                initial_feat = INITIAL_FEATURES.get(base_ds, 0)
                n_features_extracted = max(0, n_features_total - initial_feat)
                
                ds_results['baseline'] = {
                    'score': bl_data['score'].iloc[0],
                    'score_std': None,
                    'n_features': n_features_extracted,
                    'n_features_std': None,
                    'time': bl_data['time'].iloc[0],
                    'time_std': None,
                }
        
        # Add ablation variants
        for variant in ABLATION_VARIANTS:
            var_data = ds_data[ds_data['variant'] == variant]
            if var_data.empty:
                continue
            
            n_features_mean = var_data['n_features'].mean()
            n_features_std = var_data['n_features'].std() if len(var_data) > 1 else 0.0
            
            # Subtract initial features to get extracted features
            initial_feat = INITIAL_FEATURES.get(base_ds, 0)
            n_features_mean = max(0, n_features_mean - initial_feat)
            
            ds_results[variant] = {
                'score': var_data['score'].mean(),
                'score_std': var_data['score'].std() if len(var_data) > 1 else 0.0,
                'n_features': n_features_mean,
                'n_features_std': n_features_std,
                'time': var_data['time'].mean(),
                'time_std': var_data['time'].std() if len(var_data) > 1 else 0.0,
            }
        
        results[base_ds] = ds_results
    
    return results


def generate_ablation_table(results):
    """Generate ablation study table."""
    output_file = os.path.join(OUTPUT_DIR, 'ablation_results.tex')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    dataset_order = ['azt1d', 'mitbih', 'emotions', 'remc', 'mimic', 'pamap2', 'svd']
    
    # Only include variants that exist in at least one dataset
    existing_variants = set()
    for ds_results in results.values():
        existing_variants.update(ds_results.keys())
    existing_variants.discard('baseline')  # baseline is always included
    
    # Order: baseline first, then ABLATION_VARIANTS in order, but only if they exist
    all_variants = ['baseline'] + [v for v in ABLATION_VARIANTS if v in existing_variants]
    variant_labels = {'baseline': ('Baseline', ''), **VARIANT_LABELS}
    
    with open(output_file, 'w') as f:
        f.write("\\label{tab:ablation}\n")
        f.write("\\tiny\n")
        f.write("\\setlength{\\tabcolsep}{0.5pt}\n")
        f.write("\\renewcommand{\\arraystretch}{0.85}\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll" + 
                "c" * len(all_variants) + "@{}}\n")
        f.write("\\toprule\n")
        
        # Two-row header for variant names
        header_row1 = " &  & " + " & ".join([f"\\textbf{{{variant_labels[v][0]}}}" for v in all_variants])
        header_row2 = " &  & " + " & ".join([f"\\textbf{{{variant_labels[v][1]}}}" for v in all_variants])
        f.write(header_row1 + " \\\\\n")
        f.write(header_row2 + " \\\\\n")
        f.write("\\midrule\n")
        
        for ds_idx, dataset in enumerate(dataset_order):
            info = DATASET_INFO[dataset]
            ds_results = results.get(dataset, {})
            
            # Find best score rounded to 3 decimals
            decimals = 2 if info['metric'] == 'RMSE' else 3
            best_rounded_score = None
            for variant in all_variants:
                if variant in ds_results:
                    score = ds_results[variant]['score']
                    rounded_score = round(score, decimals)
                    if best_rounded_score is None:
                        best_rounded_score = rounded_score
                    elif info['direction'] == 'maximize':
                        best_rounded_score = max(best_rounded_score, rounded_score)
                    else:
                        best_rounded_score = min(best_rounded_score, rounded_score)
            
            # Format dataset label
            sign = '↑' if info['direction'] == 'maximize' else '↓'
            metric_abbrev = {'Accuracy': 'Acc', 'AUC': 'AUC', 'RMSE': 'RMSE'}[info['metric']]
            
            # Collect data
            scores = []
            features = []
            times = []
            for variant in all_variants:
                if variant in ds_results:
                    data = ds_results[variant]
                    score_str = format_score(data['score'], data['score_std'], decimals)
                    
                    # Bold if rounded score matches best rounded score
                    rounded_score = round(data['score'], decimals)
                    is_best = (best_rounded_score is not None and 
                               abs(rounded_score - best_rounded_score) < 1e-6)
                    if is_best:
                        score_str = f"\\textbf{{{score_str}}}"
                    
                    scores.append(score_str)
                    features.append(format_features(data['n_features'], data['n_features_std']))
                    times.append(format_time(data['time'], data['time_std']))
                else:
                    scores.append("-")
                    features.append("-")
                    times.append("-")
            
            # Row 1: Dataset name (multirow) + metric label + scores
            f.write(f"\\multirow{{3}}{{*}}{{\\textbf{{{info['name']}}}}} & {metric_abbrev} ({sign}) & " + " & ".join(scores) + " \\\\\n")
            # Row 2: empty (multirow) + #Feat label + features
            f.write(f" & \\# Feat. & " + " & ".join(features) + " \\\\\n")
            # Row 3: empty (multirow) + Time label + times
            f.write(f" & Time (s) & " + " & ".join(times) + " \\\\\n")
            
            if ds_idx < len(dataset_order) - 1:
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
    
if __name__ == '__main__':
    main_results = load_main_eval_data()
    ablation_results = load_ablation_data()
    
    generate_results_table(main_results)
    generate_ablation_table(ablation_results)
