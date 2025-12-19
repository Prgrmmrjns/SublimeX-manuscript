import pandas as pd
import numpy as np
import time
import warnings
import os
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patx"))
from patx import feature_extraction
from patx.models import LightGBMWrapper
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]

N_TRIALS = 500
SHOW_PROGRESS = False
# match main eval scripts more closely
N_WORKERS = -1
MAX_SAMPLES = 30000
INNER_K_FOLDS = 1
VAL_SIZE = 0.2
K_FOLDS = 5

def load_mitbih_data():
    data = pd.read_csv(ROOT / "processed_datasets/mitbih/mitbih_processed.csv")
    return data.drop('target', axis=1), data['target']

def load_svd_data():
    N_FRAMES = 700
    VOWELS = ["a_n", "i_n", "u_n"]
    df = pd.read_parquet(ROOT / "processed_datasets/svd/svd.parquet")
    channels = [df[[f"{v}_{t}" for t in range(N_FRAMES)]].astype(np.float32) for v in VOWELS]
    return channels, df["target"].astype(int)

def load_remc_data(cell_line):
    TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    df = pd.read_parquet(ROOT / "processed_datasets/remc" / f"{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in TIME_SERIES], df['target']

def save_results(new_results):
    global existing
    if new_results:
        df_new = pd.DataFrame(new_results)
        existing = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
        existing.to_csv(results_file, index=False)

def run_variant(name, X, y, folds, dataset='mitbih', metric='accuracy', n_cp=5, n_trans=5, inner_k=None, initial_features=None, existing_results=None, transforms=None, sliding_window=True, allow_shift=True, n_patterns=1, max_patterns=None, early_stop=True, n_trials=N_TRIALS):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    results = []
    for fold, (tr, te) in enumerate(folds):
        if not existing_results.empty:
            existing_fold = existing_results[(existing_results['dataset'] == dataset) & (existing_results['approach'] == name) & (existing_results['fold'] == fold)]
            if len(existing_fold) > 0:
                print(f"Fold {fold+1}: Already completed, skipping")
                results.append(existing_fold.iloc[0].to_dict())
                continue
        
        X_tr = [x.iloc[tr].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[tr].astype(np.float32)]
        X_te = [x.iloc[te].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[te].astype(np.float32)]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        t0 = time.time()
        model = LightGBMWrapper('classification', n_classes=len(np.unique(y_tr)), n_jobs=1)
        init_feat = (initial_features.iloc[tr].values, initial_features.iloc[te].values) if initial_features is not None else None
        res = feature_extraction(X_tr, y_tr.values, X_te, metric=metric, n_trials=n_trials, show_progress=SHOW_PROGRESS, n_control_points=n_cp, n_transforms=n_trans, max_samples=MAX_SAMPLES, inner_k_folds=inner_k or INNER_K_FOLDS, val_size=VAL_SIZE, n_workers=N_WORKERS, model=model, initial_features=init_feat, transforms=transforms, sliding_window=sliding_window, allow_shift=allow_shift, n_patterns=n_patterns, max_patterns=max_patterns, early_stop=early_stop)
        if metric == 'accuracy':
            preds = res['model'].predict(res['test_features'])
            score = accuracy_score(y_te.values, preds)
        else:
            preds_proba = res['model'].predict_proba(res['test_features'])
            preds = preds_proba[:, 1] if preds_proba.ndim > 1 else preds_proba
            score = roc_auc_score(y_te.values, preds)
        t = time.time() - t0
        result = {'dataset': dataset, 'approach': name, 'fold': fold, 'score': score, 'time': t, 'n_features': len(res['patterns'])}
        results.append(result)
        save_results([result])
        print(f"Fold {fold+1}: {metric.upper()}={score:.4f}, Time={t:.1f}s, Features={len(res['patterns'])}")
    scores, times, feats = [r['score'] for r in results], [r['time'] for r in results], [r['n_features'] for r in results]
    print(f"\nAverage: {metric.upper()}={np.mean(scores):.4f}±{np.std(scores):.4f}, Time={np.mean(times):.1f}±{np.std(times):.1f}s, Features={np.mean(feats):.1f}±{np.std(feats):.1f}")
    return results

def run_variant_one_fold(name, X, y, tr, te, fold, dataset='mitbih', metric='accuracy', n_cp=5, n_trans=5, inner_k=None, initial_features=None, existing_results=None, transforms=None, sliding_window=True, allow_shift=True, n_patterns=1, max_patterns=None, early_stop=True, n_trials=N_TRIALS):
    if not existing_results.empty:
        existing_fold = existing_results[(existing_results['dataset'] == dataset) & (existing_results['approach'] == name) & (existing_results['fold'] == fold)]
        if len(existing_fold) > 0:
            print(f"Fold {fold+1}: Already completed, skipping")
            return existing_fold.iloc[0].to_dict()
    X_tr = [x.iloc[tr].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[tr].astype(np.float32)]
    X_te = [x.iloc[te].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[te].astype(np.float32)]
    y_tr, y_te = y.iloc[tr], y.iloc[te]
    t0 = time.time()
    model = LightGBMWrapper('classification', n_classes=len(np.unique(y_tr)), n_jobs=1)
    init_feat = (initial_features.iloc[tr].values, initial_features.iloc[te].values) if initial_features is not None else None
    res = feature_extraction(X_tr, y_tr.values, X_te, metric=metric, n_trials=n_trials, show_progress=SHOW_PROGRESS, n_control_points=n_cp, n_transforms=n_trans, max_samples=MAX_SAMPLES, inner_k_folds=inner_k or INNER_K_FOLDS, val_size=VAL_SIZE, n_workers=N_WORKERS, model=model, initial_features=init_feat, transforms=transforms, sliding_window=sliding_window, allow_shift=allow_shift, n_patterns=n_patterns, max_patterns=max_patterns, early_stop=early_stop)
    if metric == 'accuracy':
        preds = res['model'].predict(res['test_features'])
        score = accuracy_score(y_te.values, preds)
    else:
        preds_proba = res['model'].predict_proba(res['test_features'])
        preds = preds_proba[:, 1] if preds_proba.ndim > 1 else preds_proba
        score = roc_auc_score(y_te.values, preds)
    t = time.time() - t0
    result = {'dataset': dataset, 'approach': name, 'fold': fold, 'score': score, 'time': t, 'n_features': len(res['patterns'])}
    save_results([result])
    print(f"Fold {fold+1}: {metric.upper()}={score:.4f}, Time={t:.1f}s, Features={len(res['patterns'])}")
    return result

variants = [
    ('baseline', {'n_cp': 3, 'n_trans': 5}),
    ('baseline_trials_100', {'n_cp': 3, 'n_trans': 5, 'n_trials': 100}),
    ('baseline_trials_1000', {'n_cp': 3, 'n_trans': 5, 'n_trials': 1000}),
    ('no_transforms', {'n_cp': 3, 'n_trans': 1, 'transforms': ['raw']}),
    ('no_shifting', {'n_cp': 3, 'n_trans': 5, 'sliding_window': False, 'allow_shift': False}),
    ('one_control_point', {'n_cp': 1, 'n_trans': 5}),
]

variant_names = [v[0] for v in variants]
results_file = str(ROOT / "results/ablation_study.csv")
existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
table_path = str(ROOT / "manuscript/tables/ablation_results.tex")

def latex_escape(text):
    return text.replace('_', '\\_')

def pm(mean, std, digits):
    std = 0.0 if np.isnan(std) else std
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"

def format_variant_name(approach):
    names = {
        'baseline': 'Baseline',
        'baseline_trials_100': 'Baseline (100 trials)',
        'baseline_trials_1000': 'Baseline (1000 trials)',
        'no_transforms': 'No transforms',
        'no_shifting': 'No shifting',
        'one_control_point': '1 control point',
        'parallel_optimization': 'Parallel optimization',
    }
    return names.get(approach, latex_escape(approach))

def format_dataset_name(dataset):
    names = {
        'mitbih': 'MITBIH',
        'svd': 'SVD',
        'remc_E003': 'REMC (E003)',
    }
    return names.get(dataset, latex_escape(dataset))

def write_tex_table(df, dataset_order):
    order = variant_names + ['parallel_optimization']
    lines = [
        '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc@{}}',
        '\\toprule',
        '\\textbf{Dataset} & \\textbf{Variant} & \\textbf{Score} & \\textbf{Time (s)} & \\textbf{Features} \\\\',
        '\\midrule',
    ]
    
    for i, dataset in enumerate(dataset_order):
        if i > 0:
            lines.append('\\midrule')
        
        dataset_data = df[df['dataset'] == dataset]
        best_score = -np.inf
        best_approach = None
        for approach in order:
            sub = dataset_data[dataset_data['approach'] == approach]
            if not sub.empty:
                score = sub['score'].mean()
                if score > best_score:
                    best_score = score
                    best_approach = approach
        
        for approach in order:
            sub = dataset_data[dataset_data['approach'] == approach]
            if sub.empty:
                continue
            mean = sub.mean(numeric_only=True)
            std = sub.std(numeric_only=True, ddof=0)
            
            is_best = (approach == best_approach)
            score_mean = mean['score']
            score_std = std['score'] if not np.isnan(std['score']) else 0.0
            score_str = pm(score_mean, score_std, 3)
            if is_best:
                score_str = f'$\\boldsymbol{{{score_mean:.3f} \\pm {score_std:.3f}}}$'
            
            dataset_name = format_dataset_name(dataset) if approach == order[0] else ''
            variant_name = format_variant_name(approach)
            if is_best:
                variant_name = f'\\textbf{{{variant_name}}}'
            
            row = ' & '.join([
                dataset_name,
                variant_name,
                score_str,
                pm(mean['time'], std['time'], 1),
                pm(mean['n_features'], std['n_features'], 1),
            ]) + ' \\\\'
            lines.append(row)
    
    lines.append('\\bottomrule')
    lines.append('\\end{tabular*}')
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))

def missing_results(df, fold_counts):
    required = variant_names + ['parallel_optimization']
    missing = []
    for dataset, n_folds in fold_counts.items():
        for approach in required:
            count = len(df[(df['dataset'] == dataset) & (df['approach'] == approach)])
            if count < n_folds:
                missing.append((dataset, approach))
    return missing

def maybe_write_tex_table(df, fold_counts):
    dataset_order = sorted(fold_counts.keys())
    write_tex_table(df, dataset_order)
    print(f"Ablation table written to {table_path}")

def get_done_variants(dataset_name):
    return set(existing[existing['dataset'] == dataset_name]['approach'].unique()) if not existing.empty else set()

def run_dataset(dataset_name, X, y, folds, metric, initial_features=None):
    global existing
    n_folds = len(folds)
    pending = []
    for v in variant_names:
        if len(existing[(existing['dataset'] == dataset_name) & (existing['approach'] == v)]) < n_folds:
            pending.append(v)
    if len(existing[(existing['dataset'] == dataset_name) & (existing['approach'] == 'parallel_optimization')]) < n_folds:
        pending.append('parallel_optimization')
    if len(pending) == 0:
        print(f"{dataset_name.upper()}: All variants already completed, skipping.")
        return
    print(f"{dataset_name.upper()}: Running {len(pending)} pending variants: {pending}")
    baseline_counts = {}
    if 'baseline' not in pending:
        base = existing[(existing['dataset'] == dataset_name) & (existing['approach'] == 'baseline')]
        for fold in base['fold'].unique():
            baseline_counts[int(fold)] = int(base[base['fold'] == fold]['n_features'].iloc[0])
    for name, params in variants:
        if name in pending:
            run_variant(name, X, y, folds, dataset=dataset_name, metric=metric, initial_features=initial_features, existing_results=existing, **params)
            existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
            maybe_write_tex_table(existing, fold_counts)
    if 'baseline' in pending:
        base = existing[(existing['dataset'] == dataset_name) & (existing['approach'] == 'baseline')]
        for fold in base['fold'].unique():
            baseline_counts[int(fold)] = int(base[base['fold'] == fold]['n_features'].iloc[0])
    if 'parallel_optimization' in pending:
        print(f"\n{'='*60}\nparallel_optimization\n{'='*60}")
        for fold, (tr, te) in enumerate(folds):
            k = baseline_counts.get(fold, None)
            if k is None or k <= 0:
                continue
            run_variant_one_fold('parallel_optimization', X, y, tr, te, fold, dataset=dataset_name, metric=metric, initial_features=initial_features, existing_results=existing, n_cp=3, n_trans=5, n_patterns=k, early_stop=False)
        existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
        maybe_write_tex_table(existing, fold_counts)

print("Checking MITBIH...")
X_mitbih, y_mitbih = load_mitbih_data()
folds_mitbih = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(X_mitbih, y_mitbih))
fold_counts = {'mitbih': len(folds_mitbih)}
run_dataset('mitbih', X_mitbih, y_mitbih, folds_mitbih, 'accuracy')

print("\nChecking SVD...")
channels_svd, y_svd = load_svd_data()
folds_svd = list(StratifiedKFold(2, shuffle=True, random_state=42).split(pd.concat(channels_svd, axis=1), y_svd))
fold_counts['svd'] = len(folds_svd)
run_dataset('svd', channels_svd, y_svd, folds_svd, 'auc')

print("\nChecking REMC (first cell line)...")
cell_lines = sorted([f.replace('.parquet', '') for f in os.listdir(ROOT / 'processed_datasets/remc') if f.endswith('.parquet')])
if cell_lines:
    first_cell_line = cell_lines[0]
    X_remc, y_remc = load_remc_data(first_cell_line)
    folds_remc = list(StratifiedKFold(2, shuffle=True, random_state=42).split(pd.concat(X_remc, axis=1), y_remc))
    fold_counts[f'remc_{first_cell_line}'] = len(folds_remc)
    run_dataset(f'remc_{first_cell_line}', X_remc, y_remc, folds_remc, 'auc')

print(f"\n{'='*80}\nSUMMARY BY DATASET\n{'='*80}")
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    for dataset_name in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset_name]
        print(f"\n{dataset_name.upper()}:")
        summary = dataset_df.groupby('approach').agg({'score': ['mean', 'std'], 'time': ['mean', 'std'], 'n_features': ['mean', 'std']}).round(4)
        print(summary)
    maybe_write_tex_table(df, fold_counts)
