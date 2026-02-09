"""Incremental feature analysis: train/val/test performance per added feature.

Fast version - loads saved parameters and applies them (no fit_transform).
Optimized: batch-transforms all data upfront, extracts all features once.
Uses exact same CV splits as main_eval.py and the same fixed 50/50 internal
train/validation split as core.py (random_state=42).
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from model import LightGBMModel
from preprocess import (load_emotions, load_mimic, load_mitbih, load_svd,
                        load_pamap2, load_remc, load_azt1d)
from core import TRANSFORMS, extract_feature

warnings.filterwarnings('ignore')

# Configuration (same as main_eval.py and core.py)
K_FOLDS = 5
AZT1D_TRAIN_RATIO = 0.8
VAL_SIZE = 0.5        # same as core.py
RANDOM_STATE = 42     # same as core.py
RESULTS_PATH = '../results/incremental_features.csv'
PLOT_PATH = '../elsarticle/images/incremental_features.png'
PARAMS_DIR = '../parameters'

# Standard datasets (5-fold CV)
STANDARD_DATASETS = [
    ('emotions', load_emotions),
    ('mimic', load_mimic),
    ('mitbih', load_mitbih),
    ('svd', load_svd),
]


def load_parameters(ds_name, fold_idx):
    """Load feature parameters from JSON file."""
    path = f'{PARAMS_DIR}/{ds_name}/fold{fold_idx}.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)

    path = f'{PARAMS_DIR}/{ds_name}/mean_only_fold{fold_idx}.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)

    return None


def apply_transforms_batch(X_list):
    """Apply all transforms to data once, return transformed array."""
    arrays = [
        x.values.astype(np.float32) if hasattr(x, 'values')
        else np.asarray(x, dtype=np.float32)
        for x in X_list
    ]
    data = np.stack(arrays, axis=1).astype(np.float32)
    n_samples, n_channels, n_time = data.shape

    transform_names = list(TRANSFORMS.keys())
    transformed = np.empty(
        (len(transform_names), n_samples, n_channels, n_time),
        dtype=np.float32)
    for ti, tname in enumerate(transform_names):
        transformed[ti] = TRANSFORMS[tname](
            data.reshape(-1, n_time)
        ).reshape(n_samples, n_channels, n_time)

    return transformed, n_channels, n_time, transform_names


def extract_all_features(transformed, params_list, n_channels,
                         n_time, transform_names):
    """Extract all features from pre-transformed data in one pass."""
    ctx = {
        'transformed': transformed,
        'n_channels': n_channels,
        'n_time': n_time,
        'transform_names': transform_names,
    }
    features = [extract_feature(p, ctx) for p in params_list]
    if features:
        return np.hstack(features).astype(np.float32)
    return np.empty((transformed.shape[1], 0), dtype=np.float32)


def get_val_split(y, metric):
    """Return the same deterministic train/val split used by core.py."""
    stratify = y if metric != 'rmse' else None
    return train_test_split(
        np.arange(len(y)), test_size=VAL_SIZE,
        random_state=RANDOM_STATE, stratify=stratify)


def evaluate_incremental(train_feat_all, val_feat_all,
                         test_feat_all, y_train, y_val,
                         y_test, n_features_total, metric,
                         init_train=None, init_val=None,
                         init_test=None):
    """Evaluate model incrementally by slicing pre-extracted features.

    Trains on the optimization-train split (same data SublimeX used
    for model training during feature search) and evaluates on train,
    validation, and test sets.
    """
    results = []

    for n_feat in range(1, n_features_total + 1):
        tr = train_feat_all[:, :n_feat]
        va = val_feat_all[:, :n_feat]
        te = test_feat_all[:, :n_feat]

        if init_train is not None:
            tr = np.hstack([init_train, tr])
            va = np.hstack([init_val, va])
            te = np.hstack([init_test, te])

        model = LightGBMModel(
            'regression' if metric == 'rmse' else 'classification')
        train_score = model.test(tr, y_train, tr, y_train, metric)
        val_score = model.test(tr, y_train, va, y_val, metric)
        test_score = model.test(tr, y_train, te, y_test, metric)

        results.append({
            'n_features': n_feat,
            'train_score': train_score,
            'val_score': val_score,
            'test_score': test_score,
        })

    return results


def _split_train_val(transformed_all, train_idx, test_idx,
                     y, metric, params, n_channels, n_time,
                     transform_names, static_features=None):
    """Split training fold into train/val and extract features.

    Returns (train_feat, val_feat, test_feat,
             y_train, y_val, y_test,
             init_train, init_val, init_test).
    """
    y_train_full = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values

    tr_rel, va_rel = get_val_split(y_train_full, metric)
    y_train = y_train_full[tr_rel]
    y_val = y_train_full[va_rel]

    full_train_feat = extract_all_features(
        transformed_all[:, train_idx], params,
        n_channels, n_time, transform_names)
    train_feat = full_train_feat[tr_rel]
    val_feat = full_train_feat[va_rel]
    test_feat = extract_all_features(
        transformed_all[:, test_idx], params,
        n_channels, n_time, transform_names)

    init_train = init_val = init_test = None
    if static_features is not None:
        sf_train = static_features[train_idx]
        init_train = sf_train[tr_rel]
        init_val = sf_train[va_rel]
        init_test = static_features[test_idx]

    return (train_feat, val_feat, test_feat,
            y_train, y_val, y_test,
            init_train, init_val, init_test)


def _append_results(results, ds_name, fold_idx, fold_results, metric):
    """Append fold results to the results list."""
    for r in fold_results:
        results.append({
            'dataset': ds_name,
            'fold': fold_idx,
            'n_features': r['n_features'],
            'train_score': r['train_score'],
            'val_score': r['val_score'],
            'test_score': r['test_score'],
            'metric': metric,
        })


def run_standard_dataset(ds_name, load_fn, skip_folds=None):
    """Run incremental feature experiment for a standard dataset.

    Uses exact same CV folds as main_eval.py and the same fixed
    internal train/val split as core.py.
    """
    results = []
    skip_folds = skip_folds or set()

    X, y, info = load_fn()
    metric, task = info['metric'], info['task']
    static_features = info.get('initial_features', None)

    X_list = X if isinstance(X, list) else [X]
    transformed_all, n_channels, n_time, transform_names = \
        apply_transforms_batch(X_list)

    cv_cls = StratifiedKFold if task == 'classification' else KFold
    folds = list(cv_cls(K_FOLDS, shuffle=True, random_state=42).split(
        pd.concat(X_list, axis=1), y))

    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        if fold_idx in skip_folds:
            print(f"  Fold {fold_idx}: Skipping (already completed)")
            continue

        params = load_parameters(ds_name, fold_idx)

        (train_feat, val_feat, test_feat,
         y_train, y_val, y_test,
         init_train, init_val, init_test) = _split_train_val(
            transformed_all, train_idx, test_idx, y, metric,
            params, n_channels, n_time, transform_names,
            static_features)

        fold_results = evaluate_incremental(
            train_feat, val_feat, test_feat,
            y_train, y_val, y_test, len(params), metric,
            init_train, init_val, init_test)

        _append_results(results, ds_name, fold_idx,
                        fold_results, metric)

        print(f"  Fold {fold_idx}: {len(params)} features, "
              f"val={fold_results[-1]['val_score']:.4f}, "
              f"test={fold_results[-1]['test_score']:.4f}")

    return results


def run_pamap2_first_subject(skip_folds=None):
    """PAMAP2 first subject only (LOSO fold 1)."""
    results = []
    skip_folds = skip_folds or set()
    ds_name = 'pamap2'

    if 1 in skip_folds:
        print(f"Skipping {ds_name} fold 1 (already completed)")
        return results

    X, y, info = load_pamap2()
    metric = info['metric']
    subject_ids = info['subject_ids']
    unique_subjects = np.unique(subject_ids)

    X_list = X if isinstance(X, list) else [X]
    transformed_all, n_channels, n_time, transform_names = \
        apply_transforms_batch(X_list)

    first_subject = unique_subjects[0]
    test_mask = subject_ids == first_subject
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    params = load_parameters(ds_name, 1)
    if params is None:
        print(f"  WARNING: No parameters for {ds_name}, skipping")
        return results

    (train_feat, val_feat, test_feat,
     y_train, y_val, y_test,
     init_train, init_val, init_test) = _split_train_val(
        transformed_all, train_idx, test_idx, y, metric,
        params, n_channels, n_time, transform_names)

    fold_results = evaluate_incremental(
        train_feat, val_feat, test_feat,
        y_train, y_val, y_test, len(params), metric)

    _append_results(results, ds_name, 1, fold_results, metric)

    print(f"  {len(params)} features, "
          f"val={fold_results[-1]['val_score']:.4f}, "
          f"test={fold_results[-1]['test_score']:.4f}")
    return results


def run_remc_first_cellline(skip_folds=None):
    """REMC first cell line, 5-fold CV."""
    results = []
    skip_folds = skip_folds or set()

    cell_lines = load_remc()
    cell_line = cell_lines[0]
    ds_name = f'remc_{cell_line}'

    X, y, info = load_remc(cell_line=cell_line)
    metric = info['metric']

    X_list = X if isinstance(X, list) else [X]
    transformed_all, n_channels, n_time, transform_names = \
        apply_transforms_batch(X_list)

    folds = list(StratifiedKFold(
        K_FOLDS, shuffle=True, random_state=42
    ).split(pd.concat(X_list, axis=1), y))

    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        if fold_idx in skip_folds:
            print(f"  Fold {fold_idx}: Skipping (already completed)")
            continue

        params = load_parameters(ds_name, fold_idx)
        if params is None:
            print(f"  Fold {fold_idx}: WARNING - No parameters")
            continue

        (train_feat, val_feat, test_feat,
         y_train, y_val, y_test,
         init_train, init_val, init_test) = _split_train_val(
            transformed_all, train_idx, test_idx, y, metric,
            params, n_channels, n_time, transform_names)

        fold_results = evaluate_incremental(
            train_feat, val_feat, test_feat,
            y_train, y_val, y_test, len(params), metric)

        _append_results(results, ds_name, fold_idx,
                        fold_results, metric)

        print(f"  Fold {fold_idx}: {len(params)} features, "
              f"val={fold_results[-1]['val_score']:.4f}, "
              f"test={fold_results[-1]['test_score']:.4f}")

    return results


def run_azt1d_all_patients(skip=False):
    """AZT1D all patients pooled, per-patient 80/20 temporal split."""
    results = []
    ds_name = 'azt1d'

    if skip:
        print(f"Skipping {ds_name} (already completed)")
        return results

    X, y, info = load_azt1d()
    subject_ids = info['subject_ids']
    metric = info['metric']

    X_list = X if isinstance(X, list) else [X]
    transformed_all, n_channels, n_time, transform_names = \
        apply_transforms_batch(X_list)

    params = load_parameters(ds_name, 1)
    if params is None:
        print(f"  WARNING: No parameters for {ds_name}, skipping")
        return results

    # Per-patient 80/20 temporal split (same as main_eval.py)
    tr_indices, te_indices = [], []
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        n = mask.sum()
        cutoff = int(n * AZT1D_TRAIN_RATIO)
        subj_idx = np.where(mask)[0]
        tr_indices.extend(subj_idx[:cutoff])
        te_indices.extend(subj_idx[cutoff:])
    train_idx = np.array(tr_indices)
    test_idx = np.array(te_indices)

    (train_feat, val_feat, test_feat,
     y_train, y_val, y_test,
     init_train, init_val, init_test) = _split_train_val(
        transformed_all, train_idx, test_idx, y, metric,
        params, n_channels, n_time, transform_names)

    fold_results = evaluate_incremental(
        train_feat, val_feat, test_feat,
        y_train, y_val, y_test, len(params), metric)

    _append_results(results, ds_name, 1, fold_results, metric)

    print(f"  {len(params)} features, "
          f"val={fold_results[-1]['val_score']:.4f}, "
          f"test={fold_results[-1]['test_score']:.4f}")
    return results


def create_plot(df, output_path):
    """Create seaborn plot: train/val/test performance vs # features."""
    sns.set_theme(style='whitegrid', font_scale=0.9)

    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    n_cols = min(4, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten() if n_datasets > 1 else [axes]

    palette = {
        'Train': '#2ecc71',
        'Validation': '#3498db',
        'Test': '#e74c3c',
    }

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        ds_df = df[df['dataset'] == ds_name]
        metric = ds_df['metric'].iloc[0]

        # Reshape to long format for seaborn
        long_records = []
        for _, row in ds_df.iterrows():
            for split, col in [('Train', 'train_score'),
                               ('Validation', 'val_score'),
                               ('Test', 'test_score')]:
                long_records.append({
                    'n_features': row['n_features'],
                    'Split': split,
                    'score': row[col],
                    'fold': row['fold'],
                })
        long_df = pd.DataFrame(long_records)

        sns.lineplot(
            data=long_df, x='n_features', y='score',
            hue='Split', palette=palette, ax=ax,
            linewidth=2, marker='o', markersize=3,
            err_style='band', errorbar='sd')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_xlabel('# Features')
        ylabel = ('RMSE' if metric == 'rmse'
                  else ('AUC' if metric == 'auc' else 'Acc'))
        ax.set_ylabel(ylabel)

        title = ds_name.upper()
        if ds_name.startswith('remc_'):
            title = 'REMC'
        elif ds_name.startswith('azt1d'):
            title = 'AZT1D'
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=7, title=None)
        if metric == 'rmse':
            ax.invert_yaxis()

    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    sns.reset_defaults()


def main():
    completed = {}
    all_results = []

    if os.path.exists(RESULTS_PATH):
        existing_df = pd.read_csv(RESULTS_PATH)
        # Re-run everything if old format (no val_score column)
        if 'val_score' in existing_df.columns:
            all_results = existing_df.to_dict('records')
            for _, row in existing_df.groupby(
                    ['dataset', 'fold']).first().reset_index().iterrows():
                ds_name = row['dataset']
                fold = row['fold']
                if ds_name not in completed:
                    completed[ds_name] = set()
                completed[ds_name].add(fold)
            print("Already completed:")
            for ds_name, folds in sorted(completed.items()):
                print(f"  {ds_name}: folds {sorted(folds)}")
            print()
        else:
            print("Old results without val_score found; "
                  "re-running all.\n")

    # Standard datasets
    for ds_name, load_fn in STANDARD_DATASETS:
        skip_folds = completed.get(ds_name, set())
        if len(skip_folds) >= K_FOLDS:
            print(f"Skipping {ds_name} (all folds completed)")
            continue
        print(f"Running {ds_name}...")
        results = run_standard_dataset(ds_name, load_fn, skip_folds)
        all_results.extend(results)
        pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)

    # PAMAP2
    skip_folds = completed.get('pamap2', set())
    if 1 not in skip_folds:
        print("Running pamap2...")
        all_results.extend(run_pamap2_first_subject(skip_folds))
        pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)
    else:
        print("Skipping pamap2 (already completed)")

    # REMC
    cell_lines = load_remc()
    ds_name = f'remc_{cell_lines[0]}'
    skip_folds = completed.get(ds_name, set())
    if len(skip_folds) < K_FOLDS:
        print(f"Running {ds_name}...")
        all_results.extend(run_remc_first_cellline(skip_folds))
        pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)
    else:
        print(f"Skipping {ds_name} (all folds completed)")

    # AZT1D
    azt1d_skip = 'azt1d' in completed
    print("Running AZT1D (all patients)...")
    all_results.extend(run_azt1d_all_patients(skip=azt1d_skip))
    pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)

    # Create plot
    df = pd.DataFrame(all_results)
    if not df.empty:
        create_plot(df, PLOT_PATH)
        print("\nFinal summary:")
        for ds_name in sorted(df['dataset'].unique()):
            ds_df = df[df['dataset'] == ds_name]
            metric = ds_df['metric'].iloc[0]
            max_feat = int(ds_df['n_features'].max())
            sub = ds_df[ds_df['n_features'] == max_feat]
            val = sub['val_score'].mean()
            test = sub['test_score'].mean()
            print(f"  {ds_name}: {max_feat} features, "
                  f"val={val:.4f}, test={test:.4f}")


if __name__ == '__main__':
    main()
