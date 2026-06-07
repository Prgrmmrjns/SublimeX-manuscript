"""Incremental feature analysis (Figure~3): train/val/test curves per added feature."""
import argparse, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneGroupOut
from plot_style import apply_manuscript_figure_style, mm_size, save_png

from config import K_FOLDS, PARAMETERS, ELARTICLE, INCREMENTAL_CSV, setup_sublimex_path
setup_sublimex_path()
from sublimex import LightGBMModel, TRANSFORMS, extract_feature
from sublimex.core import _get_train_val_split
from preprocess import load_emotions, load_mimic, load_mitbih, load_svd, load_pamap2, load_remc, load_azt1d

warnings.filterwarnings('ignore')
apply_manuscript_figure_style(12)

PAMAP2_FOLDS, AZT1D_TRAIN_RATIO, RANDOM_STATE = 8, 0.8, 42
RESULTS_PATH = str(INCREMENTAL_CSV)
PLOT_PATH = str(ELARTICLE / 'incremental_features')
PARAMS_DIR = str(PARAMETERS)

STANDARD_DATASETS = [
    ('emotions', load_emotions),
    ('mimic', load_mimic),
    ('mitbih', load_mitbih),
    ('svd', load_svd),
]


def load_parameters(ds_name, fold_idx):
    for name in (f'fold{fold_idx}.json', f'mean_only_fold{fold_idx}.json'):
        path = f'{PARAMS_DIR}/{ds_name}/{name}'
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return None


def apply_transforms_batch(X_list):
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


def extract_all_features(transformed, params_list, n_channels, n_time, transform_names):
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


def evaluate_incremental(train_feat_all, val_feat_all, test_feat_all,
                         y_train, y_val, y_test, n_features_total, metric,
                         init_train=None, init_val=None, init_test=None):
    results = []
    for n_feat in range(1, n_features_total + 1):
        tr = train_feat_all[:, :n_feat]
        va = val_feat_all[:, :n_feat]
        te = test_feat_all[:, :n_feat]
        if init_train is not None:
            tr = np.hstack([init_train, tr])
            va = np.hstack([init_val, va])
            te = np.hstack([init_test, te])
        model = LightGBMModel('regression' if metric == 'rmse' else 'classification')
        results.append({
            'n_features': n_feat,
            'train_score': model.test(tr, y_train, tr, y_train, metric),
            'val_score': model.test(tr, y_train, va, y_val, metric),
            'test_score': model.test(tr, y_train, te, y_test, metric),
        })
    return results


def _split_train_val(transformed_all, train_idx, test_idx, y, metric, params,
                     n_channels, n_time, transform_names, static_features=None):
    y_train_full = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values
    tr_rel, va_rel = _get_train_val_split(y_train_full, metric)
    y_train, y_val = y_train_full[tr_rel], y_train_full[va_rel]
    full_train_feat = extract_all_features(
        transformed_all[:, train_idx], params, n_channels, n_time, transform_names)
    train_feat, val_feat = full_train_feat[tr_rel], full_train_feat[va_rel]
    test_feat = extract_all_features(
        transformed_all[:, test_idx], params, n_channels, n_time, transform_names)
    init_train = init_val = init_test = None
    if static_features is not None:
        sf_train = static_features[train_idx]
        init_train, init_val, init_test = sf_train[tr_rel], sf_train[va_rel], static_features[test_idx]
    return train_feat, val_feat, test_feat, y_train, y_val, y_test, init_train, init_val, init_test


def _append_results(results, ds_name, fold_idx, fold_results, metric):
    for r in fold_results:
        results.append({
            'dataset': ds_name, 'fold': fold_idx, **r, 'metric': metric,
        })


def run_standard_dataset(ds_name, load_fn, skip_folds=None):
    results, skip_folds = [], skip_folds or set()
    X, y, info = load_fn()
    static_features = info.get('initial_features', None)
    X_list = X if isinstance(X, list) else [X]
    transformed_all, n_channels, n_time, transform_names = apply_transforms_batch(X_list)
    cv_cls = StratifiedKFold if info['task'] == 'classification' else KFold
    folds = list(cv_cls(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        if fold_idx in skip_folds:
            continue
        params = load_parameters(ds_name, fold_idx)
        split = _split_train_val(
            transformed_all, train_idx, test_idx, y, info['metric'], params,
            n_channels, n_time, transform_names, static_features)
        fold_results = evaluate_incremental(
            *split[:3], split[3], split[4], split[5], len(params), info['metric'],
            *split[6:])
        _append_results(results, ds_name, fold_idx, fold_results, info['metric'])
    return results


def run_pamap2(skip_folds=None, all_results=None, save_fn=None):
    results, skip_folds = [], skip_folds or set()
    ds_name = 'pamap2'
    X, y, info = load_pamap2()
    subject_ids = np.asarray(info['subject_ids'])
    X_list = X if isinstance(X, list) else [X]
    transformed_all, n_channels, n_time, transform_names = apply_transforms_batch(X_list)
    folds = list(LeaveOneGroupOut().split(pd.concat(X_list, axis=1), y, groups=subject_ids))
    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        if fold_idx in skip_folds:
            continue
        params = load_parameters(ds_name, fold_idx)
        if params is None:
            continue
        split = _split_train_val(
            transformed_all, train_idx, test_idx, y, info['metric'], params,
            n_channels, n_time, transform_names)
        fold_results = evaluate_incremental(
            *split[:3], split[3], split[4], split[5], len(params), info['metric'])
        _append_results(results, ds_name, fold_idx, fold_results, info['metric'])
        if all_results is not None:
            all_results.extend(results[-len(fold_results):])
            save_fn()
    return results


def run_remc_first_cellline(skip_folds=None):
    results, skip_folds = [], skip_folds or set()
    cell_line = load_remc()[0]
    ds_name = f'remc_{cell_line}'
    X, y, info = load_remc(cell_line=cell_line)
    X_list = X if isinstance(X, list) else [X]
    transformed_all, n_channels, n_time, transform_names = apply_transforms_batch(X_list)
    folds = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        if fold_idx in skip_folds:
            continue
        params = load_parameters(ds_name, fold_idx)
        if params is None:
            continue
        split = _split_train_val(
            transformed_all, train_idx, test_idx, y, info['metric'], params,
            n_channels, n_time, transform_names)
        fold_results = evaluate_incremental(
            *split[:3], split[3], split[4], split[5], len(params), info['metric'])
        _append_results(results, ds_name, fold_idx, fold_results, info['metric'])
    return results


def _azt1d_train_test_indices(subject_ids):
    tr_indices, te_indices = [], []
    for subj in np.unique(subject_ids):
        idx = np.where(subject_ids == subj)[0]
        cutoff = int(len(idx) * AZT1D_TRAIN_RATIO)
        tr_indices.extend(idx[:cutoff])
        te_indices.extend(idx[cutoff:])
    return np.array(tr_indices), np.array(te_indices)


def run_azt1d():
    """AZT1D: pooled training windows from all patients, shared discovered feature set."""
    results, ds_name = [], 'azt1d'
    X, y, info = load_azt1d()
    params = load_parameters(ds_name, 1)
    if params is None:
        return results
    X_list = X if isinstance(X, list) else [X]
    transformed_all, n_channels, n_time, transform_names = apply_transforms_batch(X_list)
    train_idx, test_idx = _azt1d_train_test_indices(np.asarray(info['subject_ids']))
    split = _split_train_val(
        transformed_all, train_idx, test_idx, y, info['metric'], params,
        n_channels, n_time, transform_names)
    fold_results = evaluate_incremental(
        *split[:3], split[3], split[4], split[5], len(params), info['metric'])
    _append_results(results, ds_name, 1, fold_results, info['metric'])
    return results


DATASET_ORDER = ['emotions', 'mimic', 'mitbih', 'svd', 'pamap2', 'remc_E003', 'azt1d']
DATASET_TITLES = {
    'emotions': 'Emotions', 'mimic': 'MIMIC-IV', 'mitbih': 'MITBIH',
    'svd': 'SVD', 'pamap2': 'PAMAP2', 'azt1d': 'AZT1D',
}
SPLIT_STYLE = {
    'train_score': ('Train', '#27ae60'),
    'val_score': ('Validation', '#2980b9'),
    'test_score': ('Test', '#c0392b'),
}


def _dataset_title(ds_name):
    return 'REMC' if ds_name.startswith('remc_') else DATASET_TITLES.get(ds_name, ds_name.upper())


def _ordered_datasets(df):
    present = set(df['dataset'].unique())
    ordered = [d for d in DATASET_ORDER if d in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _aggregate_curve(ds_df, score_col):
    agg = (ds_df.groupby('n_features')[score_col]
           .agg(mean='mean', std='std', n='count').reset_index())
    agg['std'] = agg['std'].fillna(0)
    agg.loc[agg['n'] < 2, 'std'] = 0
    return agg.sort_values('n_features')


def _plot_split_curve(ax, ds_df, score_col, label, color):
    agg = _aggregate_curve(ds_df, score_col)
    ax.plot(agg['n_features'], agg['mean'], color=color, linewidth=2.4, label=label, zorder=3)
    err = agg['std'] > 0
    if err.any():
        ax.errorbar(
            agg.loc[err, 'n_features'], agg.loc[err, 'mean'],
            yerr=agg.loc[err, 'std'], fmt='none', ecolor=color,
            elinewidth=1.3, capsize=2.5, capthick=1.3, alpha=0.32, zorder=2)


def create_plot(df):
    datasets = _ordered_datasets(df)
    n_cols = min(3, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=mm_size(148 * n_cols, 72 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        ds_df = df[df['dataset'] == ds_name]
        metric = ds_df['metric'].iloc[0]
        for score_col, (label, color) in SPLIT_STYLE.items():
            _plot_split_curve(ax, ds_df, score_col, label, color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.set_xlim(1, ds_df['n_features'].max())
        ax.set_xlabel('Number of features')
        ax.set_ylabel('RMSE' if metric == 'rmse' else ('AUC' if metric == 'auc' else 'Accuracy'))
        if metric == 'rmse':
            ax.invert_yaxis()
        ax.set_title(_dataset_title(ds_name), fontweight='bold')
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    handles = [plt.Line2D([0], [0], color=c, linewidth=2.4) for _, c in SPLIT_STYLE.values()]
    fig.legend(handles, [lab for lab, _ in SPLIT_STYLE.values()],
               loc='lower center', ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(bottom=0.1, wspace=0.32, hspace=0.42, top=0.96)
    save_png(fig, Path(PLOT_PATH), dpi=300)
    plt.close()


def _completed_folds(df):
    completed = {}
    if df.empty:
        return completed
    for _, row in df.groupby(['dataset', 'fold']).first().reset_index().iterrows():
        completed.setdefault(row['dataset'], set()).add(row['fold'])
    return completed


def _drop_datasets(results, names):
    drop = set(names)
    remc = 'remc' in drop
    drop.discard('remc')
    return [r for r in results if r['dataset'] not in drop and not (remc and str(r['dataset']).startswith('remc_'))]


def _save(results):
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)


def _run_selected(targets, force, all_results):
    completed = {} if force else _completed_folds(pd.DataFrame(all_results))
    if force and targets:
        all_results[:] = _drop_datasets(all_results, targets)

    run_all = targets is None

    if run_all or 'emotions' in targets or 'mimic' in targets or 'mitbih' in targets or 'svd' in targets:
        for ds_name, load_fn in STANDARD_DATASETS:
            if targets and ds_name not in targets:
                continue
            skip = completed.get(ds_name, set())
            if len(skip) < K_FOLDS:
                all_results.extend(run_standard_dataset(ds_name, load_fn, skip))
                _save(all_results)

    if run_all or 'pamap2' in targets:
        skip = completed.get('pamap2', set())
        if len(skip) < PAMAP2_FOLDS:
            run_pamap2(skip, all_results, _save)
            _save(all_results)

    if run_all or 'remc' in targets:
        cell_line = load_remc()[0]
        ds_name = f'remc_{cell_line}'
        skip = completed.get(ds_name, set())
        if len(skip) < K_FOLDS:
            all_results.extend(run_remc_first_cellline(skip))
            _save(all_results)

    if run_all or 'azt1d' in targets:
        if force or 'azt1d' not in completed:
            all_results.extend(run_azt1d())
            _save(all_results)


def parse_args():
    p = argparse.ArgumentParser(description='Incremental feature analysis for Figure~3.')
    p.add_argument('--datasets', nargs='+', choices=[
        'emotions', 'mimic', 'mitbih', 'svd', 'pamap2', 'remc', 'azt1d'],
        help='Run only these datasets (default: all incomplete).')
    p.add_argument('--force', action='store_true', help='Re-run selected datasets from scratch.')
    return p.parse_args()


def main():
    args = parse_args()
    all_results = pd.read_csv(RESULTS_PATH).to_dict('records') if os.path.exists(RESULTS_PATH) else []
    _run_selected(set(args.datasets) if args.datasets else None, args.force, all_results)
    create_plot(pd.DataFrame(all_results))


if __name__ == '__main__':
    main()
