"""Ablation study for SublimeX variants.

Runs on first fold only (same as main_eval). Standard datasets and REMC use fold 1 of 5-fold CV;
PAMAP2 uses first subject only. AZT1D uses all patients with per-patient 80/20 temporal splits.
"""
import pandas as pd
import numpy as np
import os
import time
import warnings
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import optuna
import core
from core import (SublimeX, mean_objective, 
                  _get_segment, _evaluate, _suggest_segment, _get_train_val_split)
from model import LightGBMModel, encode_labels
from preprocess import load_emotions, load_mimic, load_mitbih, load_remc, load_svd, load_pamap2, load_azt1d

# Aggregation functions (for ablation variants)
AGGREGATIONS = {
    'mean': lambda x: x.mean(axis=1, keepdims=True),
    'std': lambda x: x.std(axis=1, keepdims=True),
    'min': lambda x: x.min(axis=1, keepdims=True),
    'max': lambda x: x.max(axis=1, keepdims=True),
    'range': lambda x: np.ptp(x, axis=1, keepdims=True),
    'median': lambda x: np.median(x, axis=1, keepdims=True),
    'argmin': lambda x: x.argmin(axis=1, keepdims=True).astype(np.float32) / max(x.shape[1] - 1, 1),
    'argmax': lambda x: x.argmax(axis=1, keepdims=True).astype(np.float32) / max(x.shape[1] - 1, 1),
}
AGG_KEYS = list(AGGREGATIONS.keys())


def _pattern_distance(segment, w, cp0, cp1, cp2, n_time):
    """Compute min pattern distance for segment."""
    n_samples, seg_len = segment.shape
    width = min(max(2, int(w * n_time)), seg_len)
    if width > seg_len:
        return np.full((n_samples, 1), np.inf, dtype=np.float32)
    t = np.linspace(0, 1, width, dtype=np.float32)
    pattern = (1 - t)**2 * cp0 + 2 * (1 - t) * t * cp1 + t**2 * cp2
    windows = sliding_window_view(segment, window_shape=width, axis=1)
    return (np.linalg.norm(windows - pattern, axis=2) / np.sqrt(width)).min(axis=1, keepdims=True).astype(np.float32)


def _extract_feature_full(params, ctx):
    """Extract feature from saved parameters (handles all ablation variants)."""
    segment = _get_segment(params, ctx)
    if all(k in params for k in ['w', 'cp0', 'cp1', 'cp2']):
        return _pattern_distance(segment, params['w'], params['cp0'], params['cp1'], params['cp2'], ctx['n_time'])
    if 'agg' in params:
        return AGGREGATIONS[params['agg']](segment).astype(np.float32)
    return segment.mean(axis=1, keepdims=True).astype(np.float32)

# Patch core.extract_feature so SublimeX uses the full version for ablation
core.extract_feature = _extract_feature_full


# =============================================================================
# Ablation objective functions
# =============================================================================

def aggregate_objective(trial, ctx):
    """Aggregate objective with 8 aggregation choices."""
    segment = _suggest_segment(trial, ctx)
    agg = trial.suggest_categorical('agg', AGG_KEYS)
    return _evaluate(AGGREGATIONS[agg](segment).astype(np.float32), ctx)


def pattern_objective(trial, ctx):
    """Pattern-based objective with quadratic Bezier pattern."""
    segment = _suggest_segment(trial, ctx)
    w = trial.suggest_float('w', 0.05, 0.5)
    cp0, cp1, cp2 = (trial.suggest_float(f'cp{i}', 0, 1) for i in range(3))
    return _evaluate(_pattern_distance(segment, w, cp0, cp1, cp2, ctx['n_time']), ctx)


def decision_tree_objective(trial, ctx):
    """Mean objective using Decision Tree instead of LightGBM."""
    feat = _suggest_segment(trial, ctx).mean(axis=1, keepdims=True).astype(np.float32)
    X = np.hstack([ctx['current_X'], feat]) if ctx['current_X'].size else feat
    y, metric = ctx['y'], ctx['metric']
    tr, va = _get_train_val_split(y, metric)
    
    if metric == 'rmse':
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X[tr], y[tr])
        return np.sqrt(mean_squared_error(y[va], model.predict(X[va])))
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X[tr], y[tr])
    return roc_auc_score(y[va], model.predict_proba(X[va])[:, 1]) if metric == 'auc' else accuracy_score(y[va], model.predict(X[va]))


def parallel_objective(trial, ctx):
    """Parallel objective: optimize N features simultaneously."""
    n_feat = ctx.get('n_target_features', 10)
    features = []
    for i in range(n_feat):
        params = {
            'ch': trial.suggest_int(f'ch_{i}', 0, ctx['n_channels'] - 1),
            't': trial.suggest_int(f't_{i}', 0, len(ctx['transform_names']) - 1),
            'c': trial.suggest_float(f'c_{i}', 0, 1),
            'r': trial.suggest_float(f'r_{i}', 0, 1),
        }
        features.append(_get_segment(params, ctx).mean(axis=1, keepdims=True).astype(np.float32))
    return _evaluate(np.hstack(features), ctx)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Baseline feature counts per dataset (from main_eval.csv fold 1, extracted features only)
# Used for parallel variant to extract same number of features
BASELINE_FEATURES = {
    'emotions': 11,  # 11 total (no initial)
    'mitbih': 20,    # 20 total (no initial)
    'svd': 9,        # 9 total (no initial)
    'mimic': 5,      # 39 total - 34 initial = 5 extracted
    'pamap2': 18,    # 18 total (no initial)
    'remc_E003': 12, # 12 total (no initial)
    'azt1d': 16,     # 16 total (no initial)
}

# Ablation variants
VARIANTS = {
    'aggregate': {'objective_fn': aggregate_objective},
    'pattern': {'objective_fn': pattern_objective},
    'decision_tree': {'objective_fn': decision_tree_objective},
    'n_trials_1000': {'objective_fn': mean_objective, 'n_trials': 1000},
    'raw_only': {'objective_fn': mean_objective, 'transforms': {'raw': lambda d: d}},
    'nsga2': {'objective_fn': mean_objective, 'sampler': 'nsga2'},
    'parallel': {'objective_fn': parallel_objective},  # n_target_features set per dataset
}

STANDARD_DATASETS = [('emotions', load_emotions), ('mimic', load_mimic), ('mitbih', load_mitbih), ('svd', load_svd)]
K_FOLDS = 5
RESULTS_PATH = '../results/ablation_study.csv'


def run_experiment(X_train, X_test, y_train, y_test, metric, var_name, var_params, params_path=None, initial_tr=None, initial_te=None, n_target_features=None):
    t0 = time.time()
    
    # For parallel variant: extract all features in one optimization run
    if var_name == 'parallel' and n_target_features is not None:
        # Create wrapper that injects n_target_features
        original_obj = var_params['objective_fn']
        def parallel_obj_with_n(trial, ctx):
            ctx['n_target_features'] = n_target_features
            return original_obj(trial, ctx)
        var_params = {**var_params, 'objective_fn': parallel_obj_with_n}
        
        # Run single optimization study to extract all N features at once
        sublimex = SublimeX(metric=metric, **var_params)
        data = sublimex._to_array(X_train)
        n_samples, n_channels, n_time = data.shape
        transformed = sublimex._apply_transforms(data)
        model = LightGBMModel('regression' if metric == 'rmse' else 'classification')
        direction = 'minimize' if metric == 'rmse' else 'maximize'
        ctx = {
            'transformed': transformed, 'y': y_train, 'model': model, 'metric': metric,
            'n_channels': n_channels, 'n_time': n_time,
            'transform_names': sublimex.transform_names,
            'current_X': np.asarray(initial_tr, dtype=np.float32) if initial_tr is not None else np.empty((n_samples, 0), dtype=np.float32),
            'n_target_features': n_target_features,
        }
        
        sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(lambda t: parallel_obj_with_n(t, ctx), n_trials=sublimex.n_trials, n_jobs=-1)
        
        # Extract all N features from best trial
        best_params = study.best_params
        sublimex.extracted_features = []
        for i in range(n_target_features):
            feat_params = {
                'ch': int(best_params[f'ch_{i}']),
                't': int(best_params[f't_{i}']),
                'c': best_params[f'c_{i}'],
                'r': best_params[f'r_{i}'],
            }
            sublimex.extracted_features.append(feat_params)
        
        sublimex.n_channels = n_channels
        sublimex.n_time = n_time
        train_feat = sublimex.transform(X_train, initial_X=initial_tr)
        test_feat = sublimex.transform(X_test, initial_X=initial_te)
    else:
        sublimex = SublimeX(metric=metric, **var_params)
        train_feat = sublimex.fit(X_train, y_train, initial_X=initial_tr).transform(X_train, initial_X=initial_tr)
        test_feat = sublimex.transform(X_test, initial_X=initial_te)
    
    if params_path:
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        sublimex.save_features(params_path)
    
    n_feat = (initial_tr.shape[1] if initial_tr is not None else 0) + len(sublimex.extracted_features)
    
    if var_name == 'decision_tree':
        # Use decision tree for test eval (consistent with optimization)
        if metric == 'rmse':
            dt = DecisionTreeRegressor(max_depth=5, random_state=42)
            dt.fit(train_feat, y_train)
            score = np.sqrt(mean_squared_error(y_test, dt.predict(test_feat)))
        else:
            y_tr, y_te, _ = encode_labels(y_train, y_test)
            dt = DecisionTreeClassifier(max_depth=5, random_state=42)
            dt.fit(train_feat, y_tr)
            if metric == 'auc':
                score = roc_auc_score(y_te, dt.predict_proba(test_feat)[:, 1])
            else:
                score = accuracy_score(y_te, dt.predict(test_feat))
    else:
        if metric != 'rmse':
            y_tr, y_te, _ = encode_labels(y_train, y_test)
        else:
            y_tr, y_te = y_train.astype(np.float32), y_test.astype(np.float32)
        model = LightGBMModel('regression' if metric == 'rmse' else 'classification')
        score = model.test(train_feat, y_tr, test_feat, y_te, metric)
    
    return score, n_feat, time.time() - t0


def is_done(df, ds, var, fold):
    return not df.empty and ((df['dataset'] == ds) & (df['variant'] == var) & (df['fold'] == fold)).any()


def main():
    existing = pd.read_csv(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else pd.DataFrame()
    results = existing.to_dict('records') if not existing.empty else []
    
    # Standard datasets (5-fold CV)
    for ds_name, load_fn in STANDARD_DATASETS:
        if all(is_done(existing, ds_name, v, 1) for v in VARIANTS):
            print(f"Skipping {ds_name} (all variants done)")
            continue
        X, y, info = load_fn()
        X_list = X if isinstance(X, list) else [X]
        cv_cls = StratifiedKFold if info['task'] == 'classification' else KFold
        folds = list(cv_cls(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
        init = info.get('initial_features')
        for var_name, var_params in VARIANTS.items():
            for fold_idx, (tr_idx, te_idx) in enumerate(folds[:1], 1):
                if is_done(existing, ds_name, var_name, fold_idx): continue
                print(f"{ds_name} | {var_name} | Fold {fold_idx}")
                
                X_tr = [x.iloc[tr_idx] for x in X_list]
                X_te = [x.iloc[te_idx] for x in X_list]
                initial_tr = init[tr_idx] if init is not None else None
                initial_te = init[te_idx] if init is not None else None
                n_target = BASELINE_FEATURES.get(ds_name) if var_name == 'parallel' else None
                score, n_feat, elapsed = run_experiment(X_tr, X_te, y.iloc[tr_idx].values, y.iloc[te_idx].values,
                                                        info['metric'], var_name, var_params, params_path=None,
                                                        initial_tr=initial_tr, initial_te=initial_te, n_target_features=n_target)
                results.append({'dataset': ds_name, 'variant': var_name, 'fold': fold_idx,
                                'score': score, 'n_features': n_feat, 'time': elapsed, 'metric': info['metric']})
                print(f"  {info['metric']}={score:.4f}, n_feat={n_feat}, time={elapsed:.1f}s")
                pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    
    # PAMAP2 (first subject only, LOSO)
    if all(is_done(existing, 'pamap2', v, 1) for v in VARIANTS):
        print("Skipping pamap2 (all variants done)")
    else:
        X, y, info = load_pamap2()
        X_list = X if isinstance(X, list) else [X]
        subjects = info['subject_ids']
        first_subj = np.unique(subjects)[0]
        tr_idx = np.where(subjects != first_subj)[0]
        te_idx = np.where(subjects == first_subj)[0]
        
        for var_name, var_params in VARIANTS.items():
            if is_done(existing, 'pamap2', var_name, 1): continue
            print(f"pamap2 | {var_name} | LOSO (test: {first_subj})")
            
            X_tr = [x.iloc[tr_idx] for x in X_list]
            X_te = [x.iloc[te_idx] for x in X_list]
            
            n_target = BASELINE_FEATURES.get('pamap2') if var_name == 'parallel' else None
            score, n_feat, elapsed = run_experiment(X_tr, X_te, y.iloc[tr_idx].values, y.iloc[te_idx].values,
                                                    info['metric'], var_name, var_params, params_path=None,
                                                    n_target_features=n_target)
            results.append({'dataset': 'pamap2', 'variant': var_name, 'fold': 1, 'score': score,
                            'n_features': n_feat, 'time': elapsed, 'metric': info['metric'], 'test_subject': int(first_subj)})
            print(f"  {info['metric']}={score:.4f}, n_feat={n_feat}, time={elapsed:.1f}s")
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    
    # REMC (first cell line only, 5-fold CV)
    remc_ds = 'remc_E003'  # first cell line
    if all(is_done(existing, remc_ds, v, 1) for v in VARIANTS):
        print(f"Skipping {remc_ds} (all variants done)")
    else:
        cell_lines = load_remc()
        if cell_lines:
            cell_line = cell_lines[0]
            ds_name = f"remc_{cell_line}"
            X, y, info = load_remc(cell_line=cell_line)
            X_list = X if isinstance(X, list) else [X]
            folds = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
            
            for var_name, var_params in VARIANTS.items():
                for fold_idx, (tr_idx, te_idx) in enumerate(folds[:1], 1):
                    if is_done(existing, ds_name, var_name, fold_idx): continue
                    print(f"{ds_name} | {var_name} | Fold {fold_idx}")
                    
                    X_tr = [x.iloc[tr_idx] for x in X_list]
                    X_te = [x.iloc[te_idx] for x in X_list]
                    
                    n_target = BASELINE_FEATURES.get(ds_name) if var_name == 'parallel' else None
                    score, n_feat, elapsed = run_experiment(X_tr, X_te, y.iloc[tr_idx].values, y.iloc[te_idx].values,
                                                            info['metric'], var_name, var_params, params_path=None,
                                                            n_target_features=n_target)
                    results.append({'dataset': ds_name, 'variant': var_name, 'fold': fold_idx, 'score': score,
                                    'n_features': n_feat, 'time': elapsed, 'metric': info['metric'], 'cell_line': cell_line})
                    print(f"  {info['metric']}={score:.4f}, n_feat={n_feat}, time={elapsed:.1f}s")
                    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    
    # AZT1D (all patients; 80/20 temporal split per patient, pooled)
    X, y, info = load_azt1d()
    subject_ids = info['subject_ids']
    X_list = X if isinstance(X, list) else [X]
    tr_indices, te_indices = [], []
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        n = mask.sum()
        cutoff = int(n * 0.8)
        subj_idx = np.where(mask)[0]
        tr_indices.extend(subj_idx[:cutoff])
        te_indices.extend(subj_idx[cutoff:])
    tr_idx, te_idx = np.array(tr_indices), np.array(te_indices)
    ds_name = 'azt1d'

    for var_name, var_params in VARIANTS.items():
        if is_done(existing, ds_name, var_name, 1): continue
        print(f"{ds_name} | {var_name}")

        X_tr = [x.iloc[tr_idx] for x in X_list]
        X_te = [x.iloc[te_idx] for x in X_list]

        n_target = BASELINE_FEATURES.get(ds_name) if var_name == 'parallel' else None
        score, n_feat, elapsed = run_experiment(
            X_tr, X_te, y.iloc[tr_idx].values, y.iloc[te_idx].values,
            info['metric'], var_name, var_params, params_path=None,
            n_target_features=n_target)
        results.append({
            'dataset': ds_name, 'variant': var_name, 'fold': 1,
            'score': score, 'n_features': n_feat, 'time': elapsed,
            'metric': info['metric']})
        print(f"  {info['metric']}={score:.4f}, n_feat={n_feat}, time={elapsed:.1f}s")
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    
    print(f"\nDone! Results saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
