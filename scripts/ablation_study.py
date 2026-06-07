"""Ablation variants × folds. Resumes from ablation_study.csv (baseline from main_eval.csv)."""
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneGroupOut

from config import K_FOLDS, LOAD_FEATURES, ABLATION_CSV, ABLATION_PARAMS, setup_sublimex_path
setup_sublimex_path()
from sublimex import (
    SublimeX, mean_objective, random_split_mean_objective, aggregate_objective, pattern_objective,
    decision_tree_objective, extract_feature_from_params, LightGBMModel, encode_labels,
)
from preprocess import load_emotions, load_mimic, load_mitbih, load_remc, load_svd, load_pamap2, load_azt1d

warnings.filterwarnings('ignore')
RESULTS_PATH = str(ABLATION_CSV)
PARAMS_DIR = str(ABLATION_PARAMS)

VARIANTS = {
    'resampling': {'objective_fn': random_split_mean_objective},
    'aggregate': {'objective_fn': aggregate_objective, 'extract_fn': extract_feature_from_params},
    'pattern': {'objective_fn': pattern_objective, 'extract_fn': extract_feature_from_params},
    'decision_tree': {'objective_fn': decision_tree_objective},
    'n_trials_1000': {'objective_fn': mean_objective, 'n_trials': 1000},
    'raw_only': {'objective_fn': mean_objective, 'transforms': {'raw': lambda d: d}},
    'nsga2': {'objective_fn': mean_objective, 'sampler': 'nsga2'},
    'parallel': {'n_parallel': 10},
}


def _stack(X):
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return X.astype(np.float32, copy=False)
    X = [X] if not isinstance(X, list) else X
    return np.stack([np.asarray(x, dtype=np.float32) for x in X], axis=1)


def _load(load_fn):
    X, y, info = load_fn()
    init = info.get('initial_features')
    return _stack(X if isinstance(X, list) else [X]), np.asarray(y), info, np.asarray(init, np.float32) if init is not None else None


def _folds(info, y, n_folds, groups=None):
    if groups is not None:
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups=np.asarray(groups)))
    cv = StratifiedKFold if info['task'] == 'classification' else KFold
    return list(cv(n_folds, shuffle=True, random_state=42).split(np.zeros(len(y)), y))


def _temporal_fold(n, ratio=0.8):
    c = int(n * ratio)
    return [(np.arange(c), np.arange(c, n))]


def _done_keys(records):
    return {(r['dataset'], r['variant'], int(r['fold'])) for r in records if r.get('variant') != 'baseline'}


def _banner(msg):
    print(f'\n{"=" * 64}\n{msg}\n{"=" * 64}', flush=True)


def _score(metric, var, Xtr, Xte, ytr, yte):
    if var == 'decision_tree':
        Cls = DecisionTreeRegressor if metric == 'rmse' else DecisionTreeClassifier
        dt = Cls(max_depth=5, random_state=42)
        ytr, yte = (ytr, yte) if metric == 'rmse' else encode_labels(ytr, yte)[:2]
        dt.fit(Xtr, ytr)
        pred = dt.predict(Xte) if metric == 'rmse' else dt.predict_proba(Xte)
        return np.sqrt(mean_squared_error(yte, pred)) if metric == 'rmse' else (roc_auc_score(yte, pred[:, 1]) if metric == 'auc' else accuracy_score(yte, dt.predict(Xte)))
    ytr, yte = (ytr.astype(np.float32), yte.astype(np.float32)) if metric == 'rmse' else encode_labels(ytr, yte)[:2]
    return LightGBMModel('regression' if metric == 'rmse' else 'classification').test(Xtr, ytr, Xte, yte, metric, deterministic=True)


def run_dataset(ds, load_fn, results, done, folds=None, n_folds=K_FOLDS, groups=None):
    stack, y, info, init = _load(load_fn)
    folds = folds or _folds(info, y, n_folds, groups)
    n_left = sum(1 for var in VARIANTS for fi in range(1, len(folds) + 1) if (ds, var, fi) not in done)
    _banner(f'DATASET: {ds}  |  {len(folds)} folds  |  {len(VARIANTS)} variants  |  {n_left} jobs left')
    cache, metric = {}, info['metric']
    for var, vpar in VARIANTS.items():
        todo = [fi for fi in range(1, len(folds) + 1) if (ds, var, fi) not in done]
        if not todo:
            print(f'  [{ds}] variant {var}: complete (skip)', flush=True)
            continue
        print(f'  [{ds}] variant {var}: folds {todo[0]}–{todo[-1]} of {len(folds)} ({len(todo)} to run)', flush=True)
        for fi, (tr, te) in enumerate(folds, 1):
            key = (ds, var, fi)
            if key in done:
                continue
            print(f'  [{ds}] {var}  fold {fi}/{len(folds)}  …', flush=True)
            ppath = f'{PARAMS_DIR}/{ds}/fold{fi}_{var}.json'
            if var == 'resampling' and not os.path.exists(ppath):
                ppath = f'{PARAMS_DIR}/{ds}/fold{fi}_random_split.json'
            t0 = time.time()
            sx = SublimeX(metric=metric, deterministic=False, verbose=False, n_workers=-1,
                            **{k: v for k, v in vpar.items() if k != 'n_parallel'})
            tx = lambda X, sp: cache.setdefault((fi, sp, tuple(sx.transform_names)), sx._apply_transforms(sx._to_array(X)))
            tr_t, te_t = tx(stack[tr], 'tr'), tx(stack[te], 'te')
            itr, ite = (init[tr], init[te]) if init is not None else (None, None)
            if LOAD_FEATURES and os.path.exists(ppath):
                sx.load_features(ppath)
            elif var == 'parallel':
                sx.fit_parallel(stack[tr], y[tr], vpar['n_parallel'], initial_X=itr, transformed=tr_t)
            else:
                sx.fit(stack[tr], y[tr], initial_X=itr, transformed=tr_t)
            if not (LOAD_FEATURES and os.path.exists(ppath)):
                os.makedirs(os.path.dirname(ppath) or '.', exist_ok=True)
                sx.save_features(ppath)
            sc = _score(metric, var, sx.transform(stack[tr], initial_X=itr, transformed=tr_t),
                        sx.transform(stack[te], initial_X=ite, transformed=te_t), y[tr], y[te])
            row = {'dataset': ds, 'variant': var, 'fold': fi, 'score': sc,
                   'n_features': (itr.shape[1] if itr is not None else 0) + len(sx.extracted_features),
                   'time': time.time() - t0, 'metric': metric}
            if groups is not None:
                row['test_subject'] = int(np.unique(groups[te])[0])
            results.append(row)
            done.add(key)
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            print(f'  [{ds}] {var}  fold {fi}/{len(folds)}  done  {sc:.4f}  n={row["n_features"]}  {row["time"]:.0f}s', flush=True)


def run_azt1d_all_subjects(results, done):
    _, _, info, _ = _load(load_azt1d)
    sids = sorted(np.unique(info['subject_ids']))
    _banner(f'BLOCK: AZT1D per-subject  |  {len(sids)} subjects')
    for sid in sids:
        s = int(sid)
        stack, y, info, init = _load(lambda sid=s: load_azt1d(subject_id=sid))
        run_dataset(f'azt1d_s{s}', lambda sid=s: load_azt1d(subject_id=sid), results, done,
                    folds=_temporal_fold(len(y)))


if __name__ == '__main__':
    results = pd.read_csv(RESULTS_PATH).to_dict('records') if os.path.exists(RESULTS_PATH) else []
    results = [r for r in results if r.get('variant') != 'baseline']
    done = _done_keys(results)
    print(f'ablation → {RESULTS_PATH}  resume={len(done)} jobs  (baseline from main_eval.csv)', flush=True)
    for ds, fn in [('emotions', load_emotions), ('mimic', load_mimic), ('mitbih', load_mitbih), ('svd', load_svd)]:
        run_dataset(ds, fn, results, done, n_folds=K_FOLDS)
    remc = load_remc()
    if remc:
        cl = remc[0]
        run_dataset(f'remc_{cl}', lambda c=cl: load_remc(cell_line=c), results, done, n_folds=K_FOLDS)
    run_azt1d_all_subjects(results, done)
    _, _, pinfo, _ = _load(load_pamap2)
    run_dataset('pamap2', load_pamap2, results, done, n_folds=len(np.unique(pinfo['subject_ids'])),
                groups=np.asarray(pinfo['subject_ids']))
    _banner(f'FINISHED  |  {len(results)} rows in {RESULTS_PATH}')
