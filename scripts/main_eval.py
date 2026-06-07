"""Main evaluation: SublimeX and baselines. Resumes from results/main_eval.csv."""
import os
os.environ.update({
    'KMP_WARNINGS': '0', 'OMP_WARNINGS': '0', 'OMP_DISPLAY_ENV': 'FALSE',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'NUMBA_THREADING_LAYER': 'workqueue',
})
import time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform as RDST
from pycatch22 import catch22_all
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneGroupOut, train_test_split
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from config import (
    K_FOLDS, LOAD_FEATURES, SAVE_FEATURES, MAIN_EVAL_CSV, PARAMETERS,
    setup_sublimex_path,
)
setup_sublimex_path()
from sublimex import SublimeX, mean_objective, LightGBMModel
from preprocess import load_emotions, load_mimic, load_mitbih, load_remc, load_svd, load_pamap2, load_azt1d

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='overflow encountered in reduce')

RESULTS_PATH = str(MAIN_EVAL_CSV)
PARAMS_DIR = str(PARAMETERS)
STANDARD_DATASETS = [('mimic', load_mimic), ('emotions', load_emotions), ('mitbih', load_mitbih), ('svd', load_svd)]


def _ch(x, n_ch):
    return x.reshape(x.shape[0], n_ch, -1) if n_ch > 1 else x[:, None, :]


def _lgb(metric, ftr, fte, ytr, yte):
    return LightGBMModel('regression' if metric == 'rmse' else 'classification').test(ftr, ytr, fte, yte, metric, deterministic=True)


def _params_path(ds, fold):
    return os.path.join(PARAMS_DIR, ds, f'fold{fold}.json')


def _azt1d_pooled_params_path():
    return os.path.join(PARAMS_DIR, 'azt1d', 'fold1.json')


def _fit_azt1d_pooled_sublimex(splits, metric, force=False):
    ppath = _azt1d_pooled_params_path()
    sx = SublimeX(metric=metric, objective_fn=mean_objective, verbose=True, deterministic=False, n_workers=6)
    if not force and os.path.exists(ppath):
        sx.load_features(ppath)
        return sx, 0.0
    t0 = time.time()
    n_ch = splits[1][1]
    X_tr = np.vstack([splits[fi][2] for fi in splits])
    y_tr = np.concatenate([splits[fi][4] for fi in splits])
    itr = splits[1][6]
    itr_p = np.vstack([splits[fi][6] for fi in splits]) if itr is not None else None
    nt = X_tr.shape[1] // n_ch
    tr_list = [pd.DataFrame(X_tr[:, i * nt:(i + 1) * nt]) for i in range(n_ch)]
    sx.fit(tr_list, y_tr, initial_X=itr_p)
    os.makedirs(os.path.dirname(ppath), exist_ok=True)
    sx.save_features(ppath)
    return sx, time.time() - t0


def eval_sublimex_azt1d_pooled(sx, X_tr, X_te, y_tr, y_te, metric, n_ch, itr, ite):
    t0, nt = time.time(), X_tr.shape[1] // n_ch
    tr_list = [pd.DataFrame(X_tr[:, i * nt:(i + 1) * nt]) for i in range(n_ch)]
    te_list = [pd.DataFrame(X_te[:, i * nt:(i + 1) * nt]) for i in range(n_ch)]
    tr_f, te_f = sx.transform(tr_list, initial_X=itr), sx.transform(te_list, initial_X=ite)
    n_feat = (itr.shape[1] if itr is not None else 0) + len(sx.extracted_features)
    return _lgb(metric, tr_f, te_f, y_tr, y_te), time.time() - t0, n_feat


def eval_sublimex(X_tr, X_te, y_tr, y_te, metric, n_ch, ppath, itr, ite):
    t0, nt = time.time(), X_tr.shape[1] // n_ch
    tr_list = [pd.DataFrame(X_tr[:, i*nt:(i+1)*nt]) for i in range(n_ch)]
    te_list = [pd.DataFrame(X_te[:, i*nt:(i+1)*nt]) for i in range(n_ch)]
    sx = SublimeX(metric=metric, objective_fn=mean_objective, verbose=True, deterministic=False, n_workers=6)
    if LOAD_FEATURES:
        if not os.path.exists(ppath):
            raise FileNotFoundError(f'Missing saved features: {ppath}')
        sx.load_features(ppath)
    else:
        sx.fit(tr_list, y_tr, initial_X=itr)
        if SAVE_FEATURES:
            os.makedirs(os.path.dirname(ppath) or '.', exist_ok=True)
            sx.save_features(ppath)
    tr_f, te_f = sx.transform(tr_list, initial_X=itr), sx.transform(te_list, initial_X=ite)
    n_feat = (itr.shape[1] if itr is not None else 0) + len(sx.extracted_features)
    return _lgb(metric, tr_f, te_f, y_tr, y_te), time.time() - t0, n_feat


def eval_cnn(X_tr, X_te, y_tr, y_te, metric, n_ch, ppath=None, itr=None, ite=None):
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    clf = metric != 'rmse'
    tr, va, ytr, yva = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr if clf else None)
    mean, std = tr.mean(), tr.std() + 1e-8
    tr, va, te = [torch.tensor(_ch((x - mean) / std, n_ch), dtype=torch.float32).to(device) for x in [tr, va, X_te]]
    ytr = torch.tensor(ytr, dtype=torch.long if clf else torch.float32).to(device)
    yva = torch.tensor(yva, dtype=torch.long if clf else torch.float32).to(device)
    out_dim = len(np.unique(y_tr)) if clf else 1
    model = nn.Sequential(
        nn.Conv1d(n_ch, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
        nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim)).to(device)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.CrossEntropyLoss() if clf else nn.MSELoss()
    best, patience, best_state = float('inf'), 0, None
    for _ in range(300):
        model.train()
        for i in range(0, len(tr), 128):
            opt.zero_grad()
            out = model(tr[i:i+128])
            (loss_fn(out, ytr[i:i+128]) if clf else loss_fn(out.squeeze(), ytr[i:i+128])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = (loss_fn(model(va), yva) if clf else loss_fn(model(va).squeeze(), yva)).item()
        if vloss < best:
            best, patience, best_state = vloss, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif (patience := patience + 1) >= 10:
            break
    if best_state:
        model.load_state_dict(best_state)
    if itr is not None:
        feat = nn.Sequential(*list(model.children())[:-1])
        with torch.no_grad():
            ftr = feat(torch.tensor(_ch(X_tr.astype(np.float32), n_ch), dtype=torch.float32).to(device)).cpu().numpy()
            fte = feat(torch.tensor(_ch(X_te.astype(np.float32), n_ch), dtype=torch.float32).to(device)).cpu().numpy()
        return _lgb(metric, np.hstack([itr, ftr]), np.hstack([ite, fte]), y_tr, y_te), time.time() - t0, ftr.shape[1] + itr.shape[1]
    with torch.no_grad():
        pred = model(te)
        pred = (torch.softmax(pred, 1)[:, 1] if metric == 'auc' and out_dim == 2 else pred.argmax(1) if clf else pred.squeeze()).cpu().numpy()
    sc = np.sqrt(mean_squared_error(y_te, pred)) if metric == 'rmse' else (roc_auc_score(y_te, pred) if metric == 'auc' else accuracy_score(y_te, pred))
    return sc, time.time() - t0, X_tr.shape[1]


def eval_minirocket(X_tr, X_te, y_tr, y_te, metric, n_ch, ppath=None, itr=None, ite=None):
    t0 = time.time()
    tr, te = _ch(np.asarray(X_tr, np.float32), n_ch), _ch(np.asarray(X_te, np.float32), n_ch)
    rocket = MiniRocketMultivariate(random_state=42, n_jobs=1)
    ftr, fte = rocket.fit_transform(tr), rocket.transform(te)
    ftr, fte = (ftr.values if hasattr(ftr, 'values') else ftr), (fte.values if hasattr(fte, 'values') else fte)
    if itr is not None:
        ftr, fte = np.hstack([itr, ftr]), np.hstack([ite, fte])
    return _lgb(metric, ftr, fte, y_tr, y_te), time.time() - t0, ftr.shape[1]


def eval_catch22(X_tr, X_te, y_tr, y_te, metric, n_ch, ppath=None, itr=None, ite=None):
    t0, nt = time.time(), X_tr.shape[1] // n_ch
    ext = lambda arr: np.nan_to_num(np.hstack([np.array([catch22_all(x)['values'] for x in arr[:, c*nt:(c+1)*nt]]) for c in range(n_ch)]), nan=0, posinf=0, neginf=0)
    ftr, fte = ext(X_tr), ext(X_te)
    if itr is not None:
        ftr, fte = np.hstack([itr, ftr]), np.hstack([ite, fte])
    return _lgb(metric, ftr, fte, y_tr, y_te), time.time() - t0, ftr.shape[1]


def eval_tsfresh(X_tr, X_te, y_tr, y_te, metric, n_ch, ppath=None, itr=None, ite=None):
    t0 = time.time()
    X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)
    nt, fc = X_tr.shape[1] // n_ch, MinimalFCParameters()
    def ext(arr):
        parts = []
        for c in range(n_ch):
            df = pd.DataFrame(arr[:, c*nt:(c+1)*nt]).assign(id=range(len(arr)))
            long = df.melt(id_vars=['id'], var_name='time', value_name='value').dropna(subset=['value'])
            parts.append(extract_features(long, column_id='id', column_sort='time', column_value='value',
                impute_function=impute, n_jobs=0, default_fc_parameters=fc, show_warnings=False, disable_progressbar=True))
        return pd.concat(parts, axis=1)
    ftr, fte = ext(X_tr), ext(X_te)
    cols = ftr.columns.intersection(fte.columns)
    ftr, fte = ftr[cols].values, fte[cols].values
    if itr is not None:
        ftr, fte = np.hstack([itr, ftr]), np.hstack([ite, fte])
    return _lgb(metric, ftr, fte, y_tr, y_te), time.time() - t0, ftr.shape[1]


def eval_rdst(X_tr, X_te, y_tr, y_te, metric, n_ch, ppath=None, itr=None, ite=None):
    t0 = time.time()
    tr, te = _ch(np.asarray(X_tr, np.float64), n_ch), _ch(np.asarray(X_te, np.float64), n_ch)
    y_fit = y_tr if metric == 'rmse' else np.unique(np.concatenate([y_tr, y_te]), return_inverse=True)[1][:len(y_tr)]
    rdst = RDST(max_shapelets=1000, n_jobs=1, random_state=42)
    ftr, fte = rdst.fit_transform(tr, y_fit), rdst.transform(te)
    ftr, fte = (ftr.values if hasattr(ftr, 'values') else ftr), (fte.values if hasattr(fte, 'values') else fte)
    if itr is not None:
        ftr, fte = np.hstack([itr, ftr]), np.hstack([ite, fte])
    return _lgb(metric, ftr, fte, y_tr, y_te), time.time() - t0, ftr.shape[1]


ALL_APPROACHES = {'SublimeX': eval_sublimex, 'CNN': eval_cnn, 'MiniRocket': eval_minirocket,
                  'catch22': eval_catch22, 'tsfresh': eval_tsfresh, 'RDST': eval_rdst}
APPROACHES = {'SublimeX': eval_sublimex} if LOAD_FEATURES else ALL_APPROACHES


def _done_keys(records):
    return {(r['dataset'], r['approach'], int(r['fold'])) for r in records}


def _azt1d_expected():
    _, _, info = load_azt1d()
    return len(np.unique(info['subject_ids'])) * len(ALL_APPROACHES)


def _azt1d_incomplete(records):
    n = sum(1 for r in records if r.get('dataset') == 'azt1d')
    return n < _azt1d_expected()


def _rest_eval_complete(records):
    df = pd.DataFrame(records)
    if df.empty:
        return False
    n_app = len(APPROACHES)
    for ds in ['mimic', 'emotions', 'mitbih', 'svd']:
        if len(df[df['dataset'] == ds]) < K_FOLDS * n_app:
            return False
    if len(df[df['dataset'] == 'pamap2']) < 8 * n_app:
        return False
    return len(df[df['dataset'].str.startswith('remc_', na=False)]) >= K_FOLDS * n_app


def _should_run_azt1d_only(records):
    return bool(records) and _rest_eval_complete(records) and _azt1d_incomplete(records)


def run_azt1d_patients(results, done):
    """Pooled SublimeX fit on all patients' train windows; per-patient 80/20 test RMSE."""
    _, _, info = load_azt1d()
    subjects = sorted(np.unique(info['subject_ids']))
    metric = 'rmse'
    approaches = list(ALL_APPROACHES)
    print(f'AZT1D: {len(approaches)} approaches × {len(subjects)} patients (pooled SublimeX fit)', flush=True)
    splits = {}
    for fi, subj in enumerate(subjects, 1):
        X, y, pinfo = load_azt1d(subject_id=subj)
        X_list = X if isinstance(X, list) else [X]
        n_ch, n = len(X_list), len(y)
        c = int(n * 0.8)
        tr, te = np.arange(c), np.arange(c, n)
        init = pinfo.get('initial_features')
        splits[fi] = (
            subj, n_ch,
            pd.concat([x.iloc[tr] for x in X_list], axis=1).values,
            pd.concat([x.iloc[te] for x in X_list], axis=1).values,
            y.iloc[tr].values, y.iloc[te].values,
            init[tr] if init is not None else None,
            init[te] if init is not None else None,
        )
    sx, fit_share = None, 0.0
    for ai, approach in enumerate(approaches, 1):
        todo = [fi for fi in range(1, len(subjects) + 1) if ('azt1d', approach, fi) not in done]
        if not todo:
            continue
        print(f'[{ai}/{len(approaches)}] {approach} — {len(todo)} patients', flush=True)
        if approach == 'SublimeX':
            sx, fit_t = _fit_azt1d_pooled_sublimex(splits, metric, force=len(todo) == len(subjects))
            fit_share = fit_t / len(subjects)
        for fi in todo:
            subj, n_ch, X_tr, X_te, y_tr, y_te, itr, ite = splits[fi]
            if approach == 'SublimeX':
                sc, elapsed, n_feat = eval_sublimex_azt1d_pooled(sx, X_tr, X_te, y_tr, y_te, metric, n_ch, itr, ite)
                elapsed += fit_share
            else:
                sc, elapsed, n_feat = ALL_APPROACHES[approach](X_tr, X_te, y_tr, y_te, metric, n_ch, None, itr, ite)
            row = {'dataset': 'azt1d', 'approach': approach, 'fold': fi, 'score': sc, 'time': elapsed,
                   'n_features': n_feat, 'test_subject': float(subj), 'cell_line': ''}
            results.append(row)
            done.add(('azt1d', approach, fi))
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            print(f'azt1d {fi}/{len(subjects)} | {approach} | rmse={sc:.4f} | {elapsed:.1f}s | feat={n_feat}', flush=True)


def run_and_save(results, done, ds, X_tr, X_te, y_tr, y_te, metric, n_ch, fold, approach, extra=None, itr=None, ite=None):
    key = (ds, approach, fold)
    if key in done:
        return
    ppath = _params_path(ds, fold) if approach == 'SublimeX' else None
    sc, elapsed, n_feat = APPROACHES[approach](X_tr, X_te, y_tr, y_te, metric, n_ch, ppath, itr, ite)
    row = {'dataset': ds, 'approach': approach, 'fold': fold, 'score': sc, 'time': elapsed, 'n_features': n_feat}
    if extra:
        row.update(extra)
    results.append(row)
    done.add(key)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f'{ds} fold {fold} | {approach} | {metric}={sc:.4f} | {elapsed:.1f}s | feat={n_feat}', flush=True)


def _run_main_eval():
    if LOAD_FEATURES:
        for ds in ['mimic', 'emotions', 'mitbih', 'svd', 'pamap2']:
            folds = range(1, (8 if ds == 'pamap2' else K_FOLDS) + 1)
            for fi in folds:
                p = _params_path(ds, fi)
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
        for cl in load_remc():
            for fi in range(1, K_FOLDS + 1):
                p = _params_path(f'remc_{cl}', fi)
                if not os.path.exists(p):
                    raise FileNotFoundError(p)

    results = pd.read_csv(RESULTS_PATH).to_dict('records') if os.path.exists(RESULTS_PATH) else []
    done = _done_keys(results)
    print(f'load_features={LOAD_FEATURES} save_features={SAVE_FEATURES} resume={len(done)} rows → {RESULTS_PATH}', flush=True)

    for ds, load_fn in STANDARD_DATASETS:
        X, y, info = load_fn()
        X_list = X if isinstance(X, list) else [X]
        n_ch, init = len(X_list), info.get('initial_features')
        cv = StratifiedKFold if info['task'] == 'classification' else KFold
        folds = list(cv(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
        for approach in APPROACHES:
            for fi, (tr, te) in enumerate(folds, 1):
                X_tr = pd.concat([x.iloc[tr] for x in X_list], axis=1).values
                X_te = pd.concat([x.iloc[te] for x in X_list], axis=1).values
                itr = init[tr] if init is not None else None
                ite = init[te] if init is not None else None
                run_and_save(results, done, ds, X_tr, X_te, y.iloc[tr].values, y.iloc[te].values,
                             info['metric'], n_ch, fi, approach, itr=itr, ite=ite)

    X, y, info = load_pamap2()
    X_list, subjects = X if isinstance(X, list) else [X], np.asarray(info['subject_ids'])
    n_ch, init = len(X_list), info.get('initial_features')
    for approach in APPROACHES:
        for fi, (tr, te) in enumerate(LeaveOneGroupOut().split(pd.concat(X_list, axis=1), y, groups=subjects), 1):
            X_tr = pd.concat([x.iloc[tr] for x in X_list], axis=1).values
            X_te = pd.concat([x.iloc[te] for x in X_list], axis=1).values
            itr = init[tr] if init is not None else None
            ite = init[te] if init is not None else None
            run_and_save(results, done, 'pamap2', X_tr, X_te, y.iloc[tr].values, y.iloc[te].values,
                         info['metric'], n_ch, fi, approach,
                         {'test_subject': int(np.unique(subjects[te])[0])}, itr, ite)

    run_azt1d_patients(results, done)

    for cell_line in load_remc():
        ds = f'remc_{cell_line}'
        X, y, info = load_remc(cell_line=cell_line)
        X_list, n_ch = X if isinstance(X, list) else [X], len(X if isinstance(X, list) else [X])
        folds = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
        for approach in APPROACHES:
            for fi, (tr, te) in enumerate(folds, 1):
                X_tr = pd.concat([x.iloc[tr] for x in X_list], axis=1).values
                X_te = pd.concat([x.iloc[te] for x in X_list], axis=1).values
                run_and_save(results, done, ds, X_tr, X_te, y.iloc[tr].values, y.iloc[te].values,
                             info['metric'], n_ch, fi, approach, {'cell_line': cell_line})

    print(f'done ({len(results)} rows)', flush=True)


if __name__ == '__main__':
    results = pd.read_csv(RESULTS_PATH).to_dict('records') if os.path.exists(RESULTS_PATH) else []
    if _should_run_azt1d_only(results):
        print('Other datasets complete, AZT1D incomplete → running AZT1D only', flush=True)
        run_azt1d_patients(results, _done_keys(results))
        print(f'done ({len(results)} rows)', flush=True)
    else:
        _run_main_eval()
