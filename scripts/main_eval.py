"""Main evaluation script for SublimeX and baselines."""
import pandas as pd
import numpy as np
import time
import os
import warnings
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import torch
import torch.nn as nn
from pycatch22 import catch22_all
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform as RDST
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from model import LightGBMModel
from preprocess import load_emotions, load_mimic, load_mitbih, load_remc, load_svd, load_pamap2, load_azt1d
from core import SublimeX, mean_objective

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='overflow encountered in reduce')

# Configuration
K_FOLDS = 5
RESULTS_PATH = '../results/main_eval.csv'
PARAMS_DIR = '../parameters'

STANDARD_DATASETS = [('mimic', load_mimic), ('emotions', load_emotions), ('mitbih', load_mitbih), ('svd', load_svd)]

def _to_channels(arr, n_channels):
    """Reshape flat array to (n_samples, n_channels, n_time)."""
    if n_channels > 1:
        return arr.reshape(arr.shape[0], n_channels, -1)
    return arr[:, None, :]


# =============================================================================
# Baseline evaluation functions
# =============================================================================

def eval_sublimex(X_train, X_test, y_train, y_test, metric, n_channels, params_path=None, initial_tr=None, initial_te=None):
    t0 = time.time()
    n_time = X_train.shape[1] // n_channels
    train_list = [pd.DataFrame(X_train[:, i*n_time:(i+1)*n_time]) for i in range(n_channels)]
    test_list = [pd.DataFrame(X_test[:, i*n_time:(i+1)*n_time]) for i in range(n_channels)]
    
    sublimex = SublimeX(metric=metric, objective_fn=mean_objective)
    train_feat = sublimex.fit(train_list, y_train, initial_X=initial_tr).transform(train_list, initial_X=initial_tr)
    test_feat = sublimex.transform(test_list, initial_X=initial_te)
    
    #if params_path:
        #os.makedirs(os.path.dirname(params_path), exist_ok=True)
        #sublimex.save_features(params_path)
    
    model = LightGBMModel('regression' if metric == 'rmse' else 'classification')
    n_feat = (initial_tr.shape[1] if initial_tr is not None else 0) + len(sublimex.extracted_features)
    return model.test(train_feat, y_train, test_feat, y_test, metric), time.time() - t0, n_feat


def eval_cnn(X_train, X_test, y_train, y_test, metric, n_channels, params_path=None, initial_tr=None, initial_te=None):
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    is_clf = metric != 'rmse'
    
    tr, va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train if is_clf else None)
    mean, std = tr.mean(), tr.std() + 1e-8
    tr, va, te = (tr - mean) / std, (va - mean) / std, (X_test - mean) / std
    
    tr, va, te = [torch.tensor(_to_channels(x, n_channels), dtype=torch.float32).to(device) for x in [tr, va, te]]
    y_tr = torch.tensor(y_tr, dtype=torch.long if is_clf else torch.float32).to(device)
    y_va = torch.tensor(y_va, dtype=torch.long if is_clf else torch.float32).to(device)
    
    out_dim = len(np.unique(y_train)) if is_clf else 1
    model = nn.Sequential(
        nn.Conv1d(n_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
        nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim)
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
    best_state, best_loss, patience = None, float('inf'), 0
    
    for _ in range(300):
        model.train()
        for i in range(0, len(tr), 128):
            opt.zero_grad()
            out = model(tr[i:i+128])
            (loss_fn(out, y_tr[i:i+128]) if is_clf else loss_fn(out.squeeze(), y_tr[i:i+128])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = (loss_fn(model(va), y_va) if is_clf else loss_fn(model(va).squeeze(), y_va)).item()
        if val_loss < best_loss:
            best_loss, patience, best_state = val_loss, 0, {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10: break
    
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        pred = model(te)
        pred = (torch.softmax(pred, 1)[:, 1] if metric == 'auc' and out_dim == 2 else pred.argmax(1) if is_clf else pred.squeeze()).cpu().numpy()
    
    if initial_tr is not None:
        feat_model = nn.Sequential(*list(model.children())[:-1])  # penultimate layer (64-dim)
        with torch.no_grad():
            train_feat = feat_model(torch.tensor(_to_channels(X_train.astype(np.float32), n_channels), dtype=torch.float32).to(device)).cpu().numpy()
            test_feat = feat_model(torch.tensor(_to_channels(X_test.astype(np.float32), n_channels), dtype=torch.float32).to(device)).cpu().numpy()
        train_feat = np.hstack([initial_tr, train_feat])
        test_feat = np.hstack([initial_te, test_feat])
        lgb = LightGBMModel('regression' if metric == 'rmse' else 'classification')
        score = lgb.test(train_feat, y_train, test_feat, y_test, metric)
        n_feat = train_feat.shape[1]
    else:
        score = np.sqrt(mean_squared_error(y_test, pred)) if metric == 'rmse' else (roc_auc_score(y_test, pred) if metric == 'auc' else accuracy_score(y_test, pred))
        n_feat = X_train.shape[1]
    return score, time.time() - t0, n_feat


def eval_minirocket(X_train, X_test, y_train, y_test, metric, n_channels, params_path=None, initial_tr=None, initial_te=None):
    t0 = time.time()
    tr, te = [np.asarray(x, dtype=np.float32) for x in [X_train, X_test]]
    tr, te = _to_channels(tr, n_channels), _to_channels(te, n_channels)
    
    rocket = MiniRocketMultivariate(n_jobs=-1)
    ft_tr = rocket.fit_transform(tr)
    ft_te = rocket.transform(te)
    ft_tr = ft_tr.values if hasattr(ft_tr, 'values') else ft_tr
    ft_te = ft_te.values if hasattr(ft_te, 'values') else ft_te
    if initial_tr is not None:
        ft_tr = np.hstack([initial_tr, ft_tr])
        ft_te = np.hstack([initial_te, ft_te])
    model = LightGBMModel('regression' if metric == 'rmse' else 'classification')
    return model.test(ft_tr, y_train, ft_te, y_test, metric), time.time() - t0, ft_tr.shape[1]


def eval_catch22(X_train, X_test, y_train, y_test, metric, n_channels, params_path=None, initial_tr=None, initial_te=None):
    t0 = time.time()
    n_time = X_train.shape[1] // n_channels
    
    def extract(arr):
        feats = []
        for ch in range(n_channels):
            feats.append(np.array([catch22_all(x)['values'] for x in arr[:, ch*n_time:(ch+1)*n_time]]))
        return np.nan_to_num(np.hstack(feats), nan=0, posinf=0, neginf=0)
    
    ft_tr, ft_te = extract(X_train), extract(X_test)
    if initial_tr is not None:
        ft_tr = np.hstack([initial_tr, ft_tr])
        ft_te = np.hstack([initial_te, ft_te])
    model = LightGBMModel('regression' if metric == 'rmse' else 'classification')
    return model.test(ft_tr, y_train, ft_te, y_test, metric), time.time() - t0, ft_tr.shape[1]


def eval_tsfresh(X_train, X_test, y_train, y_test, metric, n_channels, params_path=None, initial_tr=None, initial_te=None):
    t0 = time.time()
    X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64), nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(np.asarray(X_test, dtype=np.float64), nan=0, posinf=0, neginf=0)
    n_time = X_train.shape[1] // n_channels
    fc = MinimalFCParameters()
    
    def extract(arr):
        feats = []
        for ch in range(n_channels):
            df = pd.DataFrame(arr[:, ch*n_time:(ch+1)*n_time]).assign(id=range(len(arr)))
            long = df.melt(id_vars=['id'], var_name='time', value_name='value').dropna(subset=['value'])
            ft = extract_features(long, column_id='id', column_sort='time', column_value='value',
                                  impute_function=impute, n_jobs=0, default_fc_parameters=fc,
                                  show_warnings=False, disable_progressbar=True)
            feats.append(ft)
        return pd.concat(feats, axis=1)
    
    ft_tr, ft_te = extract(X_train), extract(X_test)
    common = ft_tr.columns.intersection(ft_te.columns)
    ft_tr, ft_te = ft_tr[common].values, ft_te[common].values
    if initial_tr is not None:
        ft_tr = np.hstack([initial_tr, ft_tr])
        ft_te = np.hstack([initial_te, ft_te])
    model = LightGBMModel('regression' if metric == 'rmse' else 'classification')
    return model.test(ft_tr, y_train, ft_te, y_test, metric), time.time() - t0, ft_tr.shape[1]


def eval_rdst(X_train, X_test, y_train, y_test, metric, n_channels, params_path=None, initial_tr=None, initial_te=None):
    t0 = time.time()
    tr, te = [np.asarray(x, dtype=np.float64) for x in [X_train, X_test]]
    tr, te = _to_channels(tr, n_channels), _to_channels(te, n_channels)
    
    y_fit = y_train if metric == 'rmse' else np.unique(np.concatenate([y_train, y_test]), return_inverse=True)[1][:len(y_train)]
    rdst = RDST(max_shapelets=1000, n_jobs=-1, random_state=42)
    ft_tr = rdst.fit_transform(tr, y_fit)
    ft_te = rdst.transform(te)
    ft_tr = ft_tr.values if hasattr(ft_tr, 'values') else ft_tr
    ft_te = ft_te.values if hasattr(ft_te, 'values') else ft_te
    if initial_tr is not None:
        ft_tr = np.hstack([initial_tr, ft_tr])
        ft_te = np.hstack([initial_te, ft_te])
    model = LightGBMModel('regression' if metric == 'rmse' else 'classification')
    return model.test(ft_tr, y_train, ft_te, y_test, metric), time.time() - t0, ft_tr.shape[1]


APPROACHES = {'SublimeX': eval_sublimex,'CNN': eval_cnn, 'MiniRocket': eval_minirocket, 'catch22': eval_catch22, 'tsfresh': eval_tsfresh, 'RDST': eval_rdst}


# =============================================================================
# Main
# =============================================================================

def is_done(df, ds, approach, fold):
    return not df.empty and ((df['dataset'] == ds) & (df['approach'] == approach) & (df['fold'] == fold)).any()


def run_and_save(results, df, ds_name, X_tr, X_te, y_tr, y_te, metric, n_ch, fold, approach, extra=None, initial_tr=None, initial_te=None):
    if is_done(df, ds_name, approach, fold): return
    print(f"{ds_name} | {approach} | Fold {fold}")
    
    params_path = f'{PARAMS_DIR}/{ds_name}/fold{fold}.json' if approach == 'SublimeX' else None
    score, elapsed, n_feat = APPROACHES[approach](X_tr, X_te, y_tr, y_te, metric, n_ch, params_path, initial_tr=initial_tr, initial_te=initial_te)
    
    result = {'dataset': ds_name, 'approach': approach, 'fold': fold, 'score': score, 'time': elapsed, 'n_features': n_feat}
    if extra: result.update(extra)
    results.append(result)
    print(f"  {metric}={score:.4f} ({elapsed:.1f}s)")
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)


def main():
    existing = pd.read_csv(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else pd.DataFrame()
    results = existing.to_dict('records') if not existing.empty else []
    
    # Standard datasets (5-fold CV)
    for ds_name, load_fn in STANDARD_DATASETS:
        X, y, info = load_fn()
        X_list = X if isinstance(X, list) else [X]
        n_ch = len(X_list)
        folds = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
        init = info.get('initial_features')
        for approach in APPROACHES:
            for fold_idx, (tr_idx, te_idx) in enumerate(folds, 1):
                X_tr = pd.concat([x.iloc[tr_idx] for x in X_list], axis=1).values
                X_te = pd.concat([x.iloc[te_idx] for x in X_list], axis=1).values
                initial_tr = init[tr_idx] if init is not None else None
                initial_te = init[te_idx] if init is not None else None
                run_and_save(results, existing, ds_name, X_tr, X_te, y.iloc[tr_idx].values, y.iloc[te_idx].values, info['metric'], n_ch, fold_idx, approach, initial_tr=initial_tr, initial_te=initial_te)
    
    # PAMAP2 (LOSO)
    X, y, info = load_pamap2()
    X_list = X if isinstance(X, list) else [X]
    n_ch = len(X_list)
    subjects = info['subject_ids']
    folds = list(LeaveOneGroupOut().split(pd.concat(X_list, axis=1), y, groups=subjects))
    
    for approach in APPROACHES:
        for fold_idx, (tr_idx, te_idx) in enumerate(folds, 1):
            X_tr = pd.concat([x.iloc[tr_idx] for x in X_list], axis=1).values
            X_te = pd.concat([x.iloc[te_idx] for x in X_list], axis=1).values
            run_and_save(results, existing, 'pamap2', X_tr, X_te, y.iloc[tr_idx].values, y.iloc[te_idx].values,
                         info['metric'], n_ch, fold_idx, approach, {'test_subject': int(np.unique(subjects)[fold_idx-1])})


    # AZT1D (single model on all patients; 80/20 temporal split per patient to avoid leakage)
    X, y, info = load_azt1d()
    subject_ids = info['subject_ids']
    X_list = X if isinstance(X, list) else [X]
    n_ch = len(X_list)
    tr_indices, te_indices = [], []
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        n = mask.sum()
        cutoff = int(n * 0.8)
        subj_idx = np.where(mask)[0]
        tr_indices.extend(subj_idx[:cutoff])
        te_indices.extend(subj_idx[cutoff:])
    tr_idx, te_idx = np.array(tr_indices), np.array(te_indices)
    X_tr = pd.concat([x.iloc[tr_idx] for x in X_list], axis=1).values
    X_te = pd.concat([x.iloc[te_idx] for x in X_list], axis=1).values
    y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
    for approach in APPROACHES:
        run_and_save(results, existing, 'azt1d', X_tr, X_te, y_tr, y_te, info['metric'], n_ch, 1, approach)
        
    # REMC (per cell line, 5-fold CV)
    for cell_line in load_remc():
        ds_name = f"remc_{cell_line}"
        X, y, info = load_remc(cell_line=cell_line)
        X_list = X if isinstance(X, list) else [X]
        n_ch = len(X_list)
        folds = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
        
        for approach in APPROACHES:
            for fold_idx, (tr_idx, te_idx) in enumerate(folds, 1):
                X_tr = pd.concat([x.iloc[tr_idx] for x in X_list], axis=1).values
                X_te = pd.concat([x.iloc[te_idx] for x in X_list], axis=1).values
                run_and_save(results, existing, ds_name, X_tr, X_te, y.iloc[tr_idx].values, y.iloc[te_idx].values,
                             info['metric'], n_ch, fold_idx, approach, {'cell_line': cell_line})
    



if __name__ == '__main__':
    main()
