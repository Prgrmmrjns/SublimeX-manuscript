"""Probe LSTM with train-size batch: bs = min(256, max(32, n_train // 8))."""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import K_FOLDS, RESULTS, setup_sublimex_path
setup_sublimex_path()
from sublimex import LightGBMModel
from preprocess import load_emotions, load_svd, load_mimic

OUT = RESULTS / 'lstm_batch_probe.csv'
LOADERS = [('emotions', load_emotions), ('svd', load_svd), ('mimic', load_mimic)]


def _ch(x, n_ch):
    return np.asarray(x, np.float32).reshape(x.shape[0], n_ch, -1)


def _idx(y_tr, y_te):
    classes = np.unique(y_tr)
    m = {c: i for i, c in enumerate(classes)}
    return np.array([m[v] for v in y_tr]), np.array([m[v] for v in y_te]), len(classes)


def _metric_score(metric, y, pred):
    if metric == 'rmse':
        return float(np.sqrt(mean_squared_error(y, pred)))
    if metric == 'auc':
        return float(roc_auc_score(y, pred))
    return float(accuracy_score(y, pred))


def _batch(n_train):
    return int(min(256, max(32, n_train // 8)))


def eval_lstm(X_tr, X_te, y_tr, y_te, n_ch, metric, itr=None, ite=None):
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    reg = metric == 'rmse'
    ytr_i, yte_i, n_cls = (y_tr, y_te, 1) if reg else _idx(y_tr, y_te)
    idx = np.arange(len(X_tr))
    tr_i, va_i = train_test_split(idx, test_size=0.2, random_state=42, stratify=None if reg else ytr_i)
    bs = _batch(len(tr_i))
    mean, std = X_tr[tr_i].mean(), X_tr[tr_i].std() + 1e-8

    def _seq(x):
        return torch.tensor(_ch((x - mean) / std, n_ch), dtype=torch.float32).transpose(1, 2).to(device)

    tr, va, te = _seq(X_tr[tr_i]), _seq(X_tr[va_i]), _seq(X_te)
    ytr = torch.tensor(ytr_i[tr_i], dtype=torch.float32 if reg else torch.long, device=device)
    yva = torch.tensor(ytr_i[va_i], dtype=torch.float32 if reg else torch.long, device=device)
    lstm = nn.LSTM(n_ch, 64, batch_first=True).to(device)
    head = nn.Linear(64, 1 if reg else n_cls).to(device)
    opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss() if reg else nn.CrossEntropyLoss()

    def hidden(seq):
        out, _ = lstm(seq)
        return out.mean(1)

    def logits(seq):
        z = head(hidden(seq))
        return z.squeeze(-1) if reg else z

    best, best_lstm, best_head, patience = float('inf'), None, None, 0
    for _ in range(100):
        lstm.train(); head.train()
        for i in range(0, len(tr), bs):
            opt.zero_grad()
            loss_fn(logits(tr[i:i + bs]), ytr[i:i + bs]).backward()
            opt.step()
        lstm.eval(); head.eval()
        with torch.no_grad():
            vloss = loss_fn(logits(va), yva).item()
        if vloss < best:
            best, patience = vloss, 0
            best_lstm = {k: v.detach().cpu().clone() for k, v in lstm.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        elif (patience := patience + 1) >= 10:
            break
    lstm.load_state_dict(best_lstm); head.load_state_dict(best_head)
    lstm.eval(); head.eval()
    n_param = sum(p.numel() for p in list(lstm.parameters()) + list(head.parameters()))
    if itr is not None:
        with torch.no_grad():
            ftr = hidden(_seq(X_tr)).cpu().numpy()
            fte = hidden(te).cpu().numpy()
        sc = LightGBMModel('regression' if reg else 'classification').test(
            np.hstack([itr, ftr]), y_tr, np.hstack([ite, fte]), y_te, metric, deterministic=True)
        return sc, time.time() - t0, n_param + itr.shape[1], bs
    with torch.no_grad():
        z = logits(te).cpu().numpy()
    if metric == 'auc':
        pred = torch.softmax(torch.tensor(z), 1).numpy()[:, 1]
    elif metric == 'accuracy':
        pred = z.argmax(1)
    else:
        pred = z
    return _metric_score(metric, y_te if reg else yte_i, pred), time.time() - t0, n_param, bs


def folds(name, fn):
    X, y, info = fn()
    X_list = X if isinstance(X, list) else [X]
    init = info.get('initial_features')
    yv = y.values if hasattr(y, 'values') else np.asarray(y)
    splits = StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), yv)
    for fi, (tr, te) in enumerate(splits, 1):
        Xtr = pd.concat([x.iloc[tr] for x in X_list], axis=1).values
        Xte = pd.concat([x.iloc[te] for x in X_list], axis=1).values
        itr = init[tr] if init is not None else None
        ite = init[te] if init is not None else None
        yield name, fi, info['metric'], len(X_list), Xtr, Xte, yv[tr], yv[te], itr, ite


def main():
    rows = []
    for name, fn in LOADERS:
        for ds, fi, metric, n_ch, X_tr, X_te, y_tr, y_te, itr, ite in folds(name, fn):
            print(f'start {ds} fold {fi}', flush=True)
            sc, elapsed, n_feat, bs = eval_lstm(X_tr, X_te, y_tr, y_te, n_ch, metric, itr, ite)
            rows.append({'dataset': ds, 'fold': fi, 'score': sc, 'time': elapsed, 'n_features': n_feat, 'batch': bs})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f'{ds} fold {fi} | {metric}={sc:.4f} | {elapsed:.1f}s | bs={bs}', flush=True)
    df = pd.DataFrame(rows)
    print('\n=== adaptive batch ===')
    print(df.groupby('dataset').agg(bs=('batch', 'first'), score=('score', 'mean'), std=('score', 'std'), sec=('time', 'mean')).round(4).to_string())
    old = pd.read_csv(RESULTS / 'extra_baselines.csv')
    old = old[(old.approach == 'LSTM') & (old.dataset.isin(df.dataset.unique()))]
    print('\n=== batch 64 prior ===')
    print(old.groupby('dataset').agg(score=('score', 'mean'), std=('score', 'std'), sec=('time', 'mean')).round(4).to_string())


if __name__ == '__main__':
    main()
