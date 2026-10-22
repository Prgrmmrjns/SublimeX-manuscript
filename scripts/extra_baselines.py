import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneGroupOut, train_test_split

from config import K_FOLDS, RESULTS
from preprocess import load_emotions, load_svd, load_mimic, load_mitbih, load_pamap2, load_azt1d, load_remc

OUT = RESULTS / 'extra_baselines.csv'
OLD = RESULTS / 'emotions_extra_baselines.csv'
ORDER = ['LSTM', 'InceptionTime', 'HIVECOTEV2']


def _ch(x, n_ch):
    return np.asarray(x, np.float32).reshape(x.shape[0], n_ch, -1)


def _xy(X_list, idx, y):
    yv = y.values if hasattr(y, 'values') else np.asarray(y)
    return pd.concat([x.iloc[idx] for x in X_list], axis=1).values, yv[idx]


def _one(ds, fi, metric, X_list, y, tr, te):
    Xtr, ytr = _xy(X_list, tr, y)
    Xte, yte = _xy(X_list, te, y)
    return ds, fi, metric, len(X_list), Xtr, Xte, ytr, yte


def _standard(name, fn):
    X, y, info = fn()
    X_list = X if isinstance(X, list) else [X]
    cv = StratifiedKFold if info['task'] == 'classification' else KFold
    splits = cv(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y)
    return [_one(name, fi, info['metric'], X_list, y, tr, te) for fi, (tr, te) in enumerate(splits, 1)]


def _pamap():
    X, y, info = load_pamap2()
    X_list, subjects = X if isinstance(X, list) else [X], np.asarray(info['subject_ids'])
    splits = LeaveOneGroupOut().split(pd.concat(X_list, axis=1), y, groups=subjects)
    return [_one('pamap2', fi, info['metric'], X_list, y, tr, te) for fi, (tr, te) in enumerate(splits, 1)]


def _azt1d():
    _, _, info = load_azt1d()
    rows = []
    for fi, subj in enumerate(sorted(np.unique(info['subject_ids'])), 1):
        X, y, _ = load_azt1d(subject_id=subj)
        X_list = X if isinstance(X, list) else [X]
        c = int(len(y) * 0.8)
        rows.append(_one('azt1d', fi, 'rmse', X_list, y, np.arange(c), np.arange(c, len(y))))
    return rows


def _remc():
    rows = []
    for cl in load_remc():
        X, y, info = load_remc(cell_line=cl)
        X_list = X if isinstance(X, list) else [X]
        splits = StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y)
        rows += [_one(f'remc_{cl}', fi, info['metric'], X_list, y, tr, te) for fi, (tr, te) in enumerate(splits, 1)]
    return rows


def datasets():
    for name, fn in [('emotions', load_emotions), ('svd', load_svd), ('mimic', load_mimic), ('mitbih', load_mitbih)]:
        yield _standard(name, fn)
    yield _azt1d()
    yield _pamap()
    yield _remc()


def _pool(x, n_ch, target):
    s = torch.tensor(_ch(x, n_ch), dtype=torch.float32)
    s = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-8)
    k = max(1, s.shape[-1] // target)
    return s if k == 1 else F.avg_pool1d(s, k, k)


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


def eval_lstm(X_tr, X_te, y_tr, y_te, n_ch, metric):
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    reg = metric == 'rmse'
    ytr_i, yte_i, n_cls = (y_tr, y_te, 1) if reg else _idx(y_tr, y_te)
    tr_x, va_x, ytr, yva = train_test_split(X_tr, ytr_i, test_size=0.2, random_state=42, stratify=None if reg else ytr_i)
    tr, va, te = [t.to(device) for t in (_pool(tr_x, n_ch, 60), _pool(va_x, n_ch, 60), _pool(X_te, n_ch, 60))]
    tr, va = tr.transpose(1, 2), va.transpose(1, 2)
    te = te.transpose(1, 2)
    ytr = torch.tensor(ytr, dtype=torch.float32 if reg else torch.long, device=device)
    yva = torch.tensor(yva, dtype=torch.float32 if reg else torch.long, device=device)
    lstm = nn.LSTM(n_ch, 64, batch_first=True).to(device)
    head = nn.Linear(64, 1 if reg else n_cls).to(device)
    opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss() if reg else nn.CrossEntropyLoss()

    def logits(seq):
        out, _ = lstm(seq)
        z = head(out[:, -1])
        return z.squeeze(-1) if reg else z

    best, best_lstm, best_head = float('inf'), None, None
    for _ in range(100):
        lstm.train(); head.train()
        for i in range(0, len(tr), 64):
            opt.zero_grad()
            loss_fn(logits(tr[i:i + 64]), ytr[i:i + 64]).backward()
            opt.step()
        lstm.eval(); head.eval()
        with torch.no_grad():
            vloss = loss_fn(logits(va), yva).item()
        if vloss < best:
            best = vloss
            best_lstm = {k: v.detach().cpu().clone() for k, v in lstm.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    lstm.load_state_dict(best_lstm); head.load_state_dict(best_head)
    lstm.eval(); head.eval()
    with torch.no_grad():
        z = logits(te).cpu().numpy()
    if metric == 'auc':
        pred = torch.softmax(torch.tensor(z), 1).numpy()[:, 1]
    elif metric == 'accuracy':
        pred = z.argmax(1)
    else:
        pred = z
    n_param = sum(p.numel() for p in list(lstm.parameters()) + list(head.parameters()))
    return _metric_score(metric, y_te if reg else yte_i, pred), time.time() - t0, n_param


def _same(c_in, c_out, k):
    return nn.Conv1d(c_in, c_out, k, padding='same', bias=False)


class _Inc(nn.Module):
    def __init__(self, c_in, nf=32):
        super().__init__()
        self.bneck = _same(c_in, 32, 1) if c_in > 1 else nn.Identity()
        cin = 32 if c_in > 1 else c_in
        self.cs = nn.ModuleList([_same(cin, nf, k) for k in (39, 19, 9)])
        self.mp = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), _same(c_in, nf, 1))
        self.bn = nn.BatchNorm1d(nf * 4)

    def forward(self, x):
        z = self.bneck(x)
        y = torch.cat([c(z) for c in self.cs] + [self.mp(x)], 1)
        return F.relu(self.bn(y))


class _IncNet(nn.Module):
    def __init__(self, c_in, n_out):
        super().__init__()
        self.mods = nn.ModuleList()
        ch = c_in
        for _ in range(6):
            self.mods.append(_Inc(ch))
            ch = 128
        self.short = nn.ModuleList([
            nn.Sequential(_same(c_in, 128, 1), nn.BatchNorm1d(128)),
            nn.Sequential(_same(128, 128, 1), nn.BatchNorm1d(128)),
        ])
        self.head = nn.Linear(128, n_out)

    def forward(self, x):
        res, si = x, 0
        for i, m in enumerate(self.mods):
            x = m(x)
            if (i + 1) % 3 == 0:
                x = F.relu(x + self.short[si](res))
                res, si = x, si + 1
        return self.head(x.mean(-1))


def eval_inception(X_tr, X_te, y_tr, y_te, n_ch, metric):
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    reg = metric == 'rmse'
    ytr_i, yte_i, n_cls = (y_tr.astype(np.float32), y_te.astype(np.float32), 1) if reg else _idx(y_tr, y_te)
    tr = _pool(X_tr, n_ch, 140).to(device)
    te = _pool(X_te, n_ch, 140).to(device)
    ytr = torch.tensor(ytr_i, dtype=torch.float32 if reg else torch.long, device=device)
    n_out, preds, n_param = (1 if reg else n_cls), [], 0
    loss_fn = nn.MSELoss() if reg else nn.CrossEntropyLoss()
    bs = 512
    for m in range(5):
        torch.manual_seed(42 + m)
        net = _IncNet(n_ch, n_out).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        net.train()
        for _ in range(40):
            perm = torch.randperm(len(tr), device=device)
            xb, yb = tr[perm], ytr[perm]
            for i in range(0, len(xb), bs):
                opt.zero_grad(set_to_none=True)
                z = net(xb[i:i + bs])
                loss_fn(z.squeeze(-1) if reg else z, yb[i:i + bs]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            z = torch.cat([net(te[i:i + bs]) for i in range(0, len(te), bs)])
            preds.append(z.squeeze(-1) if reg else torch.softmax(z, 1))
        n_param += sum(p.numel() for p in net.parameters())
        del net
        print(f'  inception model {m + 1}/5', flush=True)
    stacked = torch.stack(preds).mean(0).cpu().numpy()
    if metric == 'auc':
        pred = stacked[:, 1]
    elif metric == 'accuracy':
        pred = stacked.argmax(1)
    else:
        pred = stacked
    return _metric_score(metric, y_te if reg else yte_i, pred), time.time() - t0, n_param


def eval_hc2(X_tr, X_te, y_tr, y_te, n_ch, metric):
    t0 = time.time()

    def norm(x):
        s = _ch(x, n_ch).astype(np.float64)
        return (s - s.mean(-1, keepdims=True)) / (s.std(-1, keepdims=True) + 1e-8)

    clf = HIVECOTEV2(random_state=42, n_jobs=-1, verbose=0)
    clf.fit(norm(X_tr), y_tr)
    if metric == 'auc':
        proba = clf.predict_proba(norm(X_te))
        classes = list(clf.classes_)
        pred = proba[:, classes.index(1) if 1 in classes else classes.index(1.0)]
    else:
        pred = clf.predict(norm(X_te))
    return _metric_score(metric, y_te, pred), time.time() - t0, 4


FNS = {'LSTM': eval_lstm, 'InceptionTime': eval_inception, 'HIVECOTEV2': eval_hc2}


def main():
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict('records')
    elif OLD.exists():
        rows = pd.read_csv(OLD).to_dict('records')
    else:
        rows = []
    done = {(r['dataset'], r['approach'], int(r['fold'])) for r in rows}
    print(f'resume={len(done)} → {OUT}', flush=True)
    for name in ORDER:
        for folds in datasets():
            for ds, fi, metric, n_ch, X_tr, X_te, y_tr, y_te in folds:
                if name == 'HIVECOTEV2' and metric == 'rmse':
                    continue
                if (ds, name, fi) in done:
                    continue
                print(f'start {ds} fold {fi} | {name}', flush=True)
                sc, elapsed, n_feat = FNS[name](X_tr, X_te, y_tr, y_te, n_ch, metric)
                rows.append({'dataset': ds, 'approach': name, 'fold': fi, 'score': sc, 'time': elapsed, 'n_features': n_feat})
                done.add((ds, name, fi))
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(f'{ds} fold {fi} | {name} | {metric}={sc:.4f} | {elapsed:.1f}s | feat={n_feat}', flush=True)
    print(pd.DataFrame(rows).groupby(['dataset', 'approach'])['score'].agg(['mean', 'std']).to_string(), flush=True)


if __name__ == '__main__':
    main()
