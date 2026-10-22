import os
os.environ.update({
    'KMP_WARNINGS': '0', 'OMP_WARNINGS': '0', 'OMP_DISPLAY_ENV': 'FALSE',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
})
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import K_FOLDS, RESULTS
from preprocess import load_emotions

OUT = RESULTS / 'emotions_extra_baselines.csv'


def _ch(x, n_ch):
    return x.reshape(x.shape[0], n_ch, -1)


def _folds():
    X, y, info = load_emotions()
    X_list = X if isinstance(X, list) else [X]
    n_ch = len(X_list)
    yv = y.values if hasattr(y, 'values') else np.asarray(y)
    splits = StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), yv)
    folds = []
    for tr, te in splits:
        folds.append((
            pd.concat([x.iloc[tr] for x in X_list], axis=1).values,
            pd.concat([x.iloc[te] for x in X_list], axis=1).values,
            yv[tr], yv[te], n_ch,
        ))
    return folds


def _lstm_seq(x, n_ch):
    s = torch.tensor(_ch(np.asarray(x, np.float32), n_ch), dtype=torch.float32)
    s = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-8)
    return nn.functional.avg_pool1d(s, kernel_size=21, stride=21).transpose(1, 2)


def eval_lstm(X_tr, X_te, y_tr, y_te, n_ch):
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tr_x, va_x, ytr, yva = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    tr, va, te = [t.to(device) for t in (_lstm_seq(tr_x, n_ch), _lstm_seq(va_x, n_ch), _lstm_seq(X_te, n_ch))]
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    yva = torch.tensor(yva, dtype=torch.long, device=device)
    n_cls, hidden = len(np.unique(y_tr)), 64
    lstm = nn.LSTM(n_ch, hidden, batch_first=True).to(device)
    head = nn.Linear(hidden, n_cls).to(device)
    opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def logits(seq):
        out, _ = lstm(seq)
        return head(out[:, -1])

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
    lstm.load_state_dict(best_lstm)
    head.load_state_dict(best_head)
    lstm.eval(); head.eval()
    with torch.no_grad():
        pred = logits(te).argmax(1).cpu().numpy()
    n_param = sum(p.numel() for p in list(lstm.parameters()) + list(head.parameters()))
    return accuracy_score(y_te, pred), time.time() - t0, n_param


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
    def __init__(self, c_in, n_cls):
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
        self.head = nn.Linear(128, n_cls)

    def forward(self, x):
        res, si = x, 0
        for i, m in enumerate(self.mods):
            x = m(x)
            if (i + 1) % 3 == 0:
                x = F.relu(x + self.short[si](res))
                res, si = x, si + 1
        return self.head(x.mean(-1))


def _it_x(x, n_ch):
    s = torch.tensor(_ch(np.asarray(x, np.float32), n_ch), dtype=torch.float32)
    s = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-8)
    return F.avg_pool1d(s, 9, 9)


def eval_inception(X_tr, X_te, y_tr, y_te, n_ch):
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tr = _it_x(X_tr, n_ch).to(device)
    te = _it_x(X_te, n_ch).to(device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    n_cls, probs, n_param = len(np.unique(y_tr)), [], 0
    for m in range(5):
        torch.manual_seed(42 + m)
        net = _IncNet(n_ch, n_cls).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        net.train()
        for _ in range(40):
            perm = torch.randperm(len(tr), device=device)
            for i in range(0, len(tr), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss_fn(net(tr[b]), ytr[b]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            probs.append(torch.softmax(net(te), 1))
        n_param += sum(p.numel() for p in net.parameters())
        print(f'  inception model {m + 1}/5', flush=True)
    pred = torch.stack(probs).mean(0).argmax(1).cpu().numpy()
    return accuracy_score(y_te, pred), time.time() - t0, n_param


def main():
    rows = pd.read_csv(OUT).to_dict('records') if OUT.exists() else []
    done = {(r['approach'], int(r['fold'])) for r in rows}
    folds = _folds()
    for name, fn in [('LSTM', eval_lstm), ('InceptionTime', eval_inception)]:
        for fi, (X_tr, X_te, y_tr, y_te, n_ch) in enumerate(folds, 1):
            if (name, fi) in done:
                continue
            sc, elapsed, n_feat = fn(X_tr, X_te, y_tr, y_te, n_ch)
            rows.append({'dataset': 'emotions', 'approach': name, 'fold': fi,
                         'score': sc, 'time': elapsed, 'n_features': n_feat})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f'emotions fold {fi} | {name} | accuracy={sc:.4f} | {elapsed:.1f}s | feat={n_feat}', flush=True)
    print(pd.DataFrame(rows).groupby('approach')['score'].agg(['mean', 'std']).to_string(), flush=True)


if __name__ == '__main__':
    main()
