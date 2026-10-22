"""UEA equal-length multivariate archive, default train/test split.

One row per dataset and method in results/uea_eval.csv.
Resumes. Does not resample and does not run the 2021 bake-off classifiers.
"""
import os
os.environ.update({
    'KMP_WARNINGS': '0', 'OMP_WARNINGS': '0', 'OMP_DISPLAY_ENV': 'FALSE',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'NUMBA_THREADING_LAYER': 'workqueue',
})
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multivariate_equal_length

import main_eval as ev
from config import RESULTS

ev.LOAD_FEATURES = False
ev.SAVE_FEATURES = False
ev.N_WORKERS = -1
OUT = RESULTS / 'uea_eval.csv'
METHODS = ('SublimeX',)


def _pack(X):
    X = np.nan_to_num(np.asarray(X, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n, c, t = X.shape
    return np.ascontiguousarray(X.reshape(n, c * t)), c


def _labels(ytr, yte):
    classes, ytr_i = np.unique(ytr, return_inverse=True)
    index = {c: i for i, c in enumerate(classes)}
    return ytr_i.astype(int), np.array([index[v] for v in yte], dtype=int)


def _run(name, Xtr, Xte, ytr, yte, n_ch):
    return ev.ALL_APPROACHES[name](Xtr, Xte, ytr, yte, 'accuracy', n_ch, None, None, None)


def main():
    done = set()
    rows = []
    if OUT.exists():
        prev = pd.read_csv(OUT)
        rows = prev.to_dict('records')
        done = {(r['dataset'], r['approach']) for r in rows}
    print(f'resume={len(done)} → {OUT}', flush=True)
    for ds in multivariate_equal_length:
        todo = [m for m in METHODS if (ds, m) not in done]
        if not todo:
            continue
        Xtr, ytr = load_classification(ds, split='train')
        Xte, yte = load_classification(ds, split='test')
        Xtr, n_ch = _pack(Xtr)
        Xte, _ = _pack(Xte)
        ytr, yte = _labels(ytr, yte)
        print(f'{ds} train={len(ytr)} test={len(yte)} ch={n_ch} len={Xtr.shape[1] // n_ch}', flush=True)
        for method in todo:
            try:
                score, elapsed, n_feat = _run(method, Xtr, Xte, ytr, yte, n_ch)
                err = ''
            except Exception as exc:
                score, elapsed, n_feat, err = float('nan'), 0.0, 0, f'{type(exc).__name__}: {exc}'
            row = {'dataset': ds, 'approach': method, 'score': score, 'time': elapsed,
                   'n_features': n_feat, 'error': err}
            rows.append(row)
            done.add((ds, method))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            msg = err or f'acc={score:.4f} {elapsed:.1f}s feat={n_feat}'
            print(f'  {method} {msg}', flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby('approach')['score'].mean().reindex(METHODS).to_string(), flush=True)


if __name__ == '__main__':
    main()
