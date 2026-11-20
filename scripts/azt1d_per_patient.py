"""Per-patient SublimeX search on AZT1D training windows. Scores test RMSE."""
import os
import time
import numpy as np
import pandas as pd

from config import PARAMETERS, RESULTS, setup_sublimex_path
setup_sublimex_path()
from sublimex import SublimeX, mean_objective, LightGBMModel
from preprocess import load_azt1d

OUT = RESULTS / 'azt1d_per_patient.csv'
PARAM_DIR = PARAMETERS / 'azt1d_per_patient'


def _lgb(ftr, fte, ytr, yte):
    return LightGBMModel('regression').test(ftr, ytr, fte, yte, 'rmse', deterministic=True)


def _lists(X, n_ch):
    nt = X.shape[1] // n_ch
    return [pd.DataFrame(X[:, i * nt:(i + 1) * nt]) for i in range(n_ch)]


def main():
    _, _, info = load_azt1d()
    subjects = sorted(np.unique(info['subject_ids']))
    rows = pd.read_csv(OUT).to_dict('records') if OUT.exists() else []
    done = {int(r['subject']) for r in rows}
    os.makedirs(PARAM_DIR, exist_ok=True)
    print(f'AZT1D per-patient search: {len(subjects)} subjects, done={len(done)}', flush=True)
    for subj in subjects:
        if int(subj) in done:
            continue
        X, y, _ = load_azt1d(subject_id=subj)
        X_list = X if isinstance(X, list) else [X]
        n_ch, n = len(X_list), len(y)
        c = int(n * 0.8)
        tr, te = np.arange(c), np.arange(c, n)
        Xtr = pd.concat([x.iloc[tr] for x in X_list], axis=1).values
        Xte = pd.concat([x.iloc[te] for x in X_list], axis=1).values
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        ppath = PARAM_DIR / f'patient_{int(subj)}.json'
        sx = SublimeX(metric='rmse', objective_fn=mean_objective, verbose=True, deterministic=False, n_workers=1)
        t0 = time.time()
        if ppath.exists():
            sx.load_features(ppath)
            fit_t = 0.0
        else:
            sx.fit(_lists(Xtr, n_ch), ytr)
            sx.save_features(ppath)
            fit_t = time.time() - t0
        tr_f, te_f = sx.transform(_lists(Xtr, n_ch)), sx.transform(_lists(Xte, n_ch))
        sc = _lgb(tr_f, te_f, ytr, yte)
        row = {'subject': int(subj), 'score': sc, 'n_features': len(sx.extracted_features),
               'n_train': int(c), 'n_test': int(n - c), 'fit_seconds': fit_t}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f'subject {int(subj)} | rmse={sc:.4f} | features={row["n_features"]} | {fit_t:.1f}s', flush=True)
    df = pd.DataFrame(rows)
    print(f'mean RMSE {df.score.mean():.4f} ± {df.score.std():.4f} | features {df.n_features.mean():.1f}', flush=True)


if __name__ == '__main__':
    main()
