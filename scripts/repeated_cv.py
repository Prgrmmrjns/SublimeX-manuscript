"""Repeated 5-fold CV (5 x 5) for SublimeX against the leaders on Emotions, SVD, and MIMIC-IV.

Repeat 0 is the main evaluation (seed 42, results/main_eval.csv and results/extra_baselines.csv);
this script adds repeats with seeds 1-4 and writes results/repeated_cv.csv. The summary uses the
Nadeau-Bengio corrected repeated k-fold t-test, because the 25 fold scores share training data.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

import main_eval as me
from extra_baselines import eval_lstm
from config import K_FOLDS, PARAMETERS, RESULTS
from preprocess import load_emotions, load_svd, load_mimic

OUT = RESULTS / 'repeated_cv.csv'
SUMMARY = RESULTS / 'repeated_cv_tests.csv'
PARAM_DIR = PARAMETERS / 'repeated_cv'
SEEDS = [1, 2, 3, 4]
PLAN = [
    ('emotions', load_emotions, ['MiniRocket', 'RDST']),
    ('svd', load_svd, ['RDST', 'catch22']),
    ('mimic', load_mimic, ['LSTM', 'tsfresh']),
]  # MIT-BIH omitted: one SublimeX fit takes ~2 h, so 20 fits were not run

me.LOAD_FEATURES, me.SAVE_FEATURES = False, True


def _run(approach, X_tr, X_te, y_tr, y_te, metric, n_ch, ppath, itr, ite):
    if approach == 'LSTM':
        return eval_lstm(X_tr, X_te, y_tr, y_te, n_ch, metric, itr, ite)
    if approach == 'SublimeX' and os.path.exists(ppath):
        me.LOAD_FEATURES = True
        try:
            return me.eval_sublimex(X_tr, X_te, y_tr, y_te, metric, n_ch, ppath, itr, ite)
        finally:
            me.LOAD_FEATURES = False
    return me.ALL_APPROACHES[approach](X_tr, X_te, y_tr, y_te, metric, n_ch, ppath, itr, ite)


def run():
    rows = pd.read_csv(OUT).to_dict('records') if OUT.exists() else []
    done = {(r['dataset'], r['approach'], int(r['seed']), int(r['fold'])) for r in rows}
    for ds, load_fn, rivals in PLAN:
        X, y, info = load_fn()
        X_list = X if isinstance(X, list) else [X]
        n_ch, init = len(X_list), info.get('initial_features')
        for seed in SEEDS:
            folds = StratifiedKFold(K_FOLDS, shuffle=True, random_state=seed).split(pd.concat(X_list, axis=1), y)
            for fi, (tr, te) in enumerate(folds, 1):
                X_tr = pd.concat([x.iloc[tr] for x in X_list], axis=1).values
                X_te = pd.concat([x.iloc[te] for x in X_list], axis=1).values
                itr = init[tr] if init is not None else None
                ite = init[te] if init is not None else None
                for approach in ['SublimeX'] + rivals:
                    if (ds, approach, seed, fi) in done:
                        continue
                    ppath = str(PARAM_DIR / ds / f'seed{seed}_fold{fi}.json')
                    sc, t, nf = _run(approach, X_tr, X_te, y.iloc[tr].values, y.iloc[te].values,
                                     info['metric'], n_ch, ppath, itr, ite)
                    rows.append({'dataset': ds, 'approach': approach, 'seed': seed, 'fold': fi,
                                 'score': sc, 'time': t, 'n_features': nf})
                    done.add((ds, approach, seed, fi))
                    pd.DataFrame(rows).to_csv(OUT, index=False)
                    print(f'{ds} seed {seed} fold {fi} | {approach} | {sc:.4f} | {t:.1f}s', flush=True)


def _all_scores():
    rep = pd.read_csv(OUT)
    main = pd.read_csv(RESULTS / 'main_eval.csv')
    extra = pd.read_csv(RESULTS / 'extra_baselines.csv')
    base = pd.concat([main, extra[extra.approach == 'LSTM']])
    base = base[base.dataset.isin([p[0] for p in PLAN])].assign(seed=42)
    cols = ['dataset', 'approach', 'seed', 'fold', 'score']
    return pd.concat([base[cols], rep[cols]], ignore_index=True)


def nadeau_bengio(diff, n_train, n_test):
    """Corrected repeated k-fold t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004)."""
    n = len(diff)
    var = np.var(diff, ddof=1)
    t = np.mean(diff) / np.sqrt((1 / n + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), n - 1)


def summarize():
    df = _all_scores()
    out = []
    for ds, _, rivals in PLAN:
        d = df[df.dataset == ds].pivot_table(index=['seed', 'fold'], columns='approach', values='score')
        for rival in rivals:
            pair = d[['SublimeX', rival]].dropna()
            diff = (pair[rival] - pair['SublimeX']).values
            t, p = nadeau_bengio(diff, K_FOLDS - 1, 1)
            out.append({'dataset': ds, 'comparison': rival, 'n_pairs': len(diff),
                        'sublimex_mean': pair['SublimeX'].mean(), 'sublimex_sd': pair['SublimeX'].std(),
                        'rival_mean': pair[rival].mean(), 'rival_sd': pair[rival].std(),
                        'mean_diff': diff.mean(), 't_corrected': t, 'p_corrected': p,
                        'rival_wins': int((diff > 0).sum())})
    res = pd.DataFrame(out)
    res.to_csv(SUMMARY, index=False)
    print(res.round(4).to_string(index=False))


if __name__ == '__main__':
    run()
    summarize()
