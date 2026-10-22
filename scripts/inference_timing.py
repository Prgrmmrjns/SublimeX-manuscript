"""Time transform + LightGBM inference only (saved SublimeX features)."""
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneGroupOut

from config import K_FOLDS, PARAMETERS, RESULTS, setup_sublimex_path
setup_sublimex_path()
from sublimex import SublimeX, mean_objective, LightGBMModel
from preprocess import load_emotions, load_mimic, load_mitbih, load_svd, load_pamap2, load_remc, load_azt1d

warnings.filterwarnings('ignore')
OUT = RESULTS / 'inference_timing.csv'


def _time_fold(sx, X_list, y, tr, te, metric, init=None):
    nt = X_list[0].shape[1]
    tr_list = [x.iloc[tr] for x in X_list]
    te_list = [x.iloc[te] for x in X_list]
    itr = init[tr] if init is not None else None
    ite = init[te] if init is not None else None
    t0 = time.time()
    ftr = sx.transform(tr_list, initial_X=itr)
    fte = sx.transform(te_list, initial_X=ite)
    LightGBMModel('regression' if metric == 'rmse' else 'classification').test(
        ftr, y.iloc[tr].values, fte, y.iloc[te].values, metric, deterministic=True)
    return time.time() - t0, ftr.shape[1]


def main():
    rows = []
    for ds, load_fn in [('mimic', load_mimic), ('emotions', load_emotions),
                        ('mitbih', load_mitbih), ('svd', load_svd)]:
        X, y, info = load_fn()
        X_list = X if isinstance(X, list) else [X]
        init = info.get('initial_features')
        cv = StratifiedKFold if info['task'] == 'classification' else KFold
        folds = list(cv(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
        for fi, (tr, te) in enumerate(folds, 1):
            sx = SublimeX(metric=info['metric'], objective_fn=mean_objective)
            sx.load_features(str(PARAMETERS / ds / f'fold{fi}.json'))
            t, nf = _time_fold(sx, X_list, y, tr, te, info['metric'], init)
            rows.append({'dataset': ds, 'fold': fi, 'infer_time': t, 'n_features': nf})
            print(f'{ds} fold{fi} infer={t:.3f}s', flush=True)

    X, y, info = load_pamap2()
    X_list, subjects = X if isinstance(X, list) else [X], np.asarray(info['subject_ids'])
    for fi, (tr, te) in enumerate(LeaveOneGroupOut().split(pd.concat(X_list, axis=1), y, groups=subjects), 1):
        sx = SublimeX(metric=info['metric'], objective_fn=mean_objective)
        sx.load_features(str(PARAMETERS / 'pamap2' / f'fold{fi}.json'))
        t, nf = _time_fold(sx, X_list, y, tr, te, info['metric'])
        rows.append({'dataset': 'pamap2', 'fold': fi, 'infer_time': t, 'n_features': nf,
                     'test_subject': int(np.unique(subjects[te])[0])})
        print(f'pamap2 fold{fi} infer={t:.3f}s', flush=True)

    _, _, ainfo = load_azt1d()
    sx = SublimeX(metric='rmse', objective_fn=mean_objective)
    sx.load_features(str(PARAMETERS / 'azt1d' / 'fold1.json'))
    for subj in sorted(np.unique(ainfo['subject_ids'])):
        X, y, pinfo = load_azt1d(subject_id=subj)
        X_list = X if isinstance(X, list) else [X]
        n = len(y)
        c = int(n * 0.8)
        tr, te = np.arange(c), np.arange(c, n)
        init = pinfo.get('initial_features')
        t, nf = _time_fold(sx, X_list, y, tr, te, 'rmse', init)
        rows.append({'dataset': 'azt1d', 'fold': int(subj), 'infer_time': t, 'n_features': nf})
    print(f'azt1d pooled infer mean={np.mean([r["infer_time"] for r in rows if r["dataset"]=="azt1d"]):.3f}s', flush=True)

    for cl in load_remc():
        X, y, info = load_remc(cell_line=cl)
        X_list = X if isinstance(X, list) else [X]
        folds = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
        for fi, (tr, te) in enumerate(folds, 1):
            p = PARAMETERS / f'remc_{cl}' / f'fold{fi}.json'
            if not p.exists():
                continue
            sx = SublimeX(metric=info['metric'], objective_fn=mean_objective)
            sx.load_features(str(p))
            t, nf = _time_fold(sx, X_list, y, tr, te, info['metric'])
            rows.append({'dataset': f'remc_{cl}', 'fold': fi, 'infer_time': t, 'n_features': nf, 'cell_line': cl})
        print(f'remc_{cl} done', flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
