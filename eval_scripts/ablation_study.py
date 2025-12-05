import pandas as pd
import numpy as np
import time
import warnings
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from core import feature_extraction
from models import LightGBMModelWrapper
from params import *
warnings.filterwarnings('ignore')

def load_mitbih_data():
    data = pd.read_csv("../processed_datasets/mitbih/mitbih_processed.csv")
    return data.drop('target', axis=1), data['target']

def save_patterns(patterns, fold, variant):
    os.makedirs(f'../json_files/mitbih_ablation/{variant}', exist_ok=True)
    data = [{**{k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in p.items()}, 'id': i+1} for i, p in enumerate(patterns)]
    with open(f'../json_files/mitbih_ablation/{variant}/fold{fold}.json', 'w') as f:
        json.dump(data, f, indent=2)

def run_variant(name, X, y, folds, n_cp=5, n_trans=5, n_pat=15, backward_elim=True):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    results = []
    for fold, (tr, te) in enumerate(folds):
        X_tr, X_te = X.iloc[tr].astype(np.float32), X.iloc[te].astype(np.float32)
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        t0 = time.time()
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_tr)))
        res = feature_extraction([X_tr], y_tr.values, [X_te], metric='accuracy', n_trials=N_TRIALS, show_progress=SHOW_PROGRESS, n_control_points=n_cp, n_patterns=n_pat, n_transforms=n_trans, max_samples=MAX_SAMPLES, inner_k_folds=INNER_K_FOLDS, early_stopping_patience=EARLY_STOPPING_PATIENCE, val_size=VAL_SIZE, n_workers=N_WORKERS, model=model, backward_elimination=backward_elim)
        save_patterns(res['patterns'], fold+1, name)
        acc = accuracy_score(y_te.values, res['model'].predict(res['test_features']))
        t = time.time() - t0
        results.append({'approach': name, 'fold': fold, 'score': acc, 'time': t, 'n_features': len(res['patterns'])})
        print(f"Fold {fold+1}: Acc={acc:.4f}, Time={t:.1f}s, Features={len(res['patterns'])}")
    scores = [r['score'] for r in results]
    times = [r['time'] for r in results]
    feats = [r['n_features'] for r in results]
    print(f"\nAverage: Acc={np.mean(scores):.4f}±{np.std(scores):.4f}, Time={np.mean(times):.1f}±{np.std(times):.1f}s, Features={np.mean(feats):.1f}±{np.std(feats):.1f}")
    return results

print("Loading MITBIH...")
X, y = load_mitbih_data()
folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(X, y))

existing = pd.read_csv('../results/mitbih_ablation.csv') if os.path.exists('../results/mitbih_ablation.csv') else pd.DataFrame()
done = set(existing['approach'].unique()) if not existing.empty else set()

variants = [
    ('baseline', {'n_cp': 5, 'n_trans': 5, 'n_pat': 15, 'backward_elim': True}),
    ('n_transforms=1', {'n_cp': 5, 'n_trans': 1, 'n_pat': 15, 'backward_elim': True}),
    ('n_transforms=3', {'n_cp': 5, 'n_trans': 3, 'n_pat': 15, 'backward_elim': True}),
    ('n_control_points=3', {'n_cp': 3, 'n_trans': 5, 'n_pat': 15, 'backward_elim': True}),
    ('n_control_points=7', {'n_cp': 7, 'n_trans': 5, 'n_pat': 15, 'backward_elim': True}),
    ('n_patterns=5', {'n_cp': 5, 'n_trans': 5, 'n_pat': 5, 'backward_elim': True}),
    ('n_patterns=10', {'n_cp': 5, 'n_trans': 5, 'n_pat': 10, 'backward_elim': True}),
    ('no_backward_elim', {'n_cp': 5, 'n_trans': 5, 'n_pat': 15, 'backward_elim': False}),
]

all_results = []
for name, params in variants:
    if name in done:
        print(f"\nSkipping {name} (done)")
        continue
    all_results.extend(run_variant(name, X, y, folds, **params))

if all_results:
    df = pd.concat([existing, pd.DataFrame(all_results)], ignore_index=True) if not existing.empty else pd.DataFrame(all_results)
    df.to_csv('../results/mitbih_ablation.csv', index=False)
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    summary = df.groupby('approach').agg({'score': ['mean', 'std'], 'time': ['mean', 'std'], 'n_features': ['mean', 'std']}).round(4)
    print(summary)
