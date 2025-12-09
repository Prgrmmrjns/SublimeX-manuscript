import pandas as pd
import numpy as np
import time
import warnings
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from patx import feature_extraction, LightGBMModelWrapper
from params import *
warnings.filterwarnings('ignore')

def load_mitbih_data():
    data = pd.read_csv("../processed_datasets/mitbih/mitbih_processed.csv")
    return data.drop('target', axis=1), data['target']

def load_svd_data():
    N_MFCC = 13
    N_FRAMES = 100
    VOWELS = ["a_n", "i_n", "u_n"]
    VOICE_FEATS = ["f0_mean", "f0_std", "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3", "hnr"]
    df = pd.read_parquet("../processed_datasets/svd/svd.parquet")
    mfcc_channels = []
    for v in VOWELS:
        for i in range(N_MFCC):
            cols = [f"{v}_mfcc{i * N_FRAMES + t}" for t in range(N_FRAMES)]
            mfcc_channels.append(df[cols].astype(np.float32))
    mfcc_cols = [c for c in df.columns if "_mfcc" in c]
    X_mfcc = df[mfcc_cols].astype(np.float32)
    voice_cols = [f"{v}_{vf}" for v in VOWELS for vf in VOICE_FEATS]
    X_voice = df[voice_cols].astype(np.float32)
    return mfcc_channels, X_voice, df["target"].astype(int)

def load_remc_data(cell_line):
    TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    df = pd.read_parquet(f"../processed_datasets/remc/{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in TIME_SERIES], df['target']

def save_results(new_results):
    global existing
    if new_results:
        df_new = pd.DataFrame(new_results)
        existing = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
        existing.to_csv(results_file, index=False)

def run_variant(name, X, y, folds, dataset='mitbih', metric='accuracy', n_cp=5, n_trans=5, n_pat=15, backward_elim=True, use_early_stopping=True, sampler='nsga2', inner_k=None, initial_features=None, existing_results=None, transforms=None):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    results = []
    for fold, (tr, te) in enumerate(folds):
        if not existing_results.empty:
            existing_fold = existing_results[(existing_results['dataset'] == dataset) & (existing_results['approach'] == name) & (existing_results['fold'] == fold)]
            if len(existing_fold) > 0:
                print(f"Fold {fold+1}: Already completed, skipping")
                results.append(existing_fold.iloc[0].to_dict())
                continue
        
        X_tr = [x.iloc[tr].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[tr].astype(np.float32)]
        X_te = [x.iloc[te].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[te].astype(np.float32)]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        t0 = time.time()
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_tr)))
        init_feat = (initial_features.iloc[tr].values, initial_features.iloc[te].values) if initial_features is not None else None
        res = feature_extraction(X_tr, y_tr.values, X_te, metric=metric, n_trials=N_TRIALS, show_progress=SHOW_PROGRESS, n_control_points=n_cp, n_patterns=n_pat, n_transforms=n_trans, max_samples=MAX_SAMPLES, inner_k_folds=inner_k or INNER_K_FOLDS, early_stopping_patience=EARLY_STOPPING_PATIENCE if use_early_stopping else None, val_size=VAL_SIZE, n_workers=N_WORKERS, model=model, backward_elimination=backward_elim, sampler=sampler, initial_features=init_feat, transforms=transforms)
        if metric == 'accuracy':
            preds = res['model'].predict(res['test_features'])
            score = accuracy_score(y_te.values, preds)
        else:
            preds_proba = res['model'].predict_proba(res['test_features'])
            preds = preds_proba[:, 1] if preds_proba.ndim > 1 else preds_proba
            score = roc_auc_score(y_te.values, preds)
        t = time.time() - t0
        result = {'dataset': dataset, 'approach': name, 'fold': fold, 'score': score, 'time': t, 'n_features': len(res['patterns'])}
        results.append(result)
        save_results([result])
        print(f"Fold {fold+1}: {metric.upper()}={score:.4f}, Time={t:.1f}s, Features={len(res['patterns'])}")
    scores, times, feats = [r['score'] for r in results], [r['time'] for r in results], [r['n_features'] for r in results]
    print(f"\nAverage: {metric.upper()}={np.mean(scores):.4f}±{np.std(scores):.4f}, Time={np.mean(times):.1f}±{np.std(times):.1f}s, Features={np.mean(feats):.1f}±{np.std(feats):.1f}")
    return results

def run_iterative_pattern_search(X, y, folds, dataset='mitbih', metric='accuracy', n_cp=5, n_trans=5, initial_features=None, existing_results=None):
    print(f"\n{'='*60}\niterative_pattern_search\n{'='*60}")
    results = []
    for fold, (tr, te) in enumerate(folds):
        if not existing_results.empty:
            existing_fold = existing_results[(existing_results['dataset'] == dataset) & (existing_results['approach'] == 'iterative_pattern_search') & (existing_results['fold'] == fold)]
            if len(existing_fold) > 0:
                print(f"Fold {fold+1}: Already completed, skipping")
                results.append(existing_fold.iloc[0].to_dict())
                continue
        
        X_tr = [x.iloc[tr].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[tr].astype(np.float32)]
        X_te = [x.iloc[te].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[te].astype(np.float32)]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        t0 = time.time()
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_tr)))
        init_feat = (initial_features.iloc[tr].values, initial_features.iloc[te].values) if initial_features is not None else None
        from sklearn.model_selection import train_test_split
        tr_idx, val_idx = train_test_split(np.arange(len(y_tr)), test_size=VAL_SIZE, random_state=42)
        y_tr_split, y_val_split = y_tr.values[tr_idx], y_tr.values[val_idx]
        all_patterns = []
        accumulated_feats_train = init_feat[0] if init_feat else np.empty((len(y_tr), 0))
        accumulated_feats_test = init_feat[1] if init_feat else np.empty((len(y_te), 0))
        best_score = -np.inf if metric != 'rmse' else np.inf
        
        for n_pat in range(1, 21):
            res = feature_extraction(X_tr, y_tr.values, X_te, metric=metric, n_trials=N_TRIALS, show_progress=False, n_control_points=n_cp, n_patterns=1, n_transforms=n_trans, max_samples=MAX_SAMPLES, inner_k_folds=INNER_K_FOLDS, early_stopping_patience=EARLY_STOPPING_PATIENCE, val_size=VAL_SIZE, n_workers=N_WORKERS, model=model.clone(), backward_elimination=False, sampler='nsga2', initial_features=(accumulated_feats_train, accumulated_feats_test))
            model_temp = model.clone()
            model_temp.fit(res['train_features'][tr_idx], y_tr_split, res['train_features'][val_idx], y_val_split)
            if metric == 'accuracy':
                val_preds = model_temp.predict(res['train_features'][val_idx])
                val_score = accuracy_score(y_val_split, val_preds)
            else:
                val_preds_proba = model_temp.predict_proba(res['train_features'][val_idx])
                val_preds = val_preds_proba[:, 1] if val_preds_proba.ndim > 1 else val_preds_proba
                val_score = roc_auc_score(y_val_split, val_preds)
            improved = val_score > best_score if metric != 'rmse' else val_score < best_score
            if improved:
                best_score = val_score
                all_patterns.append(res['patterns'][0])
                accumulated_feats_train = res['train_features']
                accumulated_feats_test = res['test_features']
                final_model = model_temp
                print(f"  Pattern {n_pat}: {metric.upper()}={val_score:.4f} (improved)")
            else:
                print(f"  Pattern {n_pat}: {metric.upper()}={val_score:.4f} (no improvement, stopping)")
                break
        
        if len(all_patterns) == 0:
            all_patterns = res['patterns']
            accumulated_feats_train = res['train_features']
            accumulated_feats_test = res['test_features']
            final_model = model.clone()
            final_model.fit(accumulated_feats_train[tr_idx], y_tr_split, accumulated_feats_train[val_idx], y_val_split)
        if metric == 'accuracy':
            preds = final_model.predict(accumulated_feats_test)
            score = accuracy_score(y_te.values, preds)
        else:
            preds_proba = final_model.predict_proba(accumulated_feats_test)
            preds = preds_proba[:, 1] if preds_proba.ndim > 1 else preds_proba
            score = roc_auc_score(y_te.values, preds)
        t = time.time() - t0
        result = {'dataset': dataset, 'approach': 'iterative_pattern_search', 'fold': fold, 'score': score, 'time': t, 'n_features': len(all_patterns)}
        results.append(result)
        save_results([result])
        print(f"Fold {fold+1}: {metric.upper()}={score:.4f}, Time={t:.1f}s, Features={len(all_patterns)}")
    scores, times, feats = [r['score'] for r in results], [r['time'] for r in results], [r['n_features'] for r in results]
    print(f"\nAverage: {metric.upper()}={np.mean(scores):.4f}±{np.std(scores):.4f}, Time={np.mean(times):.1f}±{np.std(times):.1f}s, Features={np.mean(feats):.1f}±{np.std(feats):.1f}")
    return results

variants = [
    ('baseline', {'n_cp': 5, 'n_trans': 5, 'n_pat': 15, 'backward_elim': True}),
    ('no_transforms', {'n_cp': 5, 'n_trans': 1, 'n_pat': 15, 'backward_elim': True, 'transforms': ['raw']}),
    ('n_control_points=1', {'n_cp': 1, 'n_trans': 5, 'n_pat': 15, 'backward_elim': True}),
    ('n_patterns=5', {'n_cp': 5, 'n_trans': 5, 'n_pat': 5, 'backward_elim': True}),
    ('no_backward_elim', {'n_cp': 5, 'n_trans': 5, 'n_pat': 15, 'backward_elim': False}),
    ('no_early_stopping', {'n_cp': 5, 'n_trans': 5, 'n_pat': 15, 'backward_elim': True, 'use_early_stopping': False}),
    ('tpe_sampler', {'n_cp': 5, 'n_trans': 5, 'n_pat': 15, 'backward_elim': True, 'sampler': 'tpe'}),
    ('no_inner_cv', {'n_cp': 5, 'n_trans': 5, 'n_pat': 15, 'backward_elim': True, 'inner_k': 1}),
]

variant_names = [v[0] for v in variants]
results_file = '../results/ablation_study.csv'
existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

def get_done_variants(dataset_name):
    return set(existing[existing['dataset'] == dataset_name]['approach'].unique()) if not existing.empty else set()

def run_dataset(dataset_name, X, y, folds, metric, initial_features=None):
    global existing
    done = get_done_variants(dataset_name)
    pending = [v for v in variant_names if v not in done]
    if 'iterative_pattern_search' not in done:
        pending.append('iterative_pattern_search')
    if len(pending) == 0:
        print(f"{dataset_name.upper()}: All variants already completed, skipping.")
        return
    print(f"{dataset_name.upper()}: Running {len(pending)} pending variants: {pending}")
    for name, params in variants:
        if name in pending:
            run_variant(name, X, y, folds, dataset=dataset_name, metric=metric, initial_features=initial_features, existing_results=existing, **params)
            existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
    if 'iterative_pattern_search' in pending:
        run_iterative_pattern_search(X, y, folds, dataset=dataset_name, metric=metric, initial_features=initial_features, existing_results=existing)
        existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

print("Checking MITBIH...")
X_mitbih, y_mitbih = load_mitbih_data()
folds_mitbih = list(StratifiedKFold(2, shuffle=True, random_state=42).split(X_mitbih, y_mitbih))
run_dataset('mitbih', X_mitbih, y_mitbih, folds_mitbih, 'accuracy')

print("\nChecking SVD...")
mfcc_channels_svd, X_voice_svd, y_svd = load_svd_data()
folds_svd = list(StratifiedKFold(2, shuffle=True, random_state=42).split(pd.concat(mfcc_channels_svd, axis=1), y_svd))
run_dataset('svd', mfcc_channels_svd, y_svd, folds_svd, 'auc', X_voice_svd)

print("\nChecking REMC (first cell line)...")
cell_lines = sorted([f.replace('.parquet', '') for f in os.listdir('../processed_datasets/remc') if f.endswith('.parquet')])
if cell_lines:
    first_cell_line = cell_lines[0]
    X_remc, y_remc = load_remc_data(first_cell_line)
    folds_remc = list(StratifiedKFold(2, shuffle=True, random_state=42).split(pd.concat(X_remc, axis=1), y_remc))
    run_dataset(f'remc_{first_cell_line}', X_remc, y_remc, folds_remc, 'auc')

print(f"\n{'='*80}\nSUMMARY BY DATASET\n{'='*80}")
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    for dataset_name in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset_name]
        print(f"\n{dataset_name.upper()}:")
        summary = dataset_df.groupby('approach').agg({'score': ['mean', 'std'], 'time': ['mean', 'std'], 'n_features': ['mean', 'std']}).round(4)
        print(summary)
