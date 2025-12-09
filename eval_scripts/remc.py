import pandas as pd
import numpy as np
import os
import time
import warnings
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from patx import feature_extraction, LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

CNN_ONLY = True
TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

def load_remc_data(cell_line):
    df = pd.read_parquet(f"../processed_datasets/remc/{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in TIME_SERIES], df['target']


cell_lines = sorted([f.replace('.parquet', '') for f in os.listdir('../processed_datasets/remc') if f.endswith('.parquet')])

if CNN_ONLY:
    existing = pd.read_csv('../results/remc.csv')
    results = existing[existing['approach'] != 'CNN'].to_dict('records')
    done = set()
else:
    results = []
    if os.path.exists('../results/remc.csv'):
        existing = pd.read_csv('../results/remc.csv')
        results = existing.to_dict('records')
        done_checks = existing.groupby(['cell_line', 'approach']).size()
        done = set()
        for cell_line in cell_lines:
            if all((cell_line, app) in done_checks and done_checks[(cell_line, app)] == K_FOLDS for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']):
                done.add(cell_line)
    else:
        done = set()

for cell_line in cell_lines:
    if not CNN_ONLY and cell_line in done:
        print(f"Skipping {cell_line} (already complete)")
        continue
        
    print(f"\n{'='*60}")
    print(f"Processing {cell_line}")
    print(f"{'='*60}")
    X_list, y = load_remc_data(cell_line)
    kfold_indices = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
    
    all_patterns = {}
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        input_train = [x.iloc[train_idx].astype(np.float32) for x in X_list]
        input_test = [x.iloc[val_idx].astype(np.float32) for x in X_list]
        train_concat = pd.concat([x.iloc[train_idx] for x in X_list], axis=1)
        test_concat = pd.concat([x.iloc[val_idx] for x in X_list], axis=1)
        
        if not CNN_ONLY:
            # PATX
            t0 = time.time()
            res = feature_extraction(input_train, y_train.values, input_test, metric='auc', n_trials=N_TRIALS, n_control_points=N_CONTROL_POINTS, n_patterns=N_PATTERNS, n_transforms=N_TRANSFORMS, max_samples=MAX_SAMPLES, inner_k_folds=INNER_K_FOLDS, early_stopping_patience=EARLY_STOPPING_PATIENCE, val_size=VAL_SIZE, show_progress=SHOW_PROGRESS, n_workers=N_WORKERS)
            all_patterns[f'fold_{fold+1}'] = res['patterns']
            preds_proba = res['model'].predict_proba(res['test_features'])
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            auc = roc_auc_score(y_val, preds)
            elapsed = time.time()-t0
            print(f"PATX: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
            results.append({'cell_line': cell_line, 'approach': 'PATX', 'fold': fold+1, 'score': auc, 'processing_time': elapsed, 'n_features': len(res['patterns'])})
            
            # TSFRESH
            t0 = time.time()
            test_feat, train_feat = run_tsfresh(train_concat.values, test_concat.values)
            tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
            model = LightGBMModelWrapper('classification', n_classes=2)
            model.fit(tr_f, y_tr, val_f, y_val_split)
            preds_proba = model.predict_proba(test_feat)
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            auc = roc_auc_score(y_val, preds)
            elapsed = time.time()-t0
            print(f"TSFRESH: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
            results.append({'cell_line': cell_line, 'approach': 'TSFRESH', 'fold': fold+1, 'score': auc, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
            
            # CATCH22
            t0 = time.time()
            test_feat, train_feat = run_catch22(train_concat.values, test_concat.values)
            tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
            model = LightGBMModelWrapper('classification', n_classes=2)
            model.fit(tr_f, y_tr, val_f, y_val_split)
            preds_proba = model.predict_proba(test_feat)
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            auc = roc_auc_score(y_val, preds)
            elapsed = time.time()-t0
            print(f"CATCH22: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
            results.append({'cell_line': cell_line, 'approach': 'CATCH22', 'fold': fold+1, 'score': auc, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
        
        # CNN
        t0 = time.time()
        preds = run_cnn(train_concat.values, y_train.values, test_concat.values, task_type='classification', metric='auc', num_classes=2, epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
        auc = roc_auc_score(y_val, preds)
        elapsed = time.time()-t0
        print(f"CNN: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={train_concat.shape[1]}")
        results.append({'cell_line': cell_line, 'approach': 'CNN', 'fold': fold+1, 'score': auc, 'processing_time': elapsed, 'n_features': train_concat.shape[1]})
    
    if not CNN_ONLY:
        # Save patterns
        os.makedirs('../json_files/remc', exist_ok=True)
        serializable_all_patterns = {}
        for fold_key, patterns in all_patterns.items():
            serializable_patterns = []
            for pattern in patterns:
                serializable_pattern = {}
                for key, value in pattern.items():
                    if key == 'pattern': continue
                    if isinstance(value, np.ndarray):
                        serializable_pattern[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        serializable_pattern[key] = value.item()
                    else:
                        serializable_pattern[key] = value
                serializable_patterns.append(serializable_pattern)
            serializable_all_patterns[fold_key] = serializable_patterns
        
        with open(f'../json_files/remc/pattern_parameters_{cell_line}.json', 'w') as f:
            json.dump(serializable_all_patterns, f, indent=2)
    
    # Print summary for this cell line
    print(f"\n{'='*60}")
    print(f"Summary for {cell_line}")
    print(f"{'='*60}")
    cell_line_res = pd.DataFrame([r for r in results if r['cell_line'] == cell_line])
    for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
        app_res = cell_line_res[cell_line_res['approach'] == app]
        if len(app_res) > 0:
            m_score, s_score = app_res['score'].mean(), app_res['score'].std()
            m_time, s_time = app_res['processing_time'].mean(), app_res['processing_time'].std()
            m_feat, s_feat = app_res['n_features'].mean(), app_res['n_features'].std()
            print(f"{app:8}: AUC={m_score:.4f}±{s_score:.4f}, Time={m_time:.1f}±{s_time:.1f}s, Features={m_feat:.0f}±{s_feat:.0f}")
    
    pd.DataFrame(results).to_csv('../results/remc.csv', index=False)

# Final save
pd.DataFrame(results).to_csv('../results/remc.csv', index=False)