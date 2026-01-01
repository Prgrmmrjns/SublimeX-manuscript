import pandas as pd
import numpy as np
import os
import time
import warnings
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from patx_runner import *
from tsfresh_utils import eval_tsfresh
from catch22_utils import eval_catch22
from cnn import eval_cnn
warnings.filterwarnings('ignore')

TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

def load_remc_data(cell_line):
    df = pd.read_parquet(f"../processed_datasets/remc/{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in TIME_SERIES], df['target']


cell_lines = sorted([f.replace('.parquet', '') for f in os.listdir('../processed_datasets/remc') if f.endswith('.parquet')])

results, done = [], set()
if os.path.exists('../results/remc.csv'):
    existing = pd.read_csv('../results/remc.csv')
    results = existing.to_dict('records')
    done_checks = existing.groupby(['cell_line', 'approach']).size()
    for cell_line in cell_lines:
        if all((cell_line, app) in done_checks and done_checks[(cell_line, app)] == K_FOLDS for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']):
            done.add(cell_line)

for cell_line in cell_lines:
    if cell_line in done:
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
        
        # PATX
        t0 = time.time()
        res = run_patx(input_train, y_train.values, input_test, metric='auc')
        all_patterns[f'fold_{fold+1}'] = res['patterns']
        preds_proba = res['model'].predict_proba(res['test_features'])
        preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
        auc = roc_auc_score(y_val, preds)
        elapsed = time.time()-t0
        print(f"PATX: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
        results.append({'cell_line': cell_line, 'approach': 'PATX', 'fold': fold+1, 'score': auc, 'processing_time': elapsed, 'n_features': len(res['patterns'])})
        
        auc, elapsed, n_feat = eval_tsfresh(train_concat.values, test_concat.values, y_train.values, y_val.values, metric='auc', val_size=VAL_SIZE, n_classes=2)
        print(f"TSFRESH: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
        results.append({'cell_line': cell_line, 'approach': 'TSFRESH', 'fold': fold+1, 'score': auc, 'processing_time': elapsed, 'n_features': n_feat})
        
        auc, elapsed, n_feat = eval_catch22(train_concat.values, test_concat.values, y_train.values, y_val.values, metric='auc', n_classes=2)
        print(f"CATCH22: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
        results.append({'cell_line': cell_line, 'approach': 'CATCH22', 'fold': fold+1, 'score': auc, 'processing_time': elapsed, 'n_features': n_feat})
        
        auc, elapsed, n_feat = eval_cnn(train_concat.values, test_concat.values, y_train.values, y_val.values, task_type='classification', metric='auc', num_classes=2)
        print(f"CNN: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
        results.append({'cell_line': cell_line, 'approach': 'CNN', 'fold': fold+1, 'score': auc, 'processing_time': elapsed, 'n_features': n_feat})
    
    # Save patterns
    os.makedirs('../json_files/remc', exist_ok=True)
    serializable_all_patterns = {}
    for fold_key, patterns in all_patterns.items():
        serializable_patterns = []
        for pattern in patterns:
            serializable_pattern = {}
            for key, value in pattern.items():
                if key == 'pattern': continue
                if isinstance(value, list):
                    serializable_pattern[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                elif isinstance(value, np.ndarray):
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
        m_score, s_score = app_res['score'].mean(), app_res['score'].std()
        m_time, s_time = app_res['processing_time'].mean(), app_res['processing_time'].std()
        m_feat, s_feat = app_res['n_features'].mean(), app_res['n_features'].std()
        print(f"{app:8}: AUC={m_score:.4f}±{s_score:.4f}, Time={m_time:.1f}±{s_time:.1f}s, Features={m_feat:.0f}±{s_feat:.0f}")

    pd.DataFrame(results).to_csv('../results/remc.csv', index=False)

# Final save
pd.DataFrame(results).to_csv('../results/remc.csv', index=False)