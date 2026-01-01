import pandas as pd
import numpy as np
import os
import time
import json
import warnings
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from patx_runner import *
from tsfresh_utils import eval_tsfresh
from catch22_utils import eval_catch22
from cnn import eval_cnn
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patx"))

def load_mimic_data():
    df = pd.read_csv('../processed_datasets/mimic/mimic_processed.csv')
    y = df['ARDS_FLAG']
    
    # Extract time series names from column names
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'anchor_age', 'ARDS_FLAG']]
    series_names = []
    for col in feature_cols:
        if '_hour_' in col:
            series_name = col.split('_hour_')[0]
            if series_name not in series_names:
                series_names.append(series_name)
    
    # Create list of DataFrames for each time series
    X_list = []
    for series_name in series_names:
        series_cols = [col for col in feature_cols if col.startswith(f"{series_name}_hour_")]
        series_cols.sort(key=lambda x: int(x.split('_hour_')[1]))
        X_series = df[series_cols]
        X_list.append(X_series)
    
    return X_list, y, series_names

# Load data once
X_list, y, series_names = load_mimic_data()

# Store KFold indices first
kfold_indices = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))

results = []
all_patterns = {}
for fold, (train_idx, val_idx) in enumerate(kfold_indices):
    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    input_train = [x.iloc[train_idx] for x in X_list]
    input_test = [x.iloc[val_idx] for x in X_list]
    train_concat = pd.concat([x.iloc[train_idx] for x in X_list], axis=1)
    test_concat = pd.concat([x.iloc[val_idx] for x in X_list], axis=1)
    
    t0 = time.time()
    res = run_patx(input_train, y_train, input_test, metric='accuracy')
    all_patterns[f'fold_{fold+1}'] = res['patterns']
    accuracy = accuracy_score(y_val, res['model'].predict(res['test_features']))
    elapsed = time.time()-t0
    print(f"PATX: Acc={accuracy:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
    results.append({'approach': 'PATX', 'fold': fold + 1, 'score': accuracy, 'processing_time': elapsed, 'n_features': len(res['patterns'])})
    
    acc, elapsed, n_feat = eval_tsfresh(train_concat.values, test_concat.values, y_train.values, y_val.values, metric='accuracy', val_size=VAL_SIZE, n_classes=2)
    print(f"TSFRESH: Acc={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'approach': 'TSFRESH', 'fold': fold + 1, 'score': acc, 'processing_time': elapsed, 'n_features': n_feat})
    
    acc, elapsed, n_feat = eval_catch22(train_concat.values, test_concat.values, y_train.values, y_val.values, metric='accuracy', n_classes=2)
    print(f"CATCH22: Acc={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'approach': 'CATCH22', 'fold': fold + 1, 'score': acc, 'processing_time': elapsed, 'n_features': n_feat})
    
    acc, elapsed, n_feat = eval_cnn(train_concat.values, test_concat.values, y_train.values, y_val.values, task_type='classification', metric='accuracy', num_classes=2)
    print(f"CNN: Acc={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'approach': 'CNN', 'fold': fold + 1, 'score': acc, 'processing_time': elapsed, 'n_features': n_feat})

df_res = pd.DataFrame(results)
print(f"\n{'='*60}")
print("MIMIC SUMMARY (Mean ± Std across {K_FOLDS} folds)")
print(f"{'='*60}")
for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
    app_res = df_res[df_res['approach'] == app]
    if len(app_res) > 0:
        m_acc, s_acc = app_res['score'].mean(), app_res['score'].std()
        m_time, s_time = app_res['processing_time'].mean(), app_res['processing_time'].std()
        m_feat, s_feat = app_res['n_features'].mean(), app_res['n_features'].std()
        print(f"{app:8}: Acc={m_acc:.4f}±{s_acc:.4f}, Time={m_time:.1f}±{s_time:.1f}s, Features={m_feat:.0f}±{s_feat:.0f}")

df_res.to_csv('../results/mimic.csv', index=False)

os.makedirs('../json_files/mimic', exist_ok=True)
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

with open('../json_files/mimic/pattern_parameters.json', 'w') as f:
    json.dump(serializable_all_patterns, f, indent=2)