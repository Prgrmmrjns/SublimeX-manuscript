import pandas as pd
import numpy as np
import time
import warnings
import os
import json
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

def load_mitbih_data():
    data = pd.read_csv("../processed_datasets/mitbih/mitbih_processed.csv")
    print(data.shape)
    return data.drop('target', axis=1), data['target']


input_series, y = load_mitbih_data()
kfold_indices = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(input_series, y))

results = []
all_patterns = {}
for fold, (train_idx, val_idx) in enumerate(kfold_indices):
    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train = input_series.iloc[train_idx].astype(np.float32)
    X_test = input_series.iloc[val_idx].astype(np.float32)
    
    t0 = time.time()
    res = run_patx([X_train], y_train.values, [X_test], metric='accuracy')
    all_patterns[f'fold_{fold+1}'] = res['patterns']
    accuracy = accuracy_score(y_val.values, res['model'].predict(res['test_features']))
    elapsed = time.time() - t0
    print(f"PATX: Acc={accuracy:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
    results.append({'approach': 'PATX', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': len(res['patterns'])})
    
    n_classes = len(np.unique(y_train))
    acc, elapsed, n_feat = eval_tsfresh(X_train.values, X_test.values, y_train.values, y_val.values, metric='accuracy', val_size=VAL_SIZE, n_classes=n_classes)
    print(f"TSFRESH: Acc={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'approach': 'TSFRESH', 'fold': fold+1, 'score': acc, 'processing_time': elapsed, 'n_features': n_feat})
    
    acc, elapsed, n_feat = eval_catch22(X_train.values, X_test.values, y_train.values, y_val.values, metric='accuracy', n_classes=n_classes)
    print(f"CATCH22: Acc={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'approach': 'CATCH22', 'fold': fold+1, 'score': acc, 'processing_time': elapsed, 'n_features': n_feat})
    
    acc, elapsed, n_feat = eval_cnn(X_train.values, X_test.values, y_train.values, y_val.values, task_type='classification', metric='accuracy', num_classes=n_classes)
    print(f"CNN: Acc={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'approach': 'CNN', 'fold': fold+1, 'score': acc, 'processing_time': elapsed, 'n_features': n_feat})

df_res = pd.DataFrame(results)
print("\n" + "="*60)
print(f"MIT-BIH SUMMARY (Mean ± Std across {K_FOLDS} folds)")
print("="*60)
for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
    app_res = df_res[df_res['approach'] == app]
    if len(app_res) > 0:
        scores = app_res['score'].values
        times = app_res['processing_time'].values
        features = app_res['n_features'].values
        print(f"{app:8}: Acc={np.mean(scores):.4f}±{np.std(scores):.4f}, "
              f"Time={np.mean(times):.1f}±{np.std(times):.1f}s, "
              f"Features={np.mean(features):.1f}±{np.std(features):.1f}")

# Save results
df_res.to_csv('../results/mitbih.csv', index=False)
print(f"\nResults saved to ../results/mitbih.csv")

os.makedirs('../json_files/mitbih', exist_ok=True)
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

with open('../json_files/mitbih/pattern_parameters.json', 'w') as f:
    json.dump(serializable_all_patterns, f, indent=2)
