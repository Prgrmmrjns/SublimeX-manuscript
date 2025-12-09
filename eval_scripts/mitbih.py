import pandas as pd
import numpy as np
import time
import warnings
import os
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from patx import feature_extraction, LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

CNN_ONLY = True

def load_mitbih_data():
    data = pd.read_csv("../processed_datasets/mitbih/mitbih_processed.csv")
    print(data.shape)
    return data.drop('target', axis=1), data['target']


input_series, y = load_mitbih_data()
kfold_indices = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(input_series, y))

if CNN_ONLY:
    existing = pd.read_csv('../results/mitbih.csv')
    results = existing[existing['approach'] != 'CNN'].to_dict('records')
else:
    results = []

all_patterns = {}
for fold, (train_idx, val_idx) in enumerate(kfold_indices):
    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train = input_series.iloc[train_idx].astype(np.float32)
    X_test = input_series.iloc[val_idx].astype(np.float32)
    
    if not CNN_ONLY:
        # PATX
        t0 = time.time()
        res = feature_extraction([X_train], y_train.values, [X_test], metric='accuracy', n_trials=N_TRIALS, n_control_points=N_CONTROL_POINTS, n_patterns=N_PATTERNS, n_transforms=N_TRANSFORMS, max_samples=MAX_SAMPLES, inner_k_folds=INNER_K_FOLDS, early_stopping_patience=EARLY_STOPPING_PATIENCE, val_size=VAL_SIZE, show_progress=SHOW_PROGRESS, n_workers=N_WORKERS)
        all_patterns[f'fold_{fold+1}'] = res['patterns']
        accuracy = accuracy_score(y_val.values, res['model'].predict(res['test_features']))
        elapsed = time.time() - t0
        print(f"PATX: Acc={accuracy:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
        results.append({'approach': 'PATX', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': len(res['patterns'])})
        
        # TSFRESH
        t0 = time.time()
        test_feat, train_feat = run_tsfresh(X_train.values, X_test.values)
        tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        model.fit(tr_f, y_tr, val_f, y_val_split)
        accuracy = accuracy_score(y_val, model.predict(test_feat))
        elapsed = time.time()-t0
        print(f"TSFRESH: Acc={accuracy:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({'approach': 'TSFRESH', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
        
        # CATCH22
        t0 = time.time()
        test_feat, train_feat = run_catch22(X_train.values, X_test.values)
        tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        model.fit(tr_f, y_tr, val_f, y_val_split)
        accuracy = accuracy_score(y_val, model.predict(test_feat))
        elapsed = time.time()-t0
        print(f"CATCH22: Acc={accuracy:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({'approach': 'CATCH22', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
    
    # CNN
    t0 = time.time()
    preds = run_cnn(X_train.values, y_train.values, X_test.values, task_type='classification', metric='accuracy', num_classes=len(np.unique(y_train)), epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
    accuracy = accuracy_score(y_val, preds)
    elapsed = time.time()-t0
    print(f"CNN: Acc={accuracy:.4f}, Time={elapsed:.1f}s, Features={X_train.shape[1]}")
    results.append({'approach': 'CNN', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': X_train.shape[1]})

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

if not CNN_ONLY:
    # Save all patterns to a single JSON file
    os.makedirs('../json_files/mitbih', exist_ok=True)
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

    with open('../json_files/mitbih/pattern_parameters.json', 'w') as f:
        json.dump(serializable_all_patterns, f, indent=2)
