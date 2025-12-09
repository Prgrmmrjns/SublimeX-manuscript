import pandas as pd
import numpy as np
import os
import time
import warnings
import json
import re
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from patx import feature_extraction, LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

CNN_ONLY = True
CHANNELS = ['fft_a', 'fft_b']

def load_emotions_data():
    df = pd.read_csv("../processed_datasets/emotions/emotions.csv")
    y = df['target']
    X_full = df.drop(columns=['target'])
    
    # Separate into 'a' and 'b' channels
    # Columns are fft_0_a, fft_1_a, ... and fft_0_b, ...
    # We want to group them into two dataframes where columns are time points (0..749)
    
    cols_a = [c for c in X_full.columns if c.endswith('_a')]
    cols_b = [c for c in X_full.columns if c.endswith('_b')]
    
    # Sort columns by index
    cols_a.sort(key=lambda x: int(x.split('_')[1]))
    cols_b.sort(key=lambda x: int(x.split('_')[1]))
    
    X_list = [X_full[cols_a], X_full[cols_b]]
    return X_list, y


# Load data
X_list, y = load_emotions_data()

# Store KFold indices
kfold_indices = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))

if CNN_ONLY:
    existing = pd.read_csv('../results/emotions.csv')
    results = existing[existing['approach'] != 'CNN'].to_dict('records')
else:
    results = []
all_patterns = {}

print("Running cross-validation on Emotions dataset")
for fold, (train_idx, test_idx) in enumerate(kfold_indices):
    print(f"\n{'='*60}")
    print(f"Fold {fold+1}/{K_FOLDS}")
    print(f"{'='*60}")
    
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    input_train = [x.iloc[train_idx].astype(np.float32) for x in X_list]
    input_test = [x.iloc[test_idx].astype(np.float32) for x in X_list]
    X_train_concat = pd.concat(input_train, axis=1).values
    X_test_concat = pd.concat(input_test, axis=1).values
    
    if not CNN_ONLY:
        # PATX
        t0 = time.time()
        res = feature_extraction(input_train, y_train.values, input_test, metric='accuracy', n_trials=N_TRIALS, n_control_points=N_CONTROL_POINTS, n_patterns=N_PATTERNS, n_transforms=N_TRANSFORMS, max_samples=MAX_SAMPLES, inner_k_folds=INNER_K_FOLDS, early_stopping_patience=EARLY_STOPPING_PATIENCE, val_size=VAL_SIZE, show_progress=SHOW_PROGRESS, n_workers=N_WORKERS)
        all_patterns[f'fold_{fold+1}'] = res['patterns']
        accuracy = accuracy_score(y_test, res['model'].predict(res['test_features']))
        elapsed = time.time() - t0
        print(f"PATX: Accuracy={accuracy:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
        results.append({'approach': 'PATX', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': len(res['patterns'])})
        
        # TSFRESH
        t0 = time.time()
        test_feat, train_feat = run_tsfresh(X_train_concat, X_test_concat)
        tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification')
        model.fit(tr_f, y_tr, val_f, y_val)
        preds = model.predict(test_feat)
        accuracy = accuracy_score(y_test, preds)
        elapsed = time.time() - t0
        print(f"TSFRESH: Accuracy={accuracy:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({'approach': 'TSFRESH', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
        
        # CATCH22
        t0 = time.time()
        test_feat, train_feat = run_catch22(X_train_concat, X_test_concat)
        tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification')
        model.fit(tr_f, y_tr, val_f, y_val)
        preds = model.predict(test_feat)
        accuracy = accuracy_score(y_test, preds)
        elapsed = time.time() - t0
        print(f"CATCH22: Accuracy={accuracy:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({'approach': 'CATCH22', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
    
    # CNN
    t0 = time.time()
    preds = run_cnn(X_train_concat, y_train.values, X_test_concat, task_type='classification', metric='accuracy', epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
    accuracy = accuracy_score(y_test, preds)
    elapsed = time.time() - t0
    print(f"CNN: Accuracy={accuracy:.4f}, Time={elapsed:.1f}s, Features={X_train_concat.shape[1]}")
    results.append({'approach': 'CNN', 'fold': fold+1, 'score': accuracy, 'processing_time': elapsed, 'n_features': X_train_concat.shape[1]})

df_res = pd.DataFrame(results)

print("\n" + "="*60)
print("SUMMARY RESULTS (Mean ± Std)")
print("="*60)
for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
    app_res = df_res[df_res['approach'] == app]
    scores = app_res['score'].values
    times = app_res['processing_time'].values
    features = app_res['n_features'].values
    print(f"{app:8}: Accuracy={np.mean(scores):.4f}±{np.std(scores):.4f}, "
          f"Time={np.mean(times):.1f}±{np.std(times):.1f}s, "
          f"Features={np.mean(features):.1f}±{np.std(features):.1f}")

df_res.to_csv('../results/emotions.csv', index=False)

if not CNN_ONLY:
    os.makedirs('../json_files/emotions', exist_ok=True)
    serializable_all_patterns = {}
    for fold_key, patterns in all_patterns.items():
        serializable_patterns = []
        for i, pattern in enumerate(patterns):
            p_dict = {k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in pattern.items() if k != 'pattern'}
            p_dict['pattern_id'] = i + 1
            serializable_patterns.append(p_dict)
        serializable_all_patterns[fold_key] = serializable_patterns

    with open('../json_files/emotions/pattern_parameters.json', 'w') as f:
        json.dump(serializable_all_patterns, f, indent=2)
