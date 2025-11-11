import pandas as pd
import numpy as np
import time
import warnings
import os
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from core import feature_extraction
from models import LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

def load_mitbih_data():
    data = pd.read_csv("../processed_datasets/mitbih_processed.csv")
    print(data.shape)
    return data.drop('target', axis=1), data['target']


input_series, y = load_mitbih_data()
kfold_indices = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(input_series, y))

results = []

# PATX evaluation
all_patterns = {}
patx_scores = []
patx_times = []
patx_features = []
for fold, (train_idx, val_idx) in enumerate(kfold_indices):
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train = input_series.iloc[train_idx].astype(np.float32)
    X_test = input_series.iloc[val_idx].astype(np.float32)
    t0 = time.time()
    res = feature_extraction([X_train], y_train.values, [X_test], metric='accuracy', n_trials=N_TRIALS, show_progress=SHOW_PROGRESS, n_control_points=N_CONTROL_POINTS)
    all_patterns[f'fold_{fold+1}'] = res['patterns']
    accuracy = accuracy_score(y_val.values, res['model'].predict(res['test_features']))
    processing_time = time.time() - t0
    patx_scores.append(accuracy)
    patx_times.append(processing_time)
    patx_features.append(len(res['patterns']))
    results.append({'approach': 'PATX', 'fold': fold, 'score': accuracy, 
                    'processing_time': processing_time, 'n_features': len(res['patterns'])})
    print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Time={processing_time:.1f}s, Features={len(res['patterns']):.0f}")

avg_accuracy = np.mean(patx_scores)
avg_time = np.mean(patx_times)
avg_features = np.mean(patx_features)
print(f"PATX Average: Accuracy={avg_accuracy:.4f}, Time={avg_time:.1f}s, Features={avg_features:.1f}")

# TSFRESH evaluation
tsfresh_scores = []
tsfresh_times = []
tsfresh_features = []
for fold, (train_idx, val_idx) in enumerate(kfold_indices):
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train = input_series.iloc[train_idx]
    X_test = input_series.iloc[val_idx]
    t0 = time.time()
    test_feat, train_feat = run_tsfresh(X_train.values, X_test.values)
    tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
    model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
    model.fit(tr_f, y_tr, val_f, y_val_split)
    accuracy = accuracy_score(y_val.values, model.predict(test_feat))
    processing_time = time.time() - t0
    tsfresh_scores.append(accuracy)
    tsfresh_times.append(processing_time)
    tsfresh_features.append(train_feat.shape[1])
    results.append({'approach': 'TSFRESH', 'fold': fold, 'score': accuracy, 
                    'processing_time': processing_time, 'n_features': train_feat.shape[1]})

avg_accuracy = np.mean(tsfresh_scores)
avg_time = np.mean(tsfresh_times)
avg_features = np.mean(tsfresh_features)
print(f"TSFRESH Average: Accuracy={avg_accuracy:.4f}, Time={avg_time:.1f}s, Features={avg_features:.1f}")

# CATCH22 evaluation
catch22_scores = []
catch22_times = []
catch22_features = []
for fold, (train_idx, val_idx) in enumerate(kfold_indices):
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train = input_series.iloc[train_idx]
    X_test = input_series.iloc[val_idx]
    t0 = time.time()
    test_feat, train_feat = run_catch22(X_train.values, X_test.values)
    tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
    model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
    model.fit(tr_f, y_tr, val_f, y_val_split)
    accuracy = accuracy_score(y_val.values, model.predict(test_feat))
    processing_time = time.time() - t0
    catch22_scores.append(accuracy)
    catch22_times.append(processing_time)
    catch22_features.append(train_feat.shape[1])
    results.append({'approach': 'CATCH22', 'fold': fold, 'score': accuracy, 
                    'processing_time': processing_time, 'n_features': train_feat.shape[1]})

avg_accuracy = np.mean(catch22_scores)
avg_time = np.mean(catch22_times)
avg_features = np.mean(catch22_features)
print(f"CATCH22 Average: Accuracy={avg_accuracy:.4f}, Time={avg_time:.1f}s, Features={avg_features:.1f}")

# CNN evaluation
cnn_scores = []
cnn_times = []
cnn_features = []
for fold, (train_idx, val_idx) in enumerate(kfold_indices):
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train = input_series.iloc[train_idx]
    X_test = input_series.iloc[val_idx]
    t0 = time.time()
    preds = run_cnn(X_train.values, y_train.values, X_test.values, task_type='classification', metric='accuracy', num_classes=len(np.unique(y_train)), epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
    accuracy = accuracy_score(y_val.values, preds)
    processing_time = time.time() - t0
    cnn_scores.append(accuracy)
    cnn_times.append(processing_time)
    cnn_features.append(X_train.shape[1])
    results.append({'approach': 'CNN', 'fold': fold, 'score': accuracy, 
                    'processing_time': processing_time, 'n_features': X_train.shape[1]})

avg_accuracy = np.mean(cnn_scores)
avg_time = np.mean(cnn_times)
avg_features = np.mean(cnn_features)
print(f"CNN Average: Accuracy={avg_accuracy:.4f}, Time={avg_time:.1f}s, Features={avg_features:.1f}")

df_res = pd.DataFrame(results)

# Print summary results
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

# Save results
df_res.to_csv('../results/mitbih.csv', index=False)
print(f"\nResults saved to ../results/mitbih.csv")

# Save all patterns to a single JSON file
os.makedirs('../json_files/mitbih', exist_ok=True)
serializable_all_patterns = {}
for fold_key, patterns in all_patterns.items():
    serializable_patterns = []
    for pattern in patterns:
        serializable_pattern = {}
        for key, value in pattern.items():
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
