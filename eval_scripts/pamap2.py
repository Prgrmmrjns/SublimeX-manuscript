import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from patx import feature_extraction, LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

CNN_ONLY = True

def create_windows(df, window_size=100, step_size=50):
    feature_cols = [col for col in df.columns if col not in ['time_stamp', 'activity_id', 'id']]
    all_windows, all_labels, all_subjects = [], [], []
    
    for subject_id in df['id'].unique():
        subject_df = df[df['id'] == subject_id].reset_index(drop=True)
        for activity_id in subject_df['activity_id'].unique():
            activity_df = subject_df[subject_df['activity_id'] == activity_id]
            for i in range(0, len(activity_df) - window_size + 1, step_size):
                all_windows.append(activity_df.iloc[i:i+window_size][feature_cols].values)
                all_labels.append(activity_id)
                all_subjects.append(subject_id)
    return all_windows, np.array(all_labels), np.array(all_subjects), feature_cols

def bin_time_series(data, bin_size):
    n_samples, n_timepoints = data.shape
    n_bins = n_timepoints // bin_size
    if n_bins == 0:
        return data
    return data[:, :n_bins * bin_size].reshape(n_samples, n_bins, bin_size).mean(axis=2)

def load_pamap2_data(bin_size=10):
    df = pd.read_parquet('../processed_datasets/pamap2/pamap2.parquet')
    windows, y, subjects, feature_names = create_windows(df, window_size=100, step_size=50)
    windows_array = np.array(windows)
    
    X_list = []
    for feature_idx in range(windows_array.shape[2]):
        feature_data = windows_array[:, :, feature_idx]
        if bin_size > 1:
            feature_data = bin_time_series(feature_data, bin_size)
        X_list.append(pd.DataFrame(feature_data))
    
    return X_list, pd.Series(y), subjects, feature_names

print("Running 5-fold CV on PAMAP2 (all subjects combined)")
print("="*60)

X_list, y, subjects, feature_names = load_pamap2_data(bin_size=10)
unique_labels = np.unique(y)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_mapped = np.array([label_map[label] for label in y])
n_classes = len(unique_labels)
print(f'Total windows: {len(y)}, {n_classes} activity classes\n')

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

if CNN_ONLY:
    existing = pd.read_csv('../results/pamap2.csv')
    results = existing[existing['approach'] != 'CNN'].to_dict('records')
else:
    results = []
    all_fold_patterns = {}

for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_mapped)), y_mapped), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{K_FOLDS}")
    print(f"{'='*60}\n")
    
    y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]
    input_train = [x.iloc[train_idx] for x in X_list]
    input_test = [x.iloc[test_idx] for x in X_list]
    train_concat = np.stack([x.iloc[train_idx].values for x in X_list], axis=2).reshape(len(train_idx), -1)
    test_concat = np.stack([x.iloc[test_idx].values for x in X_list], axis=2).reshape(len(test_idx), -1)
    
    if not CNN_ONLY:
        # PATX
        t0 = time.time()
        res = feature_extraction(input_train, y_train, input_test, metric='accuracy', n_trials=N_TRIALS, n_control_points=N_CONTROL_POINTS, n_patterns=N_PATTERNS, n_transforms=N_TRANSFORMS, max_samples=MAX_SAMPLES, inner_k_folds=INNER_K_FOLDS, early_stopping_patience=EARLY_STOPPING_PATIENCE, val_size=VAL_SIZE, show_progress=SHOW_PROGRESS, n_workers=N_WORKERS)
        accuracy = accuracy_score(y_test, res['model'].predict(res['test_features']))
        elapsed = time.time()-t0
        print(f"PATX: Accuracy={accuracy:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
        results.append({'fold': fold, 'approach': 'PATX', 'score': accuracy, 'processing_time': elapsed, 'n_features': len(res['patterns'])})
        
        pattern_data = [{k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in p.items() if k != 'pattern'} for p in res['patterns']]
        all_fold_patterns[f'fold_{fold}'] = pattern_data
        
        # TSFRESH
        t0 = time.time()
        test_feat, train_feat = run_tsfresh(train_concat, test_concat)
        tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train, test_size=VAL_SIZE, random_state=42, stratify=y_train)
        model = LightGBMModelWrapper('classification', n_classes=n_classes)
        model.fit(tr_f, y_tr, val_f, y_val)
        accuracy = accuracy_score(y_test, model.predict(test_feat))
        elapsed = time.time()-t0
        print(f"TSFRESH: Accuracy={accuracy:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({'fold': fold, 'approach': 'TSFRESH', 'score': accuracy, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
        
        # CATCH22
        t0 = time.time()
        test_feat, train_feat = run_catch22(train_concat, test_concat)
        tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train, test_size=VAL_SIZE, random_state=42, stratify=y_train)
        model = LightGBMModelWrapper('classification', n_classes=n_classes)
        model.fit(tr_f, y_tr, val_f, y_val)
        accuracy = accuracy_score(y_test, model.predict(test_feat))
        elapsed = time.time()-t0
        print(f"CATCH22: Accuracy={accuracy:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({'fold': fold, 'approach': 'CATCH22', 'score': accuracy, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
    
    # CNN
    t0 = time.time()
    preds = run_cnn(train_concat, y_train, test_concat, task_type='classification', metric='accuracy', num_classes=n_classes, epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
    accuracy = accuracy_score(y_test, preds)
    elapsed = time.time()-t0
    print(f"CNN: Accuracy={accuracy:.4f}, Time={elapsed:.1f}s, Features={train_concat.shape[1]}")
    results.append({'fold': fold, 'approach': 'CNN', 'score': accuracy, 'processing_time': elapsed, 'n_features': train_concat.shape[1]})

if not CNN_ONLY:
    os.makedirs('../json_files/pamap2', exist_ok=True)
    with open('../json_files/pamap2/pattern_parameters.json', 'w') as f:
        json.dump(all_fold_patterns, f, indent=2)
    print("\nSaved patterns from all folds to ../json_files/pamap2/pattern_parameters.json")

print(f"\n{'='*60}")
print(f"PAMAP2 SUMMARY (Mean ± Std across {K_FOLDS} folds)")
print(f"{'='*60}")
df_res = pd.DataFrame(results)
for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
    app_res = df_res[df_res['approach'] == app]
    print(f"{app:8}: Accuracy={app_res['score'].mean():.4f}±{app_res['score'].std():.4f}, Time={app_res['processing_time'].mean():.1f}±{app_res['processing_time'].std():.1f}s, Features={app_res['n_features'].mean():.1f}±{app_res['n_features'].std():.1f}")

df_res.to_csv('../results/pamap2.csv', index=False)
print("Results saved to ../results/pamap2.csv")
