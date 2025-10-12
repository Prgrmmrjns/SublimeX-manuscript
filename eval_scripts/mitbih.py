import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from patx import feature_extraction
from models import LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

def load_mitbih_data():
    """Load MITBIH data as univariate time series."""
    data = pd.read_csv("../processed_datasets/mitbih_processed.csv")
    X = data.drop('target', axis=1)
    y = data['target']
    # Convert to binary classification (0 vs others) to avoid multiclass issues
    y = (y == 0).astype(int)
    return X, y

# Load data once
input_series, y = load_mitbih_data()

# Store KFold indices first
kfold_indices = list(StratifiedKFold(5, shuffle=True, random_state=42).split(input_series, y))

results = []

if os.path.exists('../results/mitbih.csv'):
    existing = pd.read_csv('../results/mitbih.csv')
    results = existing.to_dict('records')
    done = set(zip(existing['approach'], existing['fold']))
else:
    done = set()

# PATX Approach
if ('PATX', 1) not in done:
    approach_results = []
    all_patterns = {}
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        input_series_train = [input_series.iloc[train_idx].astype(np.float32)]
        input_series_test = [input_series.iloc[val_idx].astype(np.float32)]
        t0 = time.time()
        res = feature_extraction(
            input_series_train, y_train.values, input_series_test,
            metric='accuracy', n_trials=N_TRIALS, show_progress=SHOW_PROGRESS
        )
        # Convert patterns to JSON-serializable format
        serializable_patterns = []
        for pattern in res['patterns']:
            serializable_pattern = {
                'pattern': pattern['pattern'].tolist(),
                'start': int(pattern['start']),
                'width': int(pattern['width']),
                'series_idx': int(pattern['series_idx'])
            }
            serializable_pattern['control_points'] = [float(cp) for cp in pattern['control_points']]
            serializable_patterns.append(serializable_pattern)
        
        # Store patterns for this fold
        all_patterns[f'fold_{fold+1}'] = serializable_patterns
        test_features = res['test_features']
        preds = res['model'].predict(test_features)
        n_feat = len(res['patterns'])
        accuracy = accuracy_score(y_val, preds)
        processing_time = time.time() - t0
        results.append({'approach': 'PATX', 'fold': fold + 1, 'score': accuracy, 'processing_time': processing_time, 'n_features': n_feat})
        approach_results.append({'accuracy': accuracy, 'time': processing_time, 'features': n_feat})
    
    # Save all patterns to single JSON file with proper formatting
    with open('../json_files/mitbih/pattern_parameters.json', 'w') as f:
        json.dump(all_patterns, f, indent=2, separators=(',', ': '))

# TSFRESH Approach
if ('TSFRESH', 1) not in done:
    approach_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        input_series_train = input_series.iloc[train_idx]
        input_series_test = input_series.iloc[val_idx]
        t0 = time.time()
        test_features, train_features = run_tsfresh(input_series_train.values, input_series_test.values)
        train_features, val_features, y_train_split, y_valid = train_test_split(train_features, y_train.values, test_size=0.2, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        model.fit(train_features, y_train_split, val_features, y_valid)
        preds = model.predict(test_features)
        n_feat = train_features.shape[1]
        accuracy = accuracy_score(y_val, preds)
        processing_time = time.time() - t0
        results.append({'approach': 'TSFRESH', 'fold': fold + 1, 'score': accuracy, 'processing_time': processing_time, 'n_features': n_feat})
        approach_results.append({'accuracy': accuracy, 'time': processing_time, 'features': n_feat})

# CATCH22 Approach
if ('CATCH22', 1) not in done:
    approach_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        input_series_train = input_series.iloc[train_idx]
        input_series_test = input_series.iloc[val_idx]
        t0 = time.time()
        test_features, train_features = run_catch22(input_series_train.values, input_series_test.values)
        train_features, val_features, y_train_split, y_valid = train_test_split(train_features, y_train.values, test_size=0.2, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        model.fit(train_features, y_train_split, val_features, y_valid)
        preds = model.predict(test_features)
        n_feat = train_features.shape[1]
        accuracy = accuracy_score(y_val, preds)
        processing_time = time.time() - t0
        results.append({'approach': 'CATCH22', 'fold': fold + 1, 'score': accuracy, 'processing_time': processing_time, 'n_features': n_feat})
        approach_results.append({'accuracy': accuracy, 'time': processing_time, 'features': n_feat})

# CNN Approach
if ('CNN', 1) not in done:
    approach_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        input_series_train = input_series.iloc[train_idx]
        input_series_test = input_series.iloc[val_idx]
        t0 = time.time()
        preds = run_cnn(input_series_train.values, y_train.values, input_series_test.values, task_type='classification', metric='accuracy', num_classes=len(np.unique(y_train)), epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
        n_feat = input_series_train.shape[1]
        accuracy = accuracy_score(y_val, preds)
        processing_time = time.time() - t0
        results.append({'approach': 'CNN', 'fold': fold + 1, 'score': accuracy, 'processing_time': processing_time, 'n_features': n_feat})
        approach_results.append({'accuracy': accuracy, 'time': processing_time, 'features': n_feat})

# Print overall summary
mitbih_results = pd.DataFrame(results)
for approach in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
    approach_results = mitbih_results[mitbih_results['approach'] == approach]
    if len(approach_results) > 0:
        mean_acc = approach_results['score'].mean()
        std_acc = approach_results['score'].std()
        mean_time = approach_results['processing_time'].mean()
        std_time = approach_results['processing_time'].std()
        mean_features = approach_results['n_features'].mean()
        std_features = approach_results['n_features'].std()
        print(f"{approach:8}: Accuracy={mean_acc:.4f}±{std_acc:.4f}, Time={mean_time:.1f}±{std_time:.1f}s, Features={mean_features:.0f}±{std_features:.0f}")

pd.DataFrame(results).to_csv('../results/mitbih.csv', index=False)
