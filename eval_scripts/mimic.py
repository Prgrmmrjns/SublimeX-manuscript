import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from patx import feature_extraction
from models import LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

def load_mimic_data():
    """Load MIMIC data with multivariate time series."""
    df = pd.read_csv('../processed_datasets/mimic_processed.csv')
    y = df['ARDS_FLAG']
    anchor_age = df['anchor_age'].values
    
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
    
    return X_list, y, anchor_age, series_names

# Load data once
X_list, y, anchor_age, series_names = load_mimic_data()

# Store KFold indices first
kfold_indices = list(StratifiedKFold(5, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))

results = []

if os.path.exists('../results/mimic.csv'):
    existing = pd.read_csv('../results/mimic.csv')
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
        input_series_train = [x.iloc[train_idx] for x in X_list]
        input_series_test = [x.iloc[val_idx] for x in X_list]
        
        # Standardize each time series
        for i in range(len(input_series_train)):
            scaler = StandardScaler()
            train_vals = input_series_train[i].values
            test_vals = input_series_test[i].values
            input_series_train[i] = pd.DataFrame(
                scaler.fit_transform(train_vals), 
                columns=input_series_train[i].columns,
                index=input_series_train[i].index
            )
            input_series_test[i] = pd.DataFrame(
                scaler.transform(test_vals), 
                columns=input_series_test[i].columns,
                index=input_series_test[i].index
            )
        
        # Prepare initial features (age)
        age_scaler = StandardScaler()
        train_init = age_scaler.fit_transform(anchor_age[train_idx].reshape(-1, 1))
        test_init = age_scaler.transform(anchor_age[val_idx].reshape(-1, 1))
        
        t0 = time.time()
        res = feature_extraction(
            input_series_train, y_train, input_series_test,
            initial_features=(train_init, test_init),
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
            # Add control_points only if it exists
            if 'control_points' in pattern:
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
    with open('../json_files/mimic/pattern_parameters.json', 'w') as f:
        json.dump(all_patterns, f, indent=2, separators=(',', ': '))

# TSFRESH Approach
if ('TSFRESH', 1) not in done:
    approach_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        input_series_train_concat = pd.concat([x.iloc[train_idx] for x in X_list], axis=1)
        input_series_test_concat = pd.concat([x.iloc[val_idx] for x in X_list], axis=1)
        t0 = time.time()
        test_features, train_features = run_tsfresh(input_series_train_concat.values, input_series_test_concat.values)
        train_features, val_features, y_train_split, y_valid = train_test_split(train_features, y_train.values, test_size=0.2, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        model.fit(train_features, y_train_split, val_features, y_valid)
        preds = model.predict(test_features)
        n_feat = train_features.shape[1]
        accuracy = accuracy_score(y_val, preds)
        processing_time = time.time() - t0
        results.append({'approach': 'TSFRESH', 'fold': fold + 1, 'score': accuracy, 'processing_time': processing_time, 'n_features': n_feat})
        approach_results.append({'accuracy': accuracy, 'time': processing_time, 'features': n_feat})

# CATCH22 Approach (univariate - use only first series)
if ('CATCH22', 1) not in done:
    approach_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        first_series_train = X_list[0].iloc[train_idx]  # Use first time series only
        first_series_test = X_list[0].iloc[val_idx]
        t0 = time.time()
        test_features, train_features = run_catch22(first_series_train.values, first_series_test.values)
        train_features, val_features, y_train_split, y_valid = train_test_split(train_features, y_train.values, test_size=0.2, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        model.fit(train_features, y_train_split, val_features, y_valid)
        preds = model.predict(test_features)
        n_feat = train_features.shape[1]
        accuracy = accuracy_score(y_val, preds)
        processing_time = time.time() - t0
        results.append({'approach': 'CATCH22', 'fold': fold + 1, 'score': accuracy, 'processing_time': processing_time, 'n_features': n_feat})
        approach_results.append({'accuracy': accuracy, 'time': processing_time, 'features': n_feat})

# CNN Approach (multivariate - concatenate all series + age)
if ('CNN', 1) not in done:
    approach_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        input_series_train_concat = pd.concat([x.iloc[train_idx] for x in X_list], axis=1)
        input_series_test_concat = pd.concat([x.iloc[val_idx] for x in X_list], axis=1)
        
        # Add age as additional feature
        input_series_train_with_age = np.column_stack([input_series_train_concat.values, anchor_age[train_idx]])
        input_series_test_with_age = np.column_stack([input_series_test_concat.values, anchor_age[val_idx]])
        
        t0 = time.time()
        preds = run_cnn(input_series_train_with_age, y_train.values, input_series_test_with_age, task_type='classification', metric='accuracy', num_classes=len(np.unique(y_train)), epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
        n_feat = input_series_train_with_age.shape[1]
        accuracy = accuracy_score(y_val, preds)
        processing_time = time.time() - t0
        results.append({'approach': 'CNN', 'fold': fold + 1, 'score': accuracy, 'processing_time': processing_time, 'n_features': n_feat})
        approach_results.append({'accuracy': accuracy, 'time': processing_time, 'features': n_feat})

# Print overall summary
mimic_results = pd.DataFrame(results)
for approach in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
    approach_results = mimic_results[mimic_results['approach'] == approach]
    if len(approach_results) > 0:
        mean_acc = approach_results['score'].mean()
        std_acc = approach_results['score'].std()
        mean_time = approach_results['processing_time'].mean()
        std_time = approach_results['processing_time'].std()
        mean_features = approach_results['n_features'].mean()
        std_features = approach_results['n_features'].std()
        print(f"{approach:8}: Accuracy={mean_acc:.4f}±{std_acc:.4f}, Time={mean_time:.1f}±{std_time:.1f}s, Features={mean_features:.0f}±{std_features:.0f}")

pd.DataFrame(results).to_csv('../results/mimic.csv', index=False)