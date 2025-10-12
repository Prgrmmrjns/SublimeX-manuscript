import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from patx import feature_extraction
from models import LightGBMModelWrapper
from params import *
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn

warnings.filterwarnings('ignore')

TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

def load_remc_data(cell_line):
    df = pd.read_parquet(f"../processed_datasets/remc/{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in TIME_SERIES], df['target']

cell_lines = sorted([f.replace('.parquet', '') for f in os.listdir('../processed_datasets/remc') if f.endswith('.parquet')])
results = []

if os.path.exists('../results/remc.csv'):
    existing = pd.read_csv('../results/remc.csv')
    results = existing.to_dict('records')
    done = set(zip(existing['cell_line'], existing['approach']))
else:
    done = set()

for cell_line in cell_lines:
    # Check if all approaches for this cell line are already done
    cell_line_approaches_done = {(cell_line, approach) for approach in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']}
    if cell_line_approaches_done.issubset(done):
        continue
    
    print(f"Processing {cell_line}")
    X_list, y = load_remc_data(cell_line)
    
    # Store KFold indices first
    kfold_indices = list(StratifiedKFold(5, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
    
    # PATX Approach
    if (cell_line, 'PATX') not in done:
        approach_results = []
        all_patterns = {}
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            input_series_train = [x.iloc[train_idx].astype(np.float32) for x in X_list]
            input_series_test = [x.iloc[val_idx].astype(np.float32) for x in X_list]
            t0 = time.time()
            res = feature_extraction(input_series_train, y_train.values, input_series_test, metric='auc', n_trials=N_TRIALS, show_progress=SHOW_PROGRESS)
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
            preds_proba = res['model'].predict_proba(test_features)
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            n_feat = len(res['patterns'])
            auc = roc_auc_score(y_val, preds)
            processing_time = time.time() - t0
            results.append({'cell_line': cell_line, 'approach': 'PATX', 'fold': fold + 1, 'score': auc, 'processing_time': processing_time, 'n_features': n_feat})
            approach_results.append({'auc': auc, 'time': processing_time, 'features': n_feat})
        
        # Save all patterns to single JSON file with proper formatting
        with open(f'../json_files/remc/pattern_parameters_{cell_line}.json', 'w') as f:
            json.dump(all_patterns, f, indent=2, separators=(',', ': '))
        
        mean_auc = sum(r['auc'] for r in approach_results) / len(approach_results)
        mean_time = sum(r['time'] for r in approach_results) / len(approach_results)
        mean_features = sum(r['features'] for r in approach_results) / len(approach_results)
        std_auc = (sum((r['auc'] - mean_auc)**2 for r in approach_results) / len(approach_results))**0.5
        std_time = (sum((r['time'] - mean_time)**2 for r in approach_results) / len(approach_results))**0.5
        std_features = (sum((r['features'] - mean_features)**2 for r in approach_results) / len(approach_results))**0.5
    
    # TSFRESH Approach
    if (cell_line, 'TSFRESH') not in done:
        approach_results = []
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            input_series_train_concat = pd.concat(X_list, axis=1).iloc[train_idx]
            input_series_test_concat = pd.concat(X_list, axis=1).iloc[val_idx]
            t0 = time.time()
            test_features, train_features = run_tsfresh(input_series_train_concat.values, input_series_test_concat.values)
            train_features, val_features, y_train_split, y_valid = train_test_split(train_features, y_train.values, test_size=0.2, random_state=42, stratify=y_train.values)
            model = LightGBMModelWrapper('classification', n_classes=2)
            model.fit(train_features, y_train_split, val_features, y_valid)
            preds_proba = model.predict_proba(test_features)
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            n_feat = train_features.shape[1]
            auc = roc_auc_score(y_val, preds)
            processing_time = time.time() - t0
            results.append({'cell_line': cell_line, 'approach': 'TSFRESH', 'fold': fold + 1, 'score': auc, 'processing_time': processing_time, 'n_features': n_feat})
            approach_results.append({'auc': auc, 'time': processing_time, 'features': n_feat})
        
        mean_auc = sum(r['auc'] for r in approach_results) / len(approach_results)
        mean_time = sum(r['time'] for r in approach_results) / len(approach_results)
        mean_features = sum(r['features'] for r in approach_results) / len(approach_results)
        std_auc = (sum((r['auc'] - mean_auc)**2 for r in approach_results) / len(approach_results))**0.5
        std_time = (sum((r['time'] - mean_time)**2 for r in approach_results) / len(approach_results))**0.5
        std_features = (sum((r['features'] - mean_features)**2 for r in approach_results) / len(approach_results))**0.5
    
    # CATCH22 Approach
    if (cell_line, 'CATCH22') not in done:
        approach_results = []
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            h3k4me3_train = X_list[0].iloc[train_idx]  # First series is H3K4me3
            h3k4me3_test = X_list[0].iloc[val_idx]
            t0 = time.time()
            test_features, train_features = run_catch22(h3k4me3_train.values, h3k4me3_test.values)
            train_features, val_features, y_train_split, y_valid = train_test_split(train_features, y_train.values, test_size=0.2, random_state=42, stratify=y_train.values)
            model = LightGBMModelWrapper('classification', n_classes=2)
            model.fit(train_features, y_train_split, val_features, y_valid)
            preds_proba = model.predict_proba(test_features)
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            n_feat = train_features.shape[1]
            auc = roc_auc_score(y_val, preds)
            processing_time = time.time() - t0
            results.append({'cell_line': cell_line, 'approach': 'CATCH22', 'fold': fold + 1, 'score': auc, 'processing_time': processing_time, 'n_features': n_feat})
            approach_results.append({'auc': auc, 'time': processing_time, 'features': n_feat})
        
        mean_auc = sum(r['auc'] for r in approach_results) / len(approach_results)
        mean_time = sum(r['time'] for r in approach_results) / len(approach_results)
        mean_features = sum(r['features'] for r in approach_results) / len(approach_results)
        std_auc = (sum((r['auc'] - mean_auc)**2 for r in approach_results) / len(approach_results))**0.5
        std_time = (sum((r['time'] - mean_time)**2 for r in approach_results) / len(approach_results))**0.5
        std_features = (sum((r['features'] - mean_features)**2 for r in approach_results) / len(approach_results))**0.5
    
    # CNN Approach
    if (cell_line, 'CNN') not in done:
        approach_results = []
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            input_series_train_concat = pd.concat(X_list, axis=1).iloc[train_idx]
            input_series_test_concat = pd.concat(X_list, axis=1).iloc[val_idx]
            t0 = time.time()
            preds = run_cnn(input_series_train_concat.values, y_train.values, input_series_test_concat.values, task_type='classification', metric='auc', num_classes=2, epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
            n_feat = input_series_train_concat.shape[1]
            auc = roc_auc_score(y_val, preds)
            processing_time = time.time() - t0
            results.append({'cell_line': cell_line, 'approach': 'CNN', 'fold': fold + 1, 'score': auc, 'processing_time': processing_time, 'n_features': n_feat})
            approach_results.append({'auc': auc, 'time': processing_time, 'features': n_feat})
        
        mean_auc = sum(r['auc'] for r in approach_results) / len(approach_results)
        mean_time = sum(r['time'] for r in approach_results) / len(approach_results)
        mean_features = sum(r['features'] for r in approach_results) / len(approach_results)
        std_auc = (sum((r['auc'] - mean_auc)**2 for r in approach_results) / len(approach_results))**0.5
        std_time = (sum((r['time'] - mean_time)**2 for r in approach_results) / len(approach_results))**0.5
        std_features = (sum((r['features'] - mean_features)**2 for r in approach_results) / len(approach_results))**0.5
    
    # Print cell line summary only if we processed this cell line
    cell_line_results = pd.DataFrame([r for r in results if r['cell_line'] == cell_line])
    for approach in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
        approach_results = cell_line_results[cell_line_results['approach'] == approach]
        if len(approach_results) > 0:
            mean_auc = approach_results['score'].mean()
            std_auc = approach_results['score'].std()
            mean_time = approach_results['processing_time'].mean()
            std_time = approach_results['processing_time'].std()
            mean_features = approach_results['n_features'].mean()
            std_features = approach_results['n_features'].std()
            print(f"{approach:8}: AUC={mean_auc:.4f}±{std_auc:.4f}, Time={mean_time:.1f}±{std_time:.1f}s, Features={mean_features:.0f}±{std_features:.0f}")
    pd.DataFrame(results).to_csv('../results/remc.csv', index=False)