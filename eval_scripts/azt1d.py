import pandas as pd
import numpy as np
import os
import time
import json
from sklearn.metrics import mean_squared_error

# Import PATX and Baseline Utils
from patx_runner import run_patx
from cnn import eval_cnn
import pyarrow.parquet as pq
from minirocket_utils import eval_minirocket
from catch22_utils import eval_catch22
from tsfresh_utils import eval_tsfresh
from rdst_utils import eval_rdst

# ------------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# ------------------------------------------------------------------------------

def load_all_azt1d():
    """
    Load combined AZT1D dataset from single parquet file.
    Returns combined data with subject_id tracking for time-based splitting.
    """
    data_file = "../processed_datasets/azt1d/azt1d_combined.parquet"
    
    # Load combined dataset with error handling for corrupted parquet files
    combined_df = pd.read_parquet(data_file)
    # Try reading with pyarrow directly and re-saving to fix corruption
    
    table = pq.read_table(data_file)
    combined_df = table.to_pandas()
    # Re-save the file to fix any corruption issues
    print("Repairing parquet file...")
    combined_df.to_parquet(data_file, index=False, engine='pyarrow')
    print("Parquet file repaired successfully.")
    
    # Extract time series features
    time_series = ['CGM', 'Insulin', 'Carbs']
    X_list = [
        combined_df[[c for c in combined_df.columns 
                     if c.startswith(f'{s}_') and c != 'CGM_current']] 
        for s in time_series
    ]
    
    return (X_list, combined_df['target'], combined_df['subject_id'],
            {'metric': 'rmse', 'task': 'regression'})

# ------------------------------------------------------------------------------
# EVALUATION LOGIC
# ------------------------------------------------------------------------------

def run_evaluation(X_list, y, subject_ids, info, results_path, patterns_path):
    """
    Evaluate on combined AZT1D dataset with per-subject time-based splits.
    For each subject, the last 20% of samples (chronologically) go to test set.
    """
    metric, task = info['metric'], info['task']


    # Per-subject time-based split: last 20% of each subject goes to test
    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    
    for subject_id in np.unique(subject_ids):
        subject_indices = np.where(subject_ids == subject_id)[0]
        n_subject = len(subject_indices)
        split_idx = int(n_subject * 0.8)  # 80% train, 20% test
        
        # First 80% for training
        train_mask[subject_indices[:split_idx]] = True
        # Last 20% for testing
        test_mask[subject_indices[split_idx:]] = True
    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    # Split data
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    input_train = [x.iloc[train_idx] for x in X_list]
    input_test = [x.iloc[test_idx] for x in X_list]
    train_concat = pd.concat(input_train, axis=1).values
    test_concat = pd.concat(input_test, axis=1).values
    n_classes = len(np.unique(y_train))
    all_fold_patterns = {}
    existing_results = pd.read_csv(results_path).to_dict('records') if os.path.exists(results_path) else []
    
    for app in ['PATX', 'CNN', 'MiniRocket', 'catch22', 'tsfresh', 'rdst']:
        t0 = time.time()
        if app == 'PATX':
            res = run_patx(input_train, y_train.values, input_test, 
                          metric=metric)
            preds = res['model'].predict(res['test_features'])
            score = np.sqrt(mean_squared_error(y_test, preds))
            n_feat = len(res['patterns'])
            p_ser = []
            for i, p in enumerate(res['patterns']):
                p_d = {'pattern_id': i + 1}
                for k, v in p.items():
                    if k == 'pattern': 
                        continue
                    p_d[k] = (v.tolist() if isinstance(v, np.ndarray) 
                             else (v.item() if hasattr(v, 'item') else v))
                p_ser.append(p_d)
            all_fold_patterns['test_split'] = p_ser
        elif app == 'CNN':
            score, _, n_feat = eval_cnn(
                train_concat, test_concat, y_train.values, y_test.values,
                task, metric, n_classes
            )
        elif app == 'MiniRocket':
            score, _, n_feat = eval_minirocket(
                train_concat, test_concat, y_train.values, y_test.values,
                task, metric, n_classes
            )
        elif app == 'catch22':
            score, _, n_feat = eval_catch22(
                train_concat, test_concat, y_train.values, y_test.values,
                metric, n_classes
            )
        elif app == 'tsfresh':  
            score, _, n_feat = eval_tsfresh(
                train_concat, test_concat, y_train.values, y_test.values,
                metric, n_classes=n_classes
            )
        elif app == 'rdst':
            score, _, n_feat = eval_rdst(
                train_concat, test_concat, y_train.values, y_test.values,
                metric, n_classes
            )
        elapsed = time.time() - t0
        print(f"  {app:10}: {score:.4f} ({elapsed:.1f}s, {n_feat} features)")
        res_d = {
            'approach': app, 'score': score, 'processing_time': elapsed, 
            'n_features': n_feat
        }
        existing_results.append(res_d)
    
    pd.DataFrame(existing_results).to_csv(results_path, index=False)
    os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
    with open(patterns_path, 'w') as f:
        json.dump(all_fold_patterns, f, indent=2)

if __name__ == "__main__":
    X_list, y, subject_ids, info = load_all_azt1d()
    run_evaluation(
        X_list, y, subject_ids, info,
        '../results/azt1d.csv',
        '../json_files/azt1d/pattern_parameters.json'
    )
