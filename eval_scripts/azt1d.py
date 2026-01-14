import pandas as pd
import numpy as np
import os
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

# Import PATX and Baseline Utils
from patx_runner import run_patx
from cnn import eval_cnn
from minirocket_utils import eval_minirocket
from catch22_utils import eval_catch22
from tsfresh_utils import eval_tsfresh

# ------------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# ------------------------------------------------------------------------------

def load_azt1d(subject_id):
    df = pd.read_parquet(f"../processed_datasets/azt1d/subject_{subject_id}.parquet")
    time_series = ['CGM', 'Insulin', 'Carbs']
    X_list = [df[[c for c in df.columns if c.startswith(f'{s}_') and c != 'CGM_current']] for s in time_series]
    return X_list, df['target'], {'initial_features': df['CGM_current'].values.reshape(-1, 1), 'metric': 'rmse', 'task': 'regression'}

# ------------------------------------------------------------------------------
# EVALUATION LOGIC
# ------------------------------------------------------------------------------

def run_evaluation(dataset_name, X_list, y, info, results_path, patterns_path, sub_id=None):
    metric, task = info['metric'], info['task']
    initial_features = info.get('initial_features')
    print(f"\n>>> {dataset_name} ({sub_id})")

    # Check if this subject is already complete
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        subset = existing_df[existing_df['subject_id'].astype(str) == str(sub_id)]
        if len(subset) > 0:
            done_approaches = set(subset['approach'].unique())
            needed = {'PATX', 'CNN', 'MiniRocket', 'catch22', 'tsfresh'}
            if needed.issubset(done_approaches):
                print(f"  Already complete, skipping.")
                return
    
    # Simple split for AZT1D
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    input_train, input_test = [x.iloc[train_idx] for x in X_list], [x.iloc[test_idx] for x in X_list]
    train_concat, test_concat = pd.concat(input_train, axis=1).values, pd.concat(input_test, axis=1).values
    init_train = initial_features[train_idx] if initial_features is not None else None
    init_test = initial_features[test_idx] if initial_features is not None else None
    init_feat = (init_train, init_test) if init_train is not None else None
    
    n_classes, n_channels, n_time = len(np.unique(y_train)), len(X_list), X_list[0].shape[1]
    all_fold_patterns = {}
    
    existing_results = pd.read_csv(results_path).to_dict('records') if os.path.exists(results_path) else []
    
    for app in ['PATX', 'CNN', 'MiniRocket', 'catch22', 'tsfresh']:
        t0 = time.time()
        if app == 'PATX':
            res = run_patx(input_train, y_train.values, input_test, metric=metric, initial_features=init_feat)
            preds = res['model'].predict(res['test_features'])
            score = np.sqrt(mean_squared_error(y_test, preds))
            n_feat = len(res['patterns'])
            
            p_ser = []
            for i, p in enumerate(res['patterns']):
                p_d = {'pattern_id': i + 1}
                for k, v in p.items():
                    if k == 'pattern': continue
                    p_d[k] = v.tolist() if isinstance(v, np.ndarray) else (v.item() if hasattr(v, 'item') else v)
                p_ser.append(p_d)
            all_fold_patterns['test_split'] = p_ser
        elif app == 'CNN': score, _, n_feat = eval_cnn(train_concat, test_concat, y_train.values, y_test.values, task, metric, n_classes)
        elif app == 'MiniRocket': score, _, n_feat = eval_minirocket(train_concat, test_concat, y_train.values, y_test.values, n_channels, n_time, task, metric, n_classes)
        elif app == 'catch22': score, _, n_feat = eval_catch22(train_concat, test_concat, y_train.values, y_test.values, metric, n_classes, init_train, init_test)
        elif app == 'tsfresh': score, _, n_feat = eval_tsfresh(train_concat, test_concat, y_train.values, y_test.values, metric, n_classes=n_classes, initial_train=init_train, initial_test=init_test)
        
        elapsed = time.time() - t0
        print(f"  {app:10}: {score:.4f} ({elapsed:.1f}s)")
        res_d = {'approach': app, 'subject_id': sub_id, 'score': score, 'processing_time': elapsed, 'n_features': n_feat}
        existing_results.append(res_d)
    
    pd.DataFrame(existing_results).to_csv(results_path, index=False)
    if all_fold_patterns:
        os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
        with open(patterns_path, 'w') as f: json.dump(all_fold_patterns, f, indent=2)

if __name__ == "__main__":
    for sid in sorted([f.replace('subject_', '').replace('.parquet', '') for f in os.listdir("../processed_datasets/azt1d/") if f.endswith('.parquet')]):
        X, y, info = load_azt1d(sid)
        run_evaluation('azt1d', X, y, info, '../results/azt1d.csv', f'../json_files/azt1d/pattern_parameters_{sid}.json', sid)
