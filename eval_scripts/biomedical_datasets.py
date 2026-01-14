import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import StratifiedKFold, KFold

# Filter LightGBM warnings about feature names
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Import PATX and Baseline Utils
from patx_runner import run_patx, serialize_patterns
from cnn import eval_cnn
from minirocket_utils import eval_minirocket
from catch22_utils import eval_catch22
from tsfresh_utils import eval_tsfresh
from rdst_utils import eval_rdst

K_FOLDS = 5

# ------------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# ------------------------------------------------------------------------------

def load_emotions():
    df = pd.read_csv("../processed_datasets/emotions/emotions.csv", dtype=np.float32)
    y = df.pop('target').astype(int)
    cols_a = sorted([c for c in df.columns if c.endswith('_a')], key=lambda x: int(x.split('_')[1]))
    cols_b = sorted([c for c in df.columns if c.endswith('_b')], key=lambda x: int(x.split('_')[1]))
    return [df[cols_a], df[cols_b]], y, {'metric': 'accuracy', 'task': 'classification'}

def load_mimic():
    df = pd.read_csv('../processed_datasets/mimic/mimic_processed.csv')
    y = df['ARDS_FLAG']
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'anchor_age', 'ARDS_FLAG']]
    series_names = sorted(list(set(col.split('_hour_')[0] for col in feature_cols if '_hour_' in col)))
    X_list = [df[sorted([c for c in feature_cols if c.startswith(f"{s}_hour_")], key=lambda x: int(x.split('_hour_')[1]))] for s in series_names]
    return X_list, y, {'metric': 'accuracy', 'task': 'classification'}

def load_mitbih():
    df = pd.read_csv("../processed_datasets/mitbih/mitbih_processed.csv")
    return [df.drop('target', axis=1)], df['target'], {'metric': 'accuracy', 'task': 'classification'}

def load_remc(cell_line):
    df = pd.read_parquet(f"../processed_datasets/remc/{cell_line}.parquet")
    time_series = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    X_list = [df[[c for c in df.columns if c.startswith(f"{s}_")]] for s in time_series]
    return X_list, df['target'], {'metric': 'auc', 'task': 'classification'}

def load_svd():
    df = pd.read_parquet("../processed_datasets/svd/svd.parquet")
    channels = [f"{v}_{p}" for v in ["a", "i", "u"] for p in ["n", "h", "l"]]
    X_list = [df[sorted([c for c in df.columns if c.startswith(f"{ch}_")], key=lambda x: int(x.split('_')[-1]))].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32) for ch in channels]
    return X_list, df["target"].astype(int), {'metric': 'accuracy', 'task': 'classification'}

def load_pamap2():
    df = pd.read_parquet('../processed_datasets/pamap2/pamap2.parquet')
    def create_windows(df, window_size=100, step_size=50):
        feature_cols = [col for col in df.columns if col not in ['time_stamp', 'activity_id', 'id']]
        all_windows, all_labels = [], []
        for sid in df['id'].unique():
            sdf = df[df['id'] == sid]
            for aid in sdf['activity_id'].unique():
                adf = sdf[sdf['activity_id'] == aid]
                for i in range(0, len(adf) - window_size + 1, step_size):
                    all_windows.append(adf.iloc[i:i+window_size][feature_cols].values)
                    all_labels.append(aid)
        return np.array(all_windows), np.array(all_labels)
    windows, y = create_windows(df)
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_mapped = pd.Series([label_map[label] for label in y])
    X_list = [pd.DataFrame(windows[:, :, i][:, : (windows.shape[1] // 10) * 10].reshape(windows.shape[0], -1, 10).mean(axis=2)) for i in range(windows.shape[2])]
    return X_list, y_mapped, {'metric': 'accuracy', 'task': 'classification'}

# ------------------------------------------------------------------------------
# EVALUATION LOGIC
# ------------------------------------------------------------------------------

def run_evaluation(dataset_name, X_list, y, info, results_path, patterns_path, sub_id=None):
    metric, task = info['metric'], info['task']
    initial_features = info.get('initial_features')
    approaches = ['PATX', 'CNN',  'MiniRocket', 'catch22', 'tsfresh', 'RDST'] #,, 'ShapeletTransform'
    print(f"\n{dataset_name}")

    # Check if this dataset/subject/cell_line is already complete (all folds for all approaches)
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        if sub_id:
            id_col = 'cell_line' if dataset_name == 'remc' else 'subject_id'
            subset = existing_df[existing_df[id_col].astype(str) == str(sub_id)]
        else:
            subset = existing_df[existing_df['dataset'] == dataset_name]
        all_complete = True
        for app in approaches:
            app_subset = subset[subset['approach'] == app]
            if len(app_subset) < K_FOLDS:
                all_complete = False
                break
        if all_complete:
            print(f"Already complete, skipping.")
            return
    
    # Pre-compute all fold splits (same splits for all approaches)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42) if task == 'classification' else KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(y)), y))
    
    # Pre-compute fold data splits
    fold_data = []
    for train_idx, test_idx in folds:
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        input_train = [x.iloc[train_idx] for x in X_list]
        input_test = [x.iloc[test_idx] for x in X_list]
        train_concat = pd.concat(input_train, axis=1).values
        test_concat = pd.concat(input_test, axis=1).values
        init_train = initial_features[train_idx] if initial_features is not None else None
        init_test = initial_features[test_idx] if initial_features is not None else None
        init_feat = (init_train, init_test) if init_train is not None else None
        n_classes = len(np.unique(y_train))
        fold_data.append({
            'y_train': y_train, 'y_test': y_test,
            'input_train': input_train, 'input_test': input_test,
            'train_concat': train_concat, 'test_concat': test_concat,
            'init_train': init_train, 'init_test': init_test,
            'init_feat': init_feat, 'n_classes': n_classes
        })

    all_fold_patterns = {}
    fold_results = {app: [] for app in approaches}

    # Run each approach across all folds before moving to next approach
    for app in approaches:
        print(f"\n{app}:")
        for fold_idx, fold_info in enumerate(fold_data, 1):
            t0 = time.time()
            
            if app == 'PATX':
                res = run_patx(fold_info['input_train'], 
                    fold_info['y_train'].values, 
                    fold_info['input_test'], 
                    metric=metric, 
                    initial_features=fold_info['init_feat'], 
                    y_test=fold_info['y_test'].values
                )
                score = res['score']
                n_feat = res['n_features']
                all_fold_patterns[f'fold_{fold_idx}'] = serialize_patterns(res['patterns'])
            elif app == 'CNN':
                score, _, n_feat = eval_cnn(
                    fold_info['train_concat'], fold_info['test_concat'],
                    fold_info['y_train'].values, fold_info['y_test'].values,
                    task, metric, fold_info['n_classes']
                )
            elif app == 'MiniRocket':
                score, _, n_feat = eval_minirocket(
                    fold_info['train_concat'], fold_info['test_concat'],
                    fold_info['y_train'].values, fold_info['y_test'].values,
                    task_type=task, metric=metric, num_classes=fold_info['n_classes']
                )
            elif app == 'catch22':
                score, _, n_feat = eval_catch22(
                    fold_info['train_concat'], fold_info['test_concat'],
                    fold_info['y_train'].values, fold_info['y_test'].values,
                    metric, fold_info['n_classes'], fold_info['init_train'], fold_info['init_test']
                )
            elif app == 'tsfresh':
                score, _, n_feat = eval_tsfresh(
                    fold_info['train_concat'], fold_info['test_concat'],
                    fold_info['y_train'].values, fold_info['y_test'].values,
                    metric, n_classes=fold_info['n_classes'],
                    initial_train=fold_info['init_train'], initial_test=fold_info['init_test']
                )
            elif app == 'RDST':
                score, _, n_feat = eval_rdst(
                    fold_info['train_concat'], fold_info['test_concat'],
                    fold_info['y_train'].values, fold_info['y_test'].values,
                    metric, fold_info['n_classes']
                )
            
            elapsed = time.time() - t0
            print(f"  Fold {fold_idx}: {score:.4f} ({elapsed:.1f}s, {n_feat} features)")
            fold_results[app].append({'fold': fold_idx, 'score': score, 'time': elapsed, 'n_features': n_feat})
    
    # Save per-fold results
    existing_results = (
        pd.read_csv(results_path).to_dict('records') if os.path.exists(results_path) else []
    )
    for app in approaches:
        for fold_res in fold_results[app]:
            res_d = {
                'dataset': dataset_name,
                'approach': app,
                'fold': fold_res['fold'],
                'score': fold_res['score'],
                'processing_time': fold_res['time'],
                'n_features': fold_res['n_features']
            }
            if dataset_name == 'remc':
                res_d['cell_line'] = sub_id
            existing_results.append(res_d)
    
    pd.DataFrame(existing_results).to_csv(results_path, index=False)
    os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
    with open(patterns_path, 'w') as f: json.dump(all_fold_patterns, f, indent=2)

if __name__ == "__main__":
    for ds in ['emotions', 'mimic', 'mitbih', 'remc', 'svd', 'pamap2']:
        if ds == 'remc':
            for cl in sorted([f.replace('.parquet', '') for f in os.listdir('../processed_datasets/remc') if f.endswith('.parquet')])[:2]:
                X, y, info = load_remc(cl)
                run_evaluation('remc', X, y, info, '../results/remc.csv', f'../json_files/remc/pattern_parameters_{cl}.json', cl)
        else:
            loader = {'emotions': load_emotions, 'mimic': load_mimic, 'mitbih': load_mitbih, 'svd': load_svd, 'pamap2': load_pamap2}[ds]
            X, y, info = loader()
            run_evaluation(ds, X, y, info, f'../results/{ds}.csv', f'../json_files/{ds}/pattern_parameters.json')
