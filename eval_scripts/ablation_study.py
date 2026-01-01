import pandas as pd
import numpy as np
import time
import warnings
import os
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patx"))
from patx import feature_extraction, get_model
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]

N_TRIALS = 1000
MAX_SAMPLES = 50000
N_TRANSFORMS = 5
INNER_K_FOLDS = 4
VAL_SIZE = 0.2
K_FOLDS = 5
N_WORKERS = -1
SHOW_PROGRESS = False

def load_mitbih_data():
    data = pd.read_csv(ROOT / "processed_datasets/mitbih/mitbih_processed.csv")
    return data.drop('target', axis=1), data['target']

def load_svd_data():
    df = pd.read_parquet(ROOT / "processed_datasets/svd/svd.parquet")
    channels = [df[[f"{v}_{t}" for t in range(700)]].astype(np.float32) for v in ["a_n", "i_n", "u_n"]]
    return channels, df["target"].astype(int)

def load_remc_data(cell_line):
    df = pd.read_parquet(ROOT / "processed_datasets/remc" / f"{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']], df['target']

def load_emotions_data():
    df = pd.read_csv(ROOT / "processed_datasets/emotions/emotions.csv", dtype=np.float32)
    y = df.pop('target').astype(int)
    cols_a = sorted([c for c in df.columns if c.endswith('_a')], key=lambda x: int(x.split('_')[1]))
    cols_b = sorted([c for c in df.columns if c.endswith('_b')], key=lambda x: int(x.split('_')[1]))
    return [df[cols_a], df[cols_b]], y

def load_mimic_data():
    df = pd.read_csv(ROOT / "processed_datasets/mimic/mimic_processed.csv")
    y = df['ARDS_FLAG'].astype(int)
    feature_cols = [c for c in df.columns if c not in ['subject_id', 'anchor_age', 'ARDS_FLAG']]
    series_names = list(dict.fromkeys(c.split('_hour_')[0] for c in feature_cols if '_hour_' in c))
    return [df[sorted([c for c in feature_cols if c.startswith(f"{s}_hour_")], key=lambda x: int(x.split('_hour_')[1]))].astype(np.float32) for s in series_names], y

def load_azt1d_data(subject_id):
    df = pd.read_parquet(ROOT / "processed_datasets/azt1d" / f"subject_{subject_id}.parquet")
    X = [df[[c for c in df.columns if c.startswith(f"{s}_") and c != 'CGM_current']].astype(np.float32) for s in ['CGM', 'Insulin', 'Carbs']]
    return X, df['target'].astype(np.float32), pd.DataFrame({'CGM_current': df['CGM_current'].astype(np.float32)})

def load_pamap2_data(bin_size=10):
    df = pd.read_parquet(ROOT / "processed_datasets/pamap2/pamap2.parquet")
    feature_cols = [c for c in df.columns if c not in ['time_stamp', 'activity_id', 'id']]
    xs, ys = [], []
    for subject_id in df['id'].unique():
        subject_df = df[df['id'] == subject_id].reset_index(drop=True)
        for activity_id in subject_df['activity_id'].unique():
            activity_df = subject_df[subject_df['activity_id'] == activity_id]
            for i in range(0, len(activity_df) - 100 + 1, 50):
                xs.append(activity_df.iloc[i:i+100][feature_cols].values)
                ys.append(activity_id)
    windows, y = np.asarray(xs), np.asarray(ys)
    X_list = []
    for j in range(windows.shape[2]):
        x = windows[:, :, j]
        n_bins = x.shape[1] // bin_size
        x = x[:, :n_bins * bin_size].reshape(x.shape[0], n_bins, bin_size).mean(axis=2) if n_bins else x
        X_list.append(pd.DataFrame(x.astype(np.float32)))
    u = np.unique(y)
    return X_list, pd.Series(y).map({v: i for i, v in enumerate(u)}).astype(int)

results_file = str(ROOT / "results/ablation_study.csv")
existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

def save_result(result):
    global existing
    existing = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
    existing.to_csv(results_file, index=False)

def run_variant(name, X, y, folds, dataset, metric, initial_features=None, **kwargs):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    for fold, (tr, te) in enumerate(folds):
        if not existing.empty and len(existing[(existing['dataset'] == dataset) & (existing['approach'] == name) & (existing['fold'] == fold)]) > 0:
            print(f"Fold {fold+1}: Already completed, skipping")
            continue
        X_tr = [x.iloc[tr].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[tr].astype(np.float32)]
        X_te = [x.iloc[te].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[te].astype(np.float32)]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        t0 = time.time()
        task_type = 'regression' if metric == 'rmse' else 'classification'
        n_classes = len(np.unique(y_tr)) if task_type == 'classification' else None
        
        model_type = kwargs.pop('model_type', 'lightgbm')
        model = get_model(model_type, task_type, n_classes=n_classes, n_jobs=1)
        
        init_feat = (initial_features.iloc[tr].values, initial_features.iloc[te].values) if initial_features is not None else None
        params = dict(n_trials=N_TRIALS, n_transforms=N_TRANSFORMS, inner_k_folds=INNER_K_FOLDS, val_size=VAL_SIZE, n_workers=N_WORKERS, show_progress=SHOW_PROGRESS)
        params.update(kwargs)
        res = feature_extraction(X_tr, y_tr.values, X_te, metric=metric, model=model, initial_features=init_feat, **params)
        if metric == 'accuracy':
            score = accuracy_score(y_te.values, res['model'].predict(res['test_features']))
        elif metric == 'rmse':
            score = float(np.sqrt(np.mean((y_te.values - res['model'].predict(res['test_features'])) ** 2)))
        else:
            preds = res['model'].predict_proba(res['test_features'])
            preds = preds[:, 1] if preds.ndim > 1 else preds
            score = roc_auc_score(y_te.values, preds)
        t = time.time() - t0
        result = {'dataset': dataset, 'approach': name, 'fold': fold, 'score': score, 'time': t, 'n_features': len(res['patterns'])}
        save_result(result)
        print(f"Fold {fold+1}: {metric.upper()}={score:.4f}, Time={t:.1f}s, Features={len(res['patterns'])}")

variants = [
    ('inner_k_1', {'inner_k_folds': 1}),
    ('trials_100', {'n_trials': 100}),
    ('no_transforms', {'n_transforms': 1, 'transforms': ['raw']}),
    ('no_shifting', {'sliding_window': False}),
    ('subsampling_0.5', {'subsample': 0.5}),
    ('pearson', {'distance_metric': 'pearson'}),
    ('one_control_point', {'pattern_type': 'bezier', 'pattern_kwargs': {'n_control_points': 1}}),
    ('polynomial_3_params', {'pattern_type': 'polynomial', 'pattern_kwargs': {'order': 2}}),
]

variant_names = [v[0] for v in variants] + ['default']

def get_main_result(dataset_name):
    base_path = ROOT / "results"
    if dataset_name == 'mitbih':
        df = pd.read_csv(base_path / "mitbih.csv")
    elif dataset_name == 'svd':
        df = pd.read_csv(base_path / "svd.csv")
    elif dataset_name == 'emotions':
        df = pd.read_csv(base_path / "emotions.csv")
    elif dataset_name == 'mimic':
        df = pd.read_csv(base_path / "mimic.csv")
    elif dataset_name.startswith('remc_'):
        cell_line = dataset_name.replace('remc_', '')
        df = pd.read_csv(base_path / "remc.csv")
        df = df[df['cell_line'] == cell_line]
    elif dataset_name.startswith('azt1d_'):
        subject_id = dataset_name.replace('azt1d_', '')
        df = pd.read_csv(base_path / "azt1d.csv")
        try:
            subject_id_val = int(subject_id)
            df = df[df['subject_id'] == subject_id_val]
        except ValueError:
            df = df[df['subject_id'] == subject_id]
    else:
        return None
    
    df = df[df['approach'] == 'PATX'].copy()
    df['approach'] = 'default'
    df = df.rename(columns={'processing_time': 'time'})
    df['dataset'] = dataset_name
    return df[['dataset', 'approach', 'score', 'time', 'n_features']]

def run_dataset(dataset_name, X, y, folds, metric, initial_features=None):
    global existing
    for name, params in variants:
        if existing.empty or len(existing[(existing['dataset'] == dataset_name) & (existing['approach'] == name)]) < len(folds):
            run_variant(name, X, y, folds, dataset=dataset_name, metric=metric, initial_features=initial_features, **params)
            existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

print("Running MITBIH...")
X_mitbih, y_mitbih = load_mitbih_data()
folds_mitbih = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(X_mitbih, y_mitbih))
run_dataset('mitbih', X_mitbih, y_mitbih, folds_mitbih, 'accuracy')

print("\nRunning SVD...")
channels_svd, y_svd = load_svd_data()
folds_svd = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(channels_svd, axis=1), y_svd))
run_dataset('svd', channels_svd, y_svd, folds_svd, 'accuracy')

print("\nRunning REMC (first cell line)...")
cell_lines = sorted([f.replace('.parquet', '') for f in os.listdir(ROOT / 'processed_datasets/remc') if f.endswith('.parquet')])
if cell_lines:
    X_remc, y_remc = load_remc_data(cell_lines[0])
    folds_remc = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_remc, axis=1), y_remc))
    run_dataset(f'remc_{cell_lines[0]}', X_remc, y_remc, folds_remc, 'auc')

print("\nRunning Emotions...")
X_emotions, y_emotions = load_emotions_data()
folds_emotions = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_emotions, axis=1), y_emotions))
run_dataset('emotions', X_emotions, y_emotions, folds_emotions, 'accuracy')

print("\nRunning MIMIC...")
X_mimic, y_mimic = load_mimic_data()
folds_mimic = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_mimic, axis=1), y_mimic))
run_dataset('mimic', X_mimic, y_mimic, folds_mimic, 'accuracy')

print("\nRunning AZT1D (first subject)...")
subject_ids = sorted([f.replace('subject_', '').replace('.parquet', '') for f in os.listdir(ROOT / 'processed_datasets/azt1d') if f.endswith('.parquet')])
if subject_ids:
    X_azt, y_azt, init_azt = load_azt1d_data(subject_ids[0])
    folds_azt = list(KFold(K_FOLDS, shuffle=True, random_state=42).split(np.zeros(len(y_azt)), y_azt))
    run_dataset(f'azt1d_{subject_ids[0]}', X_azt, y_azt, folds_azt, 'rmse', initial_features=init_azt)

print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    for dataset_name in df['dataset'].unique():
        print(f"\n{dataset_name.upper()}:")
        sub = df[(df['dataset'] == dataset_name) & (df['approach'].isin(variant_names))]
        main_res = get_main_result(dataset_name)
        if main_res is not None:
            sub = pd.concat([sub, main_res], ignore_index=True)
        print(sub.groupby('approach').agg({'score': ['mean', 'std'], 'time': ['mean', 'std'], 'n_features': ['mean', 'std']}).round(4))
