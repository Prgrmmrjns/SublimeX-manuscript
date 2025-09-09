import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from patx import PatternOptimizer
from tsfresh_utils import run_tsfresh_baseline
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

def load_azt1d_data(subject_id=None):
    if subject_id is None:
        subject_id = 1
    data_path = "AZT1D 2025/CGM Records"
    file_path = f"{data_path}/Subject {subject_id}/Subject {subject_id}.csv"
    df = pd.read_csv(file_path)
    if 'CGM' not in df.columns:
        return None
    df['datetime'] = pd.to_datetime(df['EventDateTime'])
    df['glucose'] = df['CGM']
    df['time'] = df['datetime'].dt.hour + df['datetime'].dt.minute / 60
    df = df[['datetime', 'glucose', 'time']].copy().sort_values('datetime').reset_index(drop=True)
    for lag in range(24):
        df[f'glucose_lag_{lag}'] = df['glucose'].shift(lag)
    df[f'glucose_{PREDICTION_HORIZON}'] = df['glucose'].shift(-PREDICTION_HORIZON) - df['glucose']
    df = df.dropna(subset=[f'glucose_{PREDICTION_HORIZON}'])
    target_feature = f'glucose_{PREDICTION_HORIZON}'
    base_feature_cols = [f'glucose_lag_{i}' for i in range(24)]
    X_base = df[base_feature_cols].values.astype(np.float32)
    y = df[target_feature].values.astype(np.float32)
    valid_mask = ~(np.isnan(y) | np.any(np.isnan(X_base), axis=1))
    X_base = X_base[valid_mask]
    y = y[valid_mask]
    X_train_base, X_test_base, y_train, y_test = train_test_split(X_base, y, test_size=TEST_SIZE, random_state=42)
    X_glucose = df['glucose'].values.astype(np.float32)[valid_mask]
    X_time = df['time'].values.astype(np.float32)[valid_mask]
    X_train_glucose, X_test_glucose, _, _ = train_test_split(X_glucose, y, test_size=TEST_SIZE, random_state=42)
    X_train_time, X_test_time, _, _ = train_test_split(X_time, y, test_size=TEST_SIZE, random_state=42)
    X_train_initial = np.column_stack([X_train_glucose, X_train_time])
    X_test_initial = np.column_stack([X_test_glucose, X_test_time])
    return {
        'X_train': X_train_base,
        'X_test': X_test_base,
        'y_train': y_train,
        'y_test': y_test,
        'input_data': X_train_base,
        'test_data': X_test_base,
        'dims': X_train_base.shape[1],
        'initial_features': X_train_initial,
        'test_initial_features': X_test_initial
    }


def run_patx(subject_id, d):
    model = get_model('lightgbm', 'regression', 'AZT1D')
    optimizer = PatternOptimizer(
        d['input_data'], d['y_train'], model=model,
        max_n_trials=MAX_N_TRIALS, show_progress=SHOW_PROGRESS,
        test_size=TEST_SIZE, n_jobs=-1, dataset='AZT1D', multiple_series=False,
        X_test_data=d['test_data'], polynomial_degree=POLYNOMIAL_DEGREE,
        metric='rmse', val_size=VAL_SIZE,
        initial_features=(d['initial_features'], d['test_initial_features'])
    )
    t0 = time.time()
    result = optimizer.feature_extraction()
    t1 = time.time()
    optimizer.save_parameters_to_json(f'AZT1D/{subject_id}')
    X_train, X_val = result['X_train'], result['X_val']
    y_train, y_val = result['y_train'], result['y_val']
    X_test = result['X_test']
    y_test = d['y_test']
    m = result['model']
    tr = m.predict(X_train); va = m.predict(X_val); te = m.predict(X_test)
    return {
        'subject_id': subject_id,
        'approach': 'PATX',
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, tr))),
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, va))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, te))),
        'n_features': len(result['patterns']),
        'processing_time': t1 - t0
    }

def run_tsfresh(subject_id, d):
    y_train = d['y_train']; y_test = d['y_test']
    model = get_model('lightgbm', 'regression', 'AZT1D')
    te_f, X_tr, X_val, y_tr, y_val, dt = run_tsfresh_baseline(d['X_train'], y_train, d['X_test'], task_type='regression', val_size=VAL_SIZE, n_jobs=1)
    model.train(X_tr, y_tr, X_val, y_val)
    tr = model.predict(X_tr); va = model.predict(X_val); te = model.predict(te_f)
    t1 = dt
    return {
        'subject_id': subject_id,
        'approach': 'TSFRESH',
        'train_rmse': float(np.sqrt(mean_squared_error(y_tr, tr))),
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, va))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, te))),
        'n_features': int(X_tr.shape[1]),
        'processing_time': t1
    }



results = []
for subject_id in range(1, 26):
    d = load_azt1d_data(subject_id)
    if d is None:
        print(f"Subject {subject_id}: skipped (no data)")
        continue
    r1 = run_patx(subject_id, d); print(f"Subject {subject_id} PATX test RMSE: {r1['test_rmse']:.4f}"); results.append(r1)
    r2 = run_tsfresh(subject_id, d); print(f"Subject {subject_id} TSFRESH test RMSE: {r2['test_rmse']:.4f}"); results.append(r2)
    res = run_cnn(d['X_train'], d['y_train'], d['X_test'], task_type='regression', epochs=50, lr=1e-3, val_size=VAL_SIZE)
    te = res['test_predictions']; yt = d['y_test']
    r3 = {'subject_id': subject_id, 'approach': 'CNN', 'train_rmse': res['train_score'], 'val_rmse': res['val_score'], 'test_rmse': float(np.sqrt(mean_squared_error(yt, te))), 'n_features': int(d['X_train'].shape[1]), 'processing_time': res['processing_time']}
    print(f"Subject {subject_id} CNN test RMSE: {r3['test_rmse']:.4f}"); results.append(r3)

df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/azt1d.csv', index=False)