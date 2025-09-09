import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from patx import PatternOptimizer
from params import *

# Override dataset-specific parameters for MITBIH
DATASET = 'MITBIH'
METRIC = 'accuracy'
TASK_TYPE = 'classification'
TIME_SERIES_IDENTIFIERS = []
from tsfresh_utils import run_tsfresh_baseline
from cnn import run_cnn

warnings.filterwarnings('ignore')

def load_mitbih_data():
    data = pd.read_csv("mitbih_database/mitbih_processed.csv")
    X = data.drop('target', axis=1)
    y = data['target'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size=TEST_SIZE, random_state=42, stratify=y)
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    return {
        'X_train': X_train_df,
        'X_test': X_test_df,
        'y_train': y_train,
        'y_test': y_test,
        'input_data': X_train_df,
        'test_data': X_test_df,
        'dims': X_train_df.shape[1]
    }

def run_patx():
    d = load_mitbih_data()
    y_train = d['y_train']
    y_test = d['y_test']
    input_data = d['input_data']
    test_data = d['test_data']
    n_classes = len(np.unique(y_train))
    model = get_model(TASK_TYPE, 'MITBIH', n_classes)
    optimizer = PatternOptimizer(
        input_data, y_train, model=model, max_n_trials=MAX_N_TRIALS,
        show_progress=SHOW_PROGRESS, test_size=VAL_SIZE, n_jobs=N_JOBS,
        dataset='MITBIH', multiple_series=len(TIME_SERIES_IDENTIFIERS) > 0,
        X_test_data=test_data, polynomial_degree=POLYNOMIAL_DEGREE,
        metric=METRIC, val_size=VAL_SIZE,
        initial_features=None
    )
    t0 = time.time()
    result = optimizer.feature_extraction()
    t1 = time.time()
    optimizer.save_parameters_to_json('MITBIH')
    X_train = result['X_train']
    X_val = result['X_val']
    y_train_split = result['y_train']
    y_val = result['y_val']
    X_test = result['X_test']
    m = result['model']
    train_preds = m.predict(X_train)
    val_preds = m.predict(X_val)
    test_preds = m.predict(X_test)
    train_score = accuracy_score(y_train_split, train_preds)
    val_score = accuracy_score(y_val, val_preds)
    test_score = accuracy_score(y_test, test_preds)
    return {
        'approach': 'PATX',
        'train_score': float(train_score),
        'val_score': float(val_score),
        'test_score': float(test_score),
        'n_features': len(result['patterns']),
        'processing_time': t1 - t0
    }

def run_tsfresh():
    d = load_mitbih_data()
    Xtr, Xte, ytr, yte = d['X_train'], d['X_test'], d['y_train'], d['y_test']
    n_classes = len(np.unique(ytr))
    m = get_model(TASK_TYPE, 'MITBIH', n_classes)
    te_f, X_tr, X_val, y_tr, y_val, dt = run_tsfresh_baseline(Xtr.values, ytr, Xte.values, task_type='classification', val_size=VAL_SIZE, n_jobs=1)
    m.train(X_tr, y_tr, X_val, y_val)
    tr_pred = m.predict(X_tr); val_pred = m.predict(X_val); te_pred = m.predict(te_f)
    tr_s = accuracy_score(y_tr, tr_pred); val_s = accuracy_score(y_val, val_pred); te_s = accuracy_score(yte, te_pred)
    return {
        'approach': 'TSFRESH',
        'train_score': float(tr_s),
        'val_score': float(val_s),
        'test_score': float(te_s),
        'n_features': int(X_tr.shape[1]),
        'processing_time': dt
    }

def run_cnn_baseline():
    d = load_mitbih_data()
    X, y = d['X_train'].values, d['y_train']
    Xt, yt = d['X_test'].values, d['y_test']
    n_classes = int(len(np.unique(y)))
    res = run_cnn(X, y, Xt, task_type='classification', metric='accuracy', val_size=VAL_SIZE, num_classes=n_classes, epochs=50, lr=1e-3)
    te_pred = res['test_predictions']
    te_s = accuracy_score(yt, te_pred)
    return {
        'approach': 'CNN',
        'train_score': res['train_score'],
        'val_score': res['val_score'],
        'test_score': float(te_s),
        'n_features': int(X.shape[1]),
        'processing_time': res['processing_time']
    }

results = []

r1 = run_patx(); print(f"PATX test accuracy: {r1['test_score']:.4f}"); results.append(r1)
r2 = run_tsfresh(); print(f"TSFRESH test accuracy: {r2['test_score']:.4f}"); results.append(r2)
r3 = run_cnn_baseline(); print(f"CNN test accuracy: {r3['test_score']:.4f}"); results.append(r3)

df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/mitbih.csv', index=False)
