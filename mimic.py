import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from patx import PatternOptimizer
from params import *
from tsfresh_utils import run_tsfresh_baseline
from cnn import run_cnn as run_cnn_baseline

warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the preprocessed MIMIC data."""
    
    df = pd.read_csv('mimic/mimic_processed.csv')
    
    # Separate features and target
    y = df['ARDS_FLAG'].values
    anchor_age = df['anchor_age'].values
    
    # Get all time series feature names (exclude subject_id, anchor_age, ARDS_FLAG)
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'anchor_age', 'ARDS_FLAG']]
    
    # Extract unique series names from column names
    series_names = []
    for col in feature_cols:
        if '_hour_' in col:
            series_name = col.split('_hour_')[0]
            if series_name not in series_names:
                series_names.append(series_name)
    
    # Organize features by time series
    X_list = []
    for series_name in series_names:
        series_cols = [col for col in feature_cols if col.startswith(f"{series_name}_hour_")]
        series_cols.sort(key=lambda x: int(x.split('_hour_')[1]))  # Sort by hour
        X_series = df[series_cols].values
        X_list.append(X_series)
    
    return X_list, y, anchor_age, series_names

def run_patx():
    X_list, y, anchor_age, series_names = load_processed_data()
    
    # Split data
    indices = np.arange(len(y))
    tr_idx, te_idx, y_tr, y_te = train_test_split(indices, y, test_size=0.33, random_state=42, stratify=y)
    
    # Scale each series
    X_tr_list, X_te_list = [], []
    for X in X_list:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        X_tr_list.append(X_tr)
        X_te_list.append(X_te)
    
    # Prepare initial features (anchor_age)
    age_scaler = StandardScaler()
    train_init = age_scaler.fit_transform(anchor_age[tr_idx].reshape(-1, 1))
    test_init = age_scaler.transform(anchor_age[te_idx].reshape(-1, 1))
    
    # Run PATX
    model = get_model('classification', 'MIMIC', len(np.unique(y)))
    opt = PatternOptimizer(X_tr_list, y_tr, model=model, max_n_trials=300,
                          show_progress=SHOW_PROGRESS, test_size=VAL_SIZE, n_jobs=N_JOBS,
                          dataset='MIMIC', multiple_series=True, X_test_data=X_te_list,
                          polynomial_degree=POLYNOMIAL_DEGREE, metric='accuracy', 
                          val_size=VAL_SIZE, initial_features=(train_init, test_init))
    
    t0 = time.time()
    result = opt.feature_extraction()
    t1 = time.time()
    opt.save_parameters_to_json('MIMIC')
    
    # Get predictions
    m = result['model']
    tr_pred = m.predict(result['X_train'])
    val_pred = m.predict(result['X_val'])
    te_pred = m.predict(result['X_test'])
    
    tr_acc = accuracy_score(result['y_train'], tr_pred)
    val_acc = accuracy_score(result['y_val'], val_pred)
    te_acc = accuracy_score(y_te, te_pred)
    
    return {
        'approach': 'PATX',
        'train_score': float(tr_acc),
        'val_score': float(val_acc), 
        'test_score': float(te_acc),
        'n_features': len(result['patterns']),
        'processing_time': t1 - t0
    }

def run_tsfresh():
    X_list, y, anchor_age, series_names = load_processed_data()
    
    # Split data first
    indices = np.arange(len(y))
    tr_idx, te_idx, y_tr, y_te = train_test_split(indices, y, test_size=0.33, random_state=42, stratify=y)
    
    # Extract features from all time series
    X_train_features = []
    X_val_features = []
    X_test_features = []
    total_time = 0
    
    for i, X in enumerate(X_list):
        X_tr = X[tr_idx]
        X_te = X[te_idx]
        
        te_f, Xtr, Xva, ytr, yva, dt = run_tsfresh_baseline(
            X_tr, y_tr, X_te, task_type='classification', val_size=VAL_SIZE, n_jobs=1)
        
        total_time += dt
        
        if i == 0:
            # For first series, keep the split labels
            y_train_final, y_val_final = ytr, yva
            X_train_features.append(Xtr)
            X_val_features.append(Xva)
            X_test_features.append(te_f)
        else:
            # For subsequent series, just extract features (same split)
            X_train_features.append(Xtr)
            X_val_features.append(Xva)
            X_test_features.append(te_f)
    
    # Concatenate features from all series
    X_train_combined = np.concatenate(X_train_features, axis=1)
    X_val_combined = np.concatenate(X_val_features, axis=1)
    X_test_combined = np.concatenate(X_test_features, axis=1)
    
    # Train model on combined features
    model = get_model('classification', 'MIMIC', len(np.unique(y)))
    model.train(X_train_combined, y_train_final, X_val_combined, y_val_final)
    
    tr_pred = model.predict(X_train_combined)
    val_pred = model.predict(X_val_combined)
    te_pred = model.predict(X_test_combined)
    
    return {
        'approach': 'TSFRESH',
        'train_score': float(accuracy_score(y_train_final, tr_pred)),
        'val_score': float(accuracy_score(y_val_final, val_pred)),
        'test_score': float(accuracy_score(y_te, te_pred)),
        'n_features': X_train_combined.shape[1],
        'processing_time': total_time
    }

def run_cnn():
    X_list, y, anchor_age, series_names = load_processed_data()

    
    # Concatenate all time series into one matrix (flatten all series)
    X_combined = np.concatenate(X_list, axis=1)  # Shape: (n_subjects, 29*24)
    
    # Add anchor_age as additional feature
    X_with_age = np.column_stack([X_combined, anchor_age])  # Shape: (n_subjects, 29*24+1)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_with_age, y, test_size=0.33, random_state=42, stratify=y)
    
    result = run_cnn_baseline(X_tr, y_tr, X_te, task_type='classification', 
                            metric='accuracy', val_size=VAL_SIZE, 
                            num_classes=len(np.unique(y)))
    
    te_acc = accuracy_score(y_te, result['test_predictions'])
    
    return {
        'approach': 'CNN',
        'train_score': result['train_score'],
        'val_score': result['val_score'],
        'test_score': float(te_acc),
        'n_features': X_tr.shape[1],
        'processing_time': result['processing_time']
    }

results = []

r1 = run_patx(); print(f"PATX test accuracy: {r1['test_score']:.4f}"); results.append(r1)
r2 = run_tsfresh(); print(f"TSFRESH test accuracy: {r2['test_score']:.4f}"); results.append(r2)
r3 = run_cnn(); print(f"CNN test accuracy: {r3['test_score']:.4f}"); results.append(r3)

df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/mimic.csv', index=False)