import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from patx import PatternExtractor, get_model
from params import *
from tsfresh_utils import run_tsfresh
from cnn import run_cnn

warnings.filterwarnings('ignore')

df = pd.read_csv('../processed_datasets/mimic_processed.csv')
y = df['ARDS_FLAG'].values
anchor_age = df['anchor_age'].values

feature_cols = [col for col in df.columns if col not in ['subject_id', 'anchor_age', 'ARDS_FLAG']]
series_names = []
for col in feature_cols:
    if '_hour_' in col:
        series_name = col.split('_hour_')[0]
        if series_name not in series_names:
            series_names.append(series_name)

X_list = []
for series_name in series_names:
    series_cols = [col for col in feature_cols if col.startswith(f"{series_name}_hour_")]
    series_cols.sort(key=lambda x: int(x.split('_hour_')[1]))
    X_series = df[series_cols].values
    X_list.append(X_series)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for approach in ['PATX', 'TSFRESH', 'CNN']:
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_list[0], y)):
        y_train, y_val = y[train_idx], y[val_idx]
        n_classes = len(np.unique(y_train))
        
        t0 = time.time()
        
        if approach == 'PATX':
            X_tr_list, X_val_list = [], []
            for X in X_list:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_val = scaler.transform(X[val_idx])
                X_tr_list.append(X_train)
                X_val_list.append(X_val)
            
            age_scaler = StandardScaler()
            train_init = age_scaler.fit_transform(anchor_age[train_idx].reshape(-1, 1))
            val_init = age_scaler.transform(anchor_age[val_idx].reshape(-1, 1))
            
            model = get_model('classification', 'MIMIC', n_classes)
            extractor = PatternExtractor(X_tr_list, y_train, model=model, max_n_trials=MAX_N_TRIALS,
                                  show_progress=SHOW_PROGRESS, n_jobs=N_JOBS,
                                  dataset='MIMIC', multiple_series=True, X_test=X_val_list,
                                  polynomial_degree=POLYNOMIAL_DEGREE, metric='accuracy', 
                                  val_size=VAL_SIZE, initial_features=(train_init, val_init))
            result = extractor.feature_extraction()
            if fold == 0:
                extractor.save_parameters_to_json('../json_files/MIMIC')
            model = result['model']
            test_preds = model.predict(result['X_test'])
            n_features = len(result['patterns'])
            
        elif approach == 'TSFRESH':
            # Combine all series for TSFRESH
            X_combined = np.concatenate(X_list, axis=1)
            X_train, X_val = X_combined[train_idx], X_combined[val_idx]
            
            model = get_model('classification', 'MIMIC', n_classes)
            val_f, X_tr, X_v, y_tr, y_v, _ = run_tsfresh(X_train, y_train, X_val, task_type='classification', val_size=VAL_SIZE, n_jobs=TSFRESH_N_JOBS)
            model.train(X_tr, y_tr, X_v, y_v)
            test_preds = model.predict(val_f)
            n_features = X_tr.shape[1]
            
        elif approach == 'CNN':
            X_combined = np.concatenate(X_list, axis=1)
            X_with_age = np.column_stack([X_combined, anchor_age])
            
            X_tr, X_val = X_with_age[train_idx], X_with_age[val_idx]
            
            result = run_cnn(X_tr, y_train, X_val, task_type='classification', 
                            metric='accuracy', val_size=VAL_SIZE, 
                            num_classes=n_classes, epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
            test_preds = result['test_predictions']
            n_features = X_tr.shape[1]
        
        t1 = time.time()
        score = accuracy_score(y_val, test_preds)
        
        results.append({
            'approach': approach,
            'fold': fold + 1,
            'score': float(score),
            'processing_time': float(t1 - t0),
            'n_features': int(n_features)
        })
        
        print(f"{approach} fold {fold+1}: {score:.4f}")

df = pd.DataFrame(results)
df.to_csv('../results/mimic.csv', index=False)