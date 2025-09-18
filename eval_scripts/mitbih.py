import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from patx import PatternExtractor, get_model
from params import *
from tsfresh_utils import run_tsfresh
from cnn import run_cnn

# Override dataset-specific parameters for MITBIH
DATASET = 'MITBIH'
METRIC = 'accuracy'
TASK_TYPE = 'classification'
TIME_SERIES_IDENTIFIERS = []

warnings.filterwarnings('ignore')

data = pd.read_csv("../processed_datasets/mitbih_processed.csv")
X = data.drop('target', axis=1).to_numpy()
y = data['target'].to_numpy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for approach in ['PATX', 'TSFRESH', 'CNN']:
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        n_classes = len(np.unique(y_train))
        
        t0 = time.time()
        
        if approach == 'PATX':
            X_train_df = pd.DataFrame(X_train)
            model = get_model(TASK_TYPE, 'MITBIH', n_classes)
            optimizer = PatternExtractor(
                X_train_df, y_train, model=model, max_n_trials=MAX_N_TRIALS,
                show_progress=SHOW_PROGRESS, n_jobs=N_JOBS,
                dataset='MITBIH', multiple_series=len(TIME_SERIES_IDENTIFIERS) > 0,
                X_test=pd.DataFrame(X_val), polynomial_degree=POLYNOMIAL_DEGREE,
                metric=METRIC, val_size=VAL_SIZE,
                initial_features=None
            )
            result = optimizer.feature_extraction()
            if fold == 0:
                optimizer.save_parameters_to_json('../json_files/MITBIH')
            model = result['model']
            test_preds = model.predict(result['X_test'])
            n_features = len(result['patterns'])
            
        elif approach == 'TSFRESH':
            model = get_model(TASK_TYPE, 'MITBIH', n_classes)
            val_f, X_tr, X_v, y_tr, y_v, _ = run_tsfresh(X_train, y_train, X_val, task_type='classification', val_size=VAL_SIZE, n_jobs=TSFRESH_N_JOBS)
            model.train(X_tr, y_tr, X_v, y_v)
            test_preds = model.predict(val_f)
            n_features = X_tr.shape[1]
            
        elif approach == 'CNN':
            res = run_cnn(X_train, y_train, X_val, task_type='classification', metric='accuracy', val_size=VAL_SIZE, num_classes=n_classes, epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
            test_preds = res['test_predictions']
            n_features = X_train.shape[1]
        
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
df.to_csv('../results/mitbih.csv', index=False)
