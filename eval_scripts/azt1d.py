import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from patx import PatternExtractor, get_model
from tsfresh_utils import run_tsfresh
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

def load_azt1d_data(subject_id):
    file_path = f"../processed_datasets/azt1d/subject_{subject_id}.parquet"
    df = pd.read_parquet(file_path)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['target'].values
    glucose = df['glucose'].values
    time_vals = df['time'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    glucose_train, glucose_test, _, _ = train_test_split(glucose, y, test_size=TEST_SIZE, random_state=42)
    time_train, time_test, _, _ = train_test_split(time_vals, y, test_size=TEST_SIZE, random_state=42)
    initial_train = np.column_stack([glucose_train, time_train])
    initial_test = np.column_stack([glucose_test, time_test])
    return X_train, X_test, y_train, y_test, initial_train, initial_test

results = []
files = os.listdir("../processed_datasets/azt1d/")
for file in files[:3]:
    subject_id = file.replace('subject_', '').replace('.parquet', '')
    X_train, X_test, y_train, y_test, initial_train, initial_test = load_azt1d_data(subject_id)
    for approach in ['PATX', 'TSFRESH', 'CNN']:
        t0 = time.time()
        if approach == 'PATX':
            model = get_model('lightgbm', 'regression', 'AZT1D')
            optimizer = PatternExtractor(
                X_train, y_train, model=model, max_n_trials=MAX_N_TRIALS, 
                show_progress=SHOW_PROGRESS, n_jobs=-1, 
                dataset='AZT1D', multiple_series=False, X_test=X_test, 
                polynomial_degree=POLYNOMIAL_DEGREE, metric='rmse', val_size=VAL_SIZE,
                initial_features=(initial_train, initial_test)
            )
            result = optimizer.feature_extraction()
            optimizer.save_parameters_to_json(f'../json_files/AZT1D/{subject_id}')
            m = result['model']
            test_preds = m.predict(result['X_test'])
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
            n_features = len(result['patterns'])
            
        elif approach == 'TSFRESH':
            model = get_model('lightgbm', 'regression', 'AZT1D')
            te_f, X_tr, X_val, y_tr, y_val, dt = run_tsfresh(X_train, y_train, X_test, task_type='regression', val_size=VAL_SIZE, n_jobs=TSFRESH_N_JOBS)
            model.train(X_tr, y_tr, X_val, y_val)
            test_preds = model.predict(te_f)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
            n_features = int(X_tr.shape[1])
            
        elif approach == 'CNN':
            res = run_cnn(X_train, y_train, X_test, task_type='regression', epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE, val_size=VAL_SIZE)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, res['test_predictions'])))
            n_features = int(X_train.shape[1])
        
        t1 = time.time()
        results.append({
            'subject_id': subject_id,
            'approach': approach,
            'test_rmse': test_rmse,
            'n_features': n_features,
            'processing_time': t1 - t0
        })
        print(f"Subject {subject_id} {approach} test RMSE: {test_rmse:.4f}")

df = pd.DataFrame(results)
df.to_csv('../results/azt1d.csv', index=False)