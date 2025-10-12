import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from patx import feature_extraction
from models import LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

TIME_SERIES = ['CGM', 'Insulin', 'Carbs']

def load_azt1d_data(subject_id):
    """Load AZT1D data with three time series: CGM, Insulin, Carbs."""
    file_path = f"../processed_datasets/azt1d/subject_{subject_id}.parquet"
    df = pd.read_parquet(file_path)
    
    # Create separate DataFrames for each time series
    cgm_data = df[[col for col in df.columns if col.startswith('CGM_')]]
    insulin_data = df[[col for col in df.columns if col.startswith('Insulin_')]]
    carbs_data = df[[col for col in df.columns if col.startswith('Carbs_')]]
    
    # Store time series data in a dictionary for easier access
    time_series_data = {
        'CGM': cgm_data,
        'Insulin': insulin_data,
        'Carbs': carbs_data
    }
    
    y = df['target']
    return time_series_data, y

subject_ids = sorted([f.replace('subject_', '').replace('.parquet', '') for f in os.listdir("../processed_datasets/azt1d/") if f.endswith('.parquet')])
results = []

if os.path.exists('../results/azt1d.csv'):
    existing = pd.read_csv('../results/azt1d.csv')
    results = existing.to_dict('records')
    done = set(zip(existing['subject_id'].astype(str), existing['approach']))
else:
    done = set()

for subject_id in subject_ids:
    # Check if all approaches for this subject are already done
    subject_approaches_done = {(subject_id, approach) for approach in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']}
    if subject_approaches_done.issubset(done):
        continue
    
    print(f"Processing Subject {subject_id}")
    time_series_data, y = load_azt1d_data(subject_id)
    
    # Split data once for all time series
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=42
    )
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # PATX Approach (multivariate)
    if (subject_id, 'PATX') not in done:
        # Prepare time series data for PATX
        input_series_train = [time_series_data[series].iloc[train_idx] for series in TIME_SERIES]
        input_series_test = [time_series_data[series].iloc[test_idx] for series in TIME_SERIES]
        
        # Use CGM_0 as initial feature (first time point of CGM)
        initial_features_train = time_series_data['CGM'].iloc[train_idx]['CGM_0'].values.reshape(-1, 1)
        initial_features_test = time_series_data['CGM'].iloc[test_idx]['CGM_0'].values.reshape(-1, 1)
        
        t0 = time.time()
        res = feature_extraction(
            input_series_train, y_train, input_series_test,
            metric='rmse', n_trials=N_TRIALS, show_progress=SHOW_PROGRESS,
            initial_features=(initial_features_train, initial_features_test)
        )
        # Convert patterns to JSON-serializable format
        serializable_patterns = []
        for pattern in res['patterns']:
            serializable_pattern = {
                'pattern': pattern['pattern'].tolist(),
                'start': int(pattern['start']),
                'width': int(pattern['width']),
                'series_idx': int(pattern['series_idx'])
            }
            serializable_pattern['control_points'] = [float(cp) for cp in pattern['control_points']]
            serializable_patterns.append(serializable_pattern)
        
        # Store patterns for this subject
        os.makedirs('../json_files/azt1d', exist_ok=True)
        pattern_data = {
            'subject_id': subject_id,
            'patterns': serializable_patterns
        }
        with open(f'../json_files/azt1d/pattern_parameters_{subject_id}.json', 'w') as f:
            json.dump(pattern_data, f, indent=2, separators=(',', ': '))
        test_features = res['test_features']
        preds = res['model'].predict(test_features)
        n_feat = len(res['patterns'])
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        processing_time = time.time() - t0
        results.append({'subject_id': subject_id, 'approach': 'PATX', 'score': rmse, 'processing_time': processing_time, 'n_features': n_feat})
    
    # TSFRESH Approach (multivariate - concatenate series)
    if (subject_id, 'TSFRESH') not in done:
        # Concatenate all time series data
        all_series_train = pd.concat([time_series_data[series].iloc[train_idx] for series in TIME_SERIES], axis=1)
        all_series_test = pd.concat([time_series_data[series].iloc[test_idx] for series in TIME_SERIES], axis=1)
        
        # Add CGM_0 as initial feature
        cgm0_train = time_series_data['CGM'].iloc[train_idx]['CGM_0'].values.reshape(-1, 1)
        cgm0_test = time_series_data['CGM'].iloc[test_idx]['CGM_0'].values.reshape(-1, 1)
        
        t0 = time.time()
        test_features, train_features = run_tsfresh(all_series_train.values, all_series_test.values)
        # Concatenate with CGM_0
        train_features = np.concatenate([train_features, cgm0_train], axis=1)
        test_features = np.concatenate([test_features, cgm0_test], axis=1)
        train_features, val_features, y_train_split, y_valid = train_test_split(train_features, y_train.values, test_size=0.2, random_state=42)
        model = LightGBMModelWrapper('regression')
        model.fit(train_features, y_train_split, val_features, y_valid)
        preds = model.predict(test_features)
        n_feat = train_features.shape[1]
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        processing_time = time.time() - t0
        results.append({'subject_id': subject_id, 'approach': 'TSFRESH', 'score': rmse, 'processing_time': processing_time, 'n_features': n_feat})
    
    # CATCH22 Approach (univariate - use only CGM)
    if (subject_id, 'CATCH22') not in done:
        # Use only CGM time series
        cgm_train = time_series_data['CGM'].iloc[train_idx]
        cgm_test = time_series_data['CGM'].iloc[test_idx]
        
        # Add CGM_0 as initial feature
        cgm0_train = time_series_data['CGM'].iloc[train_idx]['CGM_0'].values.reshape(-1, 1)
        cgm0_test = time_series_data['CGM'].iloc[test_idx]['CGM_0'].values.reshape(-1, 1)
        
        t0 = time.time()
        test_features, train_features = run_catch22(cgm_train.values, cgm_test.values)
        # Concatenate with CGM_0
        train_features = np.concatenate([train_features, cgm0_train], axis=1)
        test_features = np.concatenate([test_features, cgm0_test], axis=1)
        train_features, val_features, y_train_split, y_valid = train_test_split(train_features, y_train.values, test_size=0.2, random_state=42)
        model = LightGBMModelWrapper('regression')
        model.fit(train_features, y_train_split, val_features, y_valid)
        preds = model.predict(test_features)
        n_feat = train_features.shape[1]
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        processing_time = time.time() - t0
        results.append({'subject_id': subject_id, 'approach': 'CATCH22', 'score': rmse, 'processing_time': processing_time, 'n_features': n_feat})
    
    # CNN Approach (multivariate - 3 channels: CGM, Insulin, Carbs)
    if (subject_id, 'CNN') not in done:
        # Concatenate all time series data for CNN
        all_series_train = pd.concat([time_series_data[series].iloc[train_idx] for series in TIME_SERIES], axis=1)
        all_series_test = pd.concat([time_series_data[series].iloc[test_idx] for series in TIME_SERIES], axis=1)
        
        t0 = time.time()
        preds = run_cnn(all_series_train.values, y_train.values, all_series_test.values, task_type='regression', metric='rmse', epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
        n_feat = all_series_train.shape[1]
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        processing_time = time.time() - t0
        results.append({'subject_id': subject_id, 'approach': 'CNN', 'score': rmse, 'processing_time': processing_time, 'n_features': n_feat})
    
    # Print subject summary
    subject_results = pd.DataFrame([r for r in results if str(r['subject_id']) == subject_id])
    for approach in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
        approach_results = subject_results[subject_results['approach'] == approach]
        if len(approach_results) > 0:
            rmse = approach_results['score'].iloc[0]
            time_val = approach_results['processing_time'].iloc[0]
            features = approach_results['n_features'].iloc[0]
            print(f"{approach:8}: RMSE={rmse:.4f}, Time={time_val:.1f}s, Features={features:.0f}")

pd.DataFrame(results).to_csv('../results/azt1d.csv', index=False)