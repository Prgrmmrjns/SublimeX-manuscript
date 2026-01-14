import time
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

from models import LightGBMWrapper


def run_tsfresh(input_series_train, input_series_test):
    # Ensure input is numpy array of float64 to avoid precision issues
    if not isinstance(input_series_train, np.ndarray):
        input_series_train = np.array(input_series_train)
    if not isinstance(input_series_test, np.ndarray):
        input_series_test = np.array(input_series_test)
        
    input_series_train = input_series_train.astype(np.float64)
    input_series_test = input_series_test.astype(np.float64)
    
    # Replace infs with NaNs so they can be imputed
    input_series_train[np.isinf(input_series_train)] = np.nan
    input_series_test[np.isinf(input_series_test)] = np.nan

    # Handle NaNs: forward fill, then backward fill, then 0
    if np.isnan(input_series_train).any():
        df_train = pd.DataFrame(input_series_train)
        df_train = df_train.ffill(axis=1).bfill(axis=1).fillna(0)
        input_series_train = df_train.values
        
    if np.isnan(input_series_test).any():
        df_test = pd.DataFrame(input_series_test)
        df_test = df_test.ffill(axis=1).bfill(axis=1).fillna(0)
        input_series_test = df_test.values

    # Convert to long format for tsfresh
    train_df = pd.DataFrame(input_series_train)
    test_df = pd.DataFrame(input_series_test)
    
    train_df['id'] = range(len(train_df))
    test_df['id'] = range(len(test_df))
    
    train_long = train_df.melt(id_vars=['id'], var_name='time', value_name='value')
    test_long = test_df.melt(id_vars=['id'], var_name='time', value_name='value')
    
    # Final safety check: drop any remaining NaNs
    train_long = train_long.dropna(subset=['value'])
    test_long = test_long.dropna(subset=['value'])
    
    fc = MinimalFCParameters()
    train_features = extract_features(
        train_long, 
        column_id='id', 
        column_sort='time', 
        column_value='value', 
        impute_function=impute, 
        n_jobs=1, 
        default_fc_parameters=fc, 
        show_warnings=False, 
        disable_progressbar=True
    )

    test_features = extract_features(
        test_long, 
        column_id='id', 
        column_sort='time', 
        column_value='value', 
        impute_function=impute, 
        n_jobs=1, 
        default_fc_parameters=fc, 
        show_warnings=False, 
        disable_progressbar=True
    )
            
    return test_features, train_features


def eval_tsfresh(train_array, test_array, y_train, y_test, metric, val_size=0.2, n_classes=None, initial_train=None, initial_test=None):
    t0 = time.time()

    # Handle encoding for classification
    if metric != 'rmse':
        y_all = np.concatenate([y_train, y_test])
        y_encoded = np.unique(y_all, return_inverse=True)[1]
        y_train_enc = y_encoded[:len(y_train)]
        y_test_enc = y_encoded[len(y_train):]
        n_classes = len(np.unique(y_all))
    else:
        y_train_enc, y_test_enc = y_train, y_test

    try:
        test_feat, train_feat = run_tsfresh(train_array, test_array)
    except Exception as e:
        print(f"tsfresh failed: {e}")
        return 0.0, 0.0, 0
        
    # Handle case where feature extraction failed or returned different columns
    common_cols = train_feat.columns.intersection(test_feat.columns)
    train_feat = train_feat[common_cols]
    test_feat = test_feat[common_cols]
    
    # Add initial features if provided
    if initial_train is not None and initial_test is not None:
        # Convert initial features to DataFrame if needed
        if not isinstance(initial_train, pd.DataFrame):
            initial_train = pd.DataFrame(initial_train, columns=[f'initial_{i}' for i in range(initial_train.shape[1])])
        if not isinstance(initial_test, pd.DataFrame):
            initial_test = pd.DataFrame(initial_test, columns=[f'initial_{i}' for i in range(initial_test.shape[1])])
        
        train_feat = pd.concat([initial_train.reset_index(drop=True), train_feat.reset_index(drop=True)], axis=1)
        test_feat = pd.concat([initial_test.reset_index(drop=True), test_feat.reset_index(drop=True)], axis=1)
    
    X_tr = train_feat.values
    X_te = test_feat.values
    
    if X_tr.shape[1] == 0:
        return 0.0, time.time() - t0, 0

    # Use LightGBM for fair comparison
    task = 'regression' if metric == 'rmse' else 'classification'
    model = LightGBMWrapper(task_type=task, n_classes=n_classes, n_jobs=1, inner_cv=3)
    
    model.fit(X_tr, y_train_enc)
    if metric == 'rmse':
        preds = model.predict(X_te)
        from sklearn.metrics import mean_squared_error
        score = np.sqrt(mean_squared_error(y_test_enc, preds))
    elif metric == 'auc':
        proba = model.predict_proba(X_te)
        from sklearn.metrics import roc_auc_score
        if (n_classes or 2) > 2:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test_enc, classes=np.arange(n_classes))
            score = roc_auc_score(y_test_bin, proba, multi_class='ovr', average='macro')
        else:
            score = roc_auc_score(y_test_enc, proba if proba.ndim == 1 else proba[:, 1])
    else:
        preds = model.predict(X_te)
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test_enc, preds)
        
    elapsed = time.time() - t0
    return score, elapsed, X_tr.shape[1]
