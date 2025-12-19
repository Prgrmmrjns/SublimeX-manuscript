import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters


def run_tsfresh(input_series_train, input_series_test):
    train_df = pd.DataFrame(input_series_train).copy()
    test_df = pd.DataFrame(input_series_test).copy()
    train_df['id'] = range(len(train_df))
    test_df['id'] = range(len(test_df))
    train_long = train_df.melt(id_vars=['id'], var_name='time', value_name='value')
    test_long = test_df.melt(id_vars=['id'], var_name='time', value_name='value')
    fc = MinimalFCParameters()
    train_features = extract_features(train_long, column_id='id', column_sort='time', column_value='value', impute_function=impute, n_jobs=1, default_fc_parameters=fc, show_warnings=False, disable_progressbar=True)
    test_features = extract_features(test_long, column_id='id', column_sort='time', column_value='value', impute_function=impute, n_jobs=1, default_fc_parameters=fc, show_warnings=False, disable_progressbar=True)
    return test_features, train_features


def eval_tsfresh(train_array, test_array, y_train, y_test, metric, val_size=0.2, n_classes=None, initial_train=None, initial_test=None):
    t0 = time.time()
    test_feat, train_feat = run_tsfresh(train_array, test_array)
    if initial_train is not None and initial_test is not None:
        train_feat = np.hstack([initial_train, train_feat])
        test_feat = np.hstack([initial_test, test_feat])
    strat = y_train if metric != 'rmse' else None
    tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train, test_size=val_size, random_state=42, stratify=strat)
    if metric == 'rmse':
        model = Ridge().fit(tr_f, y_tr)
        preds = model.predict(test_feat)
        score = np.sqrt(mean_squared_error(y_test, preds))
    elif metric == 'auc':
        base = LogisticRegression(max_iter=200, n_jobs=1)
        model = OneVsRestClassifier(base).fit(tr_f, y_tr) if (n_classes or 2) > 2 else base.fit(tr_f, y_tr)
        score = roc_auc_score(y_test, model.predict_proba(test_feat)[:, 1])
    else:
        model = LogisticRegression(max_iter=200, n_jobs=1).fit(tr_f, y_tr)
        preds = model.predict(test_feat)
        score = accuracy_score(y_test, preds)
    elapsed = time.time() - t0
    return score, elapsed, train_feat.shape[1]