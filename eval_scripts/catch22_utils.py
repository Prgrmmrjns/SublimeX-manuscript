import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from pycatch22 import catch22_all
VAL_SIZE = 0.2


def run_catch22(input_series_train, input_series_test):
    train_features = [catch22_all(x)['values'] for x in input_series_train]
    test_features = [catch22_all(x)['values'] for x in input_series_test]
    return np.array(test_features), np.array(train_features)


def eval_catch22(train_array, test_array, y_train, y_test, metric, n_classes=None, initial_train=None, initial_test=None):
    t0 = time.time()
    test_feat, train_feat = run_catch22(train_array, test_array)
    train_feat = np.nan_to_num(train_feat, nan=0, posinf=0, neginf=0)
    test_feat = np.nan_to_num(test_feat, nan=0, posinf=0, neginf=0)
    if initial_train is not None and initial_test is not None:
        train_feat = np.hstack([initial_train, train_feat])
        test_feat = np.hstack([initial_test, test_feat])
    strat = y_train if metric != 'rmse' else None
    tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train, test_size=VAL_SIZE, random_state=42, stratify=strat)
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