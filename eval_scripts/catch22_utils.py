import time
import numpy as np
from pycatch22 import catch22_all

# Import LightGBMWrapper from models
from models import LightGBMWrapper

def run_catch22(input_series_train, input_series_test):
    train_features = [catch22_all(x)['values'] for x in input_series_train]
    test_features = [catch22_all(x)['values'] for x in input_series_test]
    return np.array(test_features), np.array(train_features)


def eval_catch22(train_array, test_array, y_train, y_test, metric, n_classes=None, initial_train=None, initial_test=None):
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

    test_feat, train_feat = run_catch22(train_array, test_array)
    train_feat = np.nan_to_num(train_feat, nan=0, posinf=0, neginf=0)
    test_feat = np.nan_to_num(test_feat, nan=0, posinf=0, neginf=0)
    
    if initial_train is not None and initial_test is not None:
        train_feat = np.hstack([initial_train, train_feat])
        test_feat = np.hstack([initial_test, test_feat])
        
    # Use LightGBM for fair comparison
    task = 'regression' if metric == 'rmse' else 'classification'
    model = LightGBMWrapper(task_type=task, n_classes=n_classes, n_jobs=1, inner_cv=3)
    
    # Fit on full train and predict on test
    model.fit(train_feat, y_train_enc)
    
    if metric == 'rmse':
        preds = model.predict(test_feat)
        from sklearn.metrics import mean_squared_error
        score = np.sqrt(mean_squared_error(y_test_enc, preds))
    elif metric == 'auc':
        proba = model.predict_proba(test_feat)
        from sklearn.metrics import roc_auc_score
        if (n_classes or 2) > 2:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test_enc, classes=np.arange(n_classes))
            score = roc_auc_score(y_test_bin, proba, multi_class='ovr', average='macro')
        else:
            score = roc_auc_score(y_test_enc, proba if proba.ndim == 1 else proba[:, 1])
    else:
        preds = model.predict(test_feat)
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test_enc, preds)
        
    elapsed = time.time() - t0
    return score, elapsed, train_feat.shape[1]
