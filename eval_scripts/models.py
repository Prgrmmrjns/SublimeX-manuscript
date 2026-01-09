import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
import warnings

# Suppress LightGBM and sklearn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def _compute_metric(model, X, y, metric):
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y, model.predict(X)))
    elif metric == 'auc':
        proba = model.predict_proba(X)
        n_classes = len(np.unique(y))
        if n_classes > 2:
            y_bin = label_binarize(y, classes=np.arange(n_classes))
            return roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
        return roc_auc_score(y, proba if proba.ndim == 1 else proba[:, 1]) if len(np.unique(y)) > 1 else 0.5
    elif metric == 'accuracy':
        return accuracy_score(y, model.predict(X))
    return 0.0

def _run_cv(model_obj, X, y, folds, metric):
    X_np = X.values if hasattr(X, 'values') else np.asarray(X)
    y_np = y.values if hasattr(y, 'values') else np.asarray(y)

    if isinstance(folds, int) and folds == 1:
        X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
        model = model_obj.clone()
        model.fit(X_tr, y_tr, X_val, y_val)
        return _compute_metric(model, X_val, y_val, metric)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42) if metric != 'rmse' else KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_np, y_np):
        model = model_obj.clone()
        model.fit(X_np[train_idx], y_np[train_idx], X_np[val_idx], y_np[val_idx])
        scores.append(_compute_metric(model, X_np[val_idx], y_np[val_idx], metric))
    return np.mean(scores)

class LightGBMWrapper:
    def __init__(self, task_type='classification', n_classes=None, **kwargs):
        self.task_type, self.n_classes = task_type, n_classes
        self.inner_cv = kwargs.pop('inner_cv', 1)
        n_jobs = kwargs.pop('num_threads', kwargs.pop('n_jobs', 1))
        
        self.params = {
            'max_depth': 3,
            'verbosity': -1,
            'n_jobs': n_jobs,
            'random_state': 42,
            'n_estimators': 100,
            'force_row_wise': True,
            'data_sample_strategy': 'goss',
            'use_quantized_grad': True
        }
        
        if task_type == 'classification':
            self.params['objective'] = 'multiclass' if n_classes and n_classes > 2 else 'binary'
        else:
            self.params['objective'] = 'regression'
            
        self.params.update(kwargs)
        self.model = None
        self._is_fitted = False
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
        y_train = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        
        M = lgb.LGBMClassifier if self.task_type == 'classification' else lgb.LGBMRegressor
        self.model = M(**self.params)
        
        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val = X_val.values if hasattr(X_val, 'values') else np.asarray(X_val)
            y_val = y_val.values if hasattr(y_val, 'values') else np.asarray(y_val)
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=10, verbose=False))
        
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        if not self._is_fitted:
            return np.zeros(len(X))
        X = X.values if hasattr(X, 'values') else np.asarray(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.task_type == 'regression':
            return self.predict(X)
        if not self._is_fitted:
            n_classes = self.n_classes if self.n_classes else 2
            if n_classes == 2:
                return np.full(len(X), 0.5)
            return np.full((len(X), n_classes), 1.0 / n_classes)
        
        X = X.values if hasattr(X, 'values') else np.asarray(X)
        proba = self.model.predict_proba(X)
        # Handle binary case specifically for consistency with PATX expectations
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    
    def score(self, X, y, metric):
        return _run_cv(self, X, y, self.inner_cv, metric)

    def clone(self):
        return LightGBMWrapper(self.task_type, self.n_classes, inner_cv=self.inner_cv, **self.params)
