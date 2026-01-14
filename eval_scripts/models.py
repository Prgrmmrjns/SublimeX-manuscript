import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
import warnings

# Filter LightGBM warnings about feature names
warnings.filterwarnings('ignore', message='X does not have valid feature names')

class LightGBMWrapper:
    """LightGBM model wrapper with evaluate method for patX framework."""
    
    def __init__(self, task_type='classification', n_classes=None, **kwargs):
        self.task_type = task_type
        self.n_classes = n_classes
        self.n_estimators = kwargs.pop('n_estimators', 100)
        self.inner_cv = kwargs.pop('inner_cv', 3)
        n_jobs = kwargs.pop('num_threads', kwargs.pop('n_jobs', 1))
        self.params = {
            'max_depth': 4,
            'learning_rate': 0.1,
            'data_sample_strategy': 'goss',
            'use_quantized_grad': True,
            'verbosity': -1,
            'n_jobs': n_jobs
        }
        if task_type == 'classification':
            self.eval_metric = 'multi_logloss' if n_classes and n_classes > 2 else 'binary_logloss'
            self.params['objective'] = 'multiclass' if n_classes and n_classes > 2 else 'binary'
            if n_classes and n_classes > 2:
                self.params['num_class'] = n_classes
        else:
            self.eval_metric = 'rmse'
            self.params['objective'] = 'regression'
        self.params.update(kwargs)
    
    def evaluate(self, X_train, y_train, X_val, y_val, metric):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        if self.inner_cv == 1:
            # Single split: use provided validation set for early stopping
            if X_val is not None and y_val is not None:
                X_val = np.asarray(X_val)
                y_val = np.asarray(y_val)
                model = self._create_model()
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=self.eval_metric,
                )
                return self._compute_metric(model, X_val, y_val, metric)
            else:
                # No validation set: fit on all training data
                model = self._create_model()
                model.fit(X_train, y_train)
                return self._compute_metric(model, X_train, y_train, metric)
        else:
            # Cross-validation
            cv = StratifiedKFold(n_splits=self.inner_cv, shuffle=True, random_state=42) if metric != 'rmse' else KFold(n_splits=self.inner_cv, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                model = self._create_model()
                model.fit(
                    X_train[train_idx], y_train[train_idx],
                    eval_set=[(X_train[val_idx], y_train[val_idx])],
                    eval_metric=self.eval_metric,
                )
                scores.append(self._compute_metric(model, X_train[val_idx], y_train[val_idx], metric))
            return np.mean(scores)
    
    def _create_model(self):
        M = lgb.LGBMClassifier if self.task_type == 'classification' else lgb.LGBMRegressor
        return M(**self.params, n_estimators=self.n_estimators)
    
    def _compute_metric(self, model, X, y, metric):
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y, model.predict(X)))
        elif metric == 'auc':
            proba = model.predict_proba(X)
            n_classes = len(np.unique(y))
            if n_classes > 2:
                y_bin = label_binarize(y, classes=np.arange(n_classes))
                return roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
            p = proba if proba.ndim == 1 else proba[:, 1]
            return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        elif metric == 'accuracy':
            return accuracy_score(y, model.predict(X))
        return 0.0
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model = self._create_model()
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        X = np.asarray(X)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
