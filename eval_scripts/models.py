import lightgbm as lgb
import numpy as np

class LightGBMModelWrapper:
    def __init__(self, task_type='classification', n_classes=None, num_threads=1):
        self.task_type = task_type
        self.n_classes = n_classes
        self.lgb_params = {
            'boosting_type': 'goss',
            'num_threads': num_threads,
            'verbosity': -1,
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.2,
            'force_row_wise': True,
        }
        if task_type == 'classification':
            if n_classes and n_classes > 2:
                self.lgb_params['objective'] = 'multiclass'
                self.lgb_params['num_class'] = n_classes
            else:
                self.lgb_params['objective'] = 'binary'
        else:
            self.lgb_params['objective'] = 'regression'
            self.lgb_params['metric'] = 'rmse'
        
        self.model = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = []
        callbacks = []
        if X_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets = [dval]
            callbacks = [lgb.early_stopping(10, verbose=False)]
            
        params = self.lgb_params.copy()
        num_boost_round = params.pop('n_estimators')
        
        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks
        )
        return self
    
    def predict(self, X):
        preds = self.model.predict(X)
        if self.task_type == 'classification':
            if self.n_classes and self.n_classes > 2:
                return np.argmax(preds, axis=1)
            else:
                return (preds > 0.5).astype(int)
        return preds
    
    def predict_proba(self, X):
        return self.model.predict(X)
    
    def clone(self):
        wrapper = LightGBMModelWrapper(self.task_type, self.n_classes)
        wrapper.lgb_params = self.lgb_params.copy()
        return wrapper

    def run_cv(self, X, y, folds, metric):
        dtrain = lgb.Dataset(X, label=y)
        params = self.lgb_params.copy()
        num_boost_round = params.pop('n_estimators')
        lgb_metric = 'rmse'
        if metric == 'auc':
            lgb_metric = 'auc_mu' if self.n_classes and self.n_classes > 2 else 'auc'
        elif metric == 'accuracy':
            lgb_metric = 'multi_error' if self.n_classes and self.n_classes > 2 else 'binary_error'
        params['metric'] = lgb_metric
        cv_kwargs = {'folds': folds} if not isinstance(folds, int) else {'nfold': folds}
        stratified = self.task_type == 'classification'
        cv_res = lgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(10, verbose=False)],
            stratified=stratified,
            **cv_kwargs
        )
        
        target_key = next((k for k in cv_res.keys() if k.endswith('-mean')), list(cv_res.keys())[0])
        values = cv_res[target_key]
        
        if 'auc' in lgb_metric:
            return max(values)
        elif 'error' in lgb_metric:
            return 1.0 - min(values)
        else:
            return min(values)
