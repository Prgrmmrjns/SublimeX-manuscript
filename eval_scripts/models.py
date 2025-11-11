import lightgbm as lgb
class LightGBMModelWrapper:
    def __init__(self, task_type='classification', n_classes=None):
        self.task_type = task_type
        self.n_classes = n_classes
        lgb_params = {
            'num_threads': 1,
            'verbosity': -1,
            'max_depth': 3,
            'n_estimators': 1000,
            'learning_rate': 0.1,
        }
        self.model = lgb.LGBMClassifier(**lgb_params) if task_type == 'classification' else lgb.LGBMRegressor(**lgb_params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        else:
            self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.task_type == 'classification':
            preds = self.model.predict_proba(X)
            return preds[:, 1] if self.n_classes == 2 else preds
        return self.model.predict(X)
    
    def clone(self):
        return LightGBMModelWrapper(self.task_type, self.n_classes)
