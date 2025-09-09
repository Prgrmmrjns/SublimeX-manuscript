import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import numpy as np

DATASET = 'REMC'
MODEL = 'catboost' 

if DATASET == 'REMC':
    METRIC = 'auc'
    TASK_TYPE = 'classification'
    TIME_SERIES_IDENTIFIERS = ['H3K4me3', 'H3K9me3', 'H3K27me3', 'H3K4me1', 'H3K36me3']
elif DATASET == 'MITBIH':
    METRIC = 'accuracy'
    TASK_TYPE = 'classification'
    TIME_SERIES_IDENTIFIERS = []
elif DATASET == 'AZT1D':
    METRIC = 'rmse'
    TASK_TYPE = 'regression'
    TIME_SERIES_IDENTIFIERS = []
    SUBJECT_ID = 1
elif DATASET == 'MIMIC':
    METRIC = 'accuracy'
    TASK_TYPE = 'classification'
    TIME_SERIES_IDENTIFIERS = ['Respiratory Rate [insp/min]', 'Heart Rate [bpm]', 'O2 saturation pulseoxymetry [%]']

# -- Dataset specific parameters --
PREDICTION_HORIZON = 12
CELL_LINE = 'E003'

# --- PatX Configuration ---
MAX_N_TRIALS = 500  # High number, will be stopped early by no improvement
N_JOBS = -1 # Single thread to avoid segmentation faults  / multiple threads only works for mitbih dataset
SHOW_PROGRESS = True
TEST_SIZE = 1/3
VAL_SIZE = 0.5
POLYNOMIAL_DEGREE = 3  # Degree of polynomial patterns (0=constant, 1=linear, 2=quadratic, etc.)

def get_lgb_params(task_type, dataset, n_classes=None):
    params = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'num_iterations': 100,
        'random_state': 42,
        'num_threads': 1,   
        'force_col_wise': True,
        'verbosity': -1,
        'data_sample_strategy': 'goss',
    }
    
    if task_type == 'classification':
        if dataset == 'REMC':
            params['objective'] = 'binary'
            params['metric'] = 'auc'
        else:
            params['objective'] = 'multiclass'
            params['metric'] = 'multi_logloss'
            if n_classes is not None:
                params['num_class'] = n_classes
    else:
        params['objective'] = 'regression'
        params['metric'] = 'rmse'
    
    return params

class LightGBMModel:
    def __init__(self, params):
        self.params = params
        self.booster = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        train_data = lgb.Dataset(X_train, label=y_train)
        self.booster = lgb.train(self.params, train_data, valid_sets=[lgb.Dataset(X_val, label=y_val, reference=train_data)], callbacks=[lgb.early_stopping(10, verbose=False)])
        return self
    
    def predict(self, X):
        preds = self.booster.predict(X)
        if self.params.get('objective') == 'multiclass':
            return np.argmax(preds, axis=1)
        elif self.params.get('objective') == 'binary':
            return (preds > 0.5).astype(int)
        return preds
    
    def predict_proba(self, X):
        preds = self.booster.predict(X)
        if self.params.get('objective') == 'binary':
            return np.column_stack([1 - preds, preds])
        else:
            return preds
    
    def predict_proba_positive(self, X):
        """Get probability of positive class for binary classification, handling 1D/2D arrays"""
        preds = self.predict_proba(X)
        if preds.ndim == 2:
            return preds[:, 1]
        return preds

def get_model(task_type, dataset, n_classes=None):
    params = get_lgb_params(task_type, dataset, n_classes)
    return LightGBMModel(params)

def evaluate_model_performance(model, X, y, METRIC):
    if METRIC == 'auc':
        if len(np.unique(y)) > 2:
            y_pred = model.predict_proba(X)
            score = roc_auc_score(y, y_pred, multi_class='ovr', average='macro')
        else:
            y_pred = model.predict_proba_positive(X)
            score = roc_auc_score(y, y_pred)
    elif METRIC == 'accuracy':
        y_pred = model.predict(X)
        score = accuracy_score(y, y_pred)
    elif METRIC == 'rmse':
        y_pred = model.predict(X)
        score = np.sqrt(mean_squared_error(y, y_pred))
    return score