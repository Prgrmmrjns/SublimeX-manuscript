import pandas as pd
import numpy as np
import time
import os
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline
from patx_runner import run_patx
# Filter LightGBM warnings about feature names
warnings.filterwarnings('ignore', message='X does not have valid feature names')


def generate_polynomial_pattern(control_points: list, width: int) -> np.ndarray:
    """Generate pattern from third-degree polynomial coefficients.
    
    The control points are interpreted as polynomial coefficients [a, b, c, d]
    for the polynomial: p(t) = a + b*t + c*t^2 + d*t^3
    where t is normalized to [0, 1].
    
    If fewer than 4 control points are provided, higher-degree terms are set to 0.
    """
    coeffs = np.zeros(4, dtype=np.float32)
    coeffs[:len(control_points)] = control_points
    t = np.linspace(0, 1, width, dtype=np.float32)
    # Polynomial: a + b*t + c*t^2 + d*t^3
    pattern = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3
    return pattern.astype(np.float32)


class LinearRegressionWrapper:
    """Wrapper for linear models with evaluate method for patX framework."""
    
    def __init__(self, task_type='classification', n_classes=None, **kwargs):
        self.task_type = task_type
        self.n_classes = n_classes
        self.inner_cv = kwargs.pop('inner_cv', 3)
        self.model = None
    
    def evaluate(self, X_train, y_train, X_val, y_val, metric):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        if self.inner_cv == 1:
            # Single split: use provided validation set
            if X_val is not None and y_val is not None:
                X_val = np.asarray(X_val)
                y_val = np.asarray(y_val)
                model = self._create_model()
                model.fit(X_train, y_train)
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
                model.fit(X_train[train_idx], y_train[train_idx])
                scores.append(self._compute_metric(model, X_train[val_idx], y_train[val_idx], metric))
            return np.mean(scores)
    
    def _create_model(self):
        """Create a new model instance."""
        if self.task_type == 'classification':
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
            )
        else:
            return make_pipeline(
                StandardScaler(),
                LinearRegression(n_jobs=1)
            )
    
    def _compute_metric(self, model, X, y, metric):
        """Compute metric for a fitted model."""
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
        """Fit the model on all training data (for final model)."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.model = self._create_model()
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        X = np.asarray(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.asarray(X)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba

N_TRIALS = 2000
VAL_SIZE = 0.2
K_FOLDS = 5
N_WORKERS = -1
SHOW_PROGRESS = False
VERBOSE = False

def load_mitbih_data():
    data = pd.read_csv("../processed_datasets/mitbih/mitbih_processed.csv")
    return data.drop('target', axis=1), data['target']

def load_svd_data():
    df = pd.read_parquet("../processed_datasets/svd/svd.parquet")
    channels = [df[[f"{v}_{t}" for t in range(700)]].astype(np.float32) for v in ["a_n", "i_n", "u_n"]]
    return channels, df["target"].astype(int)

def load_remc_data(cell_line):
    df = pd.read_parquet(f"../processed_datasets/remc/{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']], df['target']

def load_emotions_data():
    df = pd.read_csv("../processed_datasets/emotions/emotions.csv", dtype=np.float32)
    y = df.pop('target').astype(int)
    cols_a = sorted([c for c in df.columns if c.endswith('_a')], key=lambda x: int(x.split('_')[1]))
    cols_b = sorted([c for c in df.columns if c.endswith('_b')], key=lambda x: int(x.split('_')[1]))
    return [df[cols_a], df[cols_b]], y

def load_mimic_data():
    df = pd.read_csv("../processed_datasets/mimic/mimic_processed.csv")
    y = df['ARDS_FLAG'].astype(int)
    feature_cols = [c for c in df.columns if c not in ['subject_id', 'anchor_age', 'ARDS_FLAG']]
    series_names = list(dict.fromkeys(c.split('_hour_')[0] for c in feature_cols if '_hour_' in c))
    return [df[sorted([c for c in feature_cols if c.startswith(f"{s}_hour_")], key=lambda x: int(x.split('_hour_')[1]))].astype(np.float32) for s in series_names], y


def load_pamap2_data(bin_size=10):
    df = pd.read_parquet("../processed_datasets/pamap2/pamap2.parquet")
    feature_cols = [c for c in df.columns if c not in ['time_stamp', 'activity_id', 'id']]
    xs, ys = [], []
    for subject_id in df['id'].unique():
        subject_df = df[df['id'] == subject_id].reset_index(drop=True)
        for activity_id in subject_df['activity_id'].unique():
            activity_df = subject_df[subject_df['activity_id'] == activity_id]
            for i in range(0, len(activity_df) - 100 + 1, 50):
                xs.append(activity_df.iloc[i:i+100][feature_cols].values)
                ys.append(activity_id)
    windows, y = np.asarray(xs), np.asarray(ys)
    X_list = []
    for j in range(windows.shape[2]):
        x = windows[:, :, j]
        n_bins = x.shape[1] // bin_size
        x = x[:, :n_bins * bin_size].reshape(x.shape[0], n_bins, bin_size).mean(axis=2) if n_bins else x
        X_list.append(pd.DataFrame(x.astype(np.float32)))
    u = np.unique(y)
    return X_list, pd.Series(y).map({v: i for i, v in enumerate(u)}).astype(int)

results_file = "../results/ablation_study.csv"
existing = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

variants = [
    ('trials_100', {'n_trials': 10, 'n_trials_without_improvement': None}),
    ('no_transforms', {'transforms': ['raw']}),
    ('control_points_1', {'n_control_points': 1}),
    ('linear_model', {'model_class': LinearRegressionWrapper}),
    ('no_sliding_window', {'sliding_window': False}),
    ('polynomial_pattern', {'pattern_fn': generate_polynomial_pattern, 'n_control_points': 4}),
]

# Main execution
datasets = [
    ('mitbih', load_mitbih_data, 'accuracy', None, StratifiedKFold),
    ('svd', load_svd_data, 'accuracy', None, StratifiedKFold),
    ('emotions', load_emotions_data, 'accuracy', None, StratifiedKFold),
    ('mimic', load_mimic_data, 'accuracy', None, StratifiedKFold),
]

# REMC
cell_lines = sorted([f.replace('.parquet', '') for f in os.listdir('../processed_datasets/remc') if f.endswith('.parquet')])
datasets.append((f'remc_{cell_lines[0]}', lambda: load_remc_data(cell_lines[0]), 'auc', None, StratifiedKFold))

for dataset_name, load_fn, metric, initial_features, cv_class in datasets:
    print(f"\nRunning {dataset_name}...")
    data = load_fn()
    X, y = data[0], data[1]
    X_concat = pd.concat(X, axis=1) if isinstance(X, list) else X
    folds = list(cv_class(K_FOLDS, shuffle=True, random_state=42).split(X_concat, y))
    
    for name, params in variants:
        if not existing.empty and len(existing[(existing['dataset'] == dataset_name) & (existing['approach'] == name)]) >= len(folds):
            continue
        
        print(f"  {name}")
        for fold, (tr, te) in enumerate(folds):
            X_tr = [x.iloc[tr].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[tr].astype(np.float32)]
            X_te = [x.iloc[te].astype(np.float32) for x in X] if isinstance(X, list) else [X.iloc[te].astype(np.float32)]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            
            init_feat = None
            if initial_features is not None:
                init_feat = (initial_features.iloc[tr].values, initial_features.iloc[te].values)
            
            run_kwargs = {**params, 'n_trials': N_TRIALS, 'n_workers': N_WORKERS, 'show_progress': SHOW_PROGRESS, 'verbose': VERBOSE}
            t0 = time.time()
            res = run_patx(X_tr, y_tr.values, X_te, metric=metric, initial_features=init_feat, y_test=y_te.values, **run_kwargs)
            t = time.time() - t0
            result = {'dataset': dataset_name, 'approach': name, 'fold': fold, 'score': res['score'], 'time': t, 'n_features': res['n_features']}
            existing = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
            existing.to_csv(results_file, index=False)
            print(f"    Fold {fold+1}: {metric.upper()}={res['score']:.4f}, Time={t:.1f}s, Features={res['n_features']}")