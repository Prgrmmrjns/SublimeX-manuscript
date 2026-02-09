"""SublimeX: Sequential feature extraction via Bayesian optimization."""
import numpy as np
import optuna
import json
import os
import warnings
from sklearn.model_selection import train_test_split
from model import LightGBMModel

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings('ignore', message='overflow encountered in reduce')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
VAL_SIZE = 0.5
RANDOM_STATE = 42


def _get_segment(params, ctx):
    """Get segment from parameters (ch, t, c, r)."""
    ch, t = int(params['ch']), int(params['t'])
    c, r, n = params['c'], params['r'], ctx['n_time']
    half = (r * (n - 1)) * 0.5
    s = max(0, int(c * (n - 1) - half))
    e = min(n - 1, int(c * (n - 1) + half))
    return ctx['transformed'][t, :, ch, s:e+1]


def extract_feature(params, ctx):
    """Extract feature from saved parameters (mean-only)."""
    segment = _get_segment(params, ctx)
    return segment.mean(axis=1, keepdims=True).astype(np.float32)


def _get_train_val_split(y, metric):
    """Get consistent train/val split."""
    stratify = y if metric != 'rmse' else None
    return train_test_split(np.arange(len(y)), test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=stratify)


def _evaluate(feat, ctx):
    """Evaluate feature with LightGBM on train/val split."""
    X = np.hstack([ctx['current_X'], feat]) if ctx['current_X'].size else feat
    tr, va = _get_train_val_split(ctx['y'], ctx['metric'])
    return ctx['model'].evaluate(X[tr], ctx['y'][tr], X[va], ctx['y'][va], ctx['metric'])


def _suggest_segment(trial, ctx):
    """Suggest segment parameters and return segment."""
    params = {
        'ch': trial.suggest_int('ch', 0, ctx['n_channels'] - 1),
        't': trial.suggest_int('t', 0, len(ctx['transform_names']) - 1),
        'c': trial.suggest_float('c', 0, 1),
        'r': trial.suggest_float('r', 0, 1),
    }
    return _get_segment(params, ctx)


def mean_objective(trial, ctx):
    """Mean-only objective (default)."""
    return _evaluate(_suggest_segment(trial, ctx).mean(axis=1, keepdims=True).astype(np.float32), ctx)

TRANSFORMS = {
    'raw': lambda d: d,
    'zscore': lambda d: (d - d.mean(axis=-1, keepdims=True)) / (d.std(axis=-1, keepdims=True) + 1e-8),
    'derivative': lambda d: np.gradient(d, axis=-1),
    'fft': lambda d: np.abs(np.fft.fft(d, axis=-1)),
}

class SublimeX:
    def __init__(self, metric='auc', n_trials=300, verbose=False, show_progress_bar=False, transforms=None, objective_fn=None, sampler='tpe'):
        self.metric = metric
        self.n_trials = n_trials
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.transforms = transforms or TRANSFORMS
        self.objective_fn = objective_fn
        self.sampler = sampler
        self.extracted_features = []
        self.transform_names = list(self.transforms.keys())
        self.n_channels = None
        self.n_time = None

    def _apply_transforms(self, data):
        n_samples, n_channels, n_time = data.shape
        out = np.empty((len(self.transform_names), n_samples, n_channels, n_time), dtype=np.float32)
        for ti, tname in enumerate(self.transform_names):
            out[ti] = self.transforms[tname](data.reshape(-1, n_time)).reshape(n_samples, n_channels, n_time)
        return out

    def _to_array(self, input_series):
        arrays = [s.values.astype(np.float32) if hasattr(s, 'values') else np.asarray(s, dtype=np.float32) for s in input_series]
        return np.stack(arrays, axis=1).astype(np.float32)

    def fit(self, input_series, y, initial_X=None):
        """Fit the model.
        
        Args:
            input_series: List of DataFrames, one per channel
            y: Target labels
            initial_X: Optional (n_samples, n_initial) array prepended to extracted features
        """
        data = self._to_array(input_series)
        n_samples, self.n_channels, self.n_time = data.shape
        
        if self.verbose:
            print(f"\nFeature extraction: {n_samples} samples, {self.n_channels} channels, {self.n_time} time points")
        
        transformed = self._apply_transforms(data)
        model = LightGBMModel('regression' if self.metric == 'rmse' else 'classification')
        direction = 'minimize' if self.metric == 'rmse' else 'maximize'
        ctx = {
            'transformed': transformed, 'y': y, 'model': model, 'metric': self.metric,
            'n_channels': self.n_channels, 'n_time': self.n_time,
            'transform_names': self.transform_names
        }
        
        self.extracted_features = []
        self._initial_X = np.asarray(initial_X, dtype=np.float32) if initial_X is not None else None
        current_X = self._initial_X.copy() if self._initial_X is not None else np.empty((n_samples, 0), dtype=np.float32)
        best_score = float('inf') if direction == 'minimize' else -float('inf')
        is_maximize = direction == 'maximize'
        sampler = (optuna.samplers.NSGAIISampler() if self.sampler == 'nsga2' 
                   else optuna.samplers.TPESampler(multivariate=True, constant_liar=True))  
        while True:
            ctx['current_X'] = current_X
            study = optuna.create_study(direction=direction, sampler=sampler)
            study.optimize(lambda t: self.objective_fn(t, ctx), n_trials=self.n_trials, show_progress_bar=self.show_progress_bar, n_jobs=-1)
            
            improved = ((is_maximize and study.best_value > best_score) or 
                        (not is_maximize and study.best_value < best_score))
            if not improved:
                break
            
            best_score = study.best_value
            self.extracted_features.append(study.best_params)
            feat = extract_feature(study.best_params, ctx)
            current_X = np.hstack([current_X, feat]) if current_X.size else feat
            
            if self.verbose:
                print(f"Feature {len(self.extracted_features)}: {self.metric}={best_score:.5f}")
        return self

    def transform(self, input_series, initial_X=None):
        data = self._to_array(input_series)
        transformed = self._apply_transforms(data)
        ctx = {'transformed': transformed, 'n_time': self.n_time,
               'n_channels': self.n_channels, 'transform_names': self.transform_names}
        features = [extract_feature(p, ctx) for p in self.extracted_features]
        out = np.hstack(features).astype(np.float32)
        if initial_X is not None:
            out = np.hstack([np.asarray(initial_X, dtype=np.float32), out])
        elif getattr(self, '_initial_X', None) is not None:
            out = np.hstack([self._initial_X, out])
        return out

    def fit_transform(self, input_series, y):
        return self.fit(input_series, y).transform(input_series)

    def save_features(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump([{'feature_id': i + 1, **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in p.items()}} for i, p in enumerate(self.extracted_features)], f, indent=2)
