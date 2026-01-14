from __future__ import annotations
from typing import Callable, Dict, List
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import optuna
from scipy.interpolate import BSpline
import warnings

# Suppress Optuna experimental warnings
warnings.filterwarnings("ignore", message=".*multivariate.*is an experimental feature.*")
warnings.filterwarnings("ignore", message=".*group.*is an experimental feature.*")
warnings.filterwarnings("ignore", message=".*constant_liar.*is an experimental feature.*")

class TransformRegistry:
    """Small registry of time-series transforms."""

    def __init__(self) -> None:
        self._transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "raw": lambda d: d,
            "zscore": lambda d: (d - d.mean(axis=-1, keepdims=True)) / (d.std(axis=-1, keepdims=True) + 1e-6),
            "derivative": lambda d: np.gradient(d, axis=-1),
            "fft_power": self._fft_power,
        }

    def register(self, name: str, func: Callable) -> "TransformRegistry":
        self._transforms[name] = func
        return self

    def list_all(self) -> List[str]:
        return self._transforms.keys()

    def apply(self, data: np.ndarray, name: str) -> np.ndarray:
        return self._transforms[name](np.asarray(data, dtype=np.float32)).astype(np.float32)

    def _fft_power(self, data: np.ndarray) -> np.ndarray:
        # Return FFT power spectrum as-is (length = n_time // 2 + 1)
        return np.abs(np.fft.rfft(data, axis=-1)) ** 2

TRANSFORMS = TransformRegistry()


def generate_bspline_pattern(control_points: List[float], width: int) -> np.ndarray:
    """Generate B-spline pattern from control points."""
    cps = np.asarray(control_points, dtype=np.float32)
    n_cp = len(cps)
    degree = min(3, n_cp - 1)
    knots = np.concatenate([
        np.zeros(degree + 1),
        np.linspace(0, 1, n_cp - degree + 1)[1:-1],
        np.ones(degree + 1),
    ])
    t = np.linspace(0, 1, width)
    return BSpline(knots, cps, degree)(t).astype(np.float32)


def _extract_pattern_features(signal: np.ndarray, pattern: np.ndarray, start: int, end: int):
    """Compute features from the sliding window response between pattern and signal windows in [start, end]."""
    pat_len = len(pattern)
    # Slice the signal to only the required search range + pattern width
    # This keeps memory usage low while allowing full vectorization
    signal_subset = signal[:, start : end + pat_len]
    # windows shape: (n_samples, end - start + 1, pat_len)
    windows = sliding_window_view(signal_subset, pat_len, axis=1)
    # Vectorized MSE: (n_samples, num_windows)
    mse = np.mean((windows - pattern)**2, axis=-1)
    return np.min(mse, axis=1).astype(np.float32)


class PatternExtractor:
    """
    Greedily learn discriminative B-spline patterns one at a time.
    Stop when adding a new pattern no longer improves performance.
    """
    def __init__(
        self,
        model,
        metric: str = "auc",
        n_trials: int = 300,
        n_control_points: int = 3,
        show_progress: bool = True,
        n_workers: int = 1,
        transforms=None,
    ):
        self.model = model
        self.metric = metric
        self.n_trials = n_trials
        self.n_control_points = n_control_points
        self.show_progress = show_progress
        self.n_workers = n_workers
        self.transforms = transforms if transforms is not None else TRANSFORMS.list_all()
        self.patterns = []
        self.train_features = None
        self.test_features = None

    def fit(
        self,
        input_series_train,
        y_train,
        input_series_test=None,
        initial_features=None,
    ):
        """
        Extract patterns and generate features.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Prepare data
        if isinstance(input_series_train, (list, tuple)):
            X = np.stack([np.asarray(v, dtype=np.float32) for v in input_series_train], axis=1)
        else:
            X = np.asarray(input_series_train, dtype=np.float32)
            if X.ndim == 2:
                X = X[:, None, :]

        n_samples, n_channels, n_time = X.shape
        y = np.asarray(y_train)
        y = y.astype(np.float32) if self.metric == "rmse" else np.unique(y, return_inverse=True)[1].astype(np.int32)

        # Select transforms
        transform_types = self.transforms

        # Transform data - store in dict to avoid one massive 4D array
        transformed = {t: TRANSFORMS.apply(X, t) for t in transform_types}
        transformed_test = None
        if input_series_test is not None:
            if isinstance(input_series_test, (list, tuple)):
                X_test = np.stack([np.asarray(v, dtype=np.float32) for v in input_series_test], axis=1)
            else:
                X_test = np.asarray(input_series_test, dtype=np.float32)
                if X_test.ndim == 2:
                    X_test = X_test[:, None, :]
            transformed_test = {t: TRANSFORMS.apply(X_test, t) for t in transform_types}

        # Initialize features
        if initial_features is not None and initial_features[0] is not None:
            self.train_features = np.asarray(initial_features[0], dtype=np.float32)
            if self.train_features.ndim == 1:
                self.train_features = self.train_features[:, None]
        else:
            self.train_features = np.empty((n_samples, 0), dtype=np.float32)

        self.test_features = None
        if transformed_test is not None:
            if initial_features is not None and initial_features[1] is not None:
                self.test_features = np.asarray(initial_features[1], dtype=np.float32)
                if self.test_features.ndim == 1:
                    self.test_features = self.test_features[:, None]
            else:
                self.test_features = np.empty((X_test.shape[0], 0), dtype=np.float32)

        # Model for greedy search
        model_ = self.model.clone()
        direction = "minimize" if self.metric == "rmse" else "maximize"
        is_better = (lambda new, old: new < old) if self.metric == "rmse" else (lambda new, old: new > old)
        
        # Compute min signal length across all transforms (for fixed w bounds)
        min_sig_len = min(transformed[t].shape[-1] for t in transform_types)
        max_width = min(50, min_sig_len)

        # Greedy pattern search - one pattern at a time
        self.patterns = []
        best_score = float('inf') if self.metric == "rmse" else float('-inf')

        while True:
            def objective(trial):
                t_name = trial.suggest_categorical("t", transform_types)
                ch = trial.suggest_int("ch", 0, n_channels - 1) if n_channels > 1 else 0
                w = trial.suggest_int("w", 4, max_width)
                sig_len = transformed[t_name].shape[-1]
                a_frac, b_frac = trial.suggest_float("a", 0.0, 1.0), trial.suggest_float("b", 0.0, 1.0)
                max_start = sig_len - w
                start = int(min(a_frac, b_frac) * max_start)
                end = int(max(a_frac, b_frac) * max_start)
                cps = tuple(trial.suggest_float(f"c{i}", -1, 1) for i in range(self.n_control_points))
                pat = generate_bspline_pattern(list(cps), w)
                feat = _extract_pattern_features(transformed[t_name][:, ch, :], pat, start, end)
                Xf = np.hstack([self.train_features, feat[:, None]]) if self.train_features.size else feat[:, None]
                return model_.score(Xf, y, self.metric)

            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(
                    multivariate=True, 
                    group=True,
                    constant_liar=True,
                )
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.show_progress, n_jobs=self.n_workers)
            if not is_better(study.best_value, best_score):
                break

            best_score = study.best_value
            bp = study.best_trial.params
            t_name, ch, w = bp["t"], bp.get("ch", 0), bp["w"]
            a_frac, b_frac = bp["a"], bp["b"]
            sig_len = transformed[t_name].shape[-1]
            max_start = sig_len - w
            start = int(min(a_frac, b_frac) * max_start)
            end = int(max(a_frac, b_frac) * max_start)
            cps = [bp[f"c{i}"] for i in range(self.n_control_points)]
            pat = generate_bspline_pattern(cps, w)

            # Add features to train
            feat_train = _extract_pattern_features(transformed[t_name][:, ch, :], pat, start, end)
            self.train_features = np.hstack([self.train_features, feat_train[:, None]]) if self.train_features.size else feat_train[:, None]

            # Add features to test
            if transformed_test is not None:
                feat_test = _extract_pattern_features(transformed_test[t_name][:, ch, :], pat, start, end)
                self.test_features = np.hstack([self.test_features, feat_test[:, None]]) if self.test_features.size else feat_test[:, None]

            self.patterns.append({
                "pattern": pat, "control_points": cps, "width": w,
                "start_frac": min(a_frac, b_frac), "end_frac": max(a_frac, b_frac),
                "channel": ch, "transform": t_name
            })

        return self