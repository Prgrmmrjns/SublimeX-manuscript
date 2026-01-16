from typing import Callable, Dict, List
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import optuna
from scipy.interpolate import BSpline
import warnings

# Suppress Optuna experimental warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="optuna.*")

def _fft_power(data: np.ndarray) -> np.ndarray:
    """Return FFT power spectrum as-is (length = n_time // 2 + 1)."""
    return np.abs(np.fft.rfft(data, axis=-1)) ** 2

TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw": lambda d: d,
    "fft_power": _fft_power,
}

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

def _default_pattern_mapping(signal: np.ndarray, pattern: np.ndarray, start: int, end: int, 
                             sliding_window: bool = True) -> np.ndarray:
    """
    Default pattern-to-feature mapping: min cosine distance in search range.
    Optimized version using efficient numpy operations.
    
    Parameters:
    -----------
    signal : np.ndarray
        Signal array of shape (n_samples, n_time)
    pattern : np.ndarray
        Pattern array of shape (pattern_length,)
    start : int
        Start position in signal
    end : int
        End position in signal
    sliding_window : bool, default=True
        If True, use sliding window; if False, use fixed position
        
    Returns:
    --------
    np.ndarray
        Feature array of shape (n_samples,) - cosine distance (1 - cosine_similarity)
    """
    pat_len = len(pattern)
    # Pre-compute pattern norm squared (avoid sqrt until final division)
    pat_norm_sq = np.dot(pattern, pattern)
    if pat_norm_sq < 1e-16:
        pat_norm = 1.0
    else:
        pat_norm = np.sqrt(pat_norm_sq)
    
    if not sliding_window:
        pos = (start + end) // 2
        window = signal[:, pos : pos + pat_len]
        # Use np.dot for faster computation
        dot_product = np.dot(window, pattern)
        # Use sum of squares instead of einsum
        window_norm_sq = np.sum(window * window, axis=1)
        window_norm = np.sqrt(window_norm_sq)
        window_norm = np.where(window_norm < 1e-8, 1.0, window_norm)
        cosine_sim = dot_product / (window_norm * pat_norm)
        distances = 1.0 - cosine_sim
        return distances.astype(np.float32)
    else:
        signal_subset = signal[:, start : end + pat_len]
        n_samples, subset_len = signal_subset.shape
        n_windows = subset_len - pat_len + 1
        
        if n_windows <= 0:
            # Edge case: no valid windows
            return np.full(n_samples, 1.0, dtype=np.float32)
        
        # Create sliding window view (creates a view, no copy)
        windows = sliding_window_view(signal_subset, pat_len, axis=1)
        # Shape: (n_samples, n_windows, pat_len)
        
        # Use tensordot for efficient dot product: (n_samples, n_windows, pat_len) @ (pat_len,) -> (n_samples, n_windows)
        dot_product = np.tensordot(windows, pattern, axes=([2], [0]))
        
        # Compute window norms using sum of squares (faster than einsum)
        window_norm_sq = np.sum(windows * windows, axis=2)
        window_norm = np.sqrt(window_norm_sq)
        window_norm = np.where(window_norm < 1e-8, 1.0, window_norm)
        
        # Cosine similarity and distance
        cosine_sim = dot_product / (window_norm * pat_norm)
        distances = 1.0 - cosine_sim
        return np.min(distances, axis=1).astype(np.float32)

def _early_stopping_callback(n_trials_without_improvement: int):
    """Create callback to stop Optuna optimization when there are no improvements for N trials."""
    best_value = [None]
    trials_since_improvement = [0]
    
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_value is None:
            return
        if best_value[0] is None or study.best_value != best_value[0]:
            best_value[0] = study.best_value
            trials_since_improvement[0] = 0
        else:
            trials_since_improvement[0] += 1
        if trials_since_improvement[0] >= n_trials_without_improvement:
            study.stop()
    return callback


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
        n_trials_without_improvement: int = None,
        n_control_points: int = 3,
        show_progress: bool = True,
        n_workers: int = 1,
        transforms=None,
        sliding_window: bool = True,
        pattern_fn=None,
        pattern_mapping_fn=None,
        transform_registry=None,
        verbose: bool = True,
    ):
        self.model = model
        self.metric = metric
        self.n_trials = n_trials
        self.n_trials_without_improvement = n_trials_without_improvement
        self.n_control_points = n_control_points
        self.show_progress = show_progress
        self.n_workers = n_workers
        self.verbose = verbose
        self.transform_registry = transform_registry if transform_registry is not None else TRANSFORMS
        self.transforms = transforms if transforms is not None else list(self.transform_registry.keys())
        self.pattern_fn = pattern_fn if pattern_fn is not None else generate_bspline_pattern
        
        # Pattern mapping function: converts (signal, pattern, start, end) -> features
        # If None, use default that uses sliding_window parameter
        if pattern_mapping_fn is not None:
            self.pattern_mapping_fn = pattern_mapping_fn
        else:
            # Create a bound version of default that captures sliding_window value
            def default_mapping(signal, pattern, start, end):
                return _default_pattern_mapping(signal, pattern, start, end, sliding_window)
            self.pattern_mapping_fn = default_mapping
        
        self.patterns = []
        self.train_features = None
        self.test_features = None
        self._transformed_test = None  # Store transformed test data for later feature extraction
    
    def _prepare_data(self, input_series):
        """Prepare and pad input data to consistent shape."""
        if isinstance(input_series, (list, tuple)):
            arrays = [np.asarray(v, dtype=np.float32) for v in input_series]
            max_time = max(arr.shape[1] if arr.ndim == 2 else arr.shape[0] for arr in arrays)
            padded = []
            for arr in arrays:
                if arr.ndim == 2:
                    n_time = arr.shape[1]
                    if n_time < max_time:
                        arr = np.hstack([arr, np.tile(arr[:, -1:], (1, max_time - n_time))])
                    padded.append(arr)
                else:
                    arr = arr.reshape(1, -1)
                    n_time = arr.shape[1]
                    if n_time < max_time:
                        arr = np.hstack([arr, np.tile(arr[:, -1:], (1, max_time - n_time))])
                    padded.append(arr)
            return np.stack(padded, axis=1)
        else:
            X = np.asarray(input_series, dtype=np.float32)
            return X[:, None, :] if X.ndim == 2 else X
    
    
    def extract_features(self, input_series, patterns=None, initial_features=None, transformed_data=None):
        """
        Extract features from input series using patterns.
        
        Parameters:
        -----------
        input_series : array-like or list of array-like, optional
            Input time series data. If None, transformed_data must be provided.
        patterns : list, optional
            List of pattern dictionaries. If None, uses self.patterns
        initial_features : array-like, optional
            Initial features to prepend to extracted features
        transformed_data : dict, optional
            Pre-transformed data dictionary. If provided, skips data preparation and transformation.
            
        Returns:
        --------
        np.ndarray
            Extracted features array
        """
        
        # Use pre-transformed data if provided, otherwise prepare and transform
        if transformed_data is not None:
            transformed = transformed_data
            n_samples = next(iter(transformed.values())).shape[0]
        else:
            X = self._prepare_data(input_series)
            transformed = {t: self.transform_registry[t](np.asarray(X, dtype=np.float32)).astype(np.float32) for t in self.transforms}
            n_samples = X.shape[0]
        
        # Initialize features
        if initial_features is not None:
            features = np.asarray(initial_features, dtype=np.float32)
            if features.ndim == 1:
                features = features[:, None]
        else:
            features = np.empty((n_samples, 0), dtype=np.float32)
        
        # Extract features using patterns
        for pattern_info in patterns:
            t_name = pattern_info['transform']
            ch = pattern_info['channel']
            sig_len = transformed[t_name].shape[-1]
            max_start = sig_len - pattern_info['width']
            start = int(min(pattern_info['start_frac'], pattern_info['end_frac']) * max_start)
            end = int(max(pattern_info['start_frac'], pattern_info['end_frac']) * max_start)
            pat = self.pattern_fn(list(pattern_info['control_points']), pattern_info['width'])
            feat = self.pattern_mapping_fn(transformed[t_name][:, ch, :], pat, start, end)
            features = np.hstack([features, feat[:, None]])
        return features

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
        X = self._prepare_data(input_series_train)
        n_samples, n_channels, n_time = X.shape
        y = np.asarray(y_train)
        y = y.astype(np.float32) if self.metric == "rmse" else np.unique(y, return_inverse=True)[1].astype(np.int32)

        # Print verbose configuration information
        if self.verbose:
            print(f"Dataset shape: {n_samples} samples, {n_channels} channels, {n_time} time points")
            print(f"Optuna parameters: n_trials={self.n_trials}, n_trials_without_improvement={self.n_trials_without_improvement}")
            print(f"PATX config: metric={self.metric}, n_control_points={self.n_control_points}, transforms={self.transforms}")

        # Transform data - store in dict to avoid one massive 4D array
        transform_types = self.transforms
        transformed = {t: self.transform_registry[t](np.asarray(X, dtype=np.float32)).astype(np.float32) for t in transform_types}
        
        # Store test data for later feature extraction (if provided)
        if input_series_test is not None:
            X_test = self._prepare_data(input_series_test)
            self._transformed_test = {t: self.transform_registry[t](np.asarray(X_test, dtype=np.float32)).astype(np.float32) for t in transform_types}
            if self.verbose:
                print(f"Test set: {X_test.shape[0]} samples")
        else:
            self._transformed_test = None

        # Initialize train features
        init_train = initial_features[0] if initial_features and initial_features[0] is not None else None
        if init_train is not None:
            self.train_features = np.asarray(init_train, dtype=np.float32)
            if self.train_features.ndim == 1:
                self.train_features = self.train_features[:, None]
            if self.verbose:
                print(f"Initial features: {self.train_features.shape[1]} features")
        else:
            self.train_features = np.empty((n_samples, 0), dtype=np.float32)

        # Don't pass test set during pattern search
        direction = "minimize" if self.metric == "rmse" else "maximize"
        is_better = (lambda new, old: new < old) if self.metric == "rmse" else (lambda new, old: new > old)
        
        # Compute min signal length across all transforms (for width bounds)
        min_sig_len = min(transformed[t].shape[-1] for t in transform_types)
        min_width = self.n_control_points  # B-spline needs at least n_control_points

        # Greedy pattern search - one pattern at a time
        self.patterns = []
        best_score = float('inf') if self.metric == "rmse" else float('-inf')

        while True:
            def objective(trial):
                t_name = trial.suggest_categorical("t", transform_types)
                ch = trial.suggest_int("ch", 0, n_channels - 1) if n_channels > 1 else 0
                w = trial.suggest_int("w", min_width, min_sig_len)
                sig_len = transformed[t_name].shape[-1]
                a_frac, b_frac = trial.suggest_float("a", 0.0, 1.0), trial.suggest_float("b", 0.0, 1.0)
                max_start = sig_len - w
                start = int(min(a_frac, b_frac) * max_start)
                end = int(max(a_frac, b_frac) * max_start)
                cps = tuple(trial.suggest_float(f"c{i}", -1, 1) for i in range(self.n_control_points))
                pat = self.pattern_fn(list(cps), w)
                feat = self.pattern_mapping_fn(transformed[t_name][:, ch, :], pat, start, end)
                Xf = np.hstack([self.train_features, feat[:, None]]) if self.train_features.size else feat[:, None]
                
                return self.model.evaluate(Xf, y, self.metric)
            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(consider_prior=False, multivariate=True, constant_liar=True)
            )
            callbacks = []
            if self.n_trials_without_improvement is not None:
                callbacks.append(_early_stopping_callback(self.n_trials_without_improvement))
            study.optimize(
                objective, 
                n_trials=self.n_trials, 
                show_progress_bar=self.show_progress, 
                n_jobs=self.n_workers,
                callbacks=callbacks if callbacks else None
            )
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
            cps = tuple(bp[f"c{i}"] for i in range(self.n_control_points))
            if self.verbose: print(f"Pattern {len(self.patterns) + 1} added: score={best_score:.6f}, ch={ch}, transform={t_name}, cps={cps}, w={w}, start_frac={min(a_frac, b_frac):.4f}, end_frac={max(a_frac, b_frac):.4f}")

            # Add feature
            pat = self.pattern_fn(list(cps), w)
            feat_train = self.pattern_mapping_fn(transformed[t_name][:, ch, :], pat, start, end)
            self.train_features = np.hstack([self.train_features, feat_train[:, None]]) if self.train_features.size else feat_train[:, None]
            self.patterns.append({"pattern": pat, "control_points": cps, "width": w, "start_frac": min(a_frac, b_frac), "end_frac": max(a_frac, b_frac), "channel": ch, "transform": t_name})
        
        # Extract all test features at the end (if test data was provided)
        if self._transformed_test is not None:
            init_test = initial_features[1] if initial_features and initial_features[1] is not None else None
            self.test_features = self.extract_features(
                None, 
                patterns=self.patterns, 
                initial_features=init_test,
                transformed_data=self._transformed_test
            )
        else:
            self.test_features = None
        return self