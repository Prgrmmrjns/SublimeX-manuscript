import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from scipy.interpolate import BSpline
from scipy.stats import skew, kurtosis
from scipy.fft import dct
import warnings
from models import LightGBMModelWrapper
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pywt

all_transforms = [
    'raw', 'derivative', 'second_deriv', 'cumsum', 'diff', 'log1p', 'abs', 
    'sorted', 'dct', 'wavelet_db4', 'wavelet_sym4', 'wavelet_coif1', 'wavelet_haar',
    'fft_power', 'exp', 'tanh', 'sin', 'cos', 'reciprocal', 'autocorr'
]

def generate_bspline_pattern(control_points, width):
    n_cp, degree = len(control_points), min(3, len(control_points) - 1)
    knots = np.concatenate([np.zeros(degree + 1), np.linspace(0, 1, n_cp - degree + 1)[1:-1], np.ones(degree + 1)])
    width_int = int(round(width))
    return BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width_int))

_transform_funcs = {
    'derivative': lambda d: np.gradient(d, axis=-1),
    'second_deriv': lambda d: np.gradient(np.gradient(d, axis=-1), axis=-1),
    'cumsum': lambda d: np.cumsum(d, axis=-1),
    'diff': lambda d: np.diff(d, axis=-1, prepend=d[..., :1]),
    'log1p': lambda d: np.log1p(np.abs(d)),
    'abs': np.abs,
    'sorted': lambda d: np.sort(d, axis=-1),
    'dct': lambda d: dct(d, axis=-1, type=2, norm='ortho'),
    'exp': lambda d: np.exp(np.clip(d, -10, 10)),
    'tanh': np.tanh,
    'sin': np.sin,
    'cos': np.cos,
    'reciprocal': lambda d: 1.0 / (np.abs(d) + 1e-8),
}

def apply_transformation(data, transform_type):
    data = np.ascontiguousarray(data, dtype=np.float32)
    if transform_type == 'raw':
        return data
    if transform_type in _transform_funcs:
        return _transform_funcs[transform_type](data).astype(np.float32)
    if transform_type.startswith('wavelet_'):
        return wavelet_transform(data, transform_type[8:]).astype(np.float32)
    if transform_type == 'fft_power':
        return fft_power(data).astype(np.float32)
    if transform_type == 'autocorr':
        return autocorr_transform(data).astype(np.float32)
    return data

def wavelet_transform(data, wavelet):
    n_samples, n_series, n_time = data.shape
    level = min(3, int(np.log2(n_time)) - 1)
    flat = data.reshape(-1, n_time)
    result_flat = np.empty_like(flat)
    x_out = np.linspace(0, 1, n_time)
    for i in range(flat.shape[0]):
        coeffs = pywt.wavedec(flat[i], wavelet, level=level, mode='periodization')
        cat = np.concatenate(coeffs)
        result_flat[i] = np.interp(x_out, np.linspace(0, 1, len(cat)), cat)
    return result_flat.reshape(n_samples, n_series, n_time)

def fft_power(data):
    n_time = data.shape[-1]
    power = np.abs(np.fft.rfft(data, axis=-1)) ** 2
    n_freq = power.shape[-1]
    x_in, x_out = np.linspace(0, 1, n_freq), np.linspace(0, 1, n_time)
    flat_power = power.reshape(-1, n_freq)
    result = np.empty((flat_power.shape[0], n_time), dtype=np.float32)
    for i in range(flat_power.shape[0]):
        result[i] = np.interp(x_out, x_in, flat_power[i])
    return result.reshape(data.shape)

def autocorr_transform(data):
    n_samples, n_series, n_time = data.shape
    flat = data.reshape(-1, n_time)
    centered = flat - flat.mean(axis=-1, keepdims=True)
    fft_len = 2 * n_time - 1
    f = np.fft.rfft(centered, n=fft_len, axis=-1)
    acf_full = np.fft.irfft(f * np.conj(f), n=fft_len, axis=-1)
    acf = acf_full[:, :n_time]
    acf = acf / (acf[:, :1] + 1e-8)
    return acf.reshape(n_samples, n_series, n_time)

def compute_aggregate_stats(transformed_data):
    n_time = transformed_data.shape[2]
    mid = max(1, n_time // 2)
    mins = np.min(transformed_data, axis=2)
    maxs = np.max(transformed_data, axis=2)
    means = np.mean(transformed_data, axis=2)
    stds = np.std(transformed_data, axis=2)
    
    qs = np.percentile(transformed_data, [25, 75], axis=2)
    q25, q75 = qs[0], qs[1]
    
    mean1 = np.mean(transformed_data[:, :, :mid], axis=2)
    mean2 = np.mean(transformed_data[:, :, mid:], axis=2)
    std1 = np.std(transformed_data[:, :, :mid], axis=2)
    std2 = np.std(transformed_data[:, :, mid:], axis=2)

    stats = np.stack([
        means,
        np.median(transformed_data, axis=2),
        stds,
        mins, maxs,
        skew(transformed_data, axis=2, nan_policy='omit'),
        kurtosis(transformed_data, axis=2, nan_policy='omit'),
        maxs - mins,
        q25, 
        q75, 
        q75 - q25,
        mean1, 
        mean2,
        std1, 
        std2,
        mean2 - mean1,
    ], axis=2)
    return stats.reshape(transformed_data.shape[0], -1)

def _eval_transform(t, data, y, model_type, n_classes, metric):
    stats = np.nan_to_num(compute_aggregate_stats(apply_transformation(data, t)), nan=0, posinf=0, neginf=0)
    model = LightGBMModelWrapper(model_type, n_classes=n_classes, num_threads=1)
    score = model.run_cv(stats, y, 3, metric)
    return (t, score)

def select_transforms(data, y, metric, n_transforms=5):
    model_type = 'regression' if metric == 'rmse' else 'classification'
    n_classes = len(np.unique(y)) if model_type == 'classification' else 2
    worker = partial(_eval_transform, data=data, y=y, model_type=model_type, n_classes=n_classes, metric=metric)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(worker, all_transforms))
    results.sort(key=lambda x: x[1], reverse=(metric != 'rmse'))
    return [t for t, score in results[:n_transforms]]

def pattern_to_features(series_data, pattern, pattern_width, pattern_start):
    pattern_width_int = int(round(pattern_width))
    segment = series_data[:, pattern_start:pattern_start + pattern_width_int]
    return np.sqrt(np.mean((segment - pattern) ** 2, axis=1))

def batch_pattern_features(transformed_stack, params_list):
    n_samples = transformed_stack.shape[1]
    n_patterns = len(params_list)
    n_time = transformed_stack.shape[3]
    feats = np.empty((n_samples, n_patterns), dtype=np.float32)
    
    for i, (t_idx, s_idx, cps, center, width) in enumerate(params_list):
        w = int(round(width))
        start = min(max(0, int(center - w // 2)), n_time - w)
        pattern = generate_bspline_pattern(cps, width)
        segment = transformed_stack[t_idx, :, s_idx, start:start+w]
        feats[:, i] = np.sqrt(np.mean((segment - pattern) ** 2, axis=1))
    return feats

class EarlyStoppingCallback:
    def __init__(self, patience, direction):
        self.patience, self.direction = patience, direction
        self.best_value, self.best_trial = None, 0
    def __call__(self, study, trial):
        val = trial.value
        if val is None:
            return
        is_better = self.best_value is None or (val < self.best_value if self.direction == 'minimize' else val > self.best_value)
        if is_better:
            self.best_value, self.best_trial = val, trial.number
        elif trial.number - self.best_trial >= self.patience:
            study.stop()

def feature_extraction(input_series_train, y_train, input_series_test=None, initial_features=None, model=None, metric='auc', val_size=0.2, n_trials=300, n_control_points=3, n_patterns=15, n_transforms=5, max_samples=2000, inner_k_folds=3, early_stopping_patience=1000, show_progress=True, n_workers=1, backward_elimination=True):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if isinstance(input_series_train, list):
        input_series_train = np.stack([x.values if hasattr(x, 'values') else x for x in input_series_train], axis=1)
    if isinstance(input_series_test, list):
        input_series_test = np.stack([x.values if hasattr(x, 'values') else x for x in input_series_test], axis=1)
    n_input_series, n_time_points = input_series_train.shape[1], input_series_train.shape[2]
    n_samples = input_series_train.shape[0]
    print(f"\nFeature extraction: {n_samples} samples, {n_input_series} channels, {n_time_points} time points")
    y_train = np.asarray(y_train).flatten()
    if metric != 'rmse':
        unique_targets = np.unique(y_train)
        if len(unique_targets) > 2 and not np.array_equal(unique_targets, np.arange(len(unique_targets))):
            label_map = {v: i for i, v in enumerate(unique_targets)}
            y_train = np.array([label_map[y] for y in y_train])
        elif len(unique_targets) == 2 and not np.array_equal(unique_targets, [0, 1]):
            y_train = (y_train == unique_targets[1]).astype(int)
            
    transform_types = select_transforms(input_series_train, y_train, metric, n_transforms)
    print(f"Selected {len(transform_types)} transforms: {transform_types}")
    
    transformed_stack = np.zeros((len(transform_types), n_samples, n_input_series, n_time_points), dtype=np.float32)
    for i, t in enumerate(transform_types):
         transformed_stack[i] = apply_transformation(input_series_train, t)
    
    base_features = initial_features[0] if initial_features else np.empty((n_samples, 0))
    model_type = 'regression' if metric == 'rmse' else 'classification'
    n_classes = len(np.unique(y_train)) if model_type == 'classification' and len(np.unique(y_train)) > 2 else 2
    model = model or LightGBMModelWrapper(model_type, n_classes=n_classes, num_threads=1)
    train_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=val_size, random_state=42)
    max_width = min(50.0, n_time_points)
    direction = 'minimize' if metric == 'rmse' else 'maximize'
    n_search_samples = min(n_samples, max_samples)
    if n_samples > n_search_samples:
        search_indices = np.random.choice(n_samples, n_search_samples, replace=False)
        search_stack = transformed_stack[:, search_indices, :, :]
        search_y = y_train[search_indices]
        search_base_X = base_features[search_indices] if base_features.size else np.empty((n_search_samples, 0))
    else:
        search_stack = transformed_stack
        search_y = y_train
        search_base_X = base_features
    
    def objective(trial):
        params_list = []
        for p in range(n_patterns):
            s_idx = trial.suggest_int(f'p{p}_s', 0, n_input_series - 1) if n_input_series > 1 else 0
            t_idx = trial.suggest_int(f'p{p}_t', 0, len(transform_types) - 1)
            cps = tuple(trial.suggest_float(f'p{p}_c{i}', 0, 1) for i in range(n_control_points))
            center = trial.suggest_int(f'p{p}_pos', 0, n_time_points - 1)
            width = trial.suggest_float(f'p{p}_w', 2.0, max_width)
            params_list.append((t_idx, s_idx, cps, center, width))
        feats = batch_pattern_features(search_stack, params_list)
        X = np.hstack([search_base_X, feats]) if search_base_X.size else feats
        return model.run_cv(X, search_y, inner_k_folds, metric)
    
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.NSGAIISampler())
    early_stop = EarlyStoppingCallback(early_stopping_patience, direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress, n_jobs=n_workers, callbacks=[early_stop])
    params, score = study.best_trial.params, study.best_trial.value
    print(f"Best {metric}={score:.4f} (stopped at trial {len(study.trials)})")
    
    extracted_patterns = []
    params_list = []
    for p in range(n_patterns):
        s_idx = params.get(f'p{p}_s', 0)
        t_idx = params[f'p{p}_t']
        cps = tuple(params[f'p{p}_c{i}'] for i in range(n_control_points))
        center, width = params[f'p{p}_pos'], params[f'p{p}_w']
        w = int(round(width))
        start = min(max(0, int(center - w // 2)), n_time_points - w)
        extracted_patterns.append({'pattern': generate_bspline_pattern(cps, width), 'start': start, 'width': width, 'center': center, 'series_idx': s_idx, 'control_points': list(cps), 'transform_type': transform_types[t_idx]})
        params_list.append((t_idx, s_idx, cps, center, width))
    
    pattern_feats = batch_pattern_features(transformed_stack, params_list)
    selected_indices = list(range(n_patterns))
    
    if backward_elimination:
        current_best_score = score
        while True:
            worst_drop_score = float('inf') if metric == 'rmse' else -float('inf')
            worst_idx = -1
            for i in selected_indices:
                trial_indices = [idx for idx in selected_indices if idx != i]
                feats_subset = pattern_feats[:, trial_indices]
                X = np.hstack([base_features, feats_subset]) if base_features.size else feats_subset
                s = model.run_cv(X, y_train, inner_k_folds, metric)
                is_better_drop = (s < worst_drop_score) if metric == 'rmse' else (s > worst_drop_score)
                if is_better_drop:
                    worst_drop_score, worst_idx = s, i
            tolerance = 0.001
            is_acceptable = (worst_drop_score <= current_best_score + tolerance) if metric == 'rmse' else (worst_drop_score >= current_best_score - tolerance)
            if is_acceptable and worst_idx != -1:
                selected_indices.remove(worst_idx)
                current_best_score = worst_drop_score
            else:
                break
        print(f"Reduced patterns from {n_patterns} to {len(selected_indices)}")
    final_patterns = [extracted_patterns[i] for i in selected_indices]
    final_params_list = [params_list[i] for i in selected_indices]
    
    model_features = np.hstack([base_features, batch_pattern_features(transformed_stack, final_params_list)]) if base_features.size else batch_pattern_features(transformed_stack, final_params_list)
    model.fit(model_features[train_idx], y_train[train_idx], model_features[val_idx], y_train[val_idx])
    test_features = None
    if input_series_test is not None:
        transformed_stack_test = np.stack([apply_transformation(input_series_test, t) for t in transform_types])
        test_feats = batch_pattern_features(transformed_stack_test, final_params_list)
        test_features = np.hstack([initial_features[1], test_feats]) if initial_features else test_feats
    return {'patterns': final_patterns, 'train_features': model_features, 'test_features': test_features, 'model': model}
