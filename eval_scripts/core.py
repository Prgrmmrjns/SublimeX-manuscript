import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from scipy.interpolate import BSpline
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import warnings
from models import LightGBMModelWrapper
from numba import njit
from scipy import fft
from functools import lru_cache
import pywt


@lru_cache(maxsize=1000)
def generate_bspline_pattern(control_points, width, data_min, data_max):
    n_cp, degree = len(control_points), min(3, len(control_points) - 1)
    knots = np.concatenate([np.zeros(degree + 1), np.linspace(0, 1, n_cp - degree + 1)[1:-1], np.ones(degree + 1)])
    width_int = int(round(width))
    return data_min + BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width_int)) * (data_max - data_min)

def apply_transformation(data, transform_type):
    if transform_type == 'derivative': result = np.gradient(data, axis=-1)
    elif transform_type == 'cumsum': result = np.cumsum(data, axis=-1)
    elif transform_type == 'fft_power': result = fft_power_transform(data)
    elif transform_type == 'wavelet': result = wavelet_transform(data)
    elif transform_type == 'spectral_entropy': result = spectral_entropy_transform(data)
    elif transform_type == 'phase_amplitude_coupling': result = phase_amplitude_coupling_transform(data)
    else: result = data
    return result

def fft_power_transform(data):
    result = np.zeros_like(data)
    n_samples, n_series, n_time = data.shape
    for s in range(n_samples):
        for ser in range(n_series):
            fft_coeffs = fft.fft(data[s, ser, :])
            power_spectrum = np.abs(fft_coeffs) ** 2
            ifft_result = np.real(fft.ifft(power_spectrum))
            result[s, ser, :] = np.interp(np.linspace(0, 1, n_time), np.linspace(0, 1, len(ifft_result)), ifft_result)
    return result

def wavelet_transform(data):
    result = np.zeros_like(data)
    n_samples, n_series, n_time = data.shape
    for s in range(n_samples):
        for ser in range(n_series):
            series = data[s, ser, :]
            coeffs = pywt.wavedec(series, 'db4', level=3, mode='periodization')
            concatenated = np.concatenate(coeffs)
            result[s, ser, :] = np.interp(np.linspace(0, 1, n_time), np.linspace(0, 1, len(concatenated)), concatenated) if len(concatenated) != n_time else concatenated
    return result

def spectral_entropy_transform(data):
    result = np.zeros_like(data)
    n_samples, n_series, n_time = data.shape
    for s in range(n_samples):
        for ser in range(n_series):
            series = data[s, ser, :]
            psd = np.abs(fft.fft(series)) ** 2
            psd_sum = np.sum(psd)
            if psd_sum > 0:
                psd_norm = psd / psd_sum
                psd_norm = np.maximum(psd_norm, 1e-12)
                entropy = -np.sum(psd_norm * np.log2(psd_norm))
            else:
                entropy = 0.0
            result[s, ser, :] = np.full(n_time, entropy)
    return result

def phase_amplitude_coupling_transform(data):
    result = np.zeros_like(data)
    n_samples, n_series, n_time = data.shape
    for s in range(n_samples):
        for ser in range(n_series):
            series = data[s, ser, :]
            analytic_signal = fft.fft(series)
            low_freq = np.copy(analytic_signal)
            low_freq[n_time//8:] = 0
            low_freq[:n_time//8] = 0
            phase = np.angle(fft.ifft(low_freq))
            high_freq = np.copy(analytic_signal)
            high_freq[:n_time//4] = 0
            high_freq[n_time//2:] = 0
            amplitude = np.abs(fft.ifft(high_freq))
            pac = np.abs(np.mean(amplitude * np.exp(1j * phase)))
            result[s, ser, :] = np.full(n_time, pac)
    return result

@njit(fastmath=True)
def calculate_distances(series_data, pattern, pattern_width, pattern_start, use_relative=False):
    pattern_width_int = int(pattern_width)
    if pattern_start < 0 or pattern_start + pattern_width_int > series_data.shape[1]:
        return np.full(series_data.shape[0], np.inf)
    n_samples = series_data.shape[0]
    distances = np.empty(n_samples)
    for i in range(n_samples):
        series_seg = series_data[i, pattern_start:pattern_start + pattern_width_int]
        if use_relative:
            series_min, series_max = series_seg.min(), series_seg.max()
            pattern_min, pattern_max = pattern.min(), pattern.max()
            series_range = series_max - series_min
            pattern_range = pattern_max - pattern_min
            sum_sq_diff = 0.0
            for j in range(pattern_width_int):
                norm_series = (series_seg[j] - series_min) / series_range if series_range > 0 else 0.5
                norm_pattern = (pattern[j] - pattern_min) / pattern_range if pattern_range > 0 else 0.5
                diff = norm_series - norm_pattern
                sum_sq_diff += diff * diff
            distances[i] = np.sqrt(sum_sq_diff / pattern_width)
        else:
            sum_sq_diff = 0.0
            for j in range(pattern_width_int):
                diff = series_data[i, pattern_start + j] - pattern[j]
                sum_sq_diff += diff * diff
            distances[i] = np.sqrt(sum_sq_diff / pattern_width)
    return distances

def pattern_to_features(input_series, pattern_width, pattern_start, series_index=0, pattern=None, data_min=0.0, data_max=1.0, use_relative=False, shift_tolerance=0.0, max_shift_evaluations=10):
    n_time_points = input_series.shape[2]
    pattern_width_int = int(round(pattern_width))
    
    # Normalize pattern if using absolute matching
    if not use_relative and pattern is not None:
        # Scale pattern to match the data range
        pattern_range = pattern.max() - pattern.min()
        if pattern_range > 0:
            pattern = data_min + (pattern - pattern.min()) / pattern_range * (data_max - data_min)
    
    if series_index == -1:
        best = np.full(input_series.shape[0], np.inf)
        for j in range(input_series.shape[1]):
            series_best = np.full(input_series.shape[0], np.inf)
            if shift_tolerance == 0:
                shifted_start = max(0, pattern_start)
                distances = calculate_distances(input_series[:, j, :], pattern, pattern_width, shifted_start, use_relative)
                series_best = np.minimum(series_best, distances)
            else:
                max_shift = min(int(shift_tolerance * n_time_points), n_time_points - pattern_width_int)
                n_evaluations = min(max_shift_evaluations, 2 * max_shift + 1)
                stride = max(1, (2 * max_shift) // (n_evaluations - 1)) if n_evaluations > 1 else 1
                for i in range(n_evaluations):
                    shift = -max_shift + i * stride
                    shifted_start = max(0, pattern_start + shift)
                    if shifted_start + pattern_width_int <= n_time_points:
                        distances = calculate_distances(input_series[:, j, :], pattern, pattern_width, shifted_start, use_relative)
                        series_best = np.minimum(series_best, distances)
            best = np.minimum(best, series_best)
        return best
    
    best = np.full(input_series.shape[0], np.inf)
    if shift_tolerance == 0:
        shifted_start = max(0, pattern_start)
        distances = calculate_distances(input_series[:, series_index, :], pattern, pattern_width, shifted_start, use_relative)
        best = np.minimum(best, distances)
    else:
        max_shift = min(int(shift_tolerance * n_time_points), n_time_points - pattern_width_int)
        n_evaluations = min(max_shift_evaluations, 2 * max_shift + 1)
        stride = max(1, (2 * max_shift) // (n_evaluations - 1)) if n_evaluations > 1 else 1
        for i in range(n_evaluations):
            shift = -max_shift + i * stride
            shifted_start = max(0, pattern_start + shift)
            if shifted_start + pattern_width_int <= n_time_points:
                distances = calculate_distances(input_series[:, series_index, :], pattern, pattern_width, shifted_start, use_relative)
                best = np.minimum(best, distances)
    return best

def evaluate_model_performance(model, metric, cached_data):
    X_train, X_val, y_train_split, y_val = cached_data
    model = model.clone()
    model.fit(X_train, y_train_split, X_val, y_val)
    if metric == 'accuracy': return accuracy_score(y_val, model.predict(X_val))
    if metric == 'rmse': return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    y_pred = model.predict_proba(X_val)
    return roc_auc_score(y_val, y_pred) if len(np.unique(y_val)) == 2 else roc_auc_score(y_val, y_pred, multi_class='ovr', average='macro')

def feature_extraction(input_series_train, y_train, input_series_test=None, initial_features=None, model=None, metric='auc', val_size=0.2, n_trials=300, n_control_points=3, show_progress=True, max_shift_evaluations=10):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if isinstance(input_series_train, list):
        input_series_train = np.stack([x.values if hasattr(x, 'values') else x for x in input_series_train], axis=1)
    if isinstance(input_series_test, list):
        input_series_test = np.stack([x.values if hasattr(x, 'values') else x for x in input_series_test], axis=1)
    n_input_series, n_time_points = input_series_train.shape[1], input_series_train.shape[2]
    data_min, data_max = input_series_train.min(), input_series_train.max()
    transform_types = ['raw', 'derivative', 'cumsum', 'fft_power', 'wavelet', 'spectral_entropy', 'phase_amplitude_coupling']
    transformed_train = {t: apply_transformation(input_series_train, t) for t in transform_types}
    model_features_list = [initial_features[0]] if initial_features else []
    y_train = np.asarray(y_train).flatten()
    if metric != 'rmse':
        unique_targets = np.unique(y_train)
        if len(unique_targets) > 2 and not np.array_equal(unique_targets, np.arange(len(unique_targets))):
            y_train = np.array([{v: i for i, v in enumerate(unique_targets)}[y] for y in y_train])
        elif len(unique_targets) == 2 and not np.array_equal(unique_targets, [0, 1]):
            y_train = (y_train == unique_targets[1]).astype(int)
    model_type = 'regression' if metric == 'rmse' else 'classification'
    n_classes = len(np.unique(y_train)) if model_type == 'classification' and len(np.unique(y_train)) > 2 else 2
    model = model or LightGBMModelWrapper(model_type, n_classes=n_classes)
    train_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=val_size, random_state=42)
    y_train_split, y_val = y_train[train_idx], y_train[val_idx]
    transformed_train_split = {t: transformed_train[t][train_idx] for t in transform_types}
    transformed_val = {t: transformed_train[t][val_idx] for t in transform_types}
    def objective(trial):
        series_idx = trial.suggest_int('series_index', 0, n_input_series - 1) if n_input_series > 1 else 0
        transform_type = trial.suggest_categorical('transform_type', transform_types)
        use_relative = trial.suggest_categorical('use_relative', [False, True])
        shift_tolerance = trial.suggest_float('shift_tolerance', 0.0, 1.0)
        cps = [trial.suggest_float(f'cp{i}', 0, 1) for i in range(n_control_points)]
        pattern_center = trial.suggest_int('pattern_center', 0, n_time_points - 1)
        pattern_width = trial.suggest_float('pattern_width', 2.0, min(50.0, n_time_points))
        start = max(0, int(pattern_center - pattern_width // 2))
        pattern = generate_bspline_pattern(tuple(cps), pattern_width, data_min, data_max)
        train_feat = pattern_to_features(transformed_train_split[transform_type], pattern_width, start, series_idx, pattern=pattern, data_min=data_min, data_max=data_max, use_relative=use_relative, shift_tolerance=shift_tolerance)
        val_feat = pattern_to_features(transformed_val[transform_type], pattern_width, start, series_idx, pattern=pattern, data_min=data_min, data_max=data_max, use_relative=use_relative, shift_tolerance=shift_tolerance)
        X_train = np.column_stack([base_X_train, train_feat]) if base_X_train.size > 0 else train_feat.reshape(-1, 1)
        X_val = np.column_stack([base_X_val, val_feat]) if base_X_val.size > 0 else val_feat.reshape(-1, 1)
        return evaluate_model_performance(model, metric, (X_train, X_val, y_train_split, y_val))
    extracted_patterns = []
    best_score = float('inf') if metric == 'rmse' else -float('inf')
    while True:
        base_X_train = np.column_stack([f[train_idx] for f in model_features_list]) if model_features_list else np.empty((len(train_idx), 0))
        base_X_val = np.column_stack([f[val_idx] for f in model_features_list]) if model_features_list else np.empty((len(val_idx), 0))
        warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
        study = optuna.create_study(direction='minimize' if metric == 'rmse' else 'maximize', sampler=optuna.samplers.TPESampler(multivariate=True, group=True))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress, n_jobs=-1)
        score = study.best_trial.value
        improved = (score < best_score) if metric == 'rmse' else (score > best_score)
        if improved:
            best_score = score
        else:
            break
        params = study.best_trial.params
        transform_type = params.get('transform_type', 'raw')
        use_relative = params.get('use_relative', False)
        shift_tolerance = params.get('shift_tolerance', 0)
        cps = [params[f'cp{i}'] for i in range(n_control_points)]
        series_idx, pattern_center, pattern_width = params.get('series_index', 0), params['pattern_center'], params['pattern_width']
        start = max(0, int(pattern_center - pattern_width // 2))
        pattern_array = generate_bspline_pattern(tuple(cps), pattern_width, data_min, data_max)
        pattern_type = 'relative' if use_relative else 'absolute'
        print(f"Pattern {len(extracted_patterns)+1}: {metric}={score:.4f}, transform={transform_type}, type={pattern_type}, series={series_idx}, center={pattern_center}, width={pattern_width:.1f}, shift_tolerance={round(shift_tolerance, 4)}")
        extracted_patterns.append({'pattern': pattern_array, 'start': start, 'width': pattern_width, 'center': pattern_center, 'series_idx': series_idx, 'control_points': cps, 'transform_type': transform_type, 'use_relative': use_relative, 'shift_tolerance': shift_tolerance, 'score': score})
        model_features_list.append(pattern_to_features(transformed_train[transform_type], pattern_width, start, series_idx, pattern=pattern_array, data_min=data_min, data_max=data_max, use_relative=use_relative, shift_tolerance=shift_tolerance))
    model_features = np.column_stack(model_features_list) if model_features_list else np.empty((len(y_train), 0))
    model.fit(model_features[train_idx], y_train[train_idx], model_features[val_idx], y_train[val_idx])
    test_features = None
    if input_series_test is not None:
        transformed_test = {t: apply_transformation(input_series_test, t) for t in transform_types}
        test_feats = [pattern_to_features(transformed_test[p.get('transform_type', 'raw')], p['width'], p['start'], p['series_idx'], pattern=p['pattern'], data_min=data_min, data_max=data_max, use_relative=p.get('use_relative', False), shift_tolerance=p.get('shift_tolerance', 0), max_shift_evaluations=max_shift_evaluations) for p in extracted_patterns]
        all_test_feats = ([initial_features[1]] if initial_features else []) + test_feats
        test_features = np.column_stack(all_test_feats) if all_test_feats else np.empty((len(input_series_test), 0))
    return {'patterns': extracted_patterns, 'train_features': model_features, 'test_features': test_features, 'model': model}
