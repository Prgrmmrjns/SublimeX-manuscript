from typing import Optional, Union, List, Dict, Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from scipy.interpolate import BSpline
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import warnings
from .models import LightGBMModelWrapper

def generate_bspline_pattern(control_points: List[float], width: int) -> np.ndarray:
    degree = 3
    n_cp = len(control_points)
    knots = np.concatenate([np.zeros(degree + 1), np.linspace(0, 1, n_cp - degree + 1)[1:-1], np.ones(degree + 1)])
    return BSpline(knots, np.asarray(control_points), degree)(np.linspace(0, 1, width))

def pattern_to_features(input_series: np.ndarray, control_points: List[float], pattern_width: int, pattern_start: int, series_index: int = 0) -> np.ndarray:
    pattern = generate_bspline_pattern(control_points, pattern_width)
    X_region = input_series[:, series_index, pattern_start:pattern_start + pattern_width]
    return np.sqrt(((X_region - pattern) ** 2).mean(axis=1))

def evaluate_model_performance(model, metric, cached_data):
    X_train, X_val, y_train_split, y_val = cached_data
    model = model.clone()
    model.fit(X_train, y_train_split, X_val, y_val)
    if metric == 'accuracy': return accuracy_score(y_val, model.predict(X_val))
    if metric == 'rmse': return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    y_pred = model.predict_proba(X_val)
    return roc_auc_score(y_val, y_pred) if len(np.unique(y_val)) == 2 else roc_auc_score(y_val, y_pred, multi_class='ovr', average='macro')

def feature_extraction(input_series_train: Union[np.ndarray, List], 
                      y_train: np.ndarray, 
                      input_series_test: Optional[Union[np.ndarray, List]] = None,
                      initial_features: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      model: Optional[Any] = None, 
                      metric: str = 'auc', 
                      val_size: float = 0.2,
                      n_trials: int = 300, 
                      n_control_points: int = 5,
                      show_progress: bool = True) -> Dict[str, Any]:
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if isinstance(input_series_train, list):
        input_series_train = np.stack([x.values if hasattr(x, 'values') else x for x in input_series_train], axis=1)
    if isinstance(input_series_test, list):
        input_series_test = np.stack([x.values if hasattr(x, 'values') else x for x in input_series_test], axis=1)
    n_input_series, n_time_points = input_series_train.shape[1], input_series_train.shape[2]
    
    model_features_list = [initial_features[0]] if initial_features else []
    y_train = np.asarray(y_train).flatten()
    
    if metric != 'rmse':
        unique_targets = np.unique(y_train)
        if len(unique_targets) > 2 and not np.array_equal(unique_targets, np.arange(len(unique_targets))):
            y_train = np.array([{v: i for i, v in enumerate(unique_targets)}[y] for y in y_train])
        elif len(unique_targets) == 2 and not np.array_equal(unique_targets, [0, 1]):
            y_train = (y_train == unique_targets[1]).astype(int)
    
    if model is None:
        model = LightGBMModelWrapper('regression' if metric == 'rmse' else 'classification', n_classes=len(np.unique(y_train)) if metric != 'rmse' and len(np.unique(y_train)) > 2 else 2)
    
    train_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=val_size, random_state=42)
    overall_best_score = float('inf') if metric == 'rmse' else -float('inf')
    
    fast_model = LightGBMModelWrapper('regression' if metric == 'rmse' else 'classification', n_classes=len(np.unique(y_train)) if metric != 'rmse' and len(np.unique(y_train)) > 2 else 2)

    def objective(trial):
        series_idx = trial.suggest_int('series_index', 0, n_input_series - 1) if n_input_series > 1 else 0
        cps = [trial.suggest_float(f'cp{i}', 0, 1) for i in range(n_control_points)]
        width = trial.suggest_int('pattern_width', max(3, n_time_points // 4), n_time_points - 1)
        start = trial.suggest_int('pattern_start', 0, n_time_points // 2)
        if start + width > n_time_points: return float('-inf') if metric != 'rmse' else float('inf')
        feat = pattern_to_features(input_series_train, cps, width, start, series_idx)
        X = np.column_stack(model_features_list + [feat]) if model_features_list else feat.reshape(-1, 1)
        cached_data = (X[train_idx], X[val_idx], y_train[train_idx], y_train[val_idx])
        return evaluate_model_performance(fast_model, metric, cached_data)
    
    extracted_patterns = []
    while True:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
            study = optuna.create_study(
                direction='minimize' if metric == 'rmse' else 'maximize',
                sampler=optuna.samplers.TPESampler(n_startup_trials=50, warn_independent_sampling=False, multivariate=True),
                pruner=optuna.pruners.HyperbandPruner()
            )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress, n_jobs=-1)
        score = study.best_trial.value
        improved = not extracted_patterns or (metric == 'rmse' and score < overall_best_score) or (metric != 'rmse' and score > overall_best_score)
        if not improved: break
        params = study.best_trial.params
        series_idx = params.get('series_index', 0)
        cps = [params[f'cp{i}'] for i in range(n_control_points)]
        start, width = params['pattern_start'], params['pattern_width']
        extracted_patterns.append({'pattern': generate_bspline_pattern(cps, width), 'start': start, 'width': width, 'series_idx': series_idx, 'control_points': cps})
        model_features_list.append(pattern_to_features(input_series_train, cps, width, start, series_idx))
        overall_best_score = score
    
    model_features = np.column_stack(model_features_list) if model_features_list else np.empty((len(y_train), 0))
    model.fit(model_features[train_idx], y_train[train_idx], model_features[val_idx], y_train[val_idx])
    
    test_features = None
    if input_series_test is not None:
        test_feats = [pattern_to_features(input_series_test, p['control_points'], p['width'], p['start'], p['series_idx']) for p in extracted_patterns]
        all_test_feats = ([initial_features[1]] if initial_features else []) + test_feats
        test_features = np.column_stack(all_test_feats) if all_test_feats else np.empty((len(input_series_test), 0))
    return {'patterns': extracted_patterns, 'train_features': model_features, 'test_features': test_features, 'model': model}