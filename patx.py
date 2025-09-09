import numpy as np
from sklearn.model_selection import train_test_split
import optuna
import json
import os
from params import evaluate_model_performance
from params import get_model, MODEL, TASK_TYPE

optuna.logging.set_verbosity(optuna.logging.WARNING)
class PatternOptimizer:
    def __init__(self, X_train, y_train, model, max_n_trials, test_size, n_jobs, show_progress, dataset, multiple_series, X_test_data, metric, polynomial_degree, val_size, initial_features=None):
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.X_series_list = None
        if multiple_series and isinstance(X_train, list):
            self.X_series_list = [np.asarray(x, dtype=np.float32) for x in X_train]
            self.X_train = np.stack(self.X_series_list, axis=1)
        self.X_test_data = X_test_data
        self.model = model
        self.max_n_trials = max_n_trials
        self.pattern_list = []
        self.pattern_starts = []
        self.pattern_ends = []
        self.pattern_series_indices = []
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.dataset = dataset
        self.multiple_series = multiple_series
        self.metric = metric
        self.features_list = []
        self.test_size = test_size
        self.best_score = float('inf') if self.metric == 'rmse' else -float('inf')
        self.val_size = val_size
        self.polynomial_degree = polynomial_degree
        self.initial_features = initial_features
    
    def polynomial_pattern(self, coeffs, n_points):
        x = np.linspace(-1, 1, n_points, dtype=np.float16)
        coeffs = np.array(coeffs, dtype=np.float16)
        powers = np.arange(len(coeffs), dtype=np.float16)
        return np.sum(coeffs * (x[:, None] ** powers), axis=1)

    def calculate_pattern_rmse(self, X_region, pattern_values):
        return np.sqrt(np.mean((X_region - pattern_values) ** 2, axis=1))

    def objective(self, trial, dim):
        series_index = trial.suggest_int('series_index', 0, self.X_train.shape[1] - 1) if self.multiple_series else None
        pattern_start = trial.suggest_int('pattern_start', 0, dim - 2)
        pattern_width = trial.suggest_int('pattern_width', 1, dim - pattern_start)
        coeffs = [trial.suggest_float(f'c{i}', -1, 1) for i in range(self.polynomial_degree + 1)]
        X_data = self.X_train[:, series_index, :] if self.multiple_series and series_index is not None and self.X_train.ndim == 3 else self.X_train
        X_region = X_data[:, pattern_start:pattern_start + pattern_width]
        new_feature = self.calculate_pattern_rmse(X_region, self.polynomial_pattern(coeffs, pattern_width))
        X_combined = np.column_stack(self.features_list + [new_feature]) if self.features_list else new_feature.reshape(-1, 1)
        X_train, X_val, y_train, y_val = train_test_split(X_combined, self.y_train, test_size=self.val_size, random_state=42)
        model = self.model
        model.train(X_train, y_train, X_val, y_val)
        return evaluate_model_performance(model, X_val, y_val, self.metric)
    
    def feature_extraction(self, X_series_list=None):
        first_pattern = True
        if X_series_list is not None and self.multiple_series:
            self.X_series_list = [np.asarray(x, dtype=np.float32) for x in X_series_list]
            self.X_train = np.stack(self.X_series_list, axis=1)
        dim = self.X_train.shape[2] if self.multiple_series and self.X_train.ndim == 3 else self.X_train.shape[1]
        train_initial_features, test_initial_features = (None, None) if self.initial_features is None else ((np.asarray(self.initial_features[0], dtype=np.float32), np.asarray(self.initial_features[1], dtype=np.float32)) if isinstance(self.initial_features, tuple) and len(self.initial_features) == 2 else (np.asarray(self.initial_features, dtype=np.float32), None))
        if train_initial_features is not None: self.features_list = [train_initial_features]
        while True:
            study = optuna.create_study(direction="minimize" if self.metric == 'rmse' else "maximize", pruner=optuna.pruners.MedianPruner())
            study.optimize(lambda trial: self.objective(trial, dim), n_trials=self.max_n_trials, n_jobs=self.n_jobs, show_progress_bar=self.show_progress)
            if first_pattern or (self.metric == 'rmse' and study.best_value < self.best_score) or (self.metric != 'rmse' and study.best_value > self.best_score):
                self.best_score = study.best_value
                best_params = study.best_trial.params
                pattern_start = best_params['pattern_start']
                pattern_width = best_params['pattern_width']
                coeffs = [best_params[f'c{i}'] for i in range(self.polynomial_degree + 1)]
                series_index = best_params.get('series_index')
                pattern_values = self.polynomial_pattern(coeffs, pattern_width)
                pattern_end = pattern_start + pattern_width
                new_pattern = np.zeros(dim, dtype=np.float32)
                new_pattern[pattern_start:pattern_end] = pattern_values
                self.pattern_list.append(new_pattern)
                self.pattern_starts.append(pattern_start)
                self.pattern_ends.append(pattern_end)
                if self.multiple_series:
                    self.pattern_series_indices.append(series_index)
                X_data = self.X_train
                if self.multiple_series and series_index is not None and X_data.ndim == 3:
                    X_data = X_data[:, series_index, :]
                X_region = X_data[:, pattern_start:pattern_end]
                new_feature_full = self.calculate_pattern_rmse(X_region, pattern_values)
                self.features_list.append(new_feature_full)
                first_pattern = False
            else:
                break
        
        cached_features = np.column_stack(self.features_list) if self.features_list else np.empty((self.X_train.shape[0], 0))
        X_train, X_val, y_train, y_val = train_test_split(cached_features, self.y_train, test_size=self.val_size, random_state=42)
        self.model.train(X_train, y_train, X_val, y_val)
        n_test_samples = self.X_test_data[0].shape[0] if isinstance(self.X_test_data, list) else self.X_test_data.shape[0]
        n_pattern_features = len(self.pattern_list)
        n_initial_features = train_initial_features.shape[1] if train_initial_features is not None else 0
        X_test = np.empty((n_test_samples, n_initial_features + n_pattern_features), dtype=np.float32)
        X_test[:, :n_initial_features] = test_initial_features if test_initial_features is not None else 0.0
        for i, pattern in enumerate(self.pattern_list):
            series_idx = self.pattern_series_indices[i] if self.multiple_series and self.pattern_series_indices else None
            X_for_pattern = self.X_test_data[series_idx] if self.multiple_series and isinstance(self.X_test_data, list) and series_idx is not None else self.X_test_data
            X_data = np.asarray(X_for_pattern, dtype=np.float32) if not isinstance(X_for_pattern, np.ndarray) else X_for_pattern
            if self.multiple_series and series_idx is not None and X_data.ndim == 3:
                X_data = X_data[:, series_idx, :]
            start, end = self.pattern_starts[i], self.pattern_ends[i]
            X_region = X_data[:, start:end]
            pattern_feature = self.calculate_pattern_rmse(X_region, pattern[start:end]).reshape(-1, 1)
            X_test[:, n_initial_features + i:n_initial_features + i+1] = pattern_feature
        result = {'patterns': self.pattern_list,'starts': self.pattern_starts,'ends': self.pattern_ends,'features': cached_features,'X_train': X_train,'X_val': X_val,'y_train': y_train,'y_val': y_val,'X_test': X_test}
        if self.multiple_series and self.pattern_series_indices: result['series_indices'] = self.pattern_series_indices
        X_combined = np.vstack((X_train, X_val))
        y_combined = np.hstack((y_train, y_val))
        X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=self.val_size, random_state=42)
        self.model.train(X_train, y_train, X_val, y_val)
        result['model'] = self.model
        return result

    def save_parameters_to_json(self, dataset_name):
        """Save all optimized pattern parameters to a JSON file"""
        params_dict = {
            'dataset': self.dataset,
            'metric': self.metric,
            'polynomial_degree': self.polynomial_degree,
            'n_patterns': len(self.pattern_list),
            'patterns': []
        }
        for i, pattern in enumerate(self.pattern_list):
            pattern_info = {
                'pattern_id': i,
                'pattern_start': int(self.pattern_starts[i]),
                'pattern_width': int(self.pattern_ends[i] - self.pattern_starts[i]),
                'pattern_values': pattern[self.pattern_starts[i]:self.pattern_ends[i]].tolist(),
            }
            if self.multiple_series and self.pattern_series_indices:
                pattern_info['series_index'] = int(self.pattern_series_indices[i])
            params_dict['patterns'].append(pattern_info)
        
        os.makedirs(f'json_files/{dataset_name}', exist_ok=True)
        with open(f'json_files/{dataset_name}/pattern_parameters.json', 'w') as f:
            json.dump(params_dict, f, indent=2)