import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from models import LightGBMWrapper
from core import PatternExtractor

def serialize_patterns(patterns):
    """Serialize patterns for JSON storage."""
    p_ser = []
    for i, p in enumerate(patterns):
        p_d = {'pattern_id': i + 1}
        for k, v in p.items():
            if k == 'pattern':
                continue
            p_d[k] = v.tolist() if isinstance(v, np.ndarray) else (v.item() if hasattr(v, 'item') else v)
        p_ser.append(p_d)
    return p_ser

def run_patx(input_train, y_train, input_test, metric, initial_features=None, y_test=None, model_class=None, **kwargs):
    # Determine number of training samples
    if isinstance(input_train, (list, tuple)):
        if len(input_train) > 0:
            first_arr = np.asarray(input_train[0])
            n_samples = first_arr.shape[0] if first_arr.ndim >= 1 else len(input_train)
        else:
            n_samples = len(input_train)
    else:
        input_train_arr = np.asarray(input_train)
        n_samples = input_train_arr.shape[0] if input_train_arr.ndim >= 1 else 1
    
    # Set inner_cv based on sample size (can be overridden by kwargs)
    inner_cv = kwargs.pop('inner_cv', (1 if n_samples > 10000 else 3))
    
    # Determine n_classes for model
    n_classes = len(np.unique(y_train)) if metric != 'rmse' else None
    
    # Initialize model wrapper for pattern search
    search_model = LightGBMWrapper(
        task_type='classification' if metric != 'rmse' else 'regression',
        n_classes=n_classes,
        inner_cv=inner_cv
    )
    
    # Default PatternExtractor parameters
    extractor_params = {
        'model': search_model,
        'metric': metric,
        'n_trials': 2000,
        'n_trials_without_improvement': 500,
        'n_control_points': 4,
        'show_progress': False,
        'n_workers': -1,
        'verbose': False,
    }
    extractor_params.update(kwargs)
    
    # Initialize PatternExtractor
    extractor = PatternExtractor(**extractor_params)
    
    # Pattern search and feature extraction
    init_train = initial_features[0] if initial_features and initial_features[0] is not None else None
    init_test = initial_features[1] if initial_features and initial_features[1] is not None else None
    extractor.fit(input_train, y_train, input_test, initial_features=(init_train, init_test))
    
    # Train final model
    if model_class is None:
        model_class = LightGBMWrapper
    final_model = model_class(
        task_type='classification' if metric != 'rmse' else 'regression',
        n_classes=n_classes,
        inner_cv=1
    )
    y_proc = np.unique(y_train, return_inverse=True)[1] if metric != 'rmse' else np.asarray(y_train, dtype=np.float32)
    final_model.fit(extractor.train_features, y_proc)
    
    result = {
        'model': final_model,
        'test_features': extractor.test_features,
        'train_features': extractor.train_features,
        'patterns': extractor.patterns,
        'n_features': len(extractor.patterns)
    }
    
    # Compute score if y_test is provided
    if y_test is not None and extractor.test_features is not None:
        preds = final_model.predict(extractor.test_features)
        if metric == 'auc':
            pp = final_model.predict_proba(extractor.test_features)
            result['score'] = roc_auc_score(y_test, pp)
        elif metric == 'rmse':
            result['score'] = float(np.sqrt(mean_squared_error(y_test, preds)))
        else:
            result['score'] = accuracy_score(y_test, preds)
    return result