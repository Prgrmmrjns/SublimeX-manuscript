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

def run_patx(input_train, y_train, input_test, metric, initial_features=None, y_test=None, **kwargs):
    
    # Initialize model wrapper for pattern search
    model = LightGBMWrapper(
        task_type='classification' if metric != 'rmse' else 'regression',
        n_classes=len(np.unique(y_train)) if metric != 'rmse' else None,
    )
    
    # Default PatternExtractor parameters
    extractor_params = {
        'model': model,
        'metric': metric,
        'n_trials': 2000,
        'n_trials_without_improvement': 500,
        'n_control_points': 5,
        'show_progress': False,
        'n_workers': -1,
        'verbose': True,
    }
    extractor_params.update(kwargs)
    
    # Initialize PatternExtractor
    extractor = PatternExtractor(**extractor_params)
    
    # Pattern search and feature extraction
    init_train = initial_features[0] if initial_features and initial_features[0] is not None else None
    init_test = initial_features[1] if initial_features and initial_features[1] is not None else None
    extractor.fit(input_train, y_train, input_test, initial_features=(init_train, init_test))
    
    # Train final model
    y_proc = np.unique(y_train, return_inverse=True)[1] if metric != 'rmse' else np.asarray(y_train, dtype=np.float32)
    model.fit(extractor.train_features, y_proc)
    preds = model.predict(extractor.test_features)
    if metric == 'auc':
        score = model.predict_proba(extractor.test_features)
        score = roc_auc_score(y_test, score)
    elif metric == 'rmse':
        score = np.sqrt(mean_squared_error(y_test, preds))
    else:
        score = accuracy_score(y_test, preds)
    result = {
        'score': score,
        'patterns': extractor.patterns,
        'n_features': len(extractor.patterns)
    }
    return result