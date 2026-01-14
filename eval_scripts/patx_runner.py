from core import PatternExtractor
from models import LightGBMWrapper
import numpy as np

K_FOLDS = 5
VAL_SIZE = 0.2

def run_patx(input_train, y_train, input_test, metric, initial_features=None):
    inner_cv = 5
    task = "regression" if metric == "rmse" else "classification"
    n_classes = len(np.unique(y_train)) if task == "classification" else 2
    
    # Use an ensemble for pattern discovery as well for more robust evaluation
    model = LightGBMWrapper(task, n_classes=n_classes, n_jobs=1, inner_cv=inner_cv)
    pe = PatternExtractor(
        model=model,
        metric=metric,
        n_trials=1000,
        n_control_points=5,
        show_progress=False,
        n_workers=-1,
    )
    pe.fit(input_train, y_train, input_series_test=input_test, initial_features=initial_features)
    res = {
        "patterns": pe.patterns,
        "train_features": pe.train_features,
        "test_features": pe.test_features,
    }
    
    # Replicate train_final_model pipeline using direct class usage
    model.fit(pe.train_features, y_train)
    res['model'] = model
    return res
