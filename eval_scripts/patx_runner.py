import sys
from pathlib import Path

# ensure local patx import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patx"))

from patx import feature_extraction

K_FOLDS = 5
VAL_SIZE = 0.2

def run_patx(input_train, y_train, input_test, metric, initial_features=None):
    return feature_extraction(
        input_train,
        y_train,
        input_test,
        metric=metric,
        n_trials=500,
        n_control_points=3,
        n_transforms=5,
        max_samples=30000,
        inner_k_folds=1,
        val_size=0.2,
        show_progress=True,
        n_workers=-1,
        initial_features=initial_features,
    )
