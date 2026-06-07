"""Shared paths and experiment settings for all manuscript scripts."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'scripts'
DATASETS = REPO / 'datasets'
RESULTS = REPO / 'results'
PARAMETERS = REPO / 'parameters'
ABLATION_PARAMS = PARAMETERS / 'ablation'
ELARTICLE = REPO / 'elsarticle'

K_FOLDS = 5
LOAD_FEATURES = True
SAVE_FEATURES = False

MAIN_EVAL_CSV = RESULTS / 'main_eval.csv'
ABLATION_CSV = RESULTS / 'ablation_study.csv'
INCREMENTAL_CSV = RESULTS / 'incremental_features.csv'

SUBLIMEX_LOCAL = REPO / 'sublimex'


def setup_sublimex_path():
    import sys
    if SUBLIMEX_LOCAL.is_dir():
        p = str(SUBLIMEX_LOCAL)
        if p not in sys.path:
            sys.path.insert(0, p)
