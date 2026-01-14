import time
import numpy as np
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform as RDST
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from models import LightGBMWrapper

def eval_rdst(train_array, test_array, y_train, y_test, metric, n_classes=None):
    """
    Evaluate RDST (Random Dilated Shapelet Transform) with CatBoost.
    Optimized for speed by using the RDST transformer directly with 2000 shapelets.
    """
    if metric == 'rmse': return 0.0, 0.0, 0
    t0 = time.time()

    # Encode labels
    y_all = np.concatenate([y_train, y_test])
    y_encoded = np.unique(y_all, return_inverse=True)[1]
    y_train_enc, y_test_enc = y_encoded[:len(y_train)], y_encoded[len(y_train):]
    n_classes = len(np.unique(y_all))
    
    # Shape for aeon: (n_instances, n_channels, n_timepoints)
    # Convert to float64 - aeon's numba code requires consistent dtype
    train_array = np.asarray(train_array, dtype=np.float64)
    test_array = np.asarray(test_array, dtype=np.float64)
    if train_array.ndim == 2:
        train_array, test_array = train_array[:, None, :], test_array[:, None, :]
    
    # Use RDST transformer directly. 2000 shapelets (6000 features) is 
    # much faster than the default 10000 and sufficient for most datasets.
    transformer = RDST(max_shapelets=2000, random_state=42)
    X_tr_feat = transformer.fit_transform(train_array, y_train_enc)
    X_te_feat = transformer.transform(test_array)
    
    # Handle aeon returning DataFrames
    X_tr_feat = getattr(X_tr_feat, 'values', X_tr_feat)
    X_te_feat = getattr(X_te_feat, 'values', X_te_feat)
    
    model = LightGBMWrapper(task_type='classification', n_classes=n_classes, n_jobs=1, inner_cv=3)
    model.fit(X_tr_feat, y_train_enc)
    
    if metric == 'auc':
        proba = model.predict_proba(X_te_feat)
        if n_classes > 2:
            score = roc_auc_score(label_binarize(y_test_enc, classes=np.arange(n_classes)), 
                                 proba, multi_class='ovr', average='macro')
        else:
            score = roc_auc_score(y_test_enc, proba if proba.ndim == 1 else proba[:, 1])
    else:
        score = accuracy_score(y_test_enc, model.predict(X_te_feat))
            
    return score, time.time() - t0, X_tr_feat.shape[1]
