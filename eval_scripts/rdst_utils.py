import time
import numpy as np
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform as RDST
from aeon.regression.shapelet_based import RDSTRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import label_binarize
from models import LightGBMWrapper

def eval_rdst(train_array, test_array, y_train, y_test, metric, n_classes=None, initial_train=None, initial_test=None):
    t0 = time.time()

    # Shape for aeon: (n_instances, n_channels, n_timepoints)
    # Convert to float64 - aeon's numba code requires consistent dtype
    train_array = np.asarray(train_array, dtype=np.float64)
    test_array = np.asarray(test_array, dtype=np.float64)
    if train_array.ndim == 2:
        train_array, test_array = train_array[:, None, :], test_array[:, None, :]
    
    # Handle regression vs classification
    if metric == 'rmse':
        # Use RDSTRegressor for regression
        regressor = RDSTRegressor(max_shapelets=2000, random_state=42, n_jobs=1)
        regressor.fit(train_array, y_train.astype(np.float64))
        predictions = regressor.predict(test_array)
        
        # If initial features provided, fit a LightGBM model on top
        if initial_train is not None and initial_test is not None:
            # Get transformed features from regressor
            transformer = RDST(max_shapelets=2000, random_state=42)
            X_tr_feat = transformer.fit_transform(train_array, y_train.astype(np.float64))
            X_te_feat = transformer.transform(test_array)
            X_tr_feat = getattr(X_tr_feat, 'values', X_tr_feat)
            X_te_feat = getattr(X_te_feat, 'values', X_te_feat)
            
            # Combine with initial features
            X_tr_feat = np.hstack([initial_train, X_tr_feat])
            X_te_feat = np.hstack([initial_test, X_te_feat])
            
            # Fit LightGBM on combined features
            model = LightGBMWrapper(task_type='regression', n_jobs=-1)
            model.fit(X_tr_feat, y_train.astype(np.float32))
            predictions = model.predict(X_te_feat)
            n_features = X_tr_feat.shape[1]
        else:
            n_features = getattr(regressor, 'transformed_data_', None)
            if n_features is not None and len(n_features) > 0:
                n_features = n_features[0].shape[1] if hasattr(n_features[0], 'shape') else 2000 * 3
            else:
                # Default: 2000 shapelets * 3 features per shapelet
                n_features = 2000 * 3
        
        score = float(np.sqrt(mean_squared_error(y_test, predictions)))
    else:
        # Classification: use transformer + LightGBM
        # Encode labels
        y_all = np.concatenate([y_train, y_test])
        y_encoded = np.unique(y_all, return_inverse=True)[1]
        y_train_enc, y_test_enc = y_encoded[:len(y_train)], y_encoded[len(y_train):]
        n_classes = len(np.unique(y_all))
        
        # Use RDST transformer directly. 2000 shapelets (6000 features) is 
        # much faster than the default 10000 and sufficient for most datasets.
        transformer = RDST(max_shapelets=2000, random_state=42)
        X_tr_feat = transformer.fit_transform(train_array, y_train_enc)
        X_te_feat = transformer.transform(test_array)
        
        # Handle aeon returning DataFrames
        X_tr_feat = getattr(X_tr_feat, 'values', X_tr_feat)
        X_te_feat = getattr(X_te_feat, 'values', X_te_feat)
        
        # Add initial features if provided
        if initial_train is not None and initial_test is not None:
            X_tr_feat = np.hstack([initial_train, X_tr_feat])
            X_te_feat = np.hstack([initial_test, X_te_feat])
        
        model = LightGBMWrapper(task_type='classification', n_classes=n_classes, n_jobs=-1)
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
        
        n_features = X_tr_feat.shape[1]
            
    return score, time.time() - t0, n_features
