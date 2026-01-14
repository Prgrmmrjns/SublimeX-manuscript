import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from models import LightGBMWrapper

def eval_minirocket(train_array, test_array, y_train, y_test, task_type='classification', metric='accuracy', num_classes=None, initial_train=None, initial_test=None):
    t0 = time.time()
    
    # 1. Ensure 3D arrays (samples, channels, time)
    X_train = np.asarray(train_array, dtype=np.float32)
    X_test = np.asarray(test_array, dtype=np.float32)
    
    # If 2D, assume single channel
    if X_train.ndim == 2:
        X_train = X_train[:, None, :]
    if X_test.ndim == 2:
        X_test = X_test[:, None, :]
    
    # Extract dimensions
    n_samples_train, n_channels, n_time = X_train.shape
    n_samples_test = X_test.shape[0]
    
    # Ensure test has same channels and time
    if X_test.shape[1] != n_channels or X_test.shape[2] != n_time:
        raise ValueError(f"Shape mismatch: train {X_train.shape} vs test {X_test.shape}")
    
    # 2. Fit MiniRocket Transform
    minirocket = MiniRocketMultivariate()
    X_train_transform = minirocket.fit_transform(X_train)
    X_test_transform = minirocket.transform(X_test)
    
    # Ensure numpy arrays (sktime often returns DataFrames)
    if hasattr(X_train_transform, 'values'):
        X_train_transform = X_train_transform.values
    if hasattr(X_test_transform, 'values'):
        X_test_transform = X_test_transform.values
    
    # Add initial features if provided
    if initial_train is not None and initial_test is not None:
        X_train_transform = np.hstack([initial_train, X_train_transform])
        X_test_transform = np.hstack([initial_test, X_test_transform])
    
    # Encode labels
    if task_type == 'classification':
        y_all = np.concatenate([y_train, y_test])
        y_encoded = np.unique(y_all, return_inverse=True)[1]
        y_train_enc = y_encoded[:len(y_train)]
        y_test_enc = y_encoded[len(y_train):]
        n_classes = len(np.unique(y_all))
    else:
        y_train_enc, y_test_enc = y_train.astype(np.float32), y_test.astype(np.float32)
        n_classes = None

    # 3. Fit Model using PATX's unified training pipeline (LightGBM)
    model = LightGBMWrapper(task_type, n_classes=n_classes, n_jobs=1, inner_cv=1)
    model.fit(X_train_transform, y_train_enc)
    
    # 4. Predict
    predictions = model.predict(X_test_transform)
    
    elapsed = time.time() - t0
    
    # 5. Calculate Score
    if metric == 'accuracy':
        score = accuracy_score(y_test_enc, predictions)
    elif metric == 'auc':
        # Use probabilities for AUC
        probs = model.predict_proba(X_test_transform)
        if n_classes == 2:
            score = roc_auc_score(y_test_enc, probs)
        else:
            score = roc_auc_score(y_test_enc, probs, multi_class='ovr')
    elif metric == 'rmse':
        from sklearn.metrics import mean_squared_error
        score = np.sqrt(mean_squared_error(y_test_enc, predictions))
    else:
        score = accuracy_score(y_test_enc, predictions)
        
    return score, elapsed, X_train_transform.shape[1]
