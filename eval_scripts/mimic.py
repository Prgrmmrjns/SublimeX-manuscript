import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from core import feature_extraction
from models import LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

def load_mimic_data():
    """Load MIMIC data with multivariate time series."""
    df = pd.read_csv('../processed_datasets/mimic_processed.csv')
    y = df['ARDS_FLAG']
    anchor_age = df['anchor_age'].values
    
    # Extract time series names from column names
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'anchor_age', 'ARDS_FLAG']]
    series_names = []
    for col in feature_cols:
        if '_hour_' in col:
            series_name = col.split('_hour_')[0]
            if series_name not in series_names:
                series_names.append(series_name)
    
    # Create list of DataFrames for each time series
    X_list = []
    for series_name in series_names:
        series_cols = [col for col in feature_cols if col.startswith(f"{series_name}_hour_")]
        series_cols.sort(key=lambda x: int(x.split('_hour_')[1]))
        X_series = df[series_cols]
        X_list.append(X_series)
    
    return X_list, y, anchor_age, series_names

# Load data once
X_list, y, anchor_age, series_names = load_mimic_data()

# Store KFold indices first
kfold_indices = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))

results = []

if os.path.exists('../results/mimic.csv'):
    try:
        existing = pd.read_csv('../results/mimic.csv')
        if len(existing) > 0:
            results = existing.to_dict('records')
            done = set(zip(existing['approach'], existing['fold']))
        else:
            results = []
            done = set()
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        results = []
        done = set()
else:
    results = []
    done = set()

all_patterns = {}
for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        # patx
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        input_train = [x.iloc[train_idx] for x in X_list]
        input_test = [x.iloc[val_idx] for x in X_list]
        t0 = time.time()
        res = feature_extraction(input_train, y_train, input_test, metric='accuracy', n_trials=N_TRIALS, show_progress=SHOW_PROGRESS)
        all_patterns[f'fold_{fold+1}'] = res['patterns']
        accuracy = accuracy_score(y_val, res['model'].predict(res['test_features']))
        results.append({'approach': 'PATX', 'fold': fold + 1, 'score': accuracy, 
                       'processing_time': time.time()-t0, 'n_features': len(res['patterns'])})
        
        # Store pattern parameters for this fold
        os.makedirs('../json_files/mimic', exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_patterns = []
        for pattern in res['patterns']:
            serializable_pattern = {}
            for key, value in pattern.items():
                if isinstance(value, np.ndarray):
                    serializable_pattern[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_pattern[key] = value.item()
                else:
                    serializable_pattern[key] = value
            serializable_patterns.append(serializable_pattern)
        
        pattern_data = {
            'patterns': serializable_patterns,
            'fold': fold + 1,
            'n_patterns': len(res['patterns']),
            'performance': {'accuracy': accuracy, 'processing_time': time.time()-t0}
        }
        with open(f'../json_files/mimic/pattern_parameters_fold{fold+1}.json', 'w') as f:
            json.dump(pattern_data, f, indent=2)

        # Tsfresh
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_concat = pd.concat([x.iloc[train_idx] for x in X_list], axis=1)
        test_concat = pd.concat([x.iloc[val_idx] for x in X_list], axis=1)
        t0 = time.time()
        test_feat, train_feat = run_tsfresh(train_concat.values, test_concat.values)
        tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, 
                                                          random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        model.fit(tr_f, y_tr, val_f, y_val_split)
        accuracy = accuracy_score(y_val, model.predict(test_feat))
        results.append({'approach': 'TSFRESH', 'fold': fold + 1, 'score': accuracy, 
                       'processing_time': time.time()-t0, 'n_features': train_feat.shape[1]})

        #CATCH22
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        t0 = time.time()
        test_feat, train_feat = run_catch22(X_list[0].iloc[train_idx].values, X_list[0].iloc[val_idx].values)
        tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, 
                                                          random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        model.fit(tr_f, y_tr, val_f, y_val_split)
        accuracy = accuracy_score(y_val, model.predict(test_feat))
        results.append({'approach': 'CATCH22', 'fold': fold + 1, 'score': accuracy, 
                       'processing_time': time.time()-t0, 'n_features': train_feat.shape[1]})

        #CNN
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_concat = pd.concat([x.iloc[train_idx] for x in X_list], axis=1)
        test_concat = pd.concat([x.iloc[val_idx] for x in X_list], axis=1)
        train_with_age = np.column_stack([train_concat.values, anchor_age[train_idx]])
        test_with_age = np.column_stack([test_concat.values, anchor_age[val_idx]])
        t0 = time.time()
        preds = run_cnn(train_with_age, y_train.values, test_with_age, task_type='classification', 
                       metric='accuracy', num_classes=len(np.unique(y_train)), epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
        accuracy = accuracy_score(y_val, preds)
        results.append({'approach': 'CNN', 'fold': fold + 1, 'score': accuracy, 
                       'processing_time': time.time()-t0, 'n_features': train_with_age.shape[1]})

df_res = pd.DataFrame(results)
for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
    app_res = df_res[df_res['approach'] == app]
    if len(app_res) > 0:
        m_acc, s_acc = app_res['score'].mean(), app_res['score'].std()
        m_time, s_time = app_res['processing_time'].mean(), app_res['processing_time'].std()
        m_feat, s_feat = app_res['n_features'].mean(), app_res['n_features'].std()
        print(f"{app:8}: Accuracy={m_acc:.4f}±{s_acc:.4f}, Time={m_time:.1f}±{s_time:.1f}s, Features={m_feat:.0f}±{s_feat:.0f}")

df_res.to_csv('../results/mimic.csv', index=False)

# Store all patterns in a single file for easy access
if all_patterns:
    os.makedirs('../json_files/mimic', exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_all_patterns = {}
    for fold_key, patterns in all_patterns.items():
        serializable_patterns = []
        for pattern in patterns:
            serializable_pattern = {}
            for key, value in pattern.items():
                if isinstance(value, np.ndarray):
                    serializable_pattern[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_pattern[key] = value.item()
                else:
                    serializable_pattern[key] = value
            serializable_patterns.append(serializable_pattern)
        serializable_all_patterns[fold_key] = serializable_patterns
    
    with open('../json_files/mimic/pattern_parameters.json', 'w') as f:
        json.dump(serializable_all_patterns, f, indent=2)