import pandas as pd
import numpy as np
import os
import time
import warnings
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from core import feature_extraction
from models import LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

TIME_SERIES = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']

def load_remc_data(cell_line):
    df = pd.read_parquet(f"../processed_datasets/remc/{cell_line}.parquet")
    X = df[[c for c in df.columns if c != 'target']]
    return [X[[c for c in X.columns if c.startswith(f"{s}_")]] for s in TIME_SERIES], df['target']

def save_patterns_to_json(patterns, cell_line, fold):
    os.makedirs('../json_files/remc', exist_ok=True)
    pattern_data = []
    for i, pattern in enumerate(patterns):
        pattern_info = {
            'pattern_id': i + 1,
            'transform_type': pattern['transform_type'],
            'use_relative': pattern['use_relative'],
            'shift_tolerance': pattern['shift_tolerance'],
            'series_idx': pattern['series_idx'],
            'center': pattern['center'],
            'width': pattern['width'],
            'control_points': pattern['control_points'],
            'score': pattern.get('score', None)
        }
        pattern_data.append(pattern_info)
    
    filename = f'../json_files/remc/pattern_parameters_{cell_line}_fold{fold}.json'
    with open(filename, 'w') as f:
        json.dump(pattern_data, f, indent=2)

cell_lines = sorted([f.replace('.parquet', '') for f in os.listdir('../processed_datasets/remc') if f.endswith('.parquet')])
results = []

done = set()
if os.path.exists('../results/remc.csv'):
    existing = pd.read_csv('../results/remc.csv')
    results = existing.to_dict('records')
    approach_counts = existing.groupby('cell_line').size()
    done = set(approach_counts[approach_counts == 4].index)

for cell_line in cell_lines:
    if cell_line not in done:
        print(f"Processing {cell_line}")
        X_list, y = load_remc_data(cell_line)
        
        # Store KFold indices first
        kfold_indices = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X_list, axis=1), y))
        
        # PATX
        all_patterns = {}
        patx_scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            input_train = [x.iloc[train_idx].astype(np.float32) for x in X_list]
            input_test = [x.iloc[val_idx].astype(np.float32) for x in X_list]
            t0 = time.time()
            res = feature_extraction(input_train, y_train.values, input_test, metric='auc', val_size=VAL_SIZE,
                                    n_trials=N_TRIALS, show_progress=SHOW_PROGRESS, n_control_points=N_CONTROL_POINTS)
            all_patterns[f'fold_{fold+1}'] = res['patterns']
            save_patterns_to_json(res['patterns'], cell_line, fold+1)
            preds_proba = res['model'].predict_proba(res['test_features'])
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            auc = roc_auc_score(y_val, preds)
            patx_scores.append(auc)
            print(f"Fold {fold+1}: AUC={auc:.4f}, Time={time.time()-t0:.1f}s, Features={len(res['patterns']):.0f}")
        
        avg_auc = np.mean(patx_scores)
        results.append({'cell_line': cell_line, 'approach': 'PATX', 'score': avg_auc, 
                       'processing_time': time.time()-t0, 'n_features': len(res['patterns'])})

        # TSFRESH
        tsfresh_scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            all_train = pd.concat([x.iloc[train_idx] for x in X_list], axis=1)
            all_test = pd.concat([x.iloc[val_idx] for x in X_list], axis=1)
            t0 = time.time()
            test_feat, train_feat = run_tsfresh(all_train.values, all_test.values)
            tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42)
            model = LightGBMModelWrapper('classification', n_classes=2)
            model.fit(tr_f, y_tr, val_f, y_val_split)
            preds_proba = model.predict_proba(test_feat)
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            auc = roc_auc_score(y_val, preds)
            tsfresh_scores.append(auc)
        
        avg_auc = np.mean(tsfresh_scores)
        results.append({'cell_line': cell_line, 'approach': 'TSFRESH', 'score': avg_auc, 
                       'processing_time': time.time()-t0, 'n_features': train_feat.shape[1]})
    
        # CATCH22
        catch22_scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            # Use first time series (H3K4me3) for CATCH22
            train_data = X_list[0].iloc[train_idx].values
            test_data = X_list[0].iloc[val_idx].values
            t0 = time.time()
            test_feat, train_feat = run_catch22(train_data, test_data)
            tr_f, val_f, y_tr, y_val_split = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42)
            model = LightGBMModelWrapper('classification', n_classes=2)
            model.fit(tr_f, y_tr, val_f, y_val_split)
            preds_proba = model.predict_proba(test_feat)
            preds = preds_proba[:, 1] if len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1 else preds_proba
            auc = roc_auc_score(y_val, preds)
            catch22_scores.append(auc)
        
        avg_auc = np.mean(catch22_scores)
        results.append({'cell_line': cell_line, 'approach': 'CATCH22', 'score': avg_auc, 
                       'processing_time': time.time()-t0, 'n_features': train_feat.shape[1]})
    
        # CNN
        cnn_scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold_indices):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            all_train = pd.concat([x.iloc[train_idx] for x in X_list], axis=1)
            all_test = pd.concat([x.iloc[val_idx] for x in X_list], axis=1)
            t0 = time.time()
            preds = run_cnn(all_train.values, y_train.values, all_test.values, task_type='classification', 
                           metric='auc', epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
            auc = roc_auc_score(y_val, preds)
            cnn_scores.append(auc)
        
        avg_auc = np.mean(cnn_scores)
        results.append({'cell_line': cell_line, 'approach': 'CNN', 'score': avg_auc, 
                       'processing_time': time.time()-t0, 'n_features': all_train.shape[1]})
    
    # Print results for this cell line
    cell_line_res = pd.DataFrame([r for r in results if r['cell_line'] == cell_line])
    for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
        app_res = cell_line_res[cell_line_res['approach'] == app]
        if len(app_res) > 0:
            r = app_res.iloc[0]
            print(f"{app:8}: AUC={r['score']:.4f}, Time={r['processing_time']:.1f}s, Features={r['n_features']:.0f}")
    
    # Save results after each cell line
    pd.DataFrame(results).to_csv('../results/remc.csv', index=False)

# Final save
pd.DataFrame(results).to_csv('../results/remc.csv', index=False)