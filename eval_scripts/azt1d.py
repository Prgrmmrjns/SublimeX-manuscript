import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from core import feature_extraction
from models import LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings('ignore')

TIME_SERIES = ['CGM', 'Insulin', 'Carbs']

def load_azt1d_data(subject_id):
    df = pd.read_parquet(f"../processed_datasets/azt1d/subject_{subject_id}.parquet")
    ts_data = {s: df[[c for c in df.columns if c.startswith(f'{s}_')]] for s in TIME_SERIES}
    return ts_data, df['target']

subject_ids = sorted([f.replace('subject_', '').replace('.parquet', '') for f in os.listdir("../processed_datasets/azt1d/") if f.endswith('.parquet')])
results = []

if os.path.exists('../results/azt1d.csv'):
    try:
        existing = pd.read_csv('../results/azt1d.csv')
        if len(existing) > 0:
            results = existing.to_dict('records')
            done = set(zip(existing['subject_id'].astype(str), existing['approach']))
        else:
            results = []
            done = set()
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        results = []
        done = set()
else:
    results = []
    done = set()
    
print("Running performance comparison on the AZT1D dataset")
for subject_id in subject_ids:
    subject_approaches_done = {(subject_id, approach) for approach in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']}
    if subject_approaches_done.issubset(done):
        print(f"Skipping Subject {subject_id} (already complete)")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing Subject {subject_id}")
    print(f"{'='*60}")
    try:
        time_series_data, y = load_azt1d_data(subject_id)
        
        if time_series_data is None or y is None:
            continue
        
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=42)
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        input_train = [time_series_data[s].iloc[train_idx] for s in TIME_SERIES]
        input_test = [time_series_data[s].iloc[test_idx] for s in TIME_SERIES]
        init_train = time_series_data['CGM'].iloc[train_idx]['CGM_0'].values.reshape(-1, 1)
        init_test = time_series_data['CGM'].iloc[test_idx]['CGM_0'].values.reshape(-1, 1)

        # PATX
        t0 = time.time()
        res = feature_extraction(input_train, y_train, input_test, metric='rmse', n_trials=N_TRIALS, n_control_points=N_CONTROL_POINTS, n_patterns=N_PATTERNS, n_transforms=N_TRANSFORMS, max_samples=MAX_SAMPLES, inner_k_folds=INNER_K_FOLDS, early_stopping_patience=EARLY_STOPPING_PATIENCE, val_size=VAL_SIZE, show_progress=SHOW_PROGRESS, n_workers=N_WORKERS, initial_features=(init_train, init_test))
        preds = res['model'].predict(res['test_features'])
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        elapsed = time.time()-t0
        print(f"PATX: RMSE={rmse:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
        results.append({'subject_id': subject_id, 'approach': 'PATX', 'score': rmse, 'processing_time': elapsed, 'n_features': len(res['patterns'])})
        
        os.makedirs('../json_files/azt1d', exist_ok=True)
        pattern_data = []
        for i, p in enumerate(res['patterns']):
            p_dict = {k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in p.items() if k != 'pattern'}
            p_dict['pattern_id'] = i + 1
            pattern_data.append(p_dict)
        with open(f'../json_files/azt1d/pattern_parameters_{subject_id}.json', 'w') as f:
            json.dump(pattern_data, f, indent=2)

        # TSFRESH
        train_concat = pd.concat([time_series_data[s].iloc[train_idx] for s in TIME_SERIES], axis=1)
        test_concat = pd.concat([time_series_data[s].iloc[test_idx] for s in TIME_SERIES], axis=1)
        t0 = time.time()
        test_feat, train_feat = run_tsfresh(train_concat.values, test_concat.values)
        train_feat = np.hstack([init_train, train_feat])
        test_feat = np.hstack([init_test, test_feat])
        tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42)
        model = LightGBMModelWrapper('regression')
        model.fit(tr_f, y_tr, val_f, y_val)
        preds = model.predict(test_feat)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        elapsed = time.time()-t0
        print(f"TSFRESH: RMSE={rmse:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({'subject_id': subject_id, 'approach': 'TSFRESH', 'score': rmse, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
        
        # CATCH22
        t0 = time.time()
        test_feat, train_feat = run_catch22(train_concat.values, test_concat.values)
        train_feat = np.hstack([init_train, train_feat])
        test_feat = np.hstack([init_test, test_feat])
        tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42)
        model = LightGBMModelWrapper('regression')
        model.fit(tr_f, y_tr, val_f, y_val)
        preds = model.predict(test_feat)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        elapsed = time.time()-t0
        print(f"CATCH22: RMSE={rmse:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({'subject_id': subject_id, 'approach': 'CATCH22', 'score': rmse, 'processing_time': elapsed, 'n_features': train_feat.shape[1]})
        
        # CNN
        t0 = time.time()
        train_concat_with_init = np.hstack([init_train, train_concat.values])
        test_concat_with_init = np.hstack([init_test, test_concat.values])
        preds = run_cnn(train_concat_with_init, y_train.values, test_concat_with_init, task_type='regression', metric='rmse', epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        elapsed = time.time()-t0
        print(f"CNN: RMSE={rmse:.4f}, Time={elapsed:.1f}s, Features={train_concat_with_init.shape[1]}")
        results.append({'subject_id': subject_id, 'approach': 'CNN', 'score': rmse, 'processing_time': elapsed, 'n_features': train_concat_with_init.shape[1]})

        # Print summary for this subject
        print(f"\n{'='*60}")
        print(f"Summary for Subject {subject_id}")
        print(f"{'='*60}")
        subj_res = pd.DataFrame([r for r in results if str(r['subject_id']) == subject_id])
        for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
            app_res = subj_res[subj_res['approach'] == app]
            if len(app_res) > 0:
                r = app_res.iloc[0]
                print(f"{app:8}: RMSE={r['score']:.4f}, Time={r['processing_time']:.1f}s, Features={r['n_features']:.0f}")
        
        pd.DataFrame(results).to_csv('../results/azt1d.csv', index=False)
    except Exception as e:
        print(f"Skipping Subject {subject_id}: {e}")
        continue

pd.DataFrame(results).to_csv('../results/azt1d.csv', index=False)