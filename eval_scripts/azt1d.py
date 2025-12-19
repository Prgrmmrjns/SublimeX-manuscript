import pandas as pd
import numpy as np
import os
import time
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from patx_runner import *
from tsfresh_utils import eval_tsfresh
from catch22_utils import eval_catch22
from cnn import eval_cnn
warnings.filterwarnings('ignore')

TIME_SERIES = ['CGM', 'Insulin', 'Carbs']
TEST_SIZE = 0.2

def load_azt1d_data(subject_id):
    df = pd.read_parquet(f"../processed_datasets/azt1d/subject_{subject_id}.parquet")
    ts_data = {s: df[[c for c in df.columns if c.startswith(f'{s}_') and c != 'CGM_current']] for s in TIME_SERIES}
    return ts_data, df['target'], df['CGM_current']

subject_ids = sorted([f.replace('subject_', '').replace('.parquet', '') for f in os.listdir("../processed_datasets/azt1d/") if f.endswith('.parquet')])

results, done = [], set()
if os.path.exists('../results/azt1d.csv'):
    existing = pd.read_csv('../results/azt1d.csv')
    results = existing.to_dict('records')
    done = set(zip(existing['subject_id'].astype(str), existing['approach']))

print("Running performance comparison on the AZT1D dataset")
for subject_id in subject_ids:
    subject_approaches_done = {(subject_id, approach) for approach in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']}
    if subject_approaches_done.issubset(done):
        print(f"Skipping Subject {subject_id} (already complete)")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing Subject {subject_id}")
    print(f"{'='*60}")
    time_series_data, y, cgm_current = load_azt1d_data(subject_id)
    
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=42)
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    input_train = [time_series_data[s].iloc[train_idx] for s in TIME_SERIES]
    input_test = [time_series_data[s].iloc[test_idx] for s in TIME_SERIES]
    init_train = cgm_current.iloc[train_idx].values.reshape(-1, 1)
    init_test = cgm_current.iloc[test_idx].values.reshape(-1, 1)
    train_concat = pd.concat([time_series_data[s].iloc[train_idx] for s in TIME_SERIES], axis=1)
    test_concat = pd.concat([time_series_data[s].iloc[test_idx] for s in TIME_SERIES], axis=1)

    t0 = time.time()
    res = run_patx(input_train, y_train, input_test, metric='rmse', initial_features=(init_train, init_test))
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

    rmse, elapsed, n_feat = eval_tsfresh(train_concat.values, test_concat.values, y_train.values, y_test.values, metric='rmse', val_size=VAL_SIZE, initial_train=init_train, initial_test=init_test)
    print(f"TSFRESH: RMSE={rmse:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'subject_id': subject_id, 'approach': 'TSFRESH', 'score': rmse, 'processing_time': elapsed, 'n_features': n_feat})
    
    rmse, elapsed, n_feat = eval_catch22(train_concat.values, test_concat.values, y_train.values, y_test.values, metric='rmse', initial_train=init_train, initial_test=init_test)
    print(f"CATCH22: RMSE={rmse:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'subject_id': subject_id, 'approach': 'CATCH22', 'score': rmse, 'processing_time': elapsed, 'n_features': n_feat})
    
    train_concat_with_init = np.hstack([init_train, train_concat.values])
    test_concat_with_init = np.hstack([init_test, test_concat.values])
    rmse, elapsed, n_feat = eval_cnn(train_concat_with_init, test_concat_with_init, y_train.values, y_test.values, task_type='regression', metric='rmse')
    print(f"CNN: RMSE={rmse:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({'subject_id': subject_id, 'approach': 'CNN', 'score': rmse, 'processing_time': elapsed, 'n_features': n_feat})

    # Print summary for this subject
    print(f"\n{'='*60}")
    print(f"Summary for Subject {subject_id}")
    print(f"{'='*60}")
    subj_res = pd.DataFrame([r for r in results if str(r['subject_id']) == subject_id])
    for app in ['PATX', 'TSFRESH', 'CATCH22', 'CNN']:
        app_res = subj_res[subj_res['approach'] == app]
        r = app_res.iloc[0]
        print(f"{app:8}: RMSE={r['score']:.4f}, Time={r['processing_time']:.1f}s, Features={r['n_features']:.0f}")

    pd.DataFrame(results).to_csv('../results/azt1d.csv', index=False)

pd.DataFrame(results).to_csv('../results/azt1d.csv', index=False)