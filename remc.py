import pandas as pd
import os
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from patx import PatternOptimizer
from params import *
from tsfresh_utils import run_tsfresh_baseline
from cnn import run_cnn

warnings.filterwarnings('ignore')

# Get cell lines
cell_line_files = os.listdir('remc_celllines')
cell_lines = [f.replace('remc_', '').replace('.parquet', '') for f in cell_line_files if f.startswith('remc_') and f.endswith('.parquet')]
cell_lines.sort()

# Load existing results from single CSV file
results_file = 'results/remc.csv'
results = []
processed_cell_lines = set()

if os.path.exists(results_file):
    existing_df = pd.read_csv(results_file)
    results = existing_df.to_dict('records')
    processed_cell_lines = set(existing_df['cell_line'].unique())
    print(f"Loaded existing results for {len(processed_cell_lines)} cell lines: {sorted(processed_cell_lines)}")

for cell_line in cell_lines:
    if cell_line in processed_cell_lines:
        continue
    print(f"Processing {cell_line}")
    
    # Load data
    data = pd.read_parquet(f'remc_celllines/remc_{cell_line}.parquet')
    feature_cols = [col for col in data.columns if col != 'target']
    y = data['target'].values
    X_train, X_test, y_train, y_test = train_test_split(data[feature_cols].values, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    X_train_list = []
    X_test_list = []
    for series_id in TIME_SERIES_IDENTIFIERS:
        series_cols = [col for col in feature_cols if col.startswith(f"{series_id}_")]
        col_indices = [feature_cols.index(col) for col in series_cols]
        X_train_series = X_train[:, col_indices]
        X_test_series = X_test[:, col_indices]
        X_train_list.append(X_train_series)
        X_test_list.append(X_test_series)
    X_train_df = pd.DataFrame(X_train_list[0])
    X_test_df = pd.DataFrame(X_test_list[0])
    input_data, test_data = X_train_list, X_test_list
    
    # PATX
    model = get_model(MODEL, TASK_TYPE, DATASET)
    optimizer = PatternOptimizer(input_data, y_train, model=model, max_n_trials=MAX_N_TRIALS, show_progress=SHOW_PROGRESS, test_size=VAL_SIZE, n_jobs=N_JOBS, dataset=DATASET, multiple_series=len(TIME_SERIES_IDENTIFIERS) > 0, X_test_data=test_data, polynomial_degree=POLYNOMIAL_DEGREE, metric=METRIC, val_size=VAL_SIZE, initial_features=None)
    t0 = time.time()
    result = optimizer.feature_extraction()
    t1 = time.time()
    optimizer.save_parameters_to_json(f'REMC/{cell_line}')
    patterns = result['patterns']
    X_train, X_val = result['X_train'], result['X_val']
    y_train_split, y_val = result['y_train'], result['y_val']
    X_test = result['X_test']
    m = result['model']
    train_preds = m.predict_proba_positive(X_train)
    val_preds = m.predict_proba_positive(X_val)
    test_preds = m.predict_proba_positive(X_test)
    train_score = roc_auc_score(y_train_split, train_preds)
    val_score = roc_auc_score(y_val, val_preds)
    test_score = roc_auc_score(y_test, test_preds)
    r1 = {'cell_line': cell_line, 'approach': 'PATX', 'train_score': float(train_score), 'val_score': float(val_score), 'test_score': float(test_score), 'n_features': len(patterns), 'processing_time': t1 - t0}
    print(f"PatX Test score: {r1['test_score']:.4f}, processing time: {r1['processing_time']:.2f} seconds, Amount of features: {r1['n_features']}")
    
    # TSFRESH
    Xtr, Xte, ytr, yte = X_train_df, X_test_df, y_train, y_test
    m = get_model(MODEL, TASK_TYPE, DATASET)
    te_f, X_tr, X_val, y_tr, y_val, dt = run_tsfresh_baseline(Xtr.values, ytr, Xte.values, task_type='classification', val_size=VAL_SIZE, n_jobs=1)
    m.train(X_tr, y_tr, X_val, y_val)
    tr_p = m.predict_proba_positive(X_tr)
    val_p = m.predict_proba_positive(X_val)
    te_p = m.predict_proba_positive(te_f)
    tr_s = roc_auc_score(y_tr, tr_p); val_s = roc_auc_score(y_val, val_p); te_s = roc_auc_score(yte, te_p)
    r2 = {'cell_line': cell_line, 'approach': 'TSFRESH', 'train_score': float(tr_s), 'val_score': float(val_s), 'test_score': float(te_s), 'n_features': int(X_tr.shape[1]), 'processing_time': dt}
    print(f"TSFRESH Test score: {r2['test_score']:.4f}, processing time: {r2['processing_time']:.2f} seconds")
    
    # CNN
    X, y = X_train_df.values, y_train
    Xt, yt = X_test_df.values, y_test
    res = run_cnn(X, y, Xt, task_type='classification', metric='auc', val_size=VAL_SIZE, num_classes=2)
    te_p = res['test_predictions']
    te_s = roc_auc_score(yt, te_p)
    r3 = {'cell_line': cell_line, 'approach': 'CNN', 'train_score': res['train_score'], 'val_score': res['val_score'], 'test_score': float(te_s), 'n_features': int(X.shape[1]), 'processing_time': res['processing_time']}
    print(f"CNN Test score: {r3['test_score']:.4f}, processing time: {r3['processing_time']:.2f} seconds")
    
    # Add to overall results
    results.extend([r1, r2, r3])
    
    # Save updated results to single CSV file after each cell line
    df_all = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df_all.to_csv(results_file, index=False)
