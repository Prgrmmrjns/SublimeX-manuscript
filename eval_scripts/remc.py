import pandas as pd
import os
import time
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from patx import PatternExtractor, get_model
from params import *
from tsfresh_utils import run_tsfresh
from cnn import run_cnn

warnings.filterwarnings('ignore')

def load_remc_data(cell_line):
    """Load preprocessed REMC data for a specific cell line."""
    data_file = f"../processed_datasets/remc/{cell_line}.parquet"
    
    if not os.path.exists(data_file):
        print(f"{cell_line}: Processed data not found - {data_file}")
        return None
    
    df = pd.read_parquet(data_file)
    feature_cols = [col for col in df.columns if col != 'target']
    y = df['target'].values
    
    # Separate the multiple time series
    TIME_SERIES_IDENTIFIERS = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
    X_list = []
    
    for series_id in TIME_SERIES_IDENTIFIERS:
        series_cols = [col for col in feature_cols if col.startswith(f"{series_id}_")]
        if series_cols:
            series_cols.sort(key=lambda x: int(x.split('_')[1]))  # Sort by time point
            X_series = df[series_cols].values
            X_list.append(X_series)
    
    # Combined X for CV splitting and CNN/TSFRESH (which expect single matrix)
    X_combined = df[feature_cols].values
    
    return {'X_list': X_list, 'y': y, 'X': X_combined}

# Get cell lines from processed data
processed_files = [f for f in os.listdir('../processed_datasets/remc') if f.endswith('.parquet')]
cell_lines = [f.replace('.parquet', '') for f in processed_files]
cell_lines.sort()

results_file = '../results/remc.csv'
results = []
processed_cell_lines = set()

if os.path.exists(results_file):
    existing_df = pd.read_csv(results_file)
    results = existing_df.to_dict('records')
    processed_cell_lines = set(existing_df['cell_line'].unique())
    print(f"Loaded existing results for {len(processed_cell_lines)} cell lines: {sorted(processed_cell_lines)}")

for cell_line in cell_lines[:2]:
    if cell_line in processed_cell_lines:
        continue
    print(f"Processing {cell_line}")
    
    data_dict = load_remc_data(cell_line)
    if data_dict is None:
        print(f"{cell_line}: Skipped (no processed data)")
        continue
    
    X_list = data_dict['X_list']
    y = data_dict['y']
    X_combined = data_dict['X']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for approach in ['PATX', 'TSFRESH', 'CNN']:
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y)):
            y_train, y_val = y[train_idx], y[val_idx]
            
            t0 = time.time()
            
            if approach == 'PATX':
                X_train_list = [X_s[train_idx] for X_s in X_list]
                X_val_list = [X_s[val_idx] for X_s in X_list]
                
                model = get_model('lightgbm', 'classification', 'REMC')
                optimizer = PatternExtractor(X_train_list, y_train, model=model, max_n_trials=MAX_N_TRIALS, 
                                           show_progress=SHOW_PROGRESS, n_jobs=N_JOBS, 
                                           dataset='REMC', multiple_series=True, 
                                           X_test=X_val_list, polynomial_degree=POLYNOMIAL_DEGREE, 
                                           metric='auc', val_size=VAL_SIZE, initial_features=None)
                result = optimizer.feature_extraction()
                if fold == 0:
                    optimizer.save_parameters_to_json(f'../json_files/REMC/{cell_line}')
                model = result['model']
                test_preds = model.predict_proba_positive(result['X_test'])
                n_features = len(result['patterns'])
                
            elif approach == 'TSFRESH':
                X_train, X_val = X_combined[train_idx], X_combined[val_idx]
                model = get_model('lightgbm', 'classification', 'REMC')
                val_f, X_tr, X_v, y_tr, y_v, _ = run_tsfresh(X_train, y_train, X_val, task_type='classification', val_size=VAL_SIZE, n_jobs=TSFRESH_N_JOBS)
                model.train(X_tr, y_tr, X_v, y_v)
                test_preds = model.predict_proba_positive(val_f)
                n_features = X_tr.shape[1]
                
            elif approach == 'CNN':
                X_train, X_val = X_combined[train_idx], X_combined[val_idx]
                res = run_cnn(X_train, y_train, X_val, task_type='classification', metric='auc', val_size=VAL_SIZE, num_classes=2, epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
                test_preds = res['test_predictions']
                n_features = X_train.shape[1]
            
            t1 = time.time()
            score = roc_auc_score(y_val, test_preds)
            
            result = {
                'cell_line': cell_line,
                'approach': approach,
                'fold': fold + 1,
                'score': float(score),
                'processing_time': float(t1 - t0),
                'n_features': int(n_features)
            }
            
            print(f"{cell_line} {approach} fold {fold+1}: {score:.4f}")
            results.append(result)
    
    df_all = pd.DataFrame(results)
    df_all.to_csv(results_file, index=False)
