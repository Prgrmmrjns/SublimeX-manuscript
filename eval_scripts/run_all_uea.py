import pandas as pd
import numpy as np
import time
import re
import os
import json
import glob
from scipy.io import arff
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import baseline utils
from patx_runner import run_patx
from cnn import eval_cnn
from tsfresh_utils import eval_tsfresh
from catch22_utils import eval_catch22
from minirocket_utils import eval_minirocket
from rdst_utils import eval_rdst
from shapelet_transform_utils import eval_shapelet_transform

# ------------------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------------------
DATASETS = [
    "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "CharacterTrajectories", 
    "Cricket", "DuckDuckGeese", "ERing", "EigenWorms", "Epilepsy", "EthanolConcentration", 
    "FaceDetection", "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat", 
    "InsectWingbeat", "JapaneseVowels", "LSST", "Libras", "MotorImagery", "NATOPS", 
    "PEMS-SF", "PenDigits", "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", 
    "SelfRegulationSCP2", "SpokenArabicDigits", "StandWalkJump", "UWaveGestureLibrary"
]

# ------------------------------------------------------------------------------
# DATA LOADING UTILS
# ------------------------------------------------------------------------------
def load_arff_file(path):
    data, meta = arff.loadarff(path)
    
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
    return df
def bin_timeseries(df, bin_size):
    """Bin time series by averaging every bin_size consecutive points."""
    n_samples, n_timepoints = df.shape
    n_bins = n_timepoints // bin_size
    
    # Truncate to multiple of bin_size
    truncated = df.iloc[:, :n_bins * bin_size]
    
    # Reshape and average
    reshaped = truncated.values.reshape(n_samples, n_bins, bin_size)
    binned = reshaped.mean(axis=2)
    
    return pd.DataFrame(binned, index=df.index)

def get_uea_data(dataset_name):
    base_path = f"../Multivariate_arff/{dataset_name}"
    train_files = glob.glob(f"{base_path}/*_TRAIN.arff")
    test_files = glob.glob(f"{base_path}/*_TEST.arff")
    
    if not train_files or not test_files:
        raise ValueError(f"No ARFF files found for dataset {dataset_name}")
    
    if len(train_files) != len(test_files):
        raise ValueError(f"Mismatch in train/test files: {len(train_files)} train, {len(test_files)} test")
    
    def extract_dim(fname):
        match = re.search(r'Dimension(\d+)_', fname)
        return int(match.group(1)) if match else 0
        
    train_files.sort(key=extract_dim)
    test_files.sort(key=extract_dim)
    
    X_train_list = []
    X_test_list = []
    y_train = None
    y_test = None
    
    for i, (tr_f, te_f) in enumerate(zip(train_files, test_files)):
        d_train = load_arff_file(tr_f)
        d_test = load_arff_file(te_f)
        
        if i == 0:
            if 'target' in d_train.columns:
                target_col = 'target'
            elif 'class' in d_train.columns:
                target_col = 'class'
            else:
                target_col = d_train.columns[-1]
            
            y_train_raw = d_train[target_col]
            y_test_raw = d_test[target_col]
                
            le = LabelEncoder()
            # Fit on both to ensure all classes are captured
            all_labels = pd.concat([y_train_raw, y_test_raw])
            le.fit(all_labels)
            y_train = pd.Series(le.transform(y_train_raw))
            y_test = pd.Series(le.transform(y_test_raw))
        
        # Drop target column before converting to float
        d_train_feat = d_train.drop(columns=[target_col])
        d_test_feat = d_test.drop(columns=[target_col])
        
        # Bin EigenWorms dataset (100:1 ratio)
        if dataset_name == 'EigenWorms':
            d_train_feat = bin_timeseries(d_train_feat, bin_size=10)
            d_test_feat = bin_timeseries(d_test_feat, bin_size=10)
        
        X_train_list.append(d_train_feat.astype(np.float32))
        X_test_list.append(d_test_feat.astype(np.float32))
    
    if not X_train_list:
        raise ValueError(f"No data loaded for dataset {dataset_name}")
        
    return X_train_list, y_train, X_test_list, y_test

# ------------------------------------------------------------------------------
# CONFIG & SETUP
# ------------------------------------------------------------------------------
# Default Full Evaluation Settings
CNN_EPOCHS = 300
SAVE_RESULTS = True

# Check for existing results to resume
results_file = "../results/uea_benchmark_full.csv"
results = []
done = set()

if os.path.exists(results_file):
    try:
        existing = pd.read_csv(results_file)
        results = existing.to_dict('records')
        # Check completed datasets
        done_ds = existing.groupby('dataset')['approach'].unique()
        
        for ds_name, approaches in done_ds.items():
            if set(approaches) >= {'PATX', 'CNN', 'TSFRESH', 'CATCH22', 'MINIROCKET', 'RDST', 'SHAPELET_TRANSFORM'}:
                done.add(ds_name)
    except Exception as e:
        print(f"Warning: Failed to load existing results: {e}. Starting fresh.")
        results = []
        done = set()

def main():
    # ------------------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------------------
    for dataset_name in DATASETS:
        print(f"Processing {dataset_name}")
        
        # Skip if already complete
        if dataset_name in done:
            print(f"Skipping {dataset_name} (already complete)")
            continue
        
        try:
            # Wrap ENTIRE dataset processing in try-except to prevent crashes from stopping the whole loop
            try:
                X_train, y_train, X_test, y_test = get_uea_data(dataset_name)
            except Exception as e:
                print(f"ERROR: Failed to load {dataset_name}: {e}")
                print(f"Skipping {dataset_name}")
                continue

            print(f"Data shape: Train {X_train[0].shape}, Test {X_test[0].shape}, {len(X_train)} channels")
            
            all_patterns = {}
            
            # --- PATX ---
            try:
                t0 = time.time()
                res = run_patx(
                    X_train,
                    y_train.values,
                    X_test,
                    metric='accuracy'
                )
                all_patterns['test_split'] = res['patterns']
                preds = res['model'].predict(res['test_features'])
                acc_patx = accuracy_score(y_test.values, preds)
                elapsed_patx = time.time() - t0
                
                results.append({
                    'dataset': dataset_name, 'split': 'test', 'approach': 'PATX', 
                    'score': acc_patx, 'processing_time': elapsed_patx, 'n_features': len(res['patterns'])
                })
            except Exception as e:
                 print(f"Error running PATX on {dataset_name}: {e}")

            # --- BASELINES ---
            # Prepare data shapes: some baselines prefer 2D flattened, some prefer 3D (samples, channels, time)
            X_train_3d = np.stack([x.values for x in X_train], axis=1)
            X_test_3d = np.stack([x.values for x in X_test], axis=1)
            X_train_flat = X_train_3d.reshape(len(X_train_3d), -1)
            X_test_flat = X_test_3d.reshape(len(X_test_3d), -1)
            n_classes = len(np.unique(y_train))
            
            # CNN
            try:
                acc_cnn, elapsed_cnn, n_feat_cnn = eval_cnn(
                    X_train_3d, X_test_3d, y_train.values, y_test.values, 
                    task_type='classification', metric='accuracy', num_classes=n_classes
                )
                results.append({
                    'dataset': dataset_name, 'split': 'test', 'approach': 'CNN', 
                    'score': acc_cnn, 'processing_time': elapsed_cnn, 'n_features': n_feat_cnn
                })
            except Exception as e:
                print(f"Error running CNN on {dataset_name}: {e}")
            
            # TSFRESH
            try:
                acc_ts, elapsed_ts, n_feat_ts = eval_tsfresh(
                    X_train_flat, X_test_flat, y_train.values, y_test.values, 
                    metric='accuracy', n_classes=n_classes
                )
                results.append({
                    'dataset': dataset_name, 'split': 'test', 'approach': 'TSFRESH', 
                    'score': acc_ts, 'processing_time': elapsed_ts, 'n_features': n_feat_ts
                })
            except Exception as e:
                print(f"Error running TSFRESH on {dataset_name}: {e}")
            
            # CATCH22
            try:
                acc_c22, elapsed_c22, n_feat_c22 = eval_catch22(
                    X_train_flat, X_test_flat, y_train.values, y_test.values, 
                    metric='accuracy', n_classes=n_classes
                )
                results.append({
                    'dataset': dataset_name, 'split': 'test', 'approach': 'CATCH22', 
                    'score': acc_c22, 'processing_time': elapsed_c22, 'n_features': n_feat_c22
                })
            except Exception as e:
                 print(f"Error running CATCH22 on {dataset_name}: {e}")
            
            # MINIROCKET
            try:
                acc_mr, elapsed_mr, n_feat_mr = eval_minirocket(
                    X_train_3d, X_test_3d, y_train.values, y_test.values, 
                    metric='accuracy', n_classes=n_classes
                )
                results.append({
                    'dataset': dataset_name, 'split': 'test', 'approach': 'MINIROCKET', 
                    'score': acc_mr, 'processing_time': elapsed_mr, 'n_features': n_feat_mr
                })
            except Exception as e:
                print(f"Error running MINIROCKET on {dataset_name}: {e}")
            
            # RDST
            try:
                acc_rs, elapsed_rs, n_feat_rs = eval_rdst(
                    X_train_3d, X_test_3d, y_train.values, y_test.values, 
                    metric='accuracy', n_classes=n_classes
                )
                results.append({
                    'dataset': dataset_name, 'split': 'test', 'approach': 'RDST', 
                    'score': acc_rs, 'processing_time': elapsed_rs, 'n_features': n_feat_rs
                })
            except Exception as e:
                print(f"Error running RDST on {dataset_name}: {e}")
            
            # SHAPELET_TRANSFORM
            try:
                acc_st, elapsed_st, n_feat_st = eval_shapelet_transform(
                    X_train_3d, X_test_3d, y_train.values, y_test.values, 
                    metric='accuracy', n_classes=n_classes
                )
                results.append({
                    'dataset': dataset_name, 'split': 'test', 'approach': 'SHAPELET_TRANSFORM', 
                    'score': acc_st, 'processing_time': elapsed_st, 'n_features': n_feat_st
                })
            except Exception as e:
                print(f"Error running SHAPELET_TRANSFORM on {dataset_name}: {e}")
            
            # Print summary for this dataset
            ds_results = [r for r in results if r['dataset'] == dataset_name]
            if ds_results:
                print(f"Results for {dataset_name}:")
                for r in ds_results:
                    print(f"  {r['approach']:8}: Acc={r['score']:.4f}, Time={r['processing_time']:.1f}s")
            
            # Save results and patterns after each dataset
            os.makedirs('../results', exist_ok=True)
            pd.DataFrame(results).to_csv(results_file, index=False)

            json_dir = "../json_files" / dataset_name
            os.makedirs(json_dir, exist_ok=True)
            
            serializable_all_patterns = {}
            for split_key, patterns in all_patterns.items():
                serializable_patterns = []
                for pattern in patterns:
                    serializable_pattern = {}
                    for key, value in pattern.items():
                        if key == 'pattern': continue
                        if isinstance(value, list):
                            serializable_pattern[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                        elif isinstance(value, np.ndarray):
                            serializable_pattern[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            serializable_pattern[key] = value.item()
                        else:
                            serializable_pattern[key] = value
                    serializable_patterns.append(serializable_pattern)
                serializable_all_patterns[split_key] = serializable_patterns
                
            with open(f'../json_files/{dataset_name}/pattern_parameters.json', 'w') as f:
                json.dump(serializable_all_patterns, f, indent=2)

        except Exception as e:
            print(f"CRITICAL ERROR processing {dataset_name}: {e}")
            # Try to save whatever we have so far
            os.makedirs('../results', exist_ok=True)
            pd.DataFrame(results).to_csv(results_file, index=False)
            continue

if __name__ == "__main__":
    main()
