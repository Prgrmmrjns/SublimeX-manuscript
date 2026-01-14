import pandas as pd
import numpy as np
import time
import re
import os
import glob
import warnings
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

# Suppress sklearn warnings about classification vs regression
warnings.filterwarnings('ignore', message='The number of unique classes is greater than 50%')

# Import baseline utils
from patx_runner import run_patx
from cnn import eval_cnn
from tsfresh_utils import eval_tsfresh
from catch22_utils import eval_catch22
from minirocket_utils import eval_minirocket
from rdst_utils import eval_rdst

# ------------------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------------------
# Datasets with >= 1000 samples (filtered by check_uea_dataset_sizes.py)
DATASETS = [
    "CharacterTrajectories",
    "FaceDetection",
    "Handwriting",
    "InsectWingbeat",
    "LSST",
    "PenDigits",
    "PhonemeSpectra",
    "SpokenArabicDigits"
]

# ------------------------------------------------------------------------------
# DATA LOADING UTILS
# ------------------------------------------------------------------------------
def load_arff_file(path):
    try:
        data, meta = arff.loadarff(path)
    except (StopIteration, ValueError, IOError) as e:
        raise ValueError(f"Failed to load ARFF file {path}: {e}")
    
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

def clean_data(data_list):
    """
    Clean NaN and inf values from a list of DataFrames/arrays.
    Replaces inf with NaN, then forward fills, backward fills, and fills remaining with 0.
    Also ensures all arrays have the same number of samples.
    Note: Time lengths can vary across channels (handled by patX core code via padding).
    """
    cleaned = []
    n_samples = None
    
    for arr in data_list:
        # Convert to numpy array if DataFrame
        if isinstance(arr, pd.DataFrame):
            arr = arr.values
        else:
            arr = np.asarray(arr, dtype=np.float32)
        
        # Ensure 2D: (n_samples, n_timepoints)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 0:
            raise ValueError(f"Unexpected 0D array in data")
        
        # Track number of samples (should be consistent across channels)
        if n_samples is None:
            n_samples = arr.shape[0]
        elif arr.shape[0] != n_samples:
            raise ValueError(f"Inconsistent number of samples: expected {n_samples}, got {arr.shape[0]}")
        
        # Replace inf with NaN
        arr = np.where(np.isinf(arr), np.nan, arr)
        
        # Handle NaNs: forward fill along time axis, backward fill, then 0
        df_temp = pd.DataFrame(arr)
        df_temp = df_temp.ffill(axis=1).bfill(axis=1).fillna(0)
        arr = df_temp.values.astype(np.float32)
        
        # Final check: replace any remaining inf/NaN with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verify no inf/NaN remain
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            # Last resort: replace with 0
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        cleaned.append(pd.DataFrame(arr))
    
    return cleaned

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
        
        X_train_list.append(d_train_feat)
        X_test_list.append(d_test_feat)
    
    if not X_train_list:
        raise ValueError(f"No data loaded for dataset {dataset_name}")
    
    # Clean NaN/inf values and ensure consistent shapes
    X_train_list = clean_data(X_train_list)
    X_test_list = clean_data(X_test_list)
        
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
            if set(approaches) >= {'PATX', 'CNN', 'TSFRESH', 'CATCH22', 'MINIROCKET', 'RDST'}: #, 'SHAPELET_TRANSFORM'
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
        
        # Try to load and process data, skip if it fails
        try:
            X_train, y_train, X_test, y_test = get_uea_data(dataset_name)
            print(f"Data shape: Train {X_train[0].shape}, Test {X_test[0].shape}, {len(X_train)} channels")
        except (ValueError, IOError, StopIteration, Exception) as e:
            print(f"Warning: Failed to load dataset {dataset_name}: {e}")
            print(f"Skipping {dataset_name}")
            continue
        
        # Process dataset - wrap in try-except to catch any evaluation errors
        try:
            # --- PATX ---
            t0 = time.time()
            res = run_patx(
                X_train,
                y_train.values,
                X_test,
                metric='accuracy',
                y_test=y_test.values
            )
            elapsed_patx = time.time() - t0
            
            results.append({
                'dataset': dataset_name, 'split': 'test', 'approach': 'PATX', 
                'score': res['score'], 'processing_time': elapsed_patx, 'n_features': res['n_features']
            })

            # --- BASELINES ---
            # Prepare data shapes: some baselines prefer 2D flattened, some prefer 3D (samples, channels, time)
            # For baselines, we need consistent time lengths across channels, so pad to max length
            def pad_channels_to_max(data_list):
                """Pad all channels to the maximum time length."""
                max_time = max(x.shape[1] for x in data_list)
                padded = []
                for x in data_list:
                    arr = x.values
                    n_samples, n_time = arr.shape
                    if n_time < max_time:
                        # Pad with last value (forward fill) along time axis
                        padding = np.tile(arr[:, -1:], (1, max_time - n_time))
                        arr = np.hstack([arr, padding])
                    padded.append(arr)
                return padded
            
            padded_train = pad_channels_to_max(X_train)
            padded_test = pad_channels_to_max(X_test)
            X_train_3d = np.stack(padded_train, axis=1)
            X_test_3d = np.stack(padded_test, axis=1)
            X_train_flat = X_train_3d.reshape(len(X_train_3d), -1)
            X_test_flat = X_test_3d.reshape(len(X_test_3d), -1)
            n_classes = len(np.unique(y_train))
            
            # CNN
            acc_cnn, elapsed_cnn, n_feat_cnn = eval_cnn(
                X_train_3d, X_test_3d, y_train.values, y_test.values, 
                task_type='classification', metric='accuracy', num_classes=n_classes
            )
            results.append({
                'dataset': dataset_name, 'split': 'test', 'approach': 'CNN', 
                'score': acc_cnn, 'processing_time': elapsed_cnn, 'n_features': n_feat_cnn
            })
            
            # TSFRESH
            acc_ts, elapsed_ts, n_feat_ts = eval_tsfresh(
                X_train_flat, X_test_flat, y_train.values, y_test.values, 
                metric='accuracy', n_classes=n_classes
            )
            results.append({
                'dataset': dataset_name, 'split': 'test', 'approach': 'TSFRESH', 
                'score': acc_ts, 'processing_time': elapsed_ts, 'n_features': n_feat_ts
            })
            
            # CATCH22
            acc_c22, elapsed_c22, n_feat_c22 = eval_catch22(
                X_train_flat, X_test_flat, y_train.values, y_test.values, 
                metric='accuracy', n_classes=n_classes
            )
            results.append({
                'dataset': dataset_name, 'split': 'test', 'approach': 'CATCH22', 
                'score': acc_c22, 'processing_time': elapsed_c22, 'n_features': n_feat_c22
            })
            
            # MINIROCKET
            acc_mr, elapsed_mr, n_feat_mr = eval_minirocket(
                X_train_3d, X_test_3d, y_train.values, y_test.values, 
                task_type='classification', metric='accuracy', num_classes=n_classes
            )
            results.append({
                'dataset': dataset_name, 'split': 'test', 'approach': 'MINIROCKET', 
                'score': acc_mr, 'processing_time': elapsed_mr, 'n_features': n_feat_mr
            })
            
            # RDST
            acc_rs, elapsed_rs, n_feat_rs = eval_rdst(
                X_train_3d, X_test_3d, y_train.values, y_test.values, 
                metric='accuracy', n_classes=n_classes
            )
            results.append({
                'dataset': dataset_name, 'split': 'test', 'approach': 'RDST', 
                'score': acc_rs, 'processing_time': elapsed_rs, 'n_features': n_feat_rs
            })
            
            # Print summary for this dataset
            ds_results = [r for r in results if r['dataset'] == dataset_name]
            if ds_results:
                print(f"Results for {dataset_name}:")
                for r in ds_results:
                    print(f"  {r['approach']:8}: Acc={r['score']:.4f}, Time={r['processing_time']:.1f}s")
            
            # Save results after each dataset
            os.makedirs('../results', exist_ok=True)
            pd.DataFrame(results).to_csv(results_file, index=False)
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            print(f"Skipping {dataset_name} and continuing with next dataset")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
