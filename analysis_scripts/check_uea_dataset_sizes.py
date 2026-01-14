"""
Script to check sample sizes for UEA datasets.
Filters datasets to only include those with >= 1000 samples.
"""
import pandas as pd
import numpy as np
import glob
from scipy.io import arff

# All UEA datasets
ALL_DATASETS = [
    "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "CharacterTrajectories", 
    "Cricket", "DuckDuckGeese", "ERing", "EigenWorms", "Epilepsy", "EthanolConcentration", 
    "FaceDetection", "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat", 
    "InsectWingbeat", "JapaneseVowels", "LSST", "Libras", "MotorImagery", "NATOPS", 
    "PEMS-SF", "PenDigits", "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", 
    "SelfRegulationSCP2", "SpokenArabicDigits", "StandWalkJump", "UWaveGestureLibrary"
]

def load_arff_file(path):
    """Load ARFF file and return DataFrame."""
    try:
        data, meta = arff.loadarff(path)
    except (StopIteration, ValueError, IOError) as e:
        raise ValueError(f"Failed to load ARFF file {path}: {e}")
    
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
    return df

def get_dataset_sample_count(dataset_name):
    """Get total sample count (train + test) for a dataset."""
    base_path = f"../Multivariate_arff/{dataset_name}"
    train_files = glob.glob(f"{base_path}/*_TRAIN.arff")
    test_files = glob.glob(f"{base_path}/*_TEST.arff")
    
    if not train_files or not test_files:
        return None, "No ARFF files found"
    
    try:
        # Load first train file to get sample count
        d_train = load_arff_file(train_files[0])
        d_test = load_arff_file(test_files[0])
        
        n_train = len(d_train)
        n_test = len(d_test)
        total = n_train + n_test
        
        return total, None
    except Exception as e:
        return None, str(e)

def main():
    """Check all datasets and filter by sample count >= 1000."""
    results = []
    valid_datasets = []
    
    print("Checking UEA dataset sample sizes...")
    print("=" * 60)
    
    for dataset_name in ALL_DATASETS:
        total_samples, error = get_dataset_sample_count(dataset_name)
        
        if total_samples is None:
            status = f"ERROR: {error}" if error else "NOT FOUND"
            print(f"{dataset_name:30} {status}")
            results.append({
                'dataset': dataset_name,
                'total_samples': None,
                'status': status,
                'include': False
            })
        else:
            include = total_samples >= 1000
            status = "✓ INCLUDED" if include else "✗ EXCLUDED (< 1000)"
            print(f"{dataset_name:30} {total_samples:6} samples  {status}")
            
            if include:
                valid_datasets.append(dataset_name)
            
            results.append({
                'dataset': dataset_name,
                'total_samples': total_samples,
                'status': 'valid' if include else 'too_small',
                'include': include
            })
    
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total datasets checked: {len(ALL_DATASETS)}")
    print(f"  Valid datasets (>= 1000 samples): {len(valid_datasets)}")
    print(f"  Excluded datasets: {len(ALL_DATASETS) - len(valid_datasets)}")
    
    print(f"\nValid datasets to include in DATASETS list:")
    print("DATASETS = [")
    for i, ds in enumerate(valid_datasets):
        comma = "," if i < len(valid_datasets) - 1 else ""
        print(f'    "{ds}"{comma}')
    print("]")
    
    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("../results/uea_dataset_sizes.csv", index=False)
    print(f"\nResults saved to ../results/uea_dataset_sizes.csv")

if __name__ == "__main__":
    main()
