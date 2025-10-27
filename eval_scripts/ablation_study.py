import pandas as pd
import numpy as np
import time
import warnings
import os
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from core import feature_extraction
from models import LightGBMModelWrapper
from params import *
warnings.filterwarnings('ignore')

class LogisticRegressionWrapper:
    def __init__(self, random_state=42, max_iter=1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Replace infinity and NaN values with finite values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.model.predict_proba(X)
    
    def clone(self):
        return LogisticRegressionWrapper(self.random_state, self.max_iter)

def load_mitbih_data():
    data = pd.read_csv("../processed_datasets/mitbih_processed.csv")
    return data.drop('target', axis=1), data['target']

def save_patterns_to_json(patterns, fold, variant):
    os.makedirs(f'../json_files/mitbih_ablation/{variant}', exist_ok=True)
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
    
    filename = f'../json_files/mitbih_ablation/{variant}/pattern_parameters_fold{fold}.json'
    with open(filename, 'w') as f:
        json.dump(pattern_data, f, indent=2)

def run_ablation_variant(variant_name, input_series, y, kfold_indices, variant_config):
    print(f"\n{'='*60}")
    print(f"RUNNING ABLATION VARIANT: {variant_name}")
    print(f"{'='*60}")
    
    results = []
    all_patterns = {}
    
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train = input_series.iloc[train_idx].astype(np.float32)
        X_test = input_series.iloc[val_idx].astype(np.float32)
        
        t0 = time.time()
        
        # Apply variant-specific modifications
        if variant_config['no_transforms']:
            # Override core.py to only use raw data
            from core import apply_transformation
            original_apply_transformation = apply_transformation
            
            def raw_only_transformation(data, transform_type):
                return data  # Always return raw data regardless of transform_type
            
            # Monkey patch the transformation function
            import core
            core.apply_transformation = raw_only_transformation
        
        if variant_config['no_shift']:
            # Set shift tolerance to 0 for all patterns
            original_pattern_to_features = None
            from core import pattern_to_features
            
            def no_shift_pattern_to_features(*args, **kwargs):
                kwargs['shift_tolerance'] = 0.0
                return pattern_to_features(*args, **kwargs)
            
            import core
            core.pattern_to_features = no_shift_pattern_to_features
        
        if variant_config['relative_only']:
            # Force all patterns to use relative mode
            from core import pattern_to_features
            original_pattern_to_features = pattern_to_features
            
            def relative_only_pattern_to_features(*args, **kwargs):
                kwargs['use_relative'] = True
                return pattern_to_features(*args, **kwargs)
            
            import core
            core.pattern_to_features = relative_only_pattern_to_features
        
        # Run feature extraction with variant-specific model
        if variant_config['use_linear_regression']:
            model = LogisticRegressionWrapper(random_state=42, max_iter=1000)
        else:
            model = LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train)))
        
        res = feature_extraction(
            [X_train], y_train.values, [X_test], 
            metric='accuracy', 
            n_trials=variant_config.get('n_trials', N_TRIALS), 
            show_progress=SHOW_PROGRESS, 
            n_control_points=variant_config.get('n_control_points', N_CONTROL_POINTS),
            model=model
        )
        
        # Restore original functions
        if variant_config['no_transforms']:
            import core
            core.apply_transformation = original_apply_transformation
        
        if variant_config['no_shift']:
            import core
            core.pattern_to_features = pattern_to_features
        
        if variant_config['relative_only']:
            import core
            core.pattern_to_features = pattern_to_features
        
        all_patterns[f'fold_{fold+1}'] = res['patterns']
        save_patterns_to_json(res['patterns'], fold+1, variant_name)
        
        # Use the specified final model
        if variant_config['final_model'] == 'neural_network':
            # Train a simple neural network on the extracted features
            from sklearn.neural_network import MLPClassifier
            nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500)
            nn_model.fit(res['train_features'], y_train.values)
            accuracy = accuracy_score(y_val.values, nn_model.predict(res['test_features']))
        else:
            # Use the model from feature extraction
            accuracy = accuracy_score(y_val.values, res['model'].predict(res['test_features']))
        
        processing_time = time.time() - t0
        results.append({
            'approach': variant_name, 
            'fold': fold, 
            'score': accuracy, 
            'processing_time': processing_time, 
            'n_features': len(res['patterns'])
        })
        
        print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Time={processing_time:.1f}s, Features={len(res['patterns']):.0f}")
    
    # Calculate averages
    scores = [r['score'] for r in results]
    times = [r['processing_time'] for r in results]
    features = [r['n_features'] for r in results]
    
    avg_accuracy = np.mean(scores)
    avg_time = np.mean(times)
    avg_features = np.mean(features)
    std_accuracy = np.std(scores)
    std_time = np.std(times)
    std_features = np.std(features)
    
    print(f"\n{variant_name} Average: Accuracy={avg_accuracy:.4f}±{std_accuracy:.4f}, Time={avg_time:.1f}±{std_time:.1f}s, Features={avg_features:.1f}±{std_features:.1f}")
    
    return results, avg_accuracy, std_accuracy, avg_time, std_time, avg_features, std_features

# Load data
print("Loading MITBIH dataset...")
input_series, y = load_mitbih_data()
kfold_indices = list(StratifiedKFold(2, shuffle=True, random_state=42).split(input_series, y))

# Define ablation variants
variants = {
    'Baseline': {
        'no_transforms': False,
        'no_shift': False,
        'relative_only': False,
        'use_linear_regression': False,
        'final_model': 'lightgbm',
        'n_trials': 100,
        'n_control_points': N_CONTROL_POINTS
    },
    'No Transforms': {
        'no_transforms': True,
        'no_shift': False,
        'relative_only': False,
        'use_linear_regression': False,
        'final_model': 'lightgbm',
        'n_trials': 100,
        'n_control_points': N_CONTROL_POINTS
    },
    'No Shift Tolerance': {
        'no_transforms': False,
        'no_shift': True,
        'relative_only': False,
        'use_linear_regression': False,
        'final_model': 'lightgbm',
        'n_trials': 100,
        'n_control_points': N_CONTROL_POINTS
    },
    'Relative Only': {
        'no_transforms': False,
        'no_shift': False,
        'relative_only': True,
        'use_linear_regression': False,
        'final_model': 'lightgbm',
        'n_trials': 100,
        'n_control_points': N_CONTROL_POINTS
    },
    'Linear Regression': {
        'no_transforms': False,
        'no_shift': False,
        'relative_only': False,
        'use_linear_regression': True,
        'final_model': 'lightgbm',
        'n_trials': 100,
        'n_control_points': N_CONTROL_POINTS
    },
    'Neural Network Final': {
        'no_transforms': False,
        'no_shift': False,
        'relative_only': False,
        'use_linear_regression': False,
        'final_model': 'neural_network',
        'n_trials': 100,
        'n_control_points': N_CONTROL_POINTS
    }
}

# Run all variants
all_results = []
summary_results = []

for variant_name, config in variants.items():
    results, avg_acc, std_acc, avg_time, std_time, avg_feat, std_feat = run_ablation_variant(
        variant_name, input_series, y, kfold_indices, config
    )
    all_results.extend(results)
    summary_results.append({
        'variant': variant_name,
        'accuracy_mean': avg_acc,
        'accuracy_std': std_acc,
        'time_mean': avg_time,
        'time_std': std_time,
        'features_mean': avg_feat,
        'features_std': std_feat
    })

# Print summary table
print("\n" + "="*80)
print("ABLATION STUDY SUMMARY")
print("="*80)
print(f"{'Variant':<20} {'Accuracy':<15} {'Time (s)':<12} {'Features':<10}")
print("-" * 80)

for result in summary_results:
    print(f"{result['variant']:<20} {result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}  "
          f"{result['time_mean']:.1f}±{result['time_std']:.1f}    {result['features_mean']:.1f}±{result['features_std']:.1f}")

# Save detailed results
df_results = pd.DataFrame(all_results)
df_results.to_csv('../results/mitbih_ablation.csv', index=False)

# Save summary results
df_summary = pd.DataFrame(summary_results)
df_summary.to_csv('../results/mitbih_ablation_summary.csv', index=False)

print(f"\nDetailed results saved to ../results/mitbih_ablation.csv")
print(f"Summary results saved to ../results/mitbih_ablation_summary.csv")
