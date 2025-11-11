import pandas as pd
import numpy as np
import time
import warnings
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from core import feature_extraction
from models import LightGBMModelWrapper
from params import *
warnings.filterwarnings('ignore')

class LogRegWrapper:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    def clone(self):
        return LogRegWrapper()

class MLPWrapper:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    def clone(self):
        return MLPWrapper()

def load_mitbih_data():
    data = pd.read_csv("../processed_datasets/mitbih_processed.csv")
    return data.drop('target', axis=1), data['target']

def save_patterns_to_json(patterns, fold, variant):
    os.makedirs(f'../json_files/mitbih_ablation/{variant}', exist_ok=True)
    pattern_data = [{**{k: v.tolist() if isinstance(v, np.ndarray) else 
                        float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in p.items()}, 'pattern_id': i+1} 
                    for i, p in enumerate(patterns)]
    with open(f'../json_files/mitbih_ablation/{variant}/pattern_parameters_fold{fold}.json', 'w') as f:
        json.dump(pattern_data, f, indent=2)

def run_ablation_variant(variant_name, input_series, y, kfold_indices, n_control_points, n_trials, model_class=None):
    print(f"\n{'='*60}\nRUNNING: {variant_name}\n{'='*60}")
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold_indices):
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train, X_test = input_series.iloc[train_idx].astype(np.float32), input_series.iloc[val_idx].astype(np.float32)
        
        t0 = time.time()
        model = (model_class() if model_class else LightGBMModelWrapper('classification', n_classes=len(np.unique(y_train))))
        res = feature_extraction([X_train], y_train.values, [X_test], metric='accuracy', 
                                n_trials=n_trials, show_progress=SHOW_PROGRESS, 
                                n_control_points=n_control_points, model=model)
        
        save_patterns_to_json(res['patterns'], fold+1, variant_name)
        accuracy = accuracy_score(y_val.values, res['model'].predict(res['test_features']))
        processing_time = time.time() - t0
        
        results.append({'approach': variant_name, 'fold': fold, 'score': accuracy, 
                       'processing_time': processing_time, 'n_features': len(res['patterns'])})
        print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Time={processing_time:.1f}s, Features={len(res['patterns'])}")
    
    scores, times, features = [r['score'] for r in results], [r['processing_time'] for r in results], [r['n_features'] for r in results]
    avg_acc, std_acc = np.mean(scores), np.std(scores)
    avg_time, std_time = np.mean(times), np.std(times)
    avg_feat, std_feat = np.mean(features), np.std(features)
    
    print(f"\n{variant_name} Average: Accuracy={avg_acc:.4f}±{std_acc:.4f}, Time={avg_time:.1f}±{std_time:.1f}s, Features={avg_feat:.1f}±{std_feat:.1f}")
    return results, avg_acc, std_acc, avg_time, std_time, avg_feat, std_feat

print("Loading MITBIH dataset...")
input_series, y = load_mitbih_data()
kfold_indices = list(StratifiedKFold(2, shuffle=True, random_state=42).split(input_series, y))

if os.path.exists('../results/mitbih_ablation.csv'):
    existing_df = pd.read_csv('../results/mitbih_ablation.csv')
    completed_variants = set(existing_df['approach'].unique())
    print(f"Found existing results for: {', '.join(completed_variants)}")
else:
    existing_df = pd.DataFrame()
    completed_variants = set()

variants = {
    'Logistic Regression': (N_CONTROL_POINTS, 100, LogRegWrapper),
    'MLP': (N_CONTROL_POINTS, 100, MLPWrapper)
}

all_results = []
for variant_name, (n_cp, n_trials, model_cls) in variants.items():
    if variant_name in completed_variants:
        print(f"\nSkipping {variant_name} (already completed)")
        continue
    results, avg_acc, std_acc, avg_time, std_time, avg_feat, std_feat = run_ablation_variant(
        variant_name, input_series, y, kfold_indices, n_cp, n_trials, model_cls
    )
    all_results.extend(results)

if all_results:
    new_df = pd.DataFrame(all_results)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    combined_df.to_csv('../results/mitbih_ablation.csv', index=False)
    print(f"\nResults appended to ../results/mitbih_ablation.csv")
    
    summary = combined_df.groupby('approach').agg({
        'score': ['mean', 'std'],
        'processing_time': ['mean', 'std'],
        'n_features': ['mean', 'std']
    }).round(4)
    print(f"\n{'='*80}\nABLATION STUDY SUMMARY\n{'='*80}")
    print(summary)
else:
    print("\nNo new variants to run.")
