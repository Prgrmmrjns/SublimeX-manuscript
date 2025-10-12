import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from patx import PatternExtractor, get_model
import time

cell_line = 'E003'
data_file = f"../processed_datasets/remc/{cell_line}.parquet"
df = pd.read_parquet(data_file)

feature_cols = [col for col in df.columns if col != 'target']
y = df['target'].values

TIME_SERIES_IDENTIFIERS = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
X_list = []

for series_id in TIME_SERIES_IDENTIFIERS:
    series_cols = [col for col in feature_cols if col.startswith(f"{series_id}_")]
    if series_cols:
        series_cols.sort(key=lambda x: int(x.split('_')[1]))
        X_series = df[series_cols].values
        X_list.append(X_series)

X_combined = df[feature_cols].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
max_patterns = 10

fold_results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_combined, y)):
    print(f"\n{'='*60}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*60}")
    
    y_train, y_val = y[train_idx], y[test_idx]
    X_train_list = [X_s[train_idx] for X_s in X_list]
    X_val_list = [X_s[test_idx] for X_s in X_list]
    
    model = get_model('lightgbm', 'classification', 'REMC')
    extractor = PatternExtractor(
        X_train_list, y_train, 
        model=model, 
        max_n_trials=50,
        show_progress=True, 
        n_jobs=-1,
        X_test=X_val_list,
        polynomial_degree=3, 
        metric='auc', 
        val_size=0.2
    )
    
    result = extractor.feature_extraction()
    patterns = result['patterns']
    n_patterns_found = len(patterns)
    
    print(f"Found {n_patterns_found} patterns total")
    
    X_tr = result['X_train']
    X_v = result['X_test']
    y_tr, _ = train_test_split(y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    for n_features in range(1, min(max_patterns + 1, n_patterns_found + 1)):
        train_feat = X_tr[:, :n_features]
        val_feat = X_v[:, :n_features]
        
        temp_model = get_model('lightgbm', 'classification', 'REMC')
        temp_model.train(train_feat, y_tr, None, None)
        
        val_preds = temp_model.predict_proba_positive(val_feat)
        val_auc = roc_auc_score(y_val, val_preds)
        
        fold_results.append({
            'fold': fold + 1,
            'n_features': n_features,
            'test_auc': val_auc
        })
        
        print(f"  {n_features} features: AUC = {val_auc:.4f}")

results_df = pd.DataFrame(fold_results)
results_df.to_csv('../results/cumulative_features_e003.csv', index=False)

summary = results_df.groupby('n_features')['test_auc'].agg(['mean', 'std']).reset_index()
print(f"\n{'='*60}")
print("Summary:")
print(summary)

plt.figure(figsize=(10, 6))
plt.errorbar(summary['n_features'], summary['mean'], yerr=summary['std'], 
             marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Test AUC', fontsize=12)
plt.title('REMC E003: Cumulative Feature Performance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../manuscript/images/cumulative_features_e003.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/images/cumulative_features_e003.pdf', bbox_inches='tight')
print("\nPlots saved!")

