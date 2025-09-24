import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_paths(cell_line):
    json_file = os.path.join(PROJECT_ROOT, 'eval_scripts', 'json_files', 'REMC', cell_line, 'pattern_parameters.json')
    data_file = os.path.join(PROJECT_ROOT, 'processed_datasets', 'remc', f'{cell_line}.parquet')
    out_img = os.path.join(PROJECT_ROOT, 'manuscript', 'images', f'remc_{cell_line}_two_pattern_scatter.png')
    return json_file, data_file, out_img


def zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean()
    s = x.std() + 1e-8
    return (x - m) / s

def resample(values: np.ndarray, new_len: int) -> np.ndarray:
    if new_len == len(values):
        return values.copy()
    x_old = np.linspace(0.0, 1.0, num=len(values))
    x_new = np.linspace(0.0, 1.0, num=new_len)
    return np.interp(x_new, x_old, values)


def load_two_patterns(json_file):
    d = json.load(open(json_file, 'r'))
    p0, p1 = d['patterns'][0], d['patterns'][1]
    return (
        (int(p0['pattern_start']), int(p0['pattern_width']), int(p0.get('series_index', 0)),
         np.asarray(p0['pattern_values'], dtype=np.float32)),
        (int(p1['pattern_start']), int(p1['pattern_width']), int(p1.get('series_index', 0)),
         np.asarray(p1['pattern_values'], dtype=np.float32)),
    )

def split_series_columns(df: pd.DataFrame, num_series: int):
    feature_cols = [c for c in df.columns if c != 'target']
    n = len(feature_cols)
    if n % num_series != 0:
        return {0: feature_cols}
    T = n // num_series
    return {s: feature_cols[s * T : (s + 1) * T] for s in range(num_series)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python decision_boundary.py <cell_line>")
        print("Examples:")
        print("  python decision_boundary.py E003")
        print("  python decision_boundary.py E004")
        sys.exit(1)
    
    cell_line = sys.argv[1]
    
    json_file, data_file, out_img = get_paths(cell_line)
    
    (s0, w0, k0, v0), (s1, w1, k1, v1) = load_two_patterns(json_file)
    
    # Load REMC data
    df = pd.read_parquet(data_file)

    all_series = [int(p.get('series_index', 0)) for p in json.load(open(json_file, 'r'))['patterns']]
    num_series = max(all_series) + 1 if all_series else 1
    series_cols = split_series_columns(df, num_series)

    X0 = df[series_cols.get(k0, list(df.columns[:-1]))].to_numpy(dtype=np.float32)
    X1 = df[series_cols.get(k1, list(df.columns[:-1]))].to_numpy(dtype=np.float32)
    y = df['target'].to_numpy()

    # Per-sample z-score
    X0 = (X0 - X0.mean(axis=1, keepdims=True)) / (X0.std(axis=1, keepdims=True) + 1e-8)
    X1 = (X1 - X1.mean(axis=1, keepdims=True)) / (X1.std(axis=1, keepdims=True) + 1e-8)
    v0 = zscore(v0)
    v1 = zscore(v1)
    v0r = resample(v0, w0)
    v1r = resample(v1, w1)

    f0 = np.sqrt(((X0[:, s0 : s0 + w0] - v0r) ** 2).mean(axis=1))
    f1 = np.sqrt(((X1[:, s1 : s1 + w1] - v1r) ** 2).mean(axis=1))
    F = np.stack([f0, f1], axis=1)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(F, y, test_size=0.25, random_state=42, stratify=y)

    # Train LightGBM
    model = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)
    
    # Calculate and print statistics
    train_acc = model.score(X_tr, y_tr)
    test_acc = model.score(X_te, y_te)
    
    print(f"\n=== Decision Boundary Statistics for REMC {cell_line} ===")
    print(f"Dataset size: {len(F)} samples")
    print(f"Train/test split: {len(X_tr)}/{len(X_te)} samples")
    print(f"Class distribution (test): {dict(zip(*np.unique(y_te, return_counts=True)))}")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Pattern 1 (start={s0}, width={w0}): RMSE range [{f0.min():.3f}, {f0.max():.3f}]")
    print(f"Pattern 2 (start={s1}, width={w1}): RMSE range [{f1.min():.3f}, {f1.max():.3f}]")

    # Create decision boundary background
    x_min, x_max = X_te[:, 0].min() - 0.05, X_te[:, 0].max() + 0.05
    y_min, y_max = X_te[:, 1].min() - 0.05, X_te[:, 1].max() + 0.05
    
    # Create meshgrid for background prediction
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict on mesh
    if hasattr(model, 'predict_proba'):
        Z_proba = model.predict_proba(mesh_points)
        Z = np.argmax(Z_proba, axis=1)  # Get predicted class
    else:
        Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Plot decision boundary background with opacity
    unique_classes = np.unique(y_te)
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'plum'][:len(unique_classes)]
    
    # Create custom colormap for background
    bg_cmap = ListedColormap(colors[:len(unique_classes)])
    
    ax.contourf(xx, yy, Z, levels=len(unique_classes)-1, cmap=bg_cmap, alpha=0.6, zorder=2)
    
    # Plot each class separately for categorical legend
    point_colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(unique_classes)]
    
    for i, cls in enumerate(unique_classes):
        mask = y_te == cls
        ax.scatter(X_te[mask, 0], X_te[mask, 1], c=point_colors[i], label=f'Class {int(cls)}', 
                  marker='x', s=15, alpha=0.5, zorder=4, linewidth=1)
    
    ax.set_xlabel('RMSE of pattern 1')
    ax.set_ylabel('RMSE of pattern 2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved:', out_img)


if __name__ == '__main__':
    main()


