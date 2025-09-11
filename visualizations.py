import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _load_mitbih():
    df = pd.read_csv('mitbih_database/mitbih_processed.csv')
    y = df['target'].to_numpy()
    X = df.drop(columns=['target']).to_numpy(dtype=np.float32)
    return X, y

def _load_bonn():
    df = pd.read_csv('bonn_eeg_preprocessing/bonn_eeg_data.csv')
    y = df['label'].map({'Z':0,'O':1,'N':2,'F':3,'S':4}).to_numpy()
    X = df.drop(columns=['label']).to_numpy(dtype=np.float32)
    return X, y

def _load_mimic_series():
    df = pd.read_csv('mimic/mimic_processed.csv')
    y = df['ARDS_FLAG'].to_numpy()
    feature_cols = [c for c in df.columns if c not in ['subject_id','anchor_age','ARDS_FLAG']]
    series_names = []
    for c in feature_cols:
        if '_hour_' in c:
            name = c.split('_hour_')[0]
            if name not in series_names:
                series_names.append(name)
    series_list = []
    for name in series_names:
        cols = [c for c in feature_cols if c.startswith(f'{name}_hour_')]
        cols.sort(key=lambda x: int(x.split('_hour_')[1]))
        series_list.append(df[cols].to_numpy(dtype=np.float32))
    return series_list, y, series_names

def _load_remc(cell_line='E003'):
    df = pd.read_parquet(f'remc_celllines/remc_{cell_line}.parquet')
    y = df['target'].to_numpy()
    feature_cols = [c for c in df.columns if c != 'target']
    prefixes = ['H3K4me3','H3K9me3','H3K27me3','H3K4me1','H3K36me3']
    series_list = []
    for pref in prefixes:
        cols = [c for c in feature_cols if c.startswith(f'{pref}_')]
        cols.sort(key=lambda x: int(x.split('_')[-1]))
        series_list.append(df[cols].to_numpy(dtype=np.float32))
    return series_list, y

def _load_azt1d(subject_id=1, pred_h=12):
    path = f'AZT1D 2025/CGM Records/Subject {subject_id}/Subject {subject_id}.csv'
    df = pd.read_csv(path)
    if 'CGM' not in df.columns:
        return None, None
    dt = pd.to_datetime(df['EventDateTime'])
    glucose = df['CGM'].astype(np.float32)
    time_of_day = dt.dt.hour + dt.dt.minute/60.0
    X = pd.DataFrame({f'glucose_lag_{i}': glucose.shift(i) for i in range(24)})
    y = glucose.shift(-pred_h) - glucose
    data = pd.concat([X, y.rename('target')], axis=1).dropna()
    return data.drop(columns=['target']).to_numpy(dtype=np.float32), data['target'].to_numpy(dtype=np.float32)

def _compute_mae(v, Xseg):
    return np.mean(np.abs(Xseg - v[None, :]), axis=1)

def visualize_patterns(json_root='json_files'):
    items = [
        ('AZT1D', os.path.join(json_root, 'AZT1D', '1', 'pattern_parameters.json')),
        ('MITBIH', os.path.join(json_root, 'MITBIH', 'pattern_parameters.json')),
        ('BONN', os.path.join(json_root, 'BONN', 'pattern_parameters.json')),
        ('REMC', os.path.join(json_root, 'REMC', 'E003', 'pattern_parameters.json')),
        ('MIMIC', os.path.join(json_root, 'MIMIC', 'pattern_parameters.json')),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(12, 12))
    for i, (name, path) in enumerate(items):
        with open(path, 'r') as f:
            d = json.load(f)
        p = d['patterns'][0]
        v = np.asarray(p['pattern_values'], dtype=np.float32)
        s = int(p.get('pattern_start', 0))
        w = int(p.get('pattern_width', len(v)))
        sid = p.get('series_index', None)
        ax_l = axes[i, 0]
        ax_l.plot(np.arange(len(v)), v, 'b-')
        t = f'{name}: pattern 0, start {s}, width {w}'
        if sid is not None:
            t += f', series {sid}'
        ax_l.set_title(t)
        ax_l.set_xlabel('Index')
        ax_l.set_ylabel('Value')
        ax_l.grid(True, alpha=0.3)
        ax_r = axes[i, 1]
        if name == 'MITBIH':
            X, y = _load_mitbih()
            Xseg = X[:, s:s+w]
            mae = _compute_mae(v, Xseg)
            sns.histplot(x=mae, hue=y, bins=50, ax=ax_r, alpha=0.7)
            ax_r.set_title('MITBIH: MAE by class')
        elif name == 'BONN':
            X, y = _load_bonn()
            Xseg = X[:, s:s+w]
            mae = _compute_mae(v, Xseg)
            sns.histplot(x=mae, hue=y, bins=50, ax=ax_r, alpha=0.7)
            ax_r.set_title('BONN: MAE by class')
        elif name == 'MIMIC':
            series_list, y, series_names = _load_mimic_series()
            series_idx = int(sid) if sid is not None else 0
            Xseg = series_list[series_idx][:, s:s+w]
            mae = _compute_mae(v, Xseg)
            sns.histplot(x=mae, hue=y, bins=50, ax=ax_r, alpha=0.7)
            ax_r.set_title('MIMIC: MAE by class')
        elif name == 'REMC':
            series_list, y = _load_remc('E003')
            series_idx = int(sid) if sid is not None else 0
            Xseg = series_list[series_idx][:, s:s+w]
            mae = _compute_mae(v, Xseg)
            sns.histplot(x=mae, hue=y, bins=50, ax=ax_r, alpha=0.7)
            ax_r.set_title('REMC (E003): MAE by class')
        else:  # AZT1D (regression)
            X, y = _load_azt1d(1)
            if X is None:
                sns.histplot(v, bins=min(50, max(10, len(v)//2)), ax=ax_r)
                ax_r.set_title('AZT1D: pattern values')
            else:
                Xseg = X[:, s:s+w]
                mae = _compute_mae(v, Xseg)
                sns.histplot(x=mae, bins=50, ax=ax_r, alpha=0.7)
                ax_r.set_title('AZT1D: MAE distribution')
        ax_r.set_xlabel('MAE')
        ax_r.set_ylabel('Count')
        ax_r.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    os.makedirs('manuscript/images', exist_ok=True)
    out1 = 'images/patx_first_patterns_grid.png'
    out2 = 'manuscript/images/patx_first_patterns_grid.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    visualize_patterns('json_files')


