"""Preprocessing and data loading for all datasets."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample

ROOT = Path(__file__).parent.parent / 'datasets'
ELARTICLE = Path(__file__).parent.parent / 'elsarticle'
TABLE_ORDER = ['azt1d', 'mitbih', 'emotions', 'remc', 'mimic', 'pamap2', 'svd']
DISPLAY_NAMES = {
    'azt1d': 'AZT1D', 'mitbih': 'MITBIH', 'emotions': 'Emotions', 'remc': 'REMC',
    'mimic': 'MIMIC--IV', 'pamap2': 'PAMAP2', 'svd': 'SVD',
}
METRIC_DISPLAY = {'accuracy': 'Acc.', 'auc': 'AUC', 'rmse': 'RMSE'}


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_emotions():
    df = pd.read_csv(ROOT / 'emotions/emotions.csv')
    df['target'] = df['label'].map({'NEUTRAL': 0, 'NEGATIVE': 1, 'POSITIVE': 2})
    df.drop(columns=['label']).to_csv(ROOT / 'emotions/emotions_processed.csv', index=False)


def preprocess_mimic():
    """Preprocess MIMIC-IV data: extract static, aggregated, and time series features.
    
    Excludes flags: ARDS, Sepsis3, Anemia, SIRS, mortality_flag (target only).
    Returns parquet with mortality_flag, static features, aggregated means, and 24h time series.
    """
    df = pd.read_csv(ROOT / 'mimic/final_df.csv')
    
    # Static features (excluding flags: anemia, ARDS, Sepsis3, SIRS)
    STATIC = ['anchor_age', 'gender', 'diabetes', 'hypertension', 'immunosupression',
              'renal_failure', 'obesity', 'heart_failure', 'liver_disease', 'cancer', 'BMI',
              'OMED', 'NMED', 'NSURG', 'TSURG', 'CMED', 'VSURG', 'PSURG', 'GU', 'GYN', 'TRAUM']
    
    # Aggregated features (mean over 24h window)
    AGG = {
        'Creatinine (serum) [nan]': 'creatinine_mean',
        'Platelet Count [nan]': 'platelets_mean',
        'WBC [nan]': 'wbc_mean',
        'INR [nan]': 'inr_mean',
        'Lactic Acid [nan]': 'lactate_mean',
        'Total Bilirubin [nan]': 'bilirubin_mean',
        'PCO2 (Arterial) [mmHg]': 'paco2_mean',
        'PH (Arterial) [nan]': 'ph_arterial_mean',
        'Verbal Response [nan]': 'gcs_verbal_mean',
        'Motor Response [nan]': 'gcs_motor_mean',
        'Norepinephrine [mg]': 'norepinephrine_mean',
        'Epinephrine [mg]': 'epinephrine_mean',
        'Dobutamine [mg]': 'dobutamine_mean'
    }
    
    # Time series features (24 hourly values)
    TS = {
        'HR [bpm]': 'heart_rate',
        'RR [insp/min]': 'respiratory_rate',
        'SpO2 [%]': 'spo2',
        'NBPs [mmHg]': 'bp_systolic',
        'NBPd [mmHg]': 'bp_diastolic',
        'ABPm [mmHg]': 'bp_mean',
        'Temperature F [°F]': 'temperature',
        'FiO2 [nan]': 'fio2',
        'PO2 (Arterial) [mmHg]': 'pao2',
        'Total PEEP Level [cmH2O]': 'peep',
        'Eye Opening [nan]': 'gcs_eye',
        'Foley [mL]': 'urine_output'
    }
    
    # Filter to first 24h of mechanical ventilation
    df['hour_from_mv'] = df['hour'] - df['MV_start_hour']
    df = df[(df['hour_from_mv'] >= 0) & (df['hour_from_mv'] < 24)]
    df = df.dropna(subset=['subject_id', 'hour_from_mv', 'mortality_flag']).sort_values(['subject_id', 'hour_from_mv'])
    
    rows = []
    for _, g in df.groupby('subject_id'):
        if len(g) < 24:
            continue
        row = {'mortality_flag': int(g['mortality_flag'].max())}
        
        # Static features
        for f in STATIC:
            if f in g.columns:
                row[f] = 1 if g[f].iloc[0] == 'M' else (0 if f == 'gender' else g[f].iloc[0])
        
        # Aggregated features (mean over 24h)
        for orig, clean in AGG.items():
            if orig in g.columns:
                row[clean] = g[orig].dropna().mean() if len(g[orig].dropna()) > 0 else 0
        
        # Time series features (24 hourly values)
        for orig, clean in TS.items():
            if orig not in g.columns:
                continue
            vals = pd.Series(np.where(np.isinf(g[orig].values[:24]), np.nan, g[orig].values[:24]))
            vals = vals.ffill().bfill().fillna(0).values
            for h in range(24):
                row[f'{clean}_{h}'] = vals[h]
        
        rows.append(row)
    
    pd.DataFrame(rows).to_parquet(ROOT / 'mimic/mimic_processed.parquet', index=False)


def preprocess_svd():
    raw_dir, n_samples = ROOT / 'SVD/raw', 1000
    subjects = {}
    for d in raw_dir.iterdir():
        if not d.is_dir() or not (d / 'overview.csv').exists(): continue
        for _, row in pd.read_csv(d / 'overview.csv').iterrows():
            wav = d / str(row['AufnahmeID']) / 'sentences' / f"{row['AufnahmeID']}-phrase.wav"
            if wav.exists(): subjects[row['AufnahmeID']] = (wav, d.name == 'healthy')
    
    features, targets = [], []
    for wav_path, healthy in subjects.values():
        sr, audio = wavfile.read(wav_path)
        audio = audio.mean(axis=1) if audio.ndim > 1 else audio
        features.append(resample(audio[:int(sr * 3.0)], n_samples).astype(np.float32))
        targets.append(0 if healthy else 1)
    
    df = pd.DataFrame(np.array(features), columns=[f'phrase_{i}' for i in range(n_samples)])
    df['target'] = targets
    df.to_parquet(ROOT / 'SVD/svd.parquet', index=False)


def preprocess_pamap2():
    COLS = [1] + list(range(4, 7)) + list(range(10, 16)) + list(range(21, 24)) + list(range(27, 33)) + list(range(38, 41)) + list(range(44, 50))
    SENSORS = ['hand_acc_x', 'hand_acc_y', 'hand_acc_z', 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
               'hand_mag_x', 'hand_mag_y', 'hand_mag_z', 'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
               'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z', 'chest_mag_x', 'chest_mag_y', 'chest_mag_z',
               'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z', 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
               'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z']
    
    dfs = []
    for sid in range(101, 110):
        path = ROOT / f'pamap2/Protocol/subject{sid}.dat'
        if not path.exists(): continue
        df = pd.read_csv(path, sep=r'\s+', header=None).iloc[:, COLS]
        df = df[df.iloc[:, 0] != 0].interpolate().fillna(df.mean())
        df['subject_id'] = sid
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    data.columns = ['activity_id'] + SENSORS + ['subject_id']
    data[['activity_id', 'subject_id']] = data[['activity_id', 'subject_id']].astype('uint8')
    data.to_parquet(ROOT / 'pamap2/pamap2.parquet', compression='zstd', index=False)

def preprocess_azt1d():
    def response(t, onset, peak, decay):
        t = np.asarray(t)
        r = np.zeros_like(t, dtype=np.float32)
        m = (t >= onset) & (t <= onset + peak * 8)
        ta = t[m] - onset
        r[m] = ((ta / peak) ** decay) * np.exp(-decay * (ta / peak))
        return r / r.max() if r.max() > 0 else r

    dfs = []
    for sid in range(1, 26):
        try:
            df = pd.read_csv(ROOT / f'azt1d/CGM Records/Subject {sid}/Subject {sid}.csv')
            df['datetime'] = pd.to_datetime(df['EventDateTime'])
            df['glucose'], df['insulin'], df['carbs'] = df['CGM'].ffill(), df['TotalBolusInsulinDelivered'].fillna(0), df['CarbSize'].fillna(0)
            df = df[['datetime', 'glucose', 'insulin', 'carbs']].dropna(subset=['glucose']).sort_values('datetime').reset_index(drop=True)
            
            t_min = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds().values / 60
            active_ins = sum(amt * response(np.maximum(t_min - et, 0), 15, 45, 2.5) for et, amt in zip(t_min[df['insulin'] > 0], df['insulin'].values[df['insulin'] > 0]))
            active_carb = sum(amt * response(np.maximum(t_min - et, 0), 10, 35, 3) for et, amt in zip(t_min[df['carbs'] > 0], df['carbs'].values[df['carbs'] > 0]))
            
            g = df['glucose'].values
            rows = []
            for i in range(24, len(df) - 12):
                if np.any(np.isnan(g[i-24:i+13])): continue
                row = {f'CGM_{t}': g[i-24+t] for t in range(24)}
                row |= {f'Insulin_{t}': active_ins[i-24+t] for t in range(24)}
                row |= {f'Carbs_{t}': active_carb[i-24+t] for t in range(24)}
                row['target'], row['subject_id'] = g[i + 12] - g[i], sid
                rows.append(row)
            if rows: dfs.append(pd.DataFrame(rows))
        except: pass
    pd.concat(dfs, ignore_index=True).to_parquet(ROOT / 'azt1d/azt1d_all_patients.parquet', index=False)


# =============================================================================
# Data loading
# =============================================================================

def load_emotions():
    df = pd.read_csv(ROOT / 'emotions/emotions_processed.csv')
    X = [df[[c for c in df.columns if c.endswith('_a')]].astype(np.float32),
         df[[c for c in df.columns if c.endswith('_b')]].astype(np.float32)]
    return X, df['target'], {'metric': 'accuracy', 'task': 'classification'}


def load_mimic():
    """MIMIC: time series = channels we extract from; static+agg = initial features (prepended).
    
    Excludes flags: ARDS, Sepsis3, Anemia, SIRS from initial features.
    """
    path = ROOT / 'mimic/mimic_processed.parquet'
    csv_path = ROOT / 'mimic/final_df.csv'
    try:
        df = pd.read_parquet(path)
    except OSError:
        if csv_path.exists():
            preprocess_mimic()
            df = pd.read_parquet(path)
        else:
            raise
    
    # Static features (excluding flags: anemia, ARDS, Sepsis3, SIRS)
    static = ['anchor_age', 'gender', 'diabetes', 'hypertension', 'immunosupression',
              'renal_failure', 'obesity', 'heart_failure', 'liver_disease', 'cancer', 'BMI',
              'OMED', 'NMED', 'NSURG', 'TSURG', 'CMED', 'VSURG', 'PSURG', 'GU', 'GYN', 'TRAUM']
    
    # Aggregated features (mean over 24h)
    agg_cols = [c for c in df.columns if c.endswith('_mean')]
    initial_cols = [c for c in static if c in df.columns] + agg_cols
    initial_features = df[initial_cols].fillna(0).values.astype(np.float32)

    # Time series channels (24 hourly values per channel)
    series = sorted(set(c.rsplit('_', 1)[0] for c in df.columns if c.rsplit('_', 1)[-1].isdigit()))
    X = []
    for s in series:
        cols = sorted([c for c in df.columns if c.startswith(f"{s}_") and c.rsplit('_', 1)[-1].isdigit()],
                      key=lambda x: int(x.rsplit('_', 1)[1]))
        X.append(df[cols].replace([np.inf, -np.inf], np.nan).clip(upper=10000).ffill(axis=1).bfill(axis=1).fillna(0))
    
    return X, df['mortality_flag'], {'metric': 'auc', 'task': 'classification', 'initial_features': initial_features}


def load_mitbih():
    df = pd.read_csv(ROOT / 'mitbih/mitbih_processed.csv')
    return [df.drop('target', axis=1)], df['target'], {'metric': 'accuracy', 'task': 'classification'}


def load_svd():
    df = pd.read_parquet(ROOT / 'SVD/svd.parquet')
    cols = sorted([c for c in df.columns if c.startswith('phrase_')], key=lambda x: int(x.split('_')[1]))
    return [df[cols]], df['target'], {'metric': 'accuracy', 'task': 'classification'}


def load_pamap2():
    df = pd.read_parquet(ROOT / 'pamap2/pamap2.parquet')
    df = df[df['subject_id'] != 109]
    feat_cols = [c for c in df.columns if c not in ['activity_id', 'subject_id']]
    
    windows, labels, sids = [], [], []
    for sid in df['subject_id'].unique():
        for aid in df[df['subject_id'] == sid]['activity_id'].unique():
            arr = df[(df['subject_id'] == sid) & (df['activity_id'] == aid)][feat_cols].values
            for i in range(0, len(arr) - 511, 100):
                windows.append(arr[i:i+512])
                labels.append(aid)
                sids.append(sid)
    
    windows = np.array(windows, dtype=np.float32)
    label_map = {l: i for i, l in enumerate(sorted(set(labels)))}
    X = [pd.DataFrame(windows[:, :, i]) for i in range(windows.shape[2])]
    return X, pd.Series([label_map[l] for l in labels]), {'metric': 'accuracy', 'task': 'classification', 'subject_ids': np.array(sids)}


def load_remc(cell_line=None):
    """Load REMC data. If cell_line is None, return list of available cell line names."""
    remc_dir = ROOT / 'remc'
    if cell_line is None:
        if not remc_dir.exists():
            return []
        return sorted(
            p.stem.replace('remc_', '') for p in remc_dir.glob('remc_*.parquet')
        )
    df = pd.read_parquet(ROOT / f'remc/remc_{cell_line}.parquet',
                         engine='fastparquet')
    X = [df[[c for c in df.columns if c.startswith(f"{m}_")]] for m in ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']]
    return X, df['target'], {'metric': 'auc', 'task': 'classification'}


def load_azt1d(subject_id=None):
    df = pd.read_parquet(ROOT / 'azt1d/azt1d_all_patients.parquet')
    if subject_id is not None:
        df = df[df['subject_id'] == subject_id].reset_index(drop=True)
    get_cols = lambda p: sorted([c for c in df.columns if c.startswith(f'{p}_') and c.split('_')[-1].isdigit()], key=lambda x: int(x.split('_')[-1]))
    X = [df[get_cols(s)].fillna(0) for s in ['CGM', 'Insulin', 'Carbs']]
    info = {'metric': 'rmse', 'task': 'regression'}
    if subject_id is None:
        info['subject_ids'] = df['subject_id'].values
    return X, df['target'], info


# =============================================================================
# Dataset table for manuscript
# =============================================================================

def _format_int(n):
    """Format integer with thousands separator for LaTeX."""
    return f'{int(n):,}'.replace(',', '{,}')


def write_dataset_table():
    """Generate dataset characteristics table from metadata and loaded data."""
    meta_path = ROOT / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path) as f:
        metadata = json.load(f)

    def _load_remc_table():
        cell_lines = load_remc()
        if not cell_lines:
            return None, None, None
        return load_remc(cell_line=cell_lines[0])

    loaders = {
        'azt1d': load_azt1d,
        'emotions': load_emotions,
        'mimic': load_mimic,
        'mitbih': load_mitbih,
        'pamap2': load_pamap2,
        'remc': _load_remc_table,
        'svd': load_svd,
    }

    rows = []
    for key in TABLE_ORDER:
        if key not in metadata or key not in loaders:
            continue
        meta = metadata[key]
        try:
            X, y, info = loaders[key]()
        except Exception as e:
            print(f"Warning: could not load {key} for table: {e}")
            continue
        if X is None or y is None:
            continue

        X_list = X if isinstance(X, list) else [X]
        n_channels = meta.get('n_channels') or len(meta.get('ts_prefixes', [])) or (
            1 if meta.get('single_channel') else len(X_list))
        length = X_list[0].shape[1]
        metric = info.get('metric', meta.get('metric', 'accuracy'))
        task = info.get('task', meta.get('task', 'classification'))
        metric_str = METRIC_DISPLAY.get(metric, metric.upper())

        if key == 'azt1d' and 'subject_ids' in info:
            sid = info['subject_ids']
            per_subj = [np.sum(sid == s) for s in np.unique(sid)]
            mean_n, std_n = np.mean(per_subj), np.std(per_subj)
            samples_str = f'${_format_int(mean_n)} \\pm {_format_int(std_n)}$'
        else:
            samples_str = _format_int(len(y))

        if task == 'regression':
            task_str = 'Regr.'
        else:
            n_classes = len(np.unique(y))
            task_str = 'Binary' if n_classes == 2 else f'{n_classes}--class'

        name = DISPLAY_NAMES.get(key, key.upper())
        input_data = meta.get('input_data', '')
        target_feature = meta.get('target_feature', '')
        rows.append((name, samples_str, n_channels, length, metric_str, task_str,
                     input_data, target_feature))

    out_path = ELARTICLE / 'tables' / 'dataset_characteristics.tex'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '\\begin{table}[H]',
        '\\centering',
        '\\caption{Dataset Characteristics and Evaluation Metrics}',
        '\\label{tab:datasets}',
        '\\tiny',
        '\\setlength{\\tabcolsep}{2pt}',
        '\\begin{tabular*}{\\textwidth}'
        '{@{\\extracolsep{\\fill}}lcccp{0.9cm}p{0.9cm}p{2.2cm}p{2.4cm}@{}}',
        '\\toprule',
        '\\textbf{Dataset} & \\textbf{Samples} & \\textbf{Chan.} & \\textbf{Length} &',
        '\\textbf{Metric} & \\textbf{Task} & \\textbf{Input Data} & \\textbf{Target Feature} \\\\',
        '\\midrule',
    ]
    for name, samples, ch, length, m, task_str, input_data, target in rows:
        # Keep lines under 90 chars: break after Task column
        lines.append(
            f'{name} & {samples} & {ch} & {length} & {m} & {task_str} &'
        )
        lines.append(f'{input_data} & {target} \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular*}', '\\end{table}'])
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    for name, func in {'emotions': preprocess_emotions, 'mimic': preprocess_mimic, 'svd': preprocess_svd, 'pamap2': preprocess_pamap2, 'azt1d': preprocess_azt1d}.items():
        print(f"Preprocessing {name}...")
        func()
    print("Writing dataset characteristics table...")
    write_dataset_table()
