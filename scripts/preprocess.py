import json
import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

from config import DATASETS, ELARTICLE

ROOT = DATASETS
TABLE_ORDER = ['azt1d', 'mitbih', 'emotions', 'remc', 'mimic', 'pamap2', 'svd']
DISPLAY_NAMES = {
    'azt1d': 'AZT1D', 'mitbih': 'MIT-BIH', 'emotions': 'Emotions', 'remc': 'REMC',
    'mimic': 'MIMIC-IV', 'pamap2': 'PAMAP2', 'svd': 'SVD',
}
METRIC_DISPLAY = {'accuracy': 'Accuracy', 'auc': 'AUC', 'rmse': 'RMSE'}


def preprocess_emotions():
    df = pd.read_csv(ROOT / 'emotions/emotions.csv')
    df['target'] = df['label'].map({'NEUTRAL': 0, 'NEGATIVE': 1, 'POSITIVE': 2})
    df.drop(columns=['label']).to_csv(ROOT / 'emotions/emotions_processed.csv', index=False)


def preprocess_mimic():
    df = pd.read_csv(ROOT / 'mimic/final_df.csv')
    STATIC = ['anchor_age', 'gender', 'diabetes', 'hypertension', 'immunosupression',
              'renal_failure', 'obesity', 'heart_failure', 'liver_disease', 'cancer', 'BMI',
              'OMED', 'NMED', 'NSURG', 'TSURG', 'CMED', 'VSURG', 'PSURG', 'GU', 'GYN', 'TRAUM']
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
        'Dobutamine [mg]': 'dobutamine_mean',
    }
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
        'Foley [mL]': 'urine_output',
    }
    df['hour_from_mv'] = df['hour'] - df['MV_start_hour']
    df = df[(df['hour_from_mv'] >= 0) & (df['hour_from_mv'] < 24)]
    df = df.dropna(subset=['subject_id', 'hour_from_mv', 'mortality_flag']).sort_values(['subject_id', 'hour_from_mv'])
    rows = []
    for _, g in df.groupby('subject_id'):
        if len(g) < 24:
            continue
        row = {'mortality_flag': int(g['mortality_flag'].max())}
        for f in STATIC:
            if f in g.columns:
                row[f] = 1 if g[f].iloc[0] == 'M' else (0 if f == 'gender' else g[f].iloc[0])
        for orig, clean in AGG.items():
            if orig in g.columns:
                row[clean] = g[orig].dropna().mean() if len(g[orig].dropna()) > 0 else 0
        for orig, clean in TS.items():
            if orig not in g.columns:
                continue
            vals = pd.Series(np.where(np.isinf(g[orig].values[:24]), np.nan, g[orig].values[:24])).ffill().bfill().fillna(0).values
            for h in range(24):
                row[f'{clean}_{h}'] = vals[h]
        rows.append(row)
    pd.DataFrame(rows).to_parquet(ROOT / 'mimic/mimic_processed.parquet', index=False)


def preprocess_svd():
    raw_dir, n_samples = ROOT / 'SVD/raw', 1000
    subjects = {}
    for d in raw_dir.iterdir():
        if not d.is_dir() or not (d / 'overview.csv').exists():
            continue
        for _, row in pd.read_csv(d / 'overview.csv').iterrows():
            wav = d / str(row['AufnahmeID']) / 'sentences' / f"{row['AufnahmeID']}-phrase.wav"
            if wav.exists():
                subjects[row['AufnahmeID']] = (wav, d.name == 'healthy')
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
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=r'\s+', header=None).iloc[:, COLS]
        df = df[df.iloc[:, 0] != 0].interpolate().fillna(df.mean())
        df['subject_id'] = sid
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data.columns = ['activity_id'] + SENSORS + ['subject_id']
    data[['activity_id', 'subject_id']] = data[['activity_id', 'subject_id']].astype('uint8')
    data.to_parquet(ROOT / 'pamap2/pamap2.parquet', compression='zstd', index=False)


def _azt1d_subject(csv_path, sid):
    def response(t, onset, peak, decay):
        t = np.asarray(t)
        r = np.zeros_like(t, dtype=np.float32)
        m = (t >= onset) & (t <= onset + peak * 8)
        ta = t[m] - onset
        r[m] = ((ta / peak) ** decay) * np.exp(-decay * (ta / peak))
        return r / r.max() if r.max() > 0 else r

    df = pd.read_csv(csv_path).rename(columns={'Readings (CGM / BGM)': 'CGM'})
    df['datetime'] = pd.to_datetime(df['EventDateTime'])
    df['glucose'], df['insulin'], df['carbs'] = df['CGM'].ffill(), df['TotalBolusInsulinDelivered'].fillna(0), df['CarbSize'].fillna(0)
    df = df[['datetime', 'glucose', 'insulin', 'carbs']].dropna(subset=['glucose']).sort_values('datetime').reset_index(drop=True)
    t_min = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds().values / 60
    z = lambda: np.zeros(len(t_min), np.float32)
    ins_t, ins_a = t_min[df['insulin'] > 0], df['insulin'].values[df['insulin'] > 0]
    carb_t, carb_a = t_min[df['carbs'] > 0], df['carbs'].values[df['carbs'] > 0]
    active_ins = sum((amt * response(np.maximum(t_min - et, 0), 15, 45, 2.5) for et, amt in zip(ins_t, ins_a)), z())
    active_carb = sum((amt * response(np.maximum(t_min - et, 0), 10, 35, 3) for et, amt in zip(carb_t, carb_a)), z())
    g = df['glucose'].values
    hours = df['datetime'].dt.hour.values
    times = hours + df['datetime'].dt.minute.values / 60.0
    rows = []
    for i in range(24, len(df) - 12):
        if np.any(np.isnan(g[i - 24:i + 13])):
            continue
        row = {f'CGM_{t}': g[i - 24 + t] for t in range(24)}
        row |= {f'Insulin_{t}': active_ins[i - 24 + t] for t in range(24)}
        row |= {f'Carbs_{t}': active_carb[i - 24 + t] for t in range(24)}
        row['hour'], row['time'] = hours[i], times[i]
        row['target'], row['subject_id'] = g[i + 12] - g[i], sid
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def preprocess_azt1d():
    dfs = []
    for sid in range(1, 26):
        csv_path = ROOT / f'azt1d/CGM Records/Subject {sid}/Subject {sid}.csv'
        if not csv_path.exists():
            continue
        part = _azt1d_subject(csv_path, sid)
        if part is not None:
            dfs.append(part)
    pd.concat(dfs, ignore_index=True).to_parquet(ROOT / 'azt1d/azt1d_all_patients.parquet', index=False)


def load_emotions():
    df = pd.read_csv(ROOT / 'emotions/emotions_processed.csv')
    X = [df[[c for c in df.columns if c.endswith('_a')]].astype(np.float32),
         df[[c for c in df.columns if c.endswith('_b')]].astype(np.float32)]
    return X, df['target'], {'metric': 'accuracy', 'task': 'classification'}


def load_mimic():
    path = ROOT / 'mimic/mimic_processed.parquet'
    if not path.exists():
        preprocess_mimic()
    df = pd.read_parquet(path)
    static = ['anchor_age', 'gender', 'diabetes', 'hypertension', 'immunosupression',
              'renal_failure', 'obesity', 'heart_failure', 'liver_disease', 'cancer', 'BMI',
              'OMED', 'NMED', 'NSURG', 'TSURG', 'CMED', 'VSURG', 'PSURG', 'GU', 'GYN', 'TRAUM']
    agg_cols = [c for c in df.columns if c.endswith('_mean')]
    initial_cols = [c for c in static if c in df.columns] + agg_cols
    initial_features = df[initial_cols].fillna(0).values.astype(np.float32)
    series = sorted(set(c.rsplit('_', 1)[0] for c in df.columns if c.rsplit('_', 1)[-1].isdigit()))
    X = []
    for s in series:
        cols = sorted([c for c in df.columns if c.startswith(f"{s}_") and c.rsplit('_', 1)[-1].isdigit()],
                      key=lambda x: int(x.rsplit('_', 1)[1]))
        X.append(df[cols].replace([np.inf, -np.inf], np.nan).clip(upper=10000).ffill(axis=1).bfill(axis=1).fillna(0))
    return X, df['mortality_flag'], {'metric': 'auc', 'task': 'classification', 'initial_features': initial_features}


MITBIH_LABELS = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}


def preprocess_mitbih():
    raw = ROOT / 'mitbih'
    rows, targets = [], []
    recs = sorted(int(p.name.replace('annotations.txt', '')) for p in raw.glob('*annotations.txt'))
    for rec in recs:
        sig = pd.read_csv(raw / f'{rec}.csv')
        sig.columns = [c.strip().strip("'").strip('"') for c in sig.columns]
        x = sig.iloc[:, 1].to_numpy(np.float32)
        n = len(x)
        for line in open(raw / f'{rec}annotations.txt'):
            parts = line.split()
            if len(parts) < 3 or not parts[1].isdigit() or parts[2] not in MITBIH_LABELS:
                continue
            s = int(parts[1])
            a = s - 100
            w = np.zeros(200, np.float32)
            sa, sb = max(0, a), min(n, a + 200)
            w[sa - a:sa - a + (sb - sa)] = x[sa:sb]
            rows.append(w)
            targets.append(MITBIH_LABELS[parts[2]])
    df = pd.DataFrame(np.vstack(rows), columns=[f'feature_{i}' for i in range(200)])
    df['target'] = targets
    df.to_csv(raw / 'mitbih_processed.csv', index=False)


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
                windows.append(arr[i:i + 512])
                labels.append(aid)
                sids.append(sid)
    windows = np.array(windows, dtype=np.float32)
    label_map = {l: i for i, l in enumerate(sorted(set(labels)))}
    X = [pd.DataFrame(windows[:, :, i]) for i in range(windows.shape[2])]
    return X, pd.Series([label_map[l] for l in labels]), {'metric': 'accuracy', 'task': 'classification', 'subject_ids': np.array(sids)}


def load_remc(cell_line=None):
    remc_dir = ROOT / 'remc'
    if cell_line is None:
        return sorted(p.stem.replace('remc_', '') for p in remc_dir.glob('remc_*.parquet')) if remc_dir.exists() else []
    df = pd.read_parquet(ROOT / f'remc/remc_{cell_line}.parquet', engine='fastparquet')
    X = [df[[c for c in df.columns if c.startswith(f"{m}_")]] for m in ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']]
    return X, df['target'], {'metric': 'auc', 'task': 'classification'}


def load_azt1d(subject_id=None):
    df = pd.read_parquet(ROOT / 'azt1d/azt1d_all_patients.parquet', engine='fastparquet')
    if subject_id is not None:
        df = df[df['subject_id'] == subject_id].reset_index(drop=True)
    get_cols = lambda p: sorted([c for c in df.columns if c.startswith(f'{p}_') and c.split('_')[-1].isdigit()],
                                key=lambda x: int(x.split('_')[-1]))
    X = [df[get_cols(s)].fillna(0) for s in ['CGM', 'Insulin', 'Carbs']]
    info = {'metric': 'rmse', 'task': 'regression'}
    if subject_id is None:
        info['subject_ids'] = df['subject_id'].values
    return X, df['target'], info


def _format_int(n):
    return f'{int(n):,}'.replace(',', '{,}')


def _class_balance_str(y, task):
    y = np.asarray(y)
    if task == 'regression':
        return '---'
    return f'{100 * pd.Series(y).value_counts().max() / len(y):.0f}\\%'


def write_dataset_table():
    metadata = json.loads((ROOT / 'metadata.json').read_text())
    loaders = {
        'azt1d': load_azt1d, 'emotions': load_emotions, 'mimic': load_mimic,
        'mitbih': load_mitbih, 'pamap2': load_pamap2,
        'remc': lambda: load_remc(cell_line=load_remc()[0]), 'svd': load_svd,
    }
    rows = []
    for key in TABLE_ORDER:
        meta = metadata[key]
        X, y, info = loaders[key]()
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
            samples_str = f'${_format_int(np.mean(per_subj))} \\pm {_format_int(np.std(per_subj))}$'
        else:
            samples_str = _format_int(len(y))
        task_str = 'Regr.' if task == 'regression' else ('Binary' if len(np.unique(y)) == 2 else f'{len(np.unique(y))}--class')
        name = DISPLAY_NAMES.get(key, key.upper())
        shape = f'{samples_str}; ${n_channels} \\times {length}$'
        io = f"{meta.get('input_data', '')} / {meta.get('target_feature', '')}"
        rows.append((name, shape, metric_str, task_str, _class_balance_str(y, task), io))

    lines = [
        '\\begingroup',
        '\\captionsetup{justification=raggedright,singlelinecheck=false}',
        '\\begin{table}[H]\\centering',
        '\\caption{Summary of the seven evaluation benchmarks.}',
        '\\label{tab:datasets}',
        '\\tiny',
        '\\setlength{\\tabcolsep}{1pt}\\renewcommand{\\arraystretch}{0.85}',
        '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lp{2.25cm}llp{0.75cm}>{\\raggedright\\arraybackslash}p{3.35cm}@{}}',
        '\\toprule',
        '\\textbf{Dataset} & \\textbf{Samples; channels $\\times$ length} & \\textbf{Metric} & \\textbf{Task} & \\textbf{\\% max.} & \\textbf{Input / target} \\\\',
        '\\midrule',
    ]
    for name, shape, m, task_str, balance, io in rows:
        lines.append(f'{name} & {shape} & {m} & {task_str} & {balance} & {io} \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular*}', '\\end{table}', '\\endgroup'])
    ELARTICLE.mkdir(parents=True, exist_ok=True)
    (ELARTICLE / 'dataset_characteristics.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    for func in (preprocess_emotions, preprocess_mimic, preprocess_svd, preprocess_pamap2, preprocess_azt1d, preprocess_mitbih):
        func()
    write_dataset_table()
