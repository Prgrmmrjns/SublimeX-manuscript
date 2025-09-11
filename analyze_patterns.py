import os
import json
import numpy as np
import pandas as pd

ROOT = 'json_files'

def collect_pattern_rows():
    rows = []
    for dataset in sorted(os.listdir(ROOT)):
        ds_dir = os.path.join(ROOT, dataset)
        if not os.path.isdir(ds_dir):
            continue
        # gather all jsons recursively
        for dirpath, _, filenames in os.walk(ds_dir):
            for fn in filenames:
                if fn.endswith('.json'):
                    path = os.path.join(dirpath, fn)
                    try:
                        with open(path, 'r') as f:
                            d = json.load(f)
                    except Exception:
                        continue
                    patterns = d.get('patterns', [])
                    degree = d.get('polynomial_degree', None)
                    for p in patterns:
                        start = p.get('pattern_start', None)
                        width = p.get('pattern_width', None)
                        series = p.get('series_index', None)
                        rows.append({
                            'dataset': dataset,
                            'degree': degree,
                            'start': start,
                            'width': width,
                            'series_index': series,
                        })
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for dataset, g in df.groupby('dataset'):
        n = len(g)
        w = g['width'].dropna().to_numpy()
        s = g['start'].dropna().to_numpy()
        deg = g['degree'].dropna().to_numpy()
        series = g['series_index'].dropna().to_numpy()
        pieces.append({
            'Dataset': dataset,
            '#Patterns': int(n),
            'Width (median [IQR])': f"{np.median(w):.1f} [{np.percentile(w,25):.1f},{np.percentile(w,75):.1f}]" if w.size else 'NA',
            'Start (median [IQR])': f"{np.median(s):.1f} [{np.percentile(s,25):.1f},{np.percentile(s,75):.1f}]" if s.size else 'NA',
            'Degree (mode)': int(pd.Series(deg).mode().iat[0]) if deg.size else 'NA',
            '#Channels used': int(len(np.unique(series))) if series.size else 'NA',
        })
    out = pd.DataFrame(pieces).sort_values('Dataset')
    return out

def to_latex_table(df: pd.DataFrame) -> str:
    header = ['Dataset', '#Patterns', 'Width (median [IQR])', 'Start (median [IQR])', 'Degree (mode)', '#Channels used']
    df = df[header]
    latex = df.to_latex(index=False, escape=False)
    return latex

def main():
    os.makedirs('manuscript/tables', exist_ok=True)
    df = collect_pattern_rows()
    if df.empty:
        with open('manuscript/tables/pattern_stats.tex', 'w') as f:
            f.write('% No patterns found')
        return
    summ = summarize(df)
    tex = to_latex_table(summ)
    with open('manuscript/tables/pattern_stats.tex', 'w') as f:
        f.write(tex)

if __name__ == '__main__':
    main()


