import os
import json
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_ROOT = os.path.join(PROJECT_ROOT, 'eval_scripts', 'json_files')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'manuscript', 'tables')
OUT_FILE = os.path.join(TABLES_DIR, 'pattern_stats_table.tex')


def _collect_json_files(dataset_dir: str):
    files = []
    if not os.path.exists(dataset_dir):
        return files
    direct = os.path.join(dataset_dir, 'pattern_parameters.json')
    if os.path.isfile(direct):
        files.append(direct)
    for entry in os.listdir(dataset_dir):
        p = os.path.join(dataset_dir, entry)
        if os.path.isdir(p):
            cand = os.path.join(p, 'pattern_parameters.json')
            if os.path.isfile(cand):
                files.append(cand)
    return sorted(files)


def _resample(vals: np.ndarray, target_len: int = 50) -> np.ndarray:
    if vals.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    x_old = np.linspace(0, 1, num=vals.size)
    x_new = np.linspace(0, 1, num=target_len)
    return np.interp(x_new, x_old, vals)


def summarize_dataset(dataset_name: str):
    files = _collect_json_files(os.path.join(JSON_ROOT, dataset_name))
    starts, widths, patterns_per_run = [], [], []
    shapes = []
    for f in files:
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
        except Exception:
            continue
        patterns = data.get('patterns', [])
        patterns_per_run.append(int(data.get('n_patterns', len(patterns))))
        for p in patterns:
            starts.append(int(p.get('pattern_start', 0)))
            widths.append(int(p.get('pattern_width', 0)))
            vals = np.array(p.get('pattern_values', []), dtype=np.float32)
            if vals.size:
                shapes.append(_resample(vals, 50))

    starts = np.array(starts, dtype=np.float32)
    widths = np.array(widths, dtype=np.float32)
    patterns_per_run = np.array(patterns_per_run, dtype=np.float32)
    shapes = np.stack(shapes) if len(shapes) else None

    def mean_std(x):
        return (float(np.nanmean(x)) if x.size else np.nan, float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0)

    stats = {
        'num_runs': int(len(patterns_per_run)),
        'total_patterns': int(np.nansum(patterns_per_run) if patterns_per_run.size else 0),
        'patterns_run_mean_std': mean_std(patterns_per_run) if patterns_per_run.size else (np.nan, 0.0),
        'patterns_run_median': float(np.nanmedian(patterns_per_run)) if patterns_per_run.size else np.nan,
        'start_min': int(np.nanmin(starts)) if starts.size else np.nan,
        'start_max': int(np.nanmax(starts)) if starts.size else np.nan,
        'start_median': float(np.nanmedian(starts)) if starts.size else np.nan,
        'width_mean_std': mean_std(widths) if widths.size else (np.nan, 0.0),
        'width_median': float(np.nanmedian(widths)) if widths.size else np.nan,
        'num_shapes': int(shapes.shape[0]) if shapes is not None else 0,
    }
    return stats


def fmt_mean_std(m, s, decimals=1):
    if np.isnan(m):
        return 'N/A'
    return f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"


def build_table(datasets):
    rows = []
    for ds in datasets:
        st = summarize_dataset(ds)
        patterns_run = fmt_mean_std(*st['patterns_run_mean_std'], decimals=1)
        start_range = 'N/A' if np.isnan(st['start_median']) else f"{int(st['start_median'])} [{int(st['start_min'])}, {int(st['start_max'])}]"
        width_stats = 'N/A' if np.isnan(st['width_median']) else f"{int(st['width_median'])} $\\pm$ {st['width_mean_std'][1]:.1f}"
        rows.append((ds, st['num_runs'], st['total_patterns'], patterns_run, start_range, width_stats))

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Summary statistics of extracted patterns across datasets. Patterns/run is the per-run mean $\\pm$ std; start shows median [min, max]; width shows median $\\pm$ std.}\n"
        "\\label{tab:pattern_stats}\n"
        "\\begin{tabular}{|l|c|c|c|c|c|}\n"
        "\\hline\n"
        "\\textbf{Dataset} & \\textbf{Runs} & \\textbf{Total patterns} & \\textbf{Patterns/run} & \\textbf{Start (med [min,max])} & \\textbf{Width (med $\\pm$ std)} \\\\ \n"
        "\\hline\n"
    )

    body = ""
    for ds, runs, total, pr, start_rng, width_str in rows:
        body += f"{ds} & {runs} & {total} & {pr} & {start_rng} & {width_str} \\\\\n"

    footer = (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    return header + body + footer


def main():
    datasets = []
    if os.path.isdir(JSON_ROOT):
        for name in os.listdir(JSON_ROOT):
            if os.path.isdir(os.path.join(JSON_ROOT, name)):
                datasets.append(name)
    datasets = sorted(datasets)
    os.makedirs(TABLES_DIR, exist_ok=True)
    table_tex = build_table(datasets)
    with open(OUT_FILE, 'w') as f:
        f.write(table_tex)
    print(f"Wrote {OUT_FILE}")


if __name__ == '__main__':
    main()


