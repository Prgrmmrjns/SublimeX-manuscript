import os, warnings
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

from config import RESULTS, PARAMETERS, ELARTICLE

warnings.filterwarnings('ignore')
RESULTS_DIR = str(RESULTS)
PARAMS_DIR = str(PARAMETERS)
OUTPUT_DIR = str(ELARTICLE)
DATASETS = ['azt1d', 'mitbih', 'emotions', 'remc', 'mimic', 'pamap2', 'svd']
REMC_LINE = 'remc_E003'
INITIAL_FEATURES = {}

DATASET_INFO = {
    'azt1d': {'name': 'AZT1D', 'metric': 'RMSE', 'direction': 'minimize'},
    'emotions': {'name': 'Emotions', 'metric': 'Accuracy', 'direction': 'maximize'},
    'mimic': {'name': 'MIMIC-IV', 'metric': 'AUC', 'direction': 'maximize'},
    'mitbih': {'name': 'MIT-BIH', 'metric': 'Accuracy', 'direction': 'maximize'},
    'remc': {'name': 'REMC', 'metric': 'AUC', 'direction': 'maximize'},
    'pamap2': {'name': 'PAMAP2', 'metric': 'Accuracy', 'direction': 'maximize'},
    'svd': {'name': 'SVD', 'metric': 'Accuracy', 'direction': 'maximize'},
}
DIRECTION = {k: v['direction'] for k, v in DATASET_INFO.items()}
METRIC_ABBR = {'Accuracy': 'Acc', 'AUC': 'AUC', 'RMSE': 'RMSE'}

APPROACHES = ['SublimeX', 'TSFRESH', 'CATCH22', 'MiniRocket', 'RDST', 'CNN', 'LSTM']
APPROACH_NORM = {'sublimex': 'SublimeX', 'tsfresh': 'TSFRESH', 'catch22': 'CATCH22',
                 'cnn': 'CNN', 'minirocket': 'MiniRocket', 'rdst': 'RDST', 'lstm': 'LSTM'}
ABLATION_VARIANTS = ['resampling', 'aggregate', 'pattern', 'decision_tree',
                     'n_trials_1000', 'raw_only', 'nsga2', 'parallel']
VARIANT_LABELS = {
    'baseline': ('Baseline', ''),
    'resampling': ('Resampling', ''),
    'aggregate': ('Optimize', 'Aggregates'),
    'pattern': ('Pattern', 'Search'),
    'decision_tree': ('Decision', 'Tree'),
    'n_trials_1000': ('1000', 'Trials'),
    'raw_only': ('Raw', 'Only'),
    'nsga2': ('NSGA-II', ''),
    'parallel': ('Parallel', ''),
}


def base_ds(name):
    s = str(name)
    if s.startswith('remc_'): return 'remc'
    if s.startswith('azt1d_'): return 'azt1d'
    return s


def fmt_score(score, std, dec=3):
    if score is None: return '-'
    if std and std > 0: return f'{score:.{dec}f}±{std:.{dec}f}'
    return f'{score:.{dec}f}'


def fmt_feat(n, std):
    if n is None: return '-'
    if std and std > 0.5: return f'{n:.1f}±{std:.1f}'
    return f'{n:.0f}'


def fmt_time(t, std=None, t_min=None, t_max=None, use_median=False):
    if t is None: return '-'
    if use_median and t_min is not None and t_max is not None:
        return f'{t:.0f} ({t_min:.0f}--{t_max:.0f})'
    if std and std > 0: return f'{t:.0f}±{std:.0f}'
    return f'{t:.0f}'


def _bootstrap_ci(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    boots = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    return float(d.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def _stats(rows, subtract_initial=0, scalar_feat_time=False, cell_line_std=False):
    nf = max(0, rows['n_features'].mean() - subtract_initial)
    t = rows['time']
    sc_std = rows.groupby('dataset')['score'].mean().std() if cell_line_std else (rows['score'].std() if len(rows) > 1 else 0.0)
    t_mean = float(t.mean())
    use_mean = scalar_feat_time or len(t) <= 1
    return {
        'score': rows['score'].mean(),
        'score_std': float(sc_std),
        'n_features': nf,
        'n_features_std': 0.0 if scalar_feat_time else (rows['n_features'].std() if len(rows) > 1 else 0.0),
        'time': t_mean if use_mean else float(t.median()),
        'time_min': float(t.min()),
        'time_max': float(t.max()),
        'use_median_time': not use_mean,
    }


def _filter_ablation(df):
    df = df[df['variant'] != 'baseline'].copy()
    df['base_dataset'] = df['dataset'].apply(
        lambda d: 'azt1d' if str(d).startswith('azt1d') else base_ds(d))
    df = df[~((df['base_dataset'] == 'remc') & (df['dataset'] != REMC_LINE))]
    if df['dataset'].str.startswith('azt1d_s').any():
        df = df[~((df['base_dataset'] == 'azt1d') & (df['dataset'] == 'azt1d'))]
    return df


def _pick_best(means, direction):
    return min(means, key=means.get) if direction == 'minimize' else max(means, key=means.get)


def _wilcoxon(a, b):
    n = min(len(a), len(b))
    if n < 2 or np.allclose(a[:n], b[:n]): return 1.0, n
    _, p = wilcoxon(a[:n], b[:n], alternative='two-sided', method='exact')
    return float(p), n


def _fold_scores_main(df, ds, method):
    if ds == 'azt1d':
        sub = df[(df['base_dataset'] == 'azt1d') & (df['approach'] == method) & df['test_subject'].notna()]
        return sub.sort_values('test_subject')['score'].values if len(sub) else None
    sub = df[(df['base_dataset'] == ds) & (df['approach'] == method)]
    if sub.empty: return None
    if ds == 'remc':
        sub = sub.copy()
        sub['cell_line'] = sub['dataset'].astype(str).str.replace('remc_', '', regex=False)
        sub = sub.groupby('cell_line')['score'].mean().reset_index()
        return sub.sort_values('cell_line')['score'].values
    return sub.sort_values('fold')['score'].values


def _fold_scores_ablation(df, ds, variant, me=None):
    if variant == 'baseline' and me is not None:
        if ds == 'remc':
            sub = me[(me['dataset'] == REMC_LINE) & (me['approach'] == 'SublimeX')]
            return sub.sort_values('fold')['score'].values if len(sub) else None
        return _fold_scores_main(me, ds, 'SublimeX')
    sub = df[(df['base_dataset'] == ds) & (df['variant'] == variant)]
    if sub.empty: return None
    key = 'dataset' if ds == 'azt1d' else 'fold'
    return sub.sort_values(key)['score'].values


def _with_lstm(df):
    extra = pd.read_csv(os.path.join(RESULTS_DIR, 'extra_baselines.csv'))
    lstm = extra.loc[extra['approach'] == 'LSTM', ['dataset', 'approach', 'fold', 'score', 'time', 'n_features']].copy()
    lstm['test_subject'] = np.nan
    lstm['cell_line'] = ''
    ids = sorted(df.loc[df['dataset'].eq('azt1d') & df['test_subject'].notna(), 'test_subject'].unique())
    lstm.loc[lstm['dataset'].eq('azt1d'), 'test_subject'] = lstm.loc[lstm['dataset'].eq('azt1d'), 'fold'].map(
        {i + 1: s for i, s in enumerate(ids)})
    remc = lstm['dataset'].str.startswith('remc_')
    lstm.loc[remc, 'cell_line'] = lstm.loc[remc, 'dataset'].str.replace('remc_', '', regex=False)
    return pd.concat([df, lstm], ignore_index=True)


def compute_significance_tests():
    rows = []
    me = _with_lstm(pd.read_csv(os.path.join(RESULTS_DIR, 'main_eval.csv')))
    me['approach'] = me['approach'].str.lower().map(APPROACH_NORM).fillna(me['approach'])
    me['base_dataset'] = me['dataset'].apply(base_ds)
    for ds in DATASETS:
        by_m = {m: _fold_scores_main(me, ds, m) for m in APPROACHES}
        means = {m: v.mean() for m, v in by_m.items() if v is not None and len(v)}
        best = _pick_best(means, DIRECTION[ds])
        for other in APPROACHES:
            if other == best or by_m[other] is None: continue
            p, n = _wilcoxon(by_m[best], by_m[other])
            md, (lo, hi) = _bootstrap_ci(by_m[best], by_m[other])
            rows.append({'study': 'main_eval', 'dataset': ds, 'best_method': best,
                         'comparison': other, 'p_value': p, 'n_pairs': n,
                         'mean_diff': md, 'ci_low': lo, 'ci_high': hi})

    ab = _filter_ablation(pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_study.csv')))
    variants = ['baseline'] + [v for v in ABLATION_VARIANTS if v in ab['variant'].values]
    for ds in DATASETS:
        by_v = {v: _fold_scores_ablation(ab, ds, v, me=me) for v in variants}
        means = {v: vscores.mean() for v, vscores in by_v.items() if vscores is not None and len(vscores)}
        best = _pick_best(means, DIRECTION[ds])
        for other in variants:
            if other == best or by_v[other] is None: continue
            p, n = _wilcoxon(by_v[best], by_v[other])
            md, (lo, hi) = _bootstrap_ci(by_v[best], by_v[other])
            rows.append({'study': 'ablation', 'dataset': ds, 'best_method': best,
                         'comparison': other, 'p_value': p, 'n_pairs': n,
                         'mean_diff': md, 'ci_low': lo, 'ci_high': hi})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, 'significance_tests.csv'), index=False)
    return out


def _sig(study):
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'significance_tests.csv'))
    df = df[df['study'] == study]
    pairs = {(r['dataset'], r['comparison']) for _, r in df.iterrows() if r['p_value'] < 0.05}
    best = df.groupby('dataset')['best_method'].first().to_dict()
    return pairs, best


def load_main_eval_data():
    df = _with_lstm(pd.read_csv(os.path.join(RESULTS_DIR, 'main_eval.csv')))
    df['approach'] = df['approach'].str.lower().map(APPROACH_NORM).fillna(df['approach'])
    df['base_dataset'] = df['dataset'].apply(base_ds)
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for app in APPROACHES:
            rows = df[(df['base_dataset'] == ds) & (df['approach'] == app)]
            if ds == 'azt1d' and rows['test_subject'].notna().any():
                rows = rows[rows['test_subject'].notna()]
            if not rows.empty:
                out[ds][app] = _stats(rows, INITIAL_FEATURES.get(ds, 0),
                                      scalar_feat_time=(ds == 'azt1d'),
                                      cell_line_std=(ds == 'remc'))
    return out


def _sublimex_baseline_rows(ds):
    bl = pd.read_csv(os.path.join(RESULTS_DIR, 'main_eval.csv'))
    bl['approach'] = bl['approach'].str.lower().map(APPROACH_NORM).fillna(bl['approach'])
    bl = bl[bl['approach'] == 'SublimeX']
    bl['base_dataset'] = bl['dataset'].apply(base_ds)
    bsub = bl[bl['base_dataset'] == ds]
    if ds == 'remc':
        bsub = bsub[bsub['dataset'] == REMC_LINE]
    return bsub


def load_ablation_data(main_results=None):
    main_results = main_results or load_main_eval_data()
    df = _filter_ablation(pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_study.csv')))
    out = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        sub = df[df['base_dataset'] == ds]
        bsub = _sublimex_baseline_rows(ds)
        if ds == 'remc' and not bsub.empty:
            out[ds]['baseline'] = _stats(bsub, INITIAL_FEATURES.get(ds, 0))
        elif ds != 'remc' and 'SublimeX' in main_results.get(ds, {}):
            out[ds]['baseline'] = main_results[ds]['SublimeX'].copy()
        for v in ABLATION_VARIANTS:
            vrows = sub[sub['variant'] == v]
            if not vrows.empty:
                out[ds][v] = _stats(vrows, INITIAL_FEATURES.get(ds, 0))
    return out


def _best_rounded(ds_results, cols, direction, dec):
    vals = [round(ds_results[c]['score'], dec) for c in cols if c in ds_results]
    return (min if direction == 'minimize' else max)(vals) if vals else None


def _write_table(path, caption, label, results, cols, labels, study, tabcolsep='1pt',
                 arraystretch='0.85', tight_subrows=False, order=None):
    sig, best = _sig(study)
    stretch = '0.62' if tight_subrows else arraystretch
    row_end = (' \\\\[-0.85ex]\n', ' \\\\[-0.35ex]\n', ' \\\\\n') if tight_subrows else (' \\\\\n',) * 3
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f'\\begin{{table}}[H]\\centering\\caption{{{caption}}}\\label{{{label}}}\\tiny\n')
        f.write(f'\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\\renewcommand{{\\arraystretch}}{{{stretch}}}\n')
        f.write(f'\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}ll{"c" * len(cols)}@{{}}}}\n\\toprule\n')
        f.write(' &  & ' + ' & '.join(f'\\textbf{{{labels[c][0]}}}' for c in cols) + ' \\\\\n')
        f.write(' &  & ' + ' & '.join(
            f'\\textbf{{{labels[c][1]}}}' if labels[c][1] else '' for c in cols) + ' \\\\\n\\midrule\n')
        order = order or DATASETS
        for i, ds in enumerate(order):
            info, dr = DATASET_INFO[ds], results.get(ds, {})
            dec = 2 if info['metric'] == 'RMSE' else 3
            sign = '↑' if info['direction'] == 'maximize' else '↓'
            abbr = METRIC_ABBR[info['metric']]
            br = _best_rounded(dr, cols, info['direction'], dec)
            rows = {k: [] for k in ('s', 'f', 't')}
            for c in cols:
                if c not in dr:
                    rows['s'].append('-'); rows['f'].append('-'); rows['t'].append('-')
                    continue
                d = dr[c]
                sc = fmt_score(d['score'], d['score_std'], dec)
                if br is not None and abs(round(d['score'], dec) - br) < 1e-6:
                    sc = f'\\textbf{{{sc}}}'
                if c != best.get(ds) and (ds, c) in sig:
                    sc += '$^{*}$'
                rows['s'].append(sc)
                rows['f'].append(fmt_feat(d['n_features'], d['n_features_std']))
                rows['t'].append(fmt_time(
                    d['time'], d.get('time_std'), d.get('time_min'), d.get('time_max'),
                    use_median=d.get('use_median_time', False)))
            f.write(f'\\multirow{{3}}{{*}}{{\\textbf{{{info["name"]}}}}} & {abbr} ({sign}) & '
                    + ' & '.join(rows['s']) + row_end[2])
            f.write(' & \\# Feat. & ' + ' & '.join(rows['f']) + row_end[0])
            f.write(' & Time (s) & ' + ' & '.join(rows['t']) + row_end[1])
            if i < len(order) - 1:
                f.write('\\midrule\n')
        f.write('\\bottomrule\\end{tabular*}\\end{table}\n')


def generate_results_table(results):
    cap = ('Predictive score, LightGBM input size (\\# Feat.), and wall-clock time (s; median (min--max) over folds). '
           'Scores: mean $\\pm$ std over folds (AZT1D RMSE: over 23 test patients; REMC: 56 cell-line means). '
           'AZT1D \\# Feat.\\ and Time are single values. '
           'For the LSTM, \\# Feat.\\ is the number of trainable parameters. '
           'Bold: best per dataset; $^{*}$: two-sided exact Wilcoxon $p{<}0.05$ vs.\\ best '
           '(unreachable on 5-fold datasets: min.\\ $p{=}0.0625$). '
           'Arrows: higher ($\\uparrow$) or lower ($\\downarrow$) is better.')
    _write_table(os.path.join(OUTPUT_DIR, 'results_table.tex'), cap, 'tab:results',
                 results, APPROACHES, {a: (a, '') for a in APPROACHES}, 'main_eval',
                 tight_subrows=True,
                 order=[d for d in DATASETS if d != 'azt1d'] + ['azt1d'])


def generate_ablation_table(results):
    existing = {v for dr in results.values() for v in dr} - {'baseline'}
    cols = ['baseline'] + [v for v in ABLATION_VARIANTS if v in existing]
    cap = ('Ablation study (mean $\\pm$ std across folds; AZT1D across patients; PAMAP2 LOSO; REMC E003 AUC). '
           'Baseline matches SublimeX on the same E003 protocol (not the pooled 56-line Table~\\ref{tab:results} score). '
           'Time: median (min--max). Bold: best per dataset; $^{*}$: two-sided exact Wilcoxon $p{<}0.05$ vs.\\ best. '
           'Rows: metric, \\# features, runtime (s).')
    _write_table(os.path.join(OUTPUT_DIR, 'ablation_results.tex'), cap, 'tab:ablation',
                 results, cols, VARIANT_LABELS, 'ablation', tabcolsep='0.5pt')


if __name__ == '__main__':
    compute_significance_tests()
    main_results = load_main_eval_data()
    generate_results_table(main_results)
    ablation = load_ablation_data(main_results)
    generate_ablation_table(ablation)
