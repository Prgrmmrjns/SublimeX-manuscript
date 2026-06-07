"""SHAP domain interpretation for REMC E003 and AZT1D (subject 1). Writes one CSV + figure."""
import json, os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import RESULTS, PARAMETERS, ELARTICLE, setup_sublimex_path
setup_sublimex_path()
from sublimex import TRANSFORMS, extract_feature
from preprocess import load_remc, load_azt1d

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family': 'serif', 'font.size': 7})

FIG_PATH = ELARTICLE / 'domain_interpretation.png'
CSV_PATH = RESULTS / 'domain_interpretation.csv'
PARAMS = PARAMETERS
K_FOLDS = 5
SHAP_TOP_K = 5
REMC_CH = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
AZT1D_CH = ['CGM', 'Insulin', 'Carbs']
T_NAMES = list(TRANSFORMS.keys())
T_DISP = {'raw': 'raw', 'zscore': 'z-score', 'derivative': 'deriv.', 'fft': 'FFT'}
COLORS = {'High': '#E53935', 'Low': '#1976D2'}
AXIS_FS, XLABEL_FS, LEGEND_FS, PANEL_FS, SHAP_FS, TICK_FS = 5.5, 6.5, 5, 12, 6, 6
LINE_KW = dict(linewidth=2, marker='o', markersize=3.5)
KDE_KW = dict(fill=True, alpha=0.3, linewidth=1.5)


def _feat_name(p, n_time, channels):
    ch, t = int(p['ch']), int(p['t'])
    c, r = p['c'], p['r']
    center, half = c * (n_time - 1), (r * (n_time - 1)) * 0.5
    s, e = max(0, int(center - half)), min(n_time - 1, int(center + half))
    tn = T_DISP.get(T_NAMES[t], T_NAMES[t])
    return f"{channels[ch]} {tn} {s}-{e}"


def _transform_batch(X_list):
    arrays = [x.values.astype(np.float32) if hasattr(x, 'values') else np.asarray(x, np.float32) for x in X_list]
    data = np.stack(arrays, axis=1)
    n, nc, nt = data.shape
    out = np.empty((len(T_NAMES), n, nc, nt), dtype=np.float32)
    for ti, tn in enumerate(T_NAMES):
        out[ti] = TRANSFORMS[tn](data.reshape(-1, nt)).reshape(n, nc, nt)
    return out, nc, nt, T_NAMES


def _extract(transformed, params, nc, nt, tnames):
    ctx = {'transformed': transformed, 'n_channels': nc, 'n_time': nt, 'transform_names': tnames}
    parts = [extract_feature(p, ctx) for p in params]
    return np.hstack(parts).astype(np.float32) if parts else np.empty((transformed.shape[1], 0), np.float32)


def _shap_dirs(sv, X, order):
    ms = sv.mean(axis=0)
    corr = [0.0 if np.std(X[:, j]) == 0 or np.std(sv[:, j]) == 0 else float(np.corrcoef(X[:, j], sv[:, j])[0, 1])
            for j in range(sv.shape[1])]
    corr = np.asarray(corr)
    return {
        'mean_shap_signed': ms[order],
        'direction_by_mean': np.where(ms >= 0, 'positive', 'negative')[order],
        'value_effect_direction': np.where(corr >= 0, 'higher value -> higher prediction',
                                           'higher value -> lower prediction')[order],
        'feature_shap_corr': corr[order],
    }


def _run_study(X_list, y, tr, te, params, channels, regression=False, subject_id=None):
    y_tr, y_te = y.iloc[tr].values, y.iloc[te].values
    tf, nc, nt, tnames = _transform_batch(X_list)
    ftr = _extract(tf[:, tr], params, nc, nt, tnames)
    fte = _extract(tf[:, te], params, nc, nt, tnames)
    names = [_feat_name(p, nt, channels) for p in params]
    lgb = (LGBMRegressor if regression else LGBMClassifier)(max_depth=5, verbosity=-1, force_row_wise=True)
    lgb.fit(ftr, y_tr)
    pred = lgb.predict(fte) if regression else lgb.predict_proba(fte)[:, 1]
    metric = np.sqrt(mean_squared_error(y_te, pred)) if regression else roc_auc_score(y_te, pred)
    explainer = shap.TreeExplainer(lgb)
    sv = explainer.shap_values(fte)
    sv = sv[1] if isinstance(sv, list) else sv
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1]
    top = order[0]
    tp = params[top]
    ch, t = int(tp['ch']), int(tp['t'])
    center, half = tp['c'] * (nt - 1), (tp['r'] * (nt - 1)) * 0.5
    ss, se = max(0, int(center - half)), min(nt - 1, int(center + half))
    fn = TRANSFORMS[T_NAMES[t]]
    raw = X_list[ch].iloc[tr].values.astype(np.float32)
    tch = fn(raw.reshape(-1, nt)).reshape(-1, nt)
    hi_tr, lo_tr = ((y_tr > 0), (y_tr < 0)) if regression else ((y_tr == 1), (y_tr == 0))
    hi_te, lo_te = ((y_te > 0), (y_te < 0)) if regression else ((y_te == 1), (y_te == 0))
    dirs = _shap_dirs(sv, fte, order)
    feat_df = pd.DataFrame({
        'rank': np.arange(1, len(names) + 1),
        'feature_name': [names[i] for i in order],
        'mean_abs_shap': np.abs(sv).mean(axis=0)[order],
        **dirs,
    })
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1 if not regression else 0]
    summary = {
        'metric_name': 'rmse' if regression else 'auc',
        'metric_value': metric,
        'n_features': len(names),
        'top_feature_name': names[top],
        'top_feature_mean_abs_shap': float(np.abs(sv).mean(axis=0)[top]),
        'subject_id': subject_id,
        'fold': 1,
    }
    return {
        'summary': summary, 'features_df': feat_df,
        'seg_start': ss, 'seg_end': se, 'n_time': nt,
        'high_mean': tch[hi_tr].mean(0), 'low_mean': tch[lo_tr].mean(0),
        'ch_name': channels[ch], 'transform_label': T_DISP.get(T_NAMES[t], T_NAMES[t]),
        'top_name': names[top],
        'high_vals': fte[hi_te, top], 'low_vals': fte[lo_te, top],
        'sv': sv, 'top_idx': order, 'test_feat': fte, 'feat_names': names, 'base_value': base,
    }


def run_remc():
    X, y, _ = load_remc(cell_line='E003')
    tr, te = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X, axis=1), y))[0]
    with open(PARAMS / 'remc_E003' / 'fold1.json') as f:
        params = json.load(f)
    out = _run_study(X, y, tr, te, params, REMC_CH)
    out['summary']['dataset'] = 'remc_E003'
    return out


def run_azt1d(subject_id=1):
    X, y, _ = load_azt1d(subject_id=subject_id)
    n, c = len(y), int(len(y) * 0.8)
    tr, te = np.arange(c), np.arange(c, n)
    with open(PARAMS / 'azt1d' / 'fold1.json') as f:
        params = json.load(f)
    out = _run_study(X, y, tr, te, params, AZT1D_CH, regression=True, subject_id=subject_id)
    out['summary']['dataset'] = 'azt1d'
    return out


def _to_csv(remc, azt1d):
    rows = []
    for res in (remc, azt1d):
        s, fd = res['summary'], res['features_df']
        for _, r in fd.iterrows():
            rows.append({**s, **r.to_dict()})
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)


def _panel_label(ax, letter):
    ax.text(-0.10, 1.0, letter, transform=ax.transAxes, fontsize=PANEL_FS, fontweight='bold', va='bottom', ha='left')


def _short_feat(name):
    name = name.replace(' derivative', ' deriv.').replace(' z-score', ' zsc.')
    m = re.match(r'^(.+?)\s(\d+-\d+)$', name)
    return f"{m.group(1)}\n{m.group(2)}" if m else name


def _shap_y_label(name):
    name = name.replace(' derivative', ' deriv.').replace(' z-score', ' zsc.')
    m = re.match(r'^(.+?)\s+(raw|zsc\.|deriv\.|FFT|z-score)\s+(\d+-\d+)$', name)
    return f"{m.group(1)}\n{m.group(2)} {m.group(3)}" if m else _short_feat(name)


def _draw_signal(ax, d, xlab, hi, lo, letter, xtimes=None):
    sl = slice(d['seg_start'], d['seg_end'] + 1)
    xt = xtimes if xtimes is not None else np.arange(d['seg_start'], d['seg_end'] + 1)
    ax.plot(xt, d['high_mean'][sl], color=COLORS['High'], label=hi, **LINE_KW)
    ax.plot(xt, d['low_mean'][sl], color=COLORS['Low'], label=lo, **LINE_KW)
    ax.set_xlabel(xlab, fontsize=XLABEL_FS)
    ax.set_ylabel(f"{d['ch_name']} ({d['transform_label']})", fontsize=AXIS_FS)
    ax.tick_params(axis='x', labelsize=TICK_FS - 0.5)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _panel_label(ax, letter)


def _draw_density(ax, d, hi, lo, letter):
    sns.kdeplot(d['high_vals'], ax=ax, color=COLORS['High'], **KDE_KW, label=hi)
    sns.kdeplot(d['low_vals'], ax=ax, color=COLORS['Low'], **KDE_KW, label=lo)
    ax.set_xlabel(_short_feat(d['top_name']).replace('\n', ' '), fontsize=XLABEL_FS)
    ax.set_ylabel('Density', fontsize=AXIS_FS)
    ax.tick_params(axis='x', labelsize=TICK_FS - 0.5)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _panel_label(ax, letter)


def _strip_parens(s):
    return re.sub(r'\s*\(.*\)', '', s).strip() if s else s


def _place_shap_cbars(fig, pairs, shap_w=0.19, cbar_w=0.014, gap=0.004):
    claimed = {a for ax_s, ax_l in pairs for a in (ax_s, ax_l)}
    pool = [a for a in fig.axes if a not in claimed and 'feature' in (a.get_ylabel() or '').lower()]
    for ax_shap, _ in pairs:
        ps = ax_shap.get_position()
        ax_shap.set_position([ps.x0, ps.y0, shap_w, ps.height])
        cbar_x0 = ps.x0 + shap_w + gap
        best = min(pool, key=lambda a: abs(a.get_position().y0 - ps.y0) + abs(a.get_position().y1 - ps.y1), default=None)
        if best is None:
            continue
        pool.remove(best)
        pa = best.get_position()
        best._colorbar_info['aspect'] = 14
        best._axes_locator = None
        best.set_position([cbar_x0, pa.y0, cbar_w, pa.height])
        best.set_box_aspect(14)
        best.tick_params(labelsize=5)
        yl = best.get_ylabel()
        lab = _strip_parens(yl if isinstance(yl, str) else yl.get_text())
        best.yaxis.set_label_position('right')
        best.set_ylabel(lab, fontsize=5, labelpad=-11, rotation=90, va='center')
    for a in pool:
        a.set_visible(False)


def _draw_shap(ax, ax_lab, d, letter):
    idx = d['top_idx'][:SHAP_TOP_K]
    plt.sca(ax)
    shap.plots.beeswarm(shap.Explanation(
        values=d['sv'][:, idx], base_values=d['base_value'], data=d['test_feat'][:, idx],
        feature_names=[f'f{k}' for k in range(len(idx))]), show=False, max_display=SHAP_TOP_K)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=TICK_FS - 0.5)
    ax.set_xlabel('SHAP value', fontsize=XLABEL_FS)
    ax.set_ylim(-0.5, SHAP_TOP_K - 0.5)
    ax_lab.set_ylim(ax.get_ylim())
    ax_lab.axis('off')
    for y, lab in zip(ax.get_yticks(), [_shap_y_label(d['feat_names'][i]) for i in idx]):
        ax_lab.text(1.0, y, lab, ha='right', va='center', fontsize=SHAP_FS, linespacing=0.95,
                    transform=ax_lab.get_yaxis_transform())
    _panel_label(ax, letter)


def _tighten_row(ax_a, ax_b, ax_lab, ax_shap, gap_ab=0.068, gap_bc=0.042):
    pa, pb = ax_a.get_position(), ax_b.get_position()
    ax_b.set_position([pa.x1 + gap_ab, pb.y0, pb.width, pb.height])
    pb = ax_b.get_position()
    pl, ps = ax_lab.get_position(), ax_shap.get_position()
    lab_w = min(pl.width, 0.048)
    x0 = pb.x1 + gap_bc
    ax_lab.set_position([x0, pl.y0, lab_w, pl.height])
    ax_shap.set_position([x0 + lab_w + 0.006, ps.y0, ps.width, ps.height])


def _widen_shap(ax_lab, ax_shap, scale=1.0):
    pl, ps = ax_lab.get_position(), ax_shap.get_position()
    ax_shap.set_position([pl.x1 + 0.006, ps.y0, ps.width * scale, ps.height])


def make_figure(remc, azt1d):
    fig = plt.figure(figsize=(17, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], wspace=0.34, hspace=0.42,
                         left=0.055, right=0.92, top=0.975, bottom=0.095)
    gs_ct = gs[0, 2].subgridspec(1, 2, width_ratios=[0.16, 1.6], wspace=0.02)
    gs_cb = gs[1, 2].subgridspec(1, 2, width_ratios=[0.16, 1.6], wspace=0.02)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_d, ax_e = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_cl = fig.add_subplot(gs_ct[0, 0])
    ax_c = fig.add_subplot(gs_ct[0, 1], sharey=ax_cl)
    ax_fl = fig.add_subplot(gs_cb[0, 0])
    ax_f = fig.add_subplot(gs_cb[0, 1], sharey=ax_fl)
    _draw_signal(ax_a, remc, 'Bin (100 bp)', 'High expr.', 'Low expr.', 'A')
    _draw_density(ax_b, remc, 'High expr.', 'Low expr.', 'B')
    _draw_shap(ax_c, ax_cl, remc, 'C')
    nt = azt1d['n_time']
    times = [(b - (nt - 1)) * 5 for b in range(azt1d['seg_start'], azt1d['seg_end'] + 1)]
    _draw_signal(ax_d, azt1d, 'Time (min)', 'Rising', 'Falling', 'D', times)
    _draw_density(ax_e, azt1d, 'Rising', 'Falling', 'E')
    _draw_shap(ax_f, ax_fl, azt1d, 'F')
    _tighten_row(ax_a, ax_b, ax_cl, ax_c)
    _tighten_row(ax_d, ax_e, ax_fl, ax_f)
    _widen_shap(ax_cl, ax_c)
    _widen_shap(ax_fl, ax_f)
    _place_shap_cbars(fig, [(ax_c, ax_cl), (ax_f, ax_fl)])
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=350, facecolor='white', pad_inches=0.01, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for old in (RESULTS / 'domain_interpretation_summary.csv',
                RESULTS / 'domain_interpretation_features.csv',
                RESULTS / 'domain_interpretation_figure_data.pkl'):
        old.unlink(missing_ok=True)
    remc, azt1d = run_remc(), run_azt1d(1)
    _to_csv(remc, azt1d)
    make_figure(remc, azt1d)
    print(f'wrote {CSV_PATH} and {FIG_PATH}', flush=True)


if __name__ == '__main__':
    main()
