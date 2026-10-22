"""SHAP domain interpretation for REMC E003, AZT1D (subject 1), and MIT-BIH (fold 1)."""
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
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import RESULTS, PARAMETERS, ELARTICLE, setup_sublimex_path
setup_sublimex_path()
from sublimex import TRANSFORMS, extract_feature
from preprocess import load_azt1d, load_mitbih, load_remc
from plot_style import mm_size

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})

FIG_PATH = ELARTICLE / 'domain_interpretation.png'
CSV_PATH = RESULTS / 'domain_interpretation.csv'
PARAMS = PARAMETERS
K_FOLDS = 5
SHAP_TOP_K = 5
REMC_CH = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3']
AZT1D_CH = ['CGM', 'Insulin', 'Carbs']
MITBIH_CH = ['ECG']
MITBIH_CLASS = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'APC', 4: 'PVC'}
T_NAMES = list(TRANSFORMS.keys())
T_DISP = {'raw': 'raw', 'zscore': 'z-score', 'derivative': 'deriv.', 'fft': 'FFT'}
COLORS = {'High': '#E53935', 'Low': '#1976D2'}
AXIS_FS, XLABEL_FS, LEGEND_FS, PANEL_FS, SHAP_FS, TICK_FS, CBAR_FS = 6.5, 5.5, 4.2, 8, 5.5, 5.5, 4.2
LINE_KW = dict(linewidth=1.0, marker='o', markersize=1.6)
KDE_KW = dict(fill=True, alpha=0.22, linewidth=0.9)
CLASS_C = ['#1565C0', '#C62828', '#EF6C00', '#2E7D32', '#6A1B9A']


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


def _stack_shap(sv, n_cls):
    if isinstance(sv, list) and len(sv) == n_cls and n_cls > 2:
        return np.stack(sv, axis=-1)
    if isinstance(sv, np.ndarray) and sv.ndim == 3 and n_cls > 2:
        if sv.shape[-1] == n_cls:
            return sv
        if sv.shape[0] == n_cls:
            return np.moveaxis(sv, 0, -1)
    return None


def _signed_shap(sv, n_cls):
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1) if len(sv) > 1 else sv[0]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3 and sv.shape[0] == n_cls:
        sv = np.moveaxis(sv, 0, -1)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        score = np.abs(sv).mean(axis=0)
        focus = int(score.sum(axis=0).argmax()) if n_cls > 2 else (1 if sv.shape[-1] > 1 else 0)
        return sv[:, :, focus], focus
    return sv, 1


def _contrast_masks(y_tr, y_te, ftr, top, regression, class_names):
    if regression:
        return (y_tr > 0), (y_tr < 0), (y_te > 0), (y_te < 0), 'Rising', 'Falling'
    classes = [int(k) for k in np.unique(y_tr)]
    if len(classes) <= 2:
        pos, neg = classes[-1], classes[0]
        return (y_tr == pos), (y_tr == neg), (y_te == pos), (y_te == neg), 'High', 'Low'
    means = {k: float(ftr[y_tr == k, top].mean()) for k in classes}
    a, b = max(((i, j) for i in classes for j in classes if i < j), key=lambda ij: abs(means[ij[0]] - means[ij[1]]))
    hi, lo = (a, b) if means[a] >= means[b] else (b, a)
    name = class_names or {}
    return (y_tr == hi), (y_tr == lo), (y_te == hi), (y_te == lo), name.get(hi, str(hi)), name.get(lo, str(lo))


def _run_study(X_list, y, tr, te, params, channels, regression=False, subject_id=None, class_names=None):
    y_tr, y_te = y.iloc[tr].values, y.iloc[te].values
    tf, nc, nt, tnames = _transform_batch(X_list)
    ftr = _extract(tf[:, tr], params, nc, nt, tnames)
    fte = _extract(tf[:, te], params, nc, nt, tnames)
    names = [_feat_name(p, nt, channels) for p in params]
    n_cls = len(np.unique(y_tr))
    lgb = (LGBMRegressor if regression else LGBMClassifier)(max_depth=5, verbosity=-1, force_row_wise=True)
    lgb.fit(ftr, y_tr)
    if regression:
        pred = lgb.predict(fte)
        metric_name, metric = 'rmse', np.sqrt(mean_squared_error(y_te, pred))
    elif n_cls > 2:
        metric_name, metric = 'accuracy', accuracy_score(y_te, lgb.predict(fte))
    else:
        metric_name, metric = 'auc', roc_auc_score(y_te, lgb.predict_proba(fte)[:, 1])
    explainer = shap.TreeExplainer(lgb)
    raw_sv = explainer.shap_values(fte)
    stack = _stack_shap(raw_sv, n_cls)
    if stack is not None:
        class_abs = np.abs(stack).mean(axis=0)
        order = np.argsort(class_abs.mean(axis=1))[::-1]
        sv = stack.mean(axis=2)
        focus = None
    else:
        sv, focus = _signed_shap(raw_sv, n_cls)
        class_abs = None
        order = np.argsort(np.abs(sv).mean(axis=0))[::-1]
    top = order[0]
    tp = params[top]
    ch, t = int(tp['ch']), int(tp['t'])
    center, half = tp['c'] * (nt - 1), (tp['r'] * (nt - 1)) * 0.5
    ss, se = max(0, int(center - half)), min(nt - 1, int(center + half))
    fn = TRANSFORMS[T_NAMES[t]]
    raw = X_list[ch].iloc[tr].values.astype(np.float32)
    tch = fn(raw.reshape(-1, nt)).reshape(-1, nt)
    hi_tr, lo_tr, hi_te, lo_te, hi_lab, lo_lab = _contrast_masks(y_tr, y_te, ftr, top, regression, class_names)
    dirs = _shap_dirs(sv, fte, order)
    mean_abs = class_abs.mean(axis=1) if class_abs is not None else np.abs(sv).mean(axis=0)
    feat_df = pd.DataFrame({
        'rank': np.arange(1, len(names) + 1),
        'feature_name': [names[i] for i in order],
        'mean_abs_shap': mean_abs[order],
        **dirs,
    })
    ids = [int(k) for k in np.unique(y_tr)]
    labels = [(class_names or {}).get(k, str(k)) for k in ids]
    class_curves = class_vals = None
    if class_abs is not None:
        for j, lab in enumerate(labels):
            feat_df[f'abs_shap_{lab}'] = class_abs[order, j]
        class_curves = [tch[y_tr == k].mean(0) for k in ids]
        class_vals = [fte[y_te == k, top] for k in ids]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = np.asarray(base).ravel()
        base = float(base.mean() if focus is None else base[0 if regression else (focus if n_cls > 2 else min(1, len(base) - 1))])
    summary = {
        'metric_name': metric_name,
        'metric_value': metric,
        'shap_class': 'all' if class_abs is not None else None,
        'n_features': len(names),
        'top_feature_name': names[top],
        'top_feature_mean_abs_shap': float(mean_abs[top]),
        'subject_id': subject_id,
        'fold': 1,
    }
    return {
        'summary': summary, 'features_df': feat_df,
        'seg_start': ss, 'seg_end': se, 'n_time': nt,
        'high_mean': tch[hi_tr].mean(0), 'low_mean': tch[lo_tr].mean(0),
        'ch_name': channels[ch], 'transform_label': T_DISP.get(T_NAMES[t], T_NAMES[t]),
        'top_name': names[top],
        'hi_label': hi_lab, 'lo_label': lo_lab,
        'high_vals': fte[hi_te, top], 'low_vals': fte[lo_te, top],
        'class_labels': labels if class_abs is not None else None,
        'class_curves': class_curves, 'class_vals': class_vals,
        'sv': np.abs(stack).mean(axis=2) if stack is not None else sv, 'top_idx': order, 'test_feat': fte, 'feat_names': names, 'base_value': base,
    }


def run_remc():
    X, y, _ = load_remc(cell_line='E003')
    tr, te = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(pd.concat(X, axis=1), y))[0]
    with open(PARAMS / 'remc_E003' / 'fold1.json') as f:
        params = json.load(f)
    out = _run_study(X, y, tr, te, params, REMC_CH)
    out['summary']['dataset'] = 'remc_E003'
    out['hi_label'], out['lo_label'] = 'High expr.', 'Low expr.'
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


def run_mitbih():
    X, y, _ = load_mitbih()
    tr, te = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(X[0], y))[0]
    with open(PARAMS / 'mitbih' / 'fold1.json') as f:
        params = json.load(f)
    out = _run_study(X, y, tr, te, params, MITBIH_CH, class_names=MITBIH_CLASS)
    out['summary']['dataset'] = 'mitbih'
    return out


def _to_csv(*studies):
    rows = []
    for res in studies:
        s, fd = res['summary'], res['features_df']
        for _, r in fd.iterrows():
            rows.append({**s, **r.to_dict()})
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)


def _legend(ax, **kw):
    ax.legend(fontsize=LEGEND_FS, framealpha=0.9, handlelength=0.85, handletextpad=0.25,
              borderpad=0.12, labelspacing=0.12, borderaxespad=0.15, **kw)


def _panel_label(ax, letter, y=1.0):
    ax.text(-0.10, y, letter, transform=ax.transAxes, fontsize=PANEL_FS, fontweight='bold', va='bottom', ha='left')


def _short_feat(name):
    name = name.replace(' derivative', ' deriv.').replace(' z-score', ' zsc.')
    m = re.match(r'^(.+?)\s(\d+-\d+)$', name)
    return f"{m.group(1)}\n{m.group(2)}" if m else name


def _shap_y_label(name):
    name = name.replace(' derivative', ' deriv.').replace(' z-score', ' zsc.')
    m = re.match(r'^(.+?)\s+(raw|zsc\.|deriv\.|FFT|z-score)\s+(\d+-\d+)$', name)
    return f"{m.group(1)}\n{m.group(2)} {m.group(3)}" if m else _short_feat(name)


def _draw_signal(ax, d, xlab, hi, lo, letter, xtimes=None, legend_loc='upper right', label_y=1.0):
    sl = slice(d['seg_start'], d['seg_end'] + 1)
    xt = xtimes if xtimes is not None else np.arange(d['seg_start'], d['seg_end'] + 1)
    ax.plot(xt, d['high_mean'][sl], color=COLORS['High'], label=hi, **LINE_KW)
    ax.plot(xt, d['low_mean'][sl], color=COLORS['Low'], label=lo, **LINE_KW)
    ax.set_xlabel(xlab, fontsize=XLABEL_FS)
    ax.set_ylabel(f"{d['ch_name']} ({d['transform_label']})", fontsize=AXIS_FS)
    ax.tick_params(axis='x', labelsize=TICK_FS - 0.5)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    _legend(ax, loc=legend_loc)
    ax.grid(True, alpha=0.3)
    _panel_label(ax, letter, label_y)


def _draw_density(ax, d, hi, lo, letter, label_y=1.0):
    sns.kdeplot(d['high_vals'], ax=ax, color=COLORS['High'], **KDE_KW, label=hi)
    sns.kdeplot(d['low_vals'], ax=ax, color=COLORS['Low'], **KDE_KW, label=lo)
    ax.set_xlabel(_short_feat(d['top_name']).replace('\n', ' '), fontsize=XLABEL_FS)
    ax.set_ylabel('Density', fontsize=AXIS_FS)
    ax.tick_params(axis='x', labelsize=TICK_FS - 0.5)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    _legend(ax, loc='upper right')
    ax.grid(True, alpha=0.3)
    _panel_label(ax, letter, label_y)


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
        best.tick_params(labelsize=CBAR_FS)
        yl = best.get_ylabel()
        lab = _strip_parens(yl if isinstance(yl, str) else yl.get_text())
        best.yaxis.set_label_position('right')
        best.set_ylabel(lab, fontsize=CBAR_FS, labelpad=2, rotation=90, va='center')
    for a in pool:
        a.set_visible(False)


def _draw_shap(ax, ax_lab, d, letter, label_y=1.0, xlabel='SHAP value'):
    idx = d['top_idx'][:SHAP_TOP_K]
    plt.sca(ax)
    shap.plots.beeswarm(shap.Explanation(
        values=d['sv'][:, idx], base_values=d['base_value'], data=d['test_feat'][:, idx],
        feature_names=[f'f{k}' for k in range(len(idx))]), show=False, max_display=SHAP_TOP_K)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=TICK_FS - 0.5)
    ax.set_xlabel(xlabel, fontsize=XLABEL_FS)
    ax.set_ylim(-0.5, SHAP_TOP_K - 0.5)
    ax_lab.set_ylim(ax.get_ylim())
    ax_lab.axis('off')
    labs = [_shap_y_label(d['feat_names'][i]) for i in idx]
    for y, lab in zip(ax.get_yticks(), labs[::-1]):
        ax_lab.text(1.0, y, lab, ha='right', va='center', fontsize=SHAP_FS, linespacing=0.95,
                    transform=ax_lab.get_yaxis_transform())
    _panel_label(ax, letter, label_y)


def _tighten_row(ax_a, ax_b, ax_lab, ax_shap, gap_ab=0.048, gap_bc=0.028):
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


def _draw_signal_classes(ax, d, xlab, letter, xtimes=None, label_y=1.0):
    sl = slice(d['seg_start'], d['seg_end'] + 1)
    xt = xtimes if xtimes is not None else np.arange(d['seg_start'], d['seg_end'] + 1)
    for curve, lab, c in zip(d['class_curves'], d['class_labels'], CLASS_C):
        ax.plot(xt, curve[sl], color=c, label=lab, **LINE_KW)
    ax.set_xlabel(xlab, fontsize=XLABEL_FS)
    ax.set_ylabel(f"{d['ch_name']} ({d['transform_label']})", fontsize=AXIS_FS)
    ax.tick_params(labelsize=TICK_FS)
    _legend(ax, loc='upper right')
    ax.grid(True, alpha=0.3)
    _panel_label(ax, letter, label_y)


def _draw_density_classes(ax, d, letter, label_y=1.0):
    for vals, lab, c in zip(d['class_vals'], d['class_labels'], CLASS_C):
        sns.kdeplot(vals, ax=ax, color=c, **KDE_KW, label=lab)
    ax.set_xlabel(_short_feat(d['top_name']).replace('\n', ' '), fontsize=XLABEL_FS)
    ax.set_ylabel('Density', fontsize=AXIS_FS)
    ax.tick_params(labelsize=TICK_FS)
    _legend(ax, loc='upper right')
    ax.grid(True, alpha=0.3)
    _panel_label(ax, letter, label_y)


def _row(fig, gs, i, study, letter, xlab, xtimes=None, legend_loc='upper right', label_y=1.02):
    gs_s = gs[i, 2].subgridspec(1, 2, width_ratios=[0.55, 1.35], wspace=0.04)
    ax_a, ax_b = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
    ax_l = fig.add_subplot(gs_s[0, 0])
    ax_s = fig.add_subplot(gs_s[0, 1], sharey=ax_l)
    hi, lo = study['hi_label'], study['lo_label']
    _draw_signal(ax_a, study, xlab, hi, lo, letter, xtimes, legend_loc, label_y)
    _draw_density(ax_b, study, hi, lo, chr(ord(letter) + 1), label_y)
    _draw_shap(ax_s, ax_l, study, chr(ord(letter) + 2), label_y)
    _tighten_row(ax_a, ax_b, ax_l, ax_s)
    _widen_shap(ax_l, ax_s)
    return ax_s, ax_l


def make_figure(remc, azt1d, mitbih):
    fig = plt.figure(figsize=mm_size(166, 236))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.05, 1.05, 1.35], wspace=0.40, hspace=0.48,
                         left=0.07, right=0.98, top=0.97, bottom=0.05)
    pairs = [
        _row(fig, gs, 0, remc, 'A', 'Bin', legend_loc='upper left', label_y=1.0),
    ]
    nt = azt1d['n_time']
    times = [(b - (nt - 1)) * 5 for b in range(azt1d['seg_start'], azt1d['seg_end'] + 1)]
    pairs.append(_row(fig, gs, 1, azt1d, 'D', 'Time (min)', times, 'center left'))
    r0 = 101
    mt = [(b - r0) / 360 * 1000 for b in range(mitbih['seg_start'], mitbih['seg_end'] + 1)]
    ax_g, ax_h = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    gs_i = gs[2, 2].subgridspec(1, 2, width_ratios=[0.55, 1.35], wspace=0.04)
    ax_il = fig.add_subplot(gs_i[0, 0])
    ax_i = fig.add_subplot(gs_i[0, 1], sharey=ax_il)
    _draw_signal_classes(ax_g, mitbih, 'Time (ms)', 'G', mt)
    _draw_density_classes(ax_h, mitbih, 'H')
    _draw_shap(ax_i, ax_il, mitbih, 'I', xlabel='Mean |SHAP|')
    _tighten_row(ax_g, ax_h, ax_il, ax_i)
    _widen_shap(ax_il, ax_i)
    pairs.append((ax_i, ax_il))
    _place_shap_cbars(fig, pairs)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=350, facecolor='white', pad_inches=0.01, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for old in (RESULTS / 'domain_interpretation_summary.csv',
                RESULTS / 'domain_interpretation_features.csv',
                RESULTS / 'domain_interpretation_figure_data.pkl'):
        old.unlink(missing_ok=True)
    remc, azt1d, mitbih = run_remc(), run_azt1d(1), run_mitbih()
    _to_csv(remc, azt1d, mitbih)
    make_figure(remc, azt1d, mitbih)
    for res in (remc, azt1d, mitbih):
        s, fd = res['summary'], res['features_df']
        print(s['dataset'], s['metric_name'], round(float(s['metric_value']), 4),
              s['top_feature_name'], 'shap_class', s.get('shap_class'),
              res['hi_label'], round(float(np.mean(res['high_vals'])), 3),
              res['lo_label'], round(float(np.mean(res['low_vals'])), 3),
              'seg', res['seg_start'], res['seg_end'], flush=True)
        if res.get('class_labels'):
            print(' '.join(f"{lab} {float(np.mean(v)):.3f}" for lab, v in zip(res['class_labels'], res['class_vals'])), flush=True)
            print(fd.head(5).filter(regex='feature_name|mean_abs|abs_shap').to_string(index=False), flush=True)
    print(f'wrote {CSV_PATH} and {FIG_PATH}', flush=True)


if __name__ == '__main__':
    main()
