"""Domain interpretation: SHAP analysis of SublimeX on REMC E003 and AZT1D.

Uses the baseline SublimeX features (mean objective, LightGBM) from the
main evaluation.  Trains LightGBM on the extracted features, computes
SHAP values, and produces a six-panel figure:
  Row 1 (REMC E003, fold 1):
    (A) Signal within the extracted segment of the top feature by class
    (B) Class-conditional distribution of the top feature
    (C) SHAP beeswarm plot for all features
  Row 2 (AZT1D, fold 1):
    (D) Signal within the extracted segment of the top feature
        by high / low glucose change
    (E) Distribution of the top feature by glucose-change group
    (F) SHAP beeswarm plot for all features

When domain_interpretation_summary.csv and domain_interpretation_features.csv
exist and cached figure data is present, only the figure is regenerated.
"""
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import shap
from lightgbm import LGBMClassifier, LGBMRegressor

from core import TRANSFORMS, extract_feature
from preprocess import load_remc, load_azt1d

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
IMAGE_DIR = BASE_DIR / "elsarticle" / "images"
PARAMS_DIR = BASE_DIR / "parameters"
FIGURE_CACHE = RESULTS_DIR / "domain_interpretation_figure_data.pkl"

K_FOLDS = 5

# ---- REMC constants ----
REMC_CHANNEL_NAMES = [
    'H3K4me3', 'H3K4me1', 'H3K36me3',
    'H3K9me3', 'H3K27me3',
]

# ---- AZT1D constants ----
AZT1D_CHANNEL_NAMES = ['CGM', 'Insulin', 'Carbs']

TRANSFORM_NAMES = list(TRANSFORMS.keys())

TRANSFORM_DISPLAY = {
    'raw': 'raw', 'zscore': 'z-score',
    'derivative': 'deriv.', 'fft': 'FFT',
}

# ---- Figure style (shared) ----
AXIS_LABEL_FS = 22   # same for x- and y-axis labels
LEGEND_FS = 20
PANEL_LABEL_FS = 32
SHAP_FEATURE_FS = 17 # beeswarm y-axis (feature names)
COLORS = {'High': '#E53935', 'Low': '#1976D2'}
LINE_KW = dict(linewidth=2.5, marker='o', markersize=4)
KDE_KW = dict(fill=True, alpha=0.3, linewidth=2)
GRID_ALPHA = 0.3
LEGEND_LOC = 'upper right'   # same relative position in all panels


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def get_feature_name(params, n_time, channel_names):
    """Human-readable name from SublimeX parameters."""
    ch = int(params['ch'])
    t = int(params['t'])
    c, r = params['c'], params['r']
    center = c * (n_time - 1)
    half = (r * (n_time - 1)) * 0.5
    s = max(0, int(center - half))
    e = min(n_time - 1, int(center + half))
    ch_name = channel_names[ch]
    t_name = TRANSFORM_DISPLAY.get(
        TRANSFORM_NAMES[t], TRANSFORM_NAMES[t])
    return f"{ch_name} {t_name} {s}-{e}"


def apply_transforms_batch(X_list):
    """Apply all transforms to data once."""
    arrays = [
        x.values.astype(np.float32) if hasattr(x, 'values')
        else np.asarray(x, dtype=np.float32)
        for x in X_list
    ]
    data = np.stack(arrays, axis=1).astype(np.float32)
    n_samples, n_channels, n_time = data.shape
    transform_names = list(TRANSFORMS.keys())
    transformed = np.empty(
        (len(transform_names), n_samples, n_channels, n_time),
        dtype=np.float32)
    for ti, tname in enumerate(transform_names):
        transformed[ti] = TRANSFORMS[tname](
            data.reshape(-1, n_time)
        ).reshape(n_samples, n_channels, n_time)
    return transformed, n_channels, n_time, transform_names


def extract_all_features(transformed, params_list, n_channels,
                         n_time, transform_names):
    """Extract all features from pre-transformed data."""
    ctx = {
        'transformed': transformed,
        'n_channels': n_channels,
        'n_time': n_time,
        'transform_names': transform_names,
    }
    features = [extract_feature(p, ctx) for p in params_list]
    if features:
        return np.hstack(features).astype(np.float32)
    return np.empty(
        (transformed.shape[1], 0), dtype=np.float32)


# ------------------------------------------------------------------ #
#  REMC analysis                                                      #
# ------------------------------------------------------------------ #

def run_remc():
    """Run SHAP analysis on REMC E003 fold 1."""
    X_list, y, _info = load_remc(cell_line='E003')

    folds = list(StratifiedKFold(
        K_FOLDS, shuffle=True, random_state=42,
    ).split(pd.concat(X_list, axis=1), y))
    tr_idx, te_idx = folds[0]

    y_tr = y.iloc[tr_idx].values
    y_te = y.iloc[te_idx].values

    params_path = PARAMS_DIR / "remc_E003" / "fold1.json"
    with open(params_path) as f:
        params = json.load(f)

    transformed_all, n_channels, n_time, transform_names = \
        apply_transforms_batch(X_list)

    train_feat = extract_all_features(
        transformed_all[:, tr_idx], params,
        n_channels, n_time, transform_names)
    test_feat = extract_all_features(
        transformed_all[:, te_idx], params,
        n_channels, n_time, transform_names)

    feat_names = [
        get_feature_name(p, n_time, REMC_CHANNEL_NAMES)
        for p in params
    ]

    lgb = LGBMClassifier(
        max_depth=5, verbosity=-1,
        force_row_wise=True, num_threads=-1)
    lgb.fit(train_feat, y_tr)
    proba = lgb.predict_proba(test_feat)[:, 1]
    auc = roc_auc_score(y_te, proba)
    print(f"[REMC] LightGBM AUC = {auc:.4f}")
    print(f"[REMC] Features: {len(feat_names)} extracted")

    explainer = shap.TreeExplainer(lgb)
    shap_values = explainer.shap_values(test_feat)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1]

    top_global = top_idx[0]
    top_name = feat_names[top_global]
    top_params = params[top_global]

    print(f"[REMC] Top feature: {top_name} "
          f"(mean |SHAP| = {mean_abs_shap[top_global]:.4f})")
    for i, idx in enumerate(top_idx[:5], 1):
        print(f"  #{i}: {feat_names[idx]} "
              f"(|SHAP| = {mean_abs_shap[idx]:.4f})")

    # Panel A data
    ch = int(top_params['ch'])
    t = int(top_params['t'])
    c, r = top_params['c'], top_params['r']
    center = c * (n_time - 1)
    half = (r * (n_time - 1)) * 0.5
    seg_start = max(0, int(center - half))
    seg_end = min(n_time - 1, int(center + half))

    ch_name = REMC_CHANNEL_NAMES[ch]
    transform_name = TRANSFORM_NAMES[t]
    transform_fn = TRANSFORMS[transform_name]
    transform_label = TRANSFORM_DISPLAY.get(
        transform_name, transform_name)

    raw_data = X_list[ch].iloc[tr_idx].values.astype(
        np.float32)
    transformed_ch = transform_fn(
        raw_data.reshape(-1, n_time)).reshape(-1, n_time)

    high_mask = y_tr == 1
    low_mask = y_tr == 0
    high_mean = transformed_ch[high_mask].mean(axis=0)
    low_mean = transformed_ch[low_mask].mean(axis=0)

    # Panel B data
    top_feat_vals_te = test_feat[:, top_global]
    high_vals = top_feat_vals_te[y_te == 1]
    low_vals = top_feat_vals_te[y_te == 0]

    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    summary_remc = {
        'dataset': 'remc_E003', 'fold': 1,
        'method': 'SublimeX_LightGBM_SHAP',
        'auc': auc,
        'n_features': len(feat_names),
        'top_feature_name': top_name,
        'top_feature_shap': mean_abs_shap[top_global],
        'top_feature_high_expr_mean': high_vals.mean(),
        'top_feature_low_expr_mean': low_vals.mean(),
        'top_feature_separation':
            abs(high_vals.mean() - low_vals.mean()),
    }

    features_remc = pd.DataFrame({
        'rank': range(1, len(feat_names) + 1),
        'feature_name': [feat_names[i] for i in top_idx],
        'mean_abs_shap': mean_abs_shap[top_idx].tolist(),
    })

    return {
        'summary': summary_remc,
        'features_df': features_remc,
        'seg_start': seg_start,
        'seg_end': seg_end,
        'high_mean': high_mean,
        'low_mean': low_mean,
        'ch_name': ch_name,
        'transform_label': transform_label,
        'top_name': top_name,
        'high_vals': high_vals,
        'low_vals': low_vals,
        'sv': sv,
        'top_idx': top_idx,
        'test_feat': test_feat,
        'feat_names': feat_names,
        'base_value': base_value,
        'explainer': explainer,
    }


# ------------------------------------------------------------------ #
#  AZT1D analysis                                                     #
# ------------------------------------------------------------------ #

def run_azt1d():
    """Run SHAP analysis on AZT1D (fold 1 / temporal split)."""
    X_list, y, info = load_azt1d()
    subject_ids = info['subject_ids']

    # 80/20 temporal split per patient (same as main eval)
    tr_indices, te_indices = [], []
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        n = mask.sum()
        cutoff = int(n * 0.8)
        subj_idx = np.where(mask)[0]
        tr_indices.extend(subj_idx[:cutoff])
        te_indices.extend(subj_idx[cutoff:])
    tr_idx = np.array(tr_indices)
    te_idx = np.array(te_indices)

    y_tr = y.iloc[tr_idx].values
    y_te = y.iloc[te_idx].values

    params_path = PARAMS_DIR / "azt1d" / "fold1.json"
    with open(params_path) as f:
        params = json.load(f)

    transformed_all, n_channels, n_time, transform_names = \
        apply_transforms_batch(X_list)

    train_feat = extract_all_features(
        transformed_all[:, tr_idx], params,
        n_channels, n_time, transform_names)
    test_feat = extract_all_features(
        transformed_all[:, te_idx], params,
        n_channels, n_time, transform_names)

    feat_names = [
        get_feature_name(p, n_time, AZT1D_CHANNEL_NAMES)
        for p in params
    ]

    lgb = LGBMRegressor(
        max_depth=5, verbosity=-1,
        force_row_wise=True, num_threads=-1)
    lgb.fit(train_feat, y_tr)
    preds = lgb.predict(test_feat)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    print(f"\n[AZT1D] LightGBM RMSE = {rmse:.4f}")
    print(f"[AZT1D] Features: {len(feat_names)} extracted")

    explainer = shap.TreeExplainer(lgb)
    sv = explainer.shap_values(test_feat)

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1]

    top_global = top_idx[0]
    top_name = feat_names[top_global]
    top_params = params[top_global]

    print(f"[AZT1D] Top feature: {top_name} "
          f"(mean |SHAP| = {mean_abs_shap[top_global]:.4f})")
    for i, idx in enumerate(top_idx[:5], 1):
        print(f"  #{i}: {feat_names[idx]} "
              f"(|SHAP| = {mean_abs_shap[idx]:.4f})")

    # Panel D data: signal within top feature segment
    ch = int(top_params['ch'])
    t = int(top_params['t'])
    c, r = top_params['c'], top_params['r']
    center = c * (n_time - 1)
    half = (r * (n_time - 1)) * 0.5
    seg_start = max(0, int(center - half))
    seg_end = min(n_time - 1, int(center + half))

    ch_name = AZT1D_CHANNEL_NAMES[ch]
    transform_name = TRANSFORM_NAMES[t]
    transform_fn = TRANSFORMS[transform_name]
    transform_label = TRANSFORM_DISPLAY.get(
        transform_name, transform_name)

    raw_data = X_list[ch].iloc[tr_idx].values.astype(
        np.float32)
    transformed_ch = transform_fn(
        raw_data.reshape(-1, n_time)).reshape(-1, n_time)

    # Split by median glucose change (regression target)
    median_change = np.median(y_tr)
    high_mask = y_tr >= median_change
    low_mask = y_tr < median_change
    high_mean = transformed_ch[high_mask].mean(axis=0)
    low_mean = transformed_ch[low_mask].mean(axis=0)

    # Panel E data
    top_feat_vals_te = test_feat[:, top_global]
    median_change_te = np.median(y_te)
    high_vals = top_feat_vals_te[y_te >= median_change_te]
    low_vals = top_feat_vals_te[y_te < median_change_te]

    # Summary
    summary_azt1d = {
        'dataset': 'azt1d', 'fold': 1,
        'method': 'SublimeX_LightGBM_SHAP',
        'rmse': rmse,
        'n_features': len(feat_names),
        'top_feature_name': top_name,
        'top_feature_shap': mean_abs_shap[top_global],
        'top_feature_high_change_mean': high_vals.mean(),
        'top_feature_low_change_mean': low_vals.mean(),
        'top_feature_separation':
            abs(high_vals.mean() - low_vals.mean()),
    }

    features_azt1d = pd.DataFrame({
        'rank': range(1, len(feat_names) + 1),
        'feature_name': [feat_names[i] for i in top_idx],
        'mean_abs_shap': mean_abs_shap[top_idx].tolist(),
    })

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]

    return {
        'summary': summary_azt1d,
        'features_df': features_azt1d,
        'seg_start': seg_start,
        'seg_end': seg_end,
        'high_mean': high_mean,
        'low_mean': low_mean,
        'ch_name': ch_name,
        'transform_label': transform_label,
        'top_name': top_name,
        'high_vals': high_vals,
        'low_vals': low_vals,
        'sv': sv,
        'top_idx': top_idx,
        'test_feat': test_feat,
        'feat_names': feat_names,
        'base_value': base_value,
        'explainer': explainer,
        'n_time': n_time,
    }


# ------------------------------------------------------------------ #
#  Six-panel figure                                                   #
# ------------------------------------------------------------------ #

def _panel_label(ax, letter, x=-0.03, y=1.01):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FS, fontweight='bold', va='bottom')


def _draw_signal_panel(ax, high_mean, low_mean, seg_start, seg_end,
                       ch_name, transform_label, xlabel,
                       high_label, low_label, letter,
                       x_times=None):
    """Draw signal segment (panel A or D)."""
    seg_slice = slice(seg_start, seg_end + 1)
    if x_times is None:
        x_times = np.arange(seg_start, seg_end + 1)
    ax.plot(x_times, high_mean[seg_slice], color=COLORS['High'],
            label=high_label, **LINE_KW)
    ax.plot(x_times, low_mean[seg_slice], color=COLORS['Low'],
            label=low_label, **LINE_KW)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(f"{ch_name} ({transform_label})", fontsize=AXIS_LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc=LEGEND_LOC)
    ax.grid(True, alpha=GRID_ALPHA)
    if x_times is None:
        ax.set_xlim(seg_start - 0.5, seg_end + 0.5)
    _panel_label(ax, letter)


def _draw_density_panel(ax, high_vals, low_vals, top_name,
                        high_label, low_label, letter):
    """Draw density (panel B or E)."""
    sns.kdeplot(high_vals, ax=ax, color=COLORS['High'],
                **KDE_KW, label=high_label)
    sns.kdeplot(low_vals, ax=ax, color=COLORS['Low'],
                **KDE_KW, label=low_label)
    ax.set_xlabel(top_name, fontsize=AXIS_LABEL_FS)
    ax.set_ylabel('Density', fontsize=AXIS_LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc=LEGEND_LOC)
    ax.grid(True, alpha=GRID_ALPHA)
    _panel_label(ax, letter)


def _draw_shap_panel(ax, sv, top_idx, test_feat, feat_names, base_value,
                     letter):
    """Draw SHAP beeswarm (panel C or F)."""
    explanation = shap.Explanation(
        values=sv[:, top_idx],
        base_values=base_value,
        data=test_feat[:, top_idx],
        feature_names=[feat_names[i] for i in top_idx],
    )
    plt.sca(ax)
    shap.plots.beeswarm(explanation, show=False, max_display=len(feat_names))
    ax.tick_params(axis='y', labelsize=SHAP_FEATURE_FS, pad=1)
    ax.set_xlabel(ax.get_xlabel(), fontsize=AXIS_LABEL_FS)
    ax.set_ylim(-1, len(feat_names))
    _panel_label(ax, letter, x=0.0)


def make_figure(remc, azt1d):
    """Create a 2x3 figure (REMC top, AZT1D bottom)."""
    os.makedirs(IMAGE_DIR, exist_ok=True)

    base_remc = remc.get('base_value')
    if base_remc is None and 'explainer' in remc:
        ev = remc['explainer'].expected_value
        base_remc = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
    base_azt = azt1d.get('base_value')
    if base_azt is None and 'explainer' in azt1d:
        ev = azt1d['explainer'].expected_value
        base_azt = ev if not isinstance(ev, (list, np.ndarray)) else ev[0]

    fig_width, fig_height = 46, 18
    fig = plt.figure(figsize=(fig_width, fig_height))
    # Left block (A-B, D-E): moderate wspace
    gs_top_ab = fig.add_gridspec(
        1, 2, left=0.02, right=0.56,
        top=0.96, bottom=0.53, wspace=0.15)
    gs_bot_de = fig.add_gridspec(
        1, 2, left=0.02, right=0.56,
        top=0.47, bottom=0.04, wspace=0.15)
    # Right block (C, F): more gap from B/E; extend to right edge (minimal margin)
    gs_top_c = fig.add_gridspec(1, 1, left=0.65, right=0.995,
                                top=0.96, bottom=0.53)
    gs_bot_f = fig.add_gridspec(1, 1, left=0.65, right=0.995,
                                top=0.47, bottom=0.04)

    ax_a = fig.add_subplot(gs_top_ab[0, 0])
    ax_b = fig.add_subplot(gs_top_ab[0, 1])
    ax_c = fig.add_subplot(gs_top_c[0, 0])
    ax_d = fig.add_subplot(gs_bot_de[0, 0])
    ax_e = fig.add_subplot(gs_bot_de[0, 1])
    ax_f = fig.add_subplot(gs_bot_f[0, 0])

    # Row 1: REMC
    _draw_signal_panel(
        ax_a, remc['high_mean'], remc['low_mean'],
        remc['seg_start'], remc['seg_end'],
        remc['ch_name'], remc['transform_label'],
        'Bin (100 bp)',
        'High expression', 'Low expression', 'A')
    _draw_density_panel(
        ax_b, remc['high_vals'], remc['low_vals'], remc['top_name'],
        'High expression', 'Low expression', 'B')
    _draw_shap_panel(
        ax_c, remc['sv'], remc['top_idx'], remc['test_feat'],
        remc['feat_names'], base_remc, 'C')

    # Row 2: AZT1D
    n_time = azt1d['n_time']
    seg_bins = np.arange(azt1d['seg_start'], azt1d['seg_end'] + 1)
    seg_times = [(b - (n_time - 1)) * 5 for b in seg_bins]
    _draw_signal_panel(
        ax_d, azt1d['high_mean'], azt1d['low_mean'],
        azt1d['seg_start'], azt1d['seg_end'],
        azt1d['ch_name'], azt1d['transform_label'],
        'Time (min before prediction)',
        'Rising glucose', 'Falling glucose', 'D',
        x_times=seg_times)
    _draw_density_panel(
        ax_e, azt1d['high_vals'], azt1d['low_vals'], azt1d['top_name'],
        'Rising glucose', 'Falling glucose', 'E')
    _draw_shap_panel(
        ax_f, azt1d['sv'], azt1d['top_idx'], azt1d['test_feat'],
        azt1d['feat_names'], base_azt, 'F')

    out = IMAGE_DIR / "domain_interpretation.png"
    # Force size before save (SHAP or other code can resize the figure)
    fig.set_size_inches(fig_width, fig_height)
    # Do not use bbox_inches='tight' so figsize directly controls output dimensions
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close()


def _figure_data_for_cache(d):
    """Return a copy of the result dict without explainer (for pickle)."""
    out = {k: v for k, v in d.items() if k != 'explainer'}
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = RESULTS_DIR / "domain_interpretation_summary.csv"
    features_path = RESULTS_DIR / "domain_interpretation_features.csv"

    if (summary_path.exists() and features_path.exists()
            and FIGURE_CACHE.exists()):
        with open(FIGURE_CACHE, 'rb') as f:
            remc, azt1d = pickle.load(f)
        make_figure(remc, azt1d)
        return

    remc = run_remc()
    azt1d = run_azt1d()

    summary_df = pd.DataFrame([remc['summary'], azt1d['summary']])
    summary_df.to_csv(summary_path, index=False)

    remc['features_df']['dataset'] = 'remc_E003'
    azt1d['features_df']['dataset'] = 'azt1d'
    features_df = pd.concat(
        [remc['features_df'], azt1d['features_df']], ignore_index=True)
    features_df.to_csv(features_path, index=False)

    with open(FIGURE_CACHE, 'wb') as f:
        pickle.dump(
            (_figure_data_for_cache(remc), _figure_data_for_cache(azt1d)), f)

    make_figure(remc, azt1d)


if __name__ == '__main__':
    main()
