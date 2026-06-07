"""Shared matplotlib styling for manuscript figures."""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BRAND_TEAL, BRAND_RED, BRAND_NAVY = "#0d9488", "#e11d48", "#172033"


def mm_size(w_mm, h_mm):
    return w_mm / 25.4, h_mm / 25.4


def feature_series(seed=3, n=280):
    """Synthetic single-channel signal with a salient central peak (SublimeX motif)."""
    xc = np.linspace(0, 1, 13)
    yc = np.array([.30, .35, .29, .42, .55, .78, .68, .82, .60, .47, .38, .33, .30])
    x = np.linspace(0, 1, n)
    y = np.interp(x, xc, yc) + 0.012 * np.random.default_rng(seed).standard_normal(n)
    return x, y


def draw_extracted_feature(ax, theme="light", seg=(0.33, 0.66), lw=2.2):
    """Extracted SublimeX feature, styled like the input-data signal (dark line +
    blue fill) with the optimized segment window and its mean highlighted in red."""
    x, y = feature_series()
    if theme == "dark":
        ax.set_facecolor(BRAND_NAVY)
        line_c, fill_c, fill_a, box_edge, mean_c = "#2dd4bf", "#2dd4bf", 0.18, "#fb7185", "#f43f5e"
    else:
        line_c, fill_c, fill_a, box_edge, mean_c = "#111111", "#2196F3", 0.30, BRAND_RED, BRAND_RED
    s, e = seg
    m = (x >= s) & (x <= e)
    mean_v = float(y[m].mean())
    y_lo, y_hi = mean_v - 0.22, float(y[m].max()) + 0.07
    ax.fill_between(x, y, 0.08, color=fill_c, alpha=fill_a, lw=0, zorder=1)
    ax.plot(x, y, color=line_c, lw=lw, solid_capstyle="round", zorder=2)
    ax.add_patch(Rectangle((s, y_lo), e - s, y_hi - y_lo, facecolor="none",
                           edgecolor=box_edge, lw=2.4, ls=(0, (4, 3)), zorder=3))
    ax.hlines(mean_v, s, e, color=mean_c, lw=lw + 2.4, zorder=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.08, 1.0)
    ax.axis("off")
    return mean_v


def apply_manuscript_figure_style(base_font_size=12):
    mpl.rcParams.update({
        'font.size': base_font_size,
        'axes.titlesize': base_font_size + 1,
        'axes.labelsize': base_font_size,
        'legend.fontsize': base_font_size - 2,
        'xtick.labelsize': base_font_size - 1,
        'ytick.labelsize': base_font_size - 1,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.35,
    })


def save_png(fig, path, dpi=300, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight', **kwargs)


def save_png_eps(fig, path, dpi=300, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.eps'):
        fig.savefig(path.with_suffix(ext), dpi=dpi, bbox_inches='tight', **kwargs)


def save_png_pdf_eps(fig, path, dpi=300, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf', '.eps'):
        fig.savefig(path.with_suffix(ext), dpi=dpi, bbox_inches='tight', **kwargs)
