import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from plot_style import apply_manuscript_figure_style, mm_size

OUT = REPO / "elsarticle" / "clean" / "ecg_walkthrough.pdf"
FS = 400
SEG = (0.22, 0.34)
WIN = 0.80
SHADE = "#2196F3"
LINE = "#111111"
MEAN = "#e11d48"


def gauss(t, mu, sig, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)


def beat(t, p_amp):
    qrs = gauss(t, 0.0, 0.012, 1.35) + gauss(t, -0.018, 0.008, -0.28) + gauss(t, 0.022, 0.010, 0.22)
    return gauss(t, -0.16, 0.028, p_amp) + qrs + gauss(t, 0.22, 0.055, 0.38)


def one_beat(p_amp, fib, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(-0.45, 0.50, 1 / FS)
    y = beat(t, p_amp)
    if fib:
        y = y + gauss(t, -0.16, 0.045, fib)
        y = y + 0.04 * (np.sin(2 * np.pi * 6.5 * t) + 0.65 * np.sin(2 * np.pi * 8.7 * t + 1.1))
        y = y + 0.012 * rng.normal(size=t.size)
    m = (t >= -WIN / 2) & (t <= WIN / 2)
    tw, yw = t[m], y[m]
    z = (yw - yw.mean()) / yw.std()
    u = (tw - tw[0]) / WIN
    seg = (u >= SEG[0]) & (u < SEG[1])
    return u * WIN, z, float(z[seg].mean())


def scale_to(target, p_amp, seed):
    lo, hi = (0.2, 3.5) if p_amp else (0.02, 0.8)
    best = None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        amp, fb = (mid, 0.0) if p_amp else (0.0, mid)
        x, z, mu = one_beat(amp, fb, seed)
        best = (x, z, mu)
        lo, hi = (mid, hi) if mu < target else (lo, mid)
    return best


def main():
    normal = scale_to(0.58, True, 0)
    afib = scale_to(0.07, False, 1)
    apply_manuscript_figure_style(13)
    fig, axes = plt.subplots(1, 2, figsize=mm_size(220, 82), sharey=True, layout="constrained")
    shade = (SEG[0] * WIN, SEG[1] * WIN)
    for ax, (x, z, mu), title in zip(axes, (normal, afib), ("Normal rhythm", "Atrial fibrillation")):
        ax.axvspan(*shade, color=SHADE, alpha=0.30, lw=0, zorder=1)
        ax.plot(x, z, color=LINE, lw=1.5, solid_capstyle="round", zorder=2)
        ax.hlines(mu, *shade, color=MEAN, lw=2.4, zorder=3)
        ax.text(0.98, 0.96, f"mean $\\approx$ {mu:.2f}", transform=ax.transAxes, ha="right", va="top")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, WIN)
    axes[0].set_ylabel("Z-score")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(OUT, round(normal[2], 3), round(afib[2], 3))


main()
