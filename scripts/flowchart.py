"""SublimeX methodology flowchart (Figure~1) → elsarticle/flowchart.{eps,png}."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import path as mpl_path
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
from plot_style import save_png_eps

REPO_ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = REPO_ROOT / "scripts" / "_flowchart_tmp_icons"
OUT = REPO_ROOT / "elsarticle/flowchart"
FIGSIZE = (5, 3)
DERIV_C, FEAT_C = "#FF9800", "#E91E63"
SEG_START, SEG_END = 32, 68
RAW_C, FFT_C = "#4CAF50", "#9C27B0"
ARROW_LW, FEAT_VLINE_LW, FEAT_MEAN_LW = 4.8, 3.4, 4.4


def workflow_signal(n=100):
    t = np.linspace(0, 2 * np.pi, n)
    return t, np.sin(t) + 0.3 * np.sin(3 * t)


def workflow_views(n=100):
    t, signal = workflow_signal(n)
    x = np.arange(n)
    deriv = np.gradient(signal)
    y_pad = 0.08 * (deriv.max() - deriv.min())
    ylim = deriv.min() - y_pad, deriv.max() + y_pad
    return dict(t=t, x=x, signal=signal, deriv=deriv, ylim=ylim,
                seg=deriv[SEG_START:SEG_END + 1], mean=float(deriv[SEG_START:SEG_END + 1].mean()))


def _save_icon(fig, name, dpi=150):
    plt.tight_layout()
    fig.savefig(ICON_DIR / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_deriv_base(ax, v):
    ax.plot(v["x"], v["deriv"], color=DERIV_C, linewidth=1.5)
    ax.fill_between(v["x"], v["deriv"], alpha=0.3, color=DERIV_C)
    ax.set_xlim(0, len(v["x"]) - 1)
    ax.set_ylim(*v["ylim"])


def _plot_feature_on_view(ax, x, data, start, end, line_c):
    pad = 0.08 * (data.max() - data.min())
    ax.plot(x, data, color=line_c, lw=1.3)
    ax.fill_between(x, data, alpha=0.25, color=line_c)
    mean_v = float(data[start:end + 1].mean())
    for xi in (start, end):
        ax.axvline(xi, color=FEAT_C, lw=FEAT_VLINE_LW, ls=(0, (4, 3)), zorder=3)
    ax.hlines(mean_v, start, end, color=FEAT_C, lw=FEAT_MEAN_LW, zorder=4)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(data.min() - pad, data.max() + pad)
    ax.axis("off")


def _plot_segment_markers(ax, show_mean=False):
    v = workflow_views()
    _plot_deriv_base(ax, v)
    for xi in (SEG_START, SEG_END):
        ax.axvline(xi, color=FEAT_C, lw=FEAT_VLINE_LW, ls=(0, (4, 3)), zorder=3)
    if show_mean:
        ax.hlines(v["mean"], SEG_START, SEG_END, color=FEAT_C, lw=FEAT_MEAN_LW, zorder=4)
    ax.axis("off")


def create_input_icon():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    v = workflow_views()
    ax.plot(v["x"], v["signal"], "k-", linewidth=1.5)
    ax.fill_between(v["x"], v["signal"], alpha=0.3, color="#2196F3")
    y_pad = 0.08 * (v["signal"].max() - v["signal"].min())
    ax.set_xlim(0, len(v["x"]) - 1)
    ax.set_ylim(v["signal"].min() - y_pad, v["signal"].max() + y_pad)
    ax.axis("off")
    _save_icon(fig, "input_icon.png")


def create_transforms_icon():
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    v = workflow_views()
    signal = v["signal"]
    transforms = [
        ("Raw", signal, "#4CAF50"),
        ("Z-score", (signal - signal.mean()) / signal.std(), "#2196F3"),
        ("Derivative", v["deriv"], DERIV_C),
        ("FFT Power", np.abs(np.fft.rfft(signal)) ** 2, "#9C27B0"),
    ]
    for ax, (name, data, color) in zip(axes.flat, transforms):
        ax.plot(data, color=color, linewidth=1.5)
        ax.fill_between(range(len(data)), data, alpha=0.3, color=color)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.axis("off")
    _save_icon(fig, "transforms_icon.png")


def create_aggregation_icon():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_segment_markers(ax, show_mean=True)
    _save_icon(fig, "aggregation_icon.png", dpi=170)


def create_optimization_icon():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pts = np.array([0.08, 0.18, 0.32, 0.5, 0.66, 0.78, 0.9])
    y = 0.55 + 0.25 * np.sin(6 * np.pi * pts) + 0.1 * np.cos(2 * np.pi * pts)
    xg = np.sort(np.unique(np.concatenate([np.linspace(0, 1, 260), pts])))
    K = np.exp(-0.5 * ((pts[:, None] - pts[None, :]) / 0.12) ** 2) + 1e-6 * np.eye(len(pts))
    Ks = np.exp(-0.5 * ((pts[:, None] - xg[None, :]) / 0.12) ** 2)
    Kinv = np.linalg.inv(K)
    mu = Ks.T @ Kinv @ y
    s = np.sqrt(np.maximum(1 - np.sum(Ks * (Kinv @ Ks), axis=0), 0))
    ax.fill_between(xg, mu - s, mu + s, color='#90CAF9', alpha=0.55, linewidth=0)
    ax.plot(xg, mu, color='#1E88E5', lw=2.2)
    ax.scatter(pts, y, s=55, color='#222', zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(ICON_DIR / "optimization_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_ml_model_icon():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    root_x, root_y = 0.5, 0.86
    ax.add_patch(FancyBboxPatch((root_x - 0.14, root_y - 0.09), 0.28, 0.18, boxstyle="round,pad=0.01",
                                facecolor='#2196F3', edgecolor='#333', linewidth=1.5, zorder=3))
    ax.text(root_x, root_y, "x < 0.5", ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=4)
    left_x, right_x, mid_y = 0.25, 0.75, 0.48
    for bx, label in [(left_x, "y < 0.3"), (right_x, "z > 0.7")]:
        ax.add_patch(FancyBboxPatch((bx - 0.12, mid_y - 0.09), 0.24, 0.18, boxstyle="round,pad=0.01",
                                    facecolor='#4CAF50', edgecolor='#333', linewidth=1.5, zorder=3))
        ax.text(bx, mid_y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=4)
    leaf_y, lw, lh = 0.14, 0.17, 0.14
    for lx, lab in [(0.15, "Class A"), (0.35, "Class B"), (0.65, "Class A"), (0.85, "Class B")]:
        ax.add_patch(FancyBboxPatch((lx - lw / 2, leaf_y - lh / 2), lw, lh, boxstyle="round,pad=0.01",
                                    facecolor='#FF9800', edgecolor='#333', linewidth=1.5, zorder=3))
        ax.text(lx, leaf_y, lab, ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=4)
    ax.plot([root_x, left_x], [root_y - 0.08, mid_y + 0.08], 'k-', lw=2, zorder=1)
    ax.plot([root_x, right_x], [root_y - 0.08, mid_y + 0.08], 'k-', lw=2, zorder=1)
    ax.plot([left_x, 0.15], [mid_y - 0.08, leaf_y + lh / 2], 'k-', lw=1.5, zorder=1)
    ax.plot([left_x, 0.35], [mid_y - 0.08, leaf_y + lh / 2], 'k-', lw=1.5, zorder=1)
    ax.plot([right_x, 0.65], [mid_y - 0.08, leaf_y + lh / 2], 'k-', lw=1.5, zorder=1)
    ax.plot([right_x, 0.85], [mid_y - 0.08, leaf_y + lh / 2], 'k-', lw=1.5, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(ICON_DIR / "ml_model_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def workflow_feature_panels():
    v = workflow_views()
    fft = np.abs(np.fft.rfft(v["signal"])) ** 2
    x_fft = np.arange(len(fft))
    return (
        (v["x"], v["signal"], 35, 70, RAW_C),
        (v["x"], v["deriv"], SEG_START, SEG_END, DERIV_C),
        (x_fft, fft, 4, 22, FFT_C),
    )


APPEND_FEATURES_FIGSIZE = (8, 6)


def _save_features_stack_icon(panels, name, dpi=170, figsize=None):
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=figsize or (5, 2 * n))
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        _plot_feature_on_view(ax, *panel)
    fig.subplots_adjust(hspace=0.35)
    _save_icon(fig, name, dpi=dpi)


def create_append_features_icon():
    _save_features_stack_icon(workflow_feature_panels(), "append_features_icon.png",
                              figsize=APPEND_FEATURES_FIGSIZE)


def create_output_icon():
    p = workflow_feature_panels()
    _save_features_stack_icon((p[0], p[2]), "output_icon.png")


class FlowchartBox:
    W, H = 64, 17
    IMG_W, IMG_H = 29, H * 0.86
    TITLE_FS, SUBTEXT_FS = 28, 24
    INNER_TITLE_FS, INNER_SUBTEXT_FS = 30, 26

    @staticmethod
    def draw(ax, x, y, title, color, subtext=None, img_path=None, add_img_func=None,
             title_fs=None, subtext_fs=None):
        w, h = FlowchartBox.W, FlowchartBox.H
        title_fs = title_fs or FlowchartBox.TITLE_FS
        subtext_fs = subtext_fs or FlowchartBox.SUBTEXT_FS
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=1",
                                    facecolor=color, edgecolor='#333', linewidth=2, zorder=2))
        img_x, text_x = x - w / 2 + FlowchartBox.IMG_W / 2 + 2, x + w * 0.26
        if img_path and add_img_func:
            add_img_func(img_path, img_x, y, FlowchartBox.IMG_W, FlowchartBox.IMG_H)
        ax.text(text_x, y + h * 0.30, title, ha='center', va='center',
                fontsize=title_fs, fontweight='bold', color='#222', zorder=4,
                linespacing=1.05)
        ax.text(text_x, y - h * 0.18, subtext, ha='center', va='center',
                fontsize=subtext_fs, color='#444', zorder=4, linespacing=1.15)
        return y - h / 2


def draw_flowchart():
    fig, ax = plt.subplots(figsize=(26, 26))
    cx, V_GAP = 66, 3.5
    FE_V_GAP, FE_H_GAP, FE_ROW_GAP = 1, 10, 5
    FE_PAD_TOP, FE_PAD_BOTTOM, FE_SIDE_PAD = 2.5, 1.5, 2
    OUTER_PAD_X, OUTER_PAD_Y = 2.5, 1.2
    H = FlowchartBox.H
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def add_img(img_path, x, y, w, h):
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xr, yr = xl[1] - xl[0], yl[1] - yl[0]
        ins = ax.inset_axes([(x - w / 2 - xl[0]) / xr, (y - h / 2 - yl[0]) / yr, w / xr, h / yr],
                            transform=ax.transAxes)
        ins.imshow(mpimg.imread(img_path))
        ins.axis('off')

    def box_xy(x, y, side):
        w, h = FlowchartBox.W / 2, FlowchartBox.H / 2
        return {'left': (x - w, y), 'right': (x + w, y), 'top': (x, y + h),
                'bottom': (x, y - h)}[side]

    def add_arrow(x1, y1, x2, y2, color='#555', lw=ARROW_LW):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    shrinkA=0, shrinkB=0, mutation_scale=26), zorder=3)

    def add_no_improvement_hook(exit_x, exit_y, dest_x, final_top, hook_y):
        verts = [(exit_x, exit_y), (exit_x, hook_y), (dest_x, hook_y), (dest_x, final_top)]
        path = mpl_path.Path(verts, [mpl_path.Path.MOVETO] + [mpl_path.Path.LINETO] * 3)
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle='->', color='#C62828', lw=ARROW_LW,
                                     mutation_scale=26, shrinkA=0, shrinkB=0, zorder=5))
        ax.text(dest_x - 2.5, (hook_y + final_top) / 2 + 0.5, 'if no improvement', fontsize=23,
                color='#C62828', ha='right', va='center', fontweight='bold', zorder=5)

    c_input, c_process, c_output, c_feature = '#42A5F5', '#81C784', '#78909C', '#C8E6C9'
    box_w = FlowchartBox.W * 2 + FE_H_GAP

    y_in = 120
    y_pre = y_in - H - V_GAP
    fe_top = y_pre - H / 2 - V_GAP
    row1_y = fe_top - FE_PAD_TOP - FE_V_GAP - H / 2
    row2_y = row1_y - H - FE_ROW_GAP
    fe_bottom = row2_y - H / 2 - FE_PAD_BOTTOM
    y_out = fe_bottom - V_GAP - H / 2
    grid_y = (row1_y + row2_y) / 2
    outer_l = cx - box_w / 2 - OUTER_PAD_X
    outer_r = cx + box_w / 2 + OUTER_PAD_X
    outer_b = y_out - H / 2 - OUTER_PAD_Y
    outer_t = y_in + H / 2 + OUTER_PAD_Y
    ax.set_xlim(outer_l - 0.4, outer_r + 0.4)
    ax.set_ylim(outer_b - 0.4, outer_t + 0.4)

    FlowchartBox.draw(ax, cx, y_in, "Input Data", c_input,
                      subtext="Time series\n(univariate or\nmultivariate)",
                      img_path=str(ICON_DIR / "input_icon.png"), add_img_func=add_img)
    FlowchartBox.draw(ax, cx, y_pre, "Precompute\nTransformations", c_process,
                      subtext="Apply 4 transforms\nper channel: raw, z-score,\nderivative, FFT power",
                      img_path=str(ICON_DIR / "transforms_icon.png"), add_img_func=add_img)
    add_arrow(cx, y_in - H / 2, cx, y_pre + H / 2)
    add_arrow(cx, y_pre - H / 2, cx, fe_top)

    ax.add_patch(FancyBboxPatch((cx - box_w / 2 - FE_SIDE_PAD, fe_bottom),
                                box_w + 2 * FE_SIDE_PAD, fe_top - fe_bottom,
                                boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor=c_feature, edgecolor='#2E7D32', linewidth=2.5,
                                zorder=0, alpha=0.6))
    ax.text(cx, fe_top - 0.5, "Iterative Feature Optimization",
            fontsize=30, color='#1B5E20', ha='center', va='top', fontweight='bold')

    col1_x = cx - FlowchartBox.W / 2 - FE_H_GAP / 2
    col2_x = cx + FlowchartBox.W / 2 + FE_H_GAP / 2
    inner_fs = dict(title_fs=FlowchartBox.INNER_TITLE_FS, subtext_fs=FlowchartBox.INNER_SUBTEXT_FS)

    FlowchartBox.draw(ax, col1_x, row1_y, "Feature Parameter\nSearch", c_process,
                      subtext="Optimizer proposes\ncandidate channel,\ntransform, segment",
                      img_path=str(ICON_DIR / "optimization_icon.png"), add_img_func=add_img, **inner_fs)
    FlowchartBox.draw(ax, col2_x, row1_y, "Extract Segment &\nCompute Mean", c_process,
                      subtext="Select segment [start:end]\non transformed data;\ncompute segment mean",
                      img_path=str(ICON_DIR / "aggregation_icon.png"), add_img_func=add_img, **inner_fs)
    FlowchartBox.draw(ax, col1_x, row2_y, "Append Feature", c_process,
                      subtext="Add feature to set;\nrepeat search on\nexpanded feature matrix",
                      img_path=str(ICON_DIR / "append_features_icon.png"), add_img_func=add_img, **inner_fs)
    FlowchartBox.draw(ax, col2_x, row2_y, "Evaluate Model", c_process,
                      subtext="Train LightGBM with\ncurrent features;\nget validation score",
                      img_path=str(ICON_DIR / "ml_model_icon.png"), add_img_func=add_img, **inner_fs)

    add_arrow(*box_xy(col1_x, row1_y, 'right'), *box_xy(col2_x, row1_y, 'left'))
    add_arrow(*box_xy(col2_x, row1_y, 'bottom'), *box_xy(col2_x, row2_y, 'top'))
    add_arrow(*box_xy(col2_x, row2_y, 'left'), *box_xy(col1_x, row2_y, 'right'))
    add_arrow(*box_xy(col1_x, row2_y, 'top'), *box_xy(col1_x, row1_y, 'bottom'))
    ax.text(cx, grid_y, 'repeat while improving',
            fontsize=24, color='#1B5E20', ha='center', va='center', fontweight='bold')

    final_top = y_out + H / 2
    hook_y = fe_bottom - 2.0
    FlowchartBox.draw(ax, cx, y_out, "Final Output", c_output,
                      subtext="Interpretable segment-\nbased features +\ntrain/test feature matrices",
                      img_path=str(ICON_DIR / "output_icon.png"), add_img_func=add_img)
    add_no_improvement_hook(*box_xy(col2_x, row2_y, 'bottom'), cx, final_top, hook_y)

    ax.add_patch(FancyBboxPatch((outer_l, outer_b), outer_r - outer_l, outer_t - outer_b,
                                boxstyle="round,pad=0.12,rounding_size=2",
                                facecolor='#F8FAFC', edgecolor='#455A64', linewidth=2.5,
                                zorder=-2))

    save_png_eps(fig, OUT, dpi=300, facecolor='white')
    plt.close(fig)


def main():
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (create_input_icon, create_transforms_icon, create_aggregation_icon,
               create_optimization_icon, create_ml_model_icon,
               create_append_features_icon, create_output_icon):
        fn()
    draw_flowchart()


if __name__ == "__main__":
    main()
