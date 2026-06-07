"""Graphical abstract: Iterative Feature Optimization panel → elsarticle/graphical_abstract.png."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import path as mpl_path
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from flowchart import (
    ARROW_LW,
    FlowchartBox,
    ICON_DIR,
    create_aggregation_icon,
    create_append_features_icon,
    create_ml_model_icon,
    create_optimization_icon,
)
from plot_style import save_png

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "elsarticle" / "graphical_abstract"
C_PROCESS, C_FEATURE = "#81C784", "#C8E6C9"
CX, Y_IN, V_GAP = 66, 120, 3.5
FE_V_GAP, FE_H_GAP, FE_ROW_GAP = 1, 10, 5
FE_PAD_TOP, FE_PAD_BOTTOM, FE_SIDE_PAD = 2.5, 1.5, 2
FLOWCHART_FIG = 26
STOP_EXTEND = 14
STOP_COLOR = "#C62828"
INNER_BOXES = [
    ("Feature Parameter\nSearch", "optimization_icon.png"),
    ("Extract Segment &\nCompute Mean", "aggregation_icon.png"),
    ("Append Feature", "append_features_icon.png"),
    ("Evaluate Model", "ml_model_icon.png"),
]


def fe_layout(cx=CX):
    H = FlowchartBox.H
    y_pre = Y_IN - H - V_GAP
    fe_top = y_pre - H / 2 - V_GAP
    row1_y = fe_top - FE_PAD_TOP - FE_V_GAP - H / 2
    row2_y = row1_y - H - FE_ROW_GAP
    fe_bottom = row2_y - H / 2 - FE_PAD_BOTTOM
    box_w = FlowchartBox.W * 2 + FE_H_GAP
    return dict(
        cx=cx, fe_top=fe_top, fe_bottom=fe_bottom, row1_y=row1_y, row2_y=row2_y,
        grid_y=(row1_y + row2_y) / 2,
        fe_left=cx - box_w / 2 - FE_SIDE_PAD,
        fe_right=cx + box_w / 2 + FE_SIDE_PAD,
        col1_x=cx - FlowchartBox.W / 2 - FE_H_GAP / 2,
        col2_x=cx + FlowchartBox.W / 2 + FE_H_GAP / 2,
    )


def add_img(ax, path, cx, cy, w, h):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    xr, yr = xl[1] - xl[0], yl[1] - yl[0]
    ins = ax.inset_axes(
        [(cx - w / 2 - xl[0]) / xr, (cy - h / 2 - yl[0]) / yr, w / xr, h / yr],
        transform=ax.transAxes)
    ins.imshow(mpimg.imread(path))
    ins.axis("off")


def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=ARROW_LW,
                                shrinkA=0, shrinkB=0, mutation_scale=26), zorder=3)


def box_xy(x, y, side):
    w, h = FlowchartBox.W / 2, FlowchartBox.H / 2
    return {"left": (x - w, y), "right": (x + w, y), "top": (x, y + h),
            "bottom": (x, y - h)}[side]


def add_no_improvement_exit(ax, lay):
    exit_x, exit_y = box_xy(lay["col2_x"], lay["row2_y"], "right")
    hook_x = lay["fe_right"] + 2.5
    end_x = lay["fe_right"] + STOP_EXTEND
    y = lay["grid_y"]
    verts = [(exit_x, exit_y), (hook_x, exit_y), (hook_x, y), (end_x, y)]
    path = mpl_path.Path(verts, [mpl_path.Path.MOVETO] + [mpl_path.Path.LINETO] * 3)
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="->", color=STOP_COLOR, lw=ARROW_LW,
                                 mutation_scale=26, shrinkA=0, shrinkB=0, zorder=5))
    ax.text((hook_x + end_x) / 2, y + 1.2, "if no\nimprovement", fontsize=23, color=STOP_COLOR,
            ha="center", va="bottom", fontweight="bold", zorder=5, linespacing=1.05)


def draw_inner_box(ax, x, y, title, img):
    w, h = FlowchartBox.W, FlowchartBox.H
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=1",
        facecolor=C_PROCESS, edgecolor="#333", linewidth=2, zorder=2))
    add_img(ax, str(ICON_DIR / img), x - w / 2 + FlowchartBox.IMG_W / 2 + 2, y,
            FlowchartBox.IMG_W, FlowchartBox.IMG_H)
    ax.text(x + w * 0.26, y + h * 0.30, title, ha="center", va="center",
            fontsize=FlowchartBox.INNER_TITLE_FS, fontweight="bold", color="#222",
            zorder=4, linespacing=1.05)


def main():
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (create_optimization_icon, create_aggregation_icon,
               create_append_features_icon, create_ml_model_icon):
        fn()

    lay = fe_layout()
    y0, y1 = lay["fe_bottom"] - 0.4, lay["fe_top"] + 0.4
    x0 = lay["fe_left"] - 0.4
    x1 = lay["fe_right"] + STOP_EXTEND + 1.5
    fig_w = FLOWCHART_FIG * (x1 - x0) / 143
    fig_h = FLOWCHART_FIG * (y1 - y0) / 108

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.add_patch(FancyBboxPatch(
        (lay["fe_left"], lay["fe_bottom"]), lay["fe_right"] - lay["fe_left"],
        lay["fe_top"] - lay["fe_bottom"], boxstyle="round,pad=0.25,rounding_size=1.2",
        facecolor=C_FEATURE, edgecolor="#2E7D32", linewidth=2.5, zorder=0, alpha=0.6))

    positions = [(lay["col1_x"], lay["row1_y"]), (lay["col2_x"], lay["row1_y"]),
                 (lay["col1_x"], lay["row2_y"]), (lay["col2_x"], lay["row2_y"])]
    for (x, y), (title, img) in zip(positions, INNER_BOXES):
        draw_inner_box(ax, x, y, title, img)

    add_arrow(ax, *box_xy(lay["col1_x"], lay["row1_y"], "right"),
              *box_xy(lay["col2_x"], lay["row1_y"], "left"))
    add_arrow(ax, *box_xy(lay["col2_x"], lay["row1_y"], "bottom"),
              *box_xy(lay["col2_x"], lay["row2_y"], "top"))
    add_arrow(ax, *box_xy(lay["col2_x"], lay["row2_y"], "left"),
              *box_xy(lay["col1_x"], lay["row2_y"], "right"))
    add_arrow(ax, *box_xy(lay["col1_x"], lay["row2_y"], "top"),
              *box_xy(lay["col1_x"], lay["row1_y"], "bottom"))
    ax.text(lay["cx"], lay["grid_y"], "repeat while improving", fontsize=24, color="#1B5E20",
            ha="center", va="center", fontweight="bold")
    add_no_improvement_exit(ax, lay)

    save_png(fig, OUT, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
