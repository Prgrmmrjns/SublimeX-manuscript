import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
import numpy as np
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ELSCARTICLE_FLOWCHART_DIR = REPO_ROOT / "elsarticle" / "images" / "flowchart"
TEMP_ICON_DIR = REPO_ROOT / "analysis_scripts" / "_flowchart_tmp_icons"

def create_input_icon():
    fig, ax = plt.subplots(figsize=(4, 1.5))
    t = np.linspace(0, 10, 200)
    signal = np.sin(t) + 0.5 * np.sin(3*t) + 0.3 * np.random.randn(200)
    ax.plot(t, signal, 'k-', linewidth=1.5)
    ax.fill_between(t, signal, alpha=0.3, color='#2196F3')
    ax.set_xlim(0, 10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "input_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_transforms_icon():
    fig, axes = plt.subplots(1, 3, figsize=(4, 1.2))
    t = np.linspace(0, 2*np.pi, 100)
    signal = np.sin(t) + 0.3 * np.sin(3*t)
    zscore = (signal - signal.mean()) / signal.std()
    fft_pow = np.abs(np.fft.rfft(signal)) ** 2
    transforms = [
        ("Raw", signal, "#4CAF50"),
        ("Zscore", zscore, "#2196F3"),
        ("FFT pow", fft_pow, "#9C27B0"),
    ]
    for ax, (name, data, color) in zip(axes, transforms):
        ax.plot(data, color=color, linewidth=1.5)
        ax.fill_between(range(len(data)), data, alpha=0.3, color=color)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "transforms_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_bspline_icon():
    fig, ax = plt.subplots(figsize=(3, 2))
    t = np.linspace(0, 1, 120, dtype=np.float32)
    # Cubic Bezier with 4 control points (c0, c1, c2, c3)
    # Make control points more distinct so curve clearly shows all 4
    c0, c1, c2, c3 = 0.20, 0.90, 0.30, 0.70
    # Correct cubic Bezier formula: B(t) = (1-t)³c₀ + 3(1-t)²t c₁ + 3(1-t)t² c₂ + t³c₃
    pattern = (1 - t) ** 3 * c0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * c3
    ax.plot(t, pattern, 'r-', linewidth=3)
    # Control points: in Bezier curves, c0 and c3 are on the curve (start/end)
    # c1 and c2 are control points that influence the shape
    # Position them at their actual x-coordinates (0, 1/3, 2/3, 1) for visualization
    t_control = np.array([0.0, 1/3, 2/3, 1.0])
    y_control = np.array([c0, c1, c2, c3])
    ax.scatter(t_control, y_control, c='blue', s=90, zorder=5, marker='o')
    ax.text(0.0, c0 - 0.08, "c0", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(1/3, c1 + 0.08, "c1", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(2/3, c2 - 0.08, "c2", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(1.0, c3 + 0.08, "c3", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.fill_between(t, pattern - 0.06, pattern + 0.06, alpha=0.18, color='red')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "bspline_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_sliding_mse_icon():
    """
    Visualize sliding-window matching: slide the pattern across a search interval,
    compute MSE per window, and take the minimum.
    """
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    n = 140
    x = np.arange(n)
    signal = (0.25 * np.sin(np.linspace(0, 5 * np.pi, n)) +
              0.08 * np.random.randn(n) + 0.5)

    # Pattern (cubic spline) to slide
    width = 28
    t = np.linspace(0, 1, width, dtype=np.float32)
    c0, c1, c2 = 0.30, 0.78, 0.38
    pattern = (1 - t) ** 2 * c0 + 2 * (1 - t) * t * c1 + t ** 2 * c2

    # Search interval (constrain overlays so windows stay fully inside)
    start, end = 38, 96
    end = min(end, n - 1)
    max_pos = max(start, end - width)

    positions = np.arange(start, max_pos + 1)
    mse = np.array([np.mean((signal[pos:pos + width] - pattern) ** 2) for pos in positions])
    best_idx = int(np.argmin(mse))
    best_pos = int(positions[best_idx])
    ax.axvspan(start, end, color="#C8E6C9", alpha=0.9, zorder=0)
    ax.plot(x, signal, color="#222", lw=1.6, alpha=0.9, zorder=2)
    ax.text((start + end) / 2, 0.985, "search interval", ha="center", va="top", fontsize=11, fontweight="bold", color="#1B5E20")
    example_positions = [start, start + (max_pos - start) // 3, start + 2 * (max_pos - start) // 3]
    example_positions = [int(p) for p in example_positions if p <= max_pos]
    for i, pos in enumerate(example_positions):
        ax.plot(x[pos:pos + width], pattern, color="#E91E63", lw=2.0,
                alpha=0.25 + 0.20 * i, zorder=3)

    # Best match window (always fully inside interval)
    ax.plot(x[best_pos:best_pos + width], pattern, color="#E91E63", lw=2.9, alpha=0.98, zorder=4)
    ax.annotate("best match\n(min MSE)",
                xy=(best_pos + width * 0.55,
                    pattern[int(width * 0.55)]),
                xytext=(best_pos + width * 0.55, 0.10),
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.4),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#1565C0")

    ax.set_xlim(0, n - 1)
    ax.set_ylim(0.02, 1.0)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "sliding_mse_icon.png", dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()

def create_optimization_icon():
    fig, ax = plt.subplots(figsize=(5, 2.5))
    pts = np.array([0.08, 0.18, 0.32, 0.5, 0.66, 0.78, 0.9])
    y = 0.55 + 0.25*np.sin(6*np.pi*pts) + 0.1*np.cos(2*np.pi*pts)
    x = np.sort(np.unique(np.concatenate([np.linspace(0, 1, 260), pts])))
    K = np.exp(-0.5*((pts[:, None]-pts[None, :])/0.12)**2) + 1e-6*np.eye(len(pts))
    Ks = np.exp(-0.5*((pts[:, None]-x[None, :])/0.12)**2)
    Kinv = np.linalg.inv(K)
    mu = (Ks.T @ Kinv @ y)
    var = 1 - np.sum(Ks * (Kinv @ Ks), axis=0)
    s = np.sqrt(np.maximum(var, 0))   
    ax.fill_between(x, mu-s, mu+s, color='#90CAF9', alpha=0.55, linewidth=0)
    ax.plot(x, mu, color='#1E88E5', lw=2.2)
    ax.scatter(pts, y, s=55, color='#222', zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Bayesian optimization", fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "optimization_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_propose_icon():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.random.rand(30) 
    y = np.random.rand(30)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 30))
    for i in range(30):
        ax.scatter(x[i], y[i], c=[colors[i]], s=40 + i*2, alpha=0.7, edgecolors='#333', linewidth=0.5)
    ax.annotate('', xy=(0.8, 0.85), xytext=(0.2, 0.15),
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=2, 
                               connectionstyle='arc3,rad=0.3'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "propose_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_ml_model_icon():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    # Decision tree structure: root -> internal nodes -> leaves
    # Root node
    root_x, root_y = 0.5, 0.85
    root_box = FancyBboxPatch((root_x - 0.12, root_y - 0.08), 0.24, 0.16,
                              boxstyle="round,pad=0.01", facecolor='#2196F3',
                              edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(root_box)
    ax.text(root_x, root_y, "x < 0.5", ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=4)
    
    # Internal nodes (level 1)
    left_x, right_x = 0.25, 0.75
    mid_y = 0.5
    # Left internal node
    left_box = FancyBboxPatch((left_x - 0.1, mid_y - 0.08), 0.2, 0.16,
                              boxstyle="round,pad=0.01", facecolor='#4CAF50',
                              edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(left_box)
    ax.text(left_x, mid_y, "y < 0.3", ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    
    # Right internal node
    right_box = FancyBboxPatch((right_x - 0.1, mid_y - 0.08), 0.2, 0.16,
                               boxstyle="round,pad=0.01", facecolor='#4CAF50',
                               edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(right_box)
    ax.text(right_x, mid_y, "z > 0.7", ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)
    
    # Leaf nodes (level 2)
    leaf_y = 0.15
    leaf_width, leaf_height = 0.15, 0.12
    
    # Left leaves
    leaf1_x, leaf2_x = 0.15, 0.35
    leaf1 = FancyBboxPatch((leaf1_x - leaf_width/2, leaf_y - leaf_height/2),
                           leaf_width, leaf_height,
                           boxstyle="round,pad=0.01", facecolor='#FF9800',
                           edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(leaf1)
    ax.text(leaf1_x, leaf_y, "Class A", ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=4)
    
    leaf2 = FancyBboxPatch((leaf2_x - leaf_width/2, leaf_y - leaf_height/2),
                           leaf_width, leaf_height,
                           boxstyle="round,pad=0.01", facecolor='#FF9800',
                           edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(leaf2)
    ax.text(leaf2_x, leaf_y, "Class B", ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=4)
    
    # Right leaves
    leaf3_x, leaf4_x = 0.65, 0.85
    leaf3 = FancyBboxPatch((leaf3_x - leaf_width/2, leaf_y - leaf_height/2),
                           leaf_width, leaf_height,
                           boxstyle="round,pad=0.01", facecolor='#FF9800',
                           edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(leaf3)
    ax.text(leaf3_x, leaf_y, "Class A", ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=4)
    
    leaf4 = FancyBboxPatch((leaf4_x - leaf_width/2, leaf_y - leaf_height/2),
                           leaf_width, leaf_height,
                           boxstyle="round,pad=0.01", facecolor='#FF9800',
                           edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(leaf4)
    ax.text(leaf4_x, leaf_y, "Class B", ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=4)
    
    # Branches (edges)
    # Root to internal nodes
    ax.plot([root_x, left_x], [root_y - 0.08, mid_y + 0.08], 'k-', lw=2, zorder=1)
    ax.plot([root_x, right_x], [root_y - 0.08, mid_y + 0.08], 'k-', lw=2, zorder=1)
    
    # Internal nodes to leaves
    ax.plot([left_x, leaf1_x], [mid_y - 0.08, leaf_y + leaf_height/2], 'k-', lw=1.5, zorder=1)
    ax.plot([left_x, leaf2_x], [mid_y - 0.08, leaf_y + leaf_height/2], 'k-', lw=1.5, zorder=1)
    ax.plot([right_x, leaf3_x], [mid_y - 0.08, leaf_y + leaf_height/2], 'k-', lw=1.5, zorder=1)
    ax.plot([right_x, leaf4_x], [mid_y - 0.08, leaf_y + leaf_height/2], 'k-', lw=1.5, zorder=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "ml_model_icon.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()

def create_output_icon():
    fig, ax = plt.subplots(figsize=(4, 1.8))
    for i, (color, offset) in enumerate(zip(['#E91E63', '#00BCD4', '#8BC34A'], [0, 0.25, 0.5])):
        t = np.linspace(0, 1, 30)
        pattern = np.sin(t * (i+2) * np.pi) * 0.12 + 0.4 + offset
        ax.plot(t + i*1.1, pattern, color=color, linewidth=3)
        ax.text(0.5 + i*1.1, 0.22 + offset, f'P{i+1}', fontsize=12, ha='center', fontweight='bold')
    ax.set_xlim(-0.1, 3.4)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "output_icon.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()

TEMP_ICON_DIR.mkdir(parents=True, exist_ok=True)
create_input_icon()
create_transforms_icon()
create_bspline_icon()
create_sliding_mse_icon()
create_optimization_icon()
create_propose_icon()
create_ml_model_icon()
create_output_icon()



def draw_flowchart(output_path):
    # Calculate figure size to match content exactly (no extra margins)
    fig, ax = plt.subplots(figsize=(15, 21))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 117)  # Increased to 117 to show top border of Input Data box
    ax.axis('off')
    # Remove all margins from the figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def add_img(img_path, x, y, w, h):
        img = mpimg.imread(img_path)
        img_x = (x - w/2) / 100
        img_y = (y - h/2) / 115
        ax_img = ax.inset_axes([img_x, img_y, w/100, h/115], transform=ax.transAxes)
        ax_img.imshow(img)
        ax_img.axis('off')

    def add_box(x, y, w, h, title, color, fontsize=None, subtext=None, img_path=None):
        shadow = FancyBboxPatch((x - w/2 + 0.4, y - h/2 - 0.4), w, h,
                                boxstyle="round,pad=0.02,rounding_size=1",
                                facecolor='#00000018', edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=1",
                             facecolor=color, edgecolor='#333', linewidth=2, zorder=2)
        ax.add_patch(box)
        img_offset = 9 if img_path else 0
        text_x = x + img_offset
        ax.text(text_x, y + h*0.25, title, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', color='#222', zorder=4)
        ax.text(text_x, y - h*0.1, subtext, ha='center', va='center', 
                fontsize=fontsize-2, color='#444', zorder=4, linespacing=1.35)
        add_img(img_path, x - w/2 + 10, y, 18, h*0.8)
        return y - h/2

    def add_arrow(x1, y1, x2, y2, color=None, lw=None):
        if color is None:
            color = arrow_color
        if lw is None:
            lw = arrow_lw
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)
    
    c_input = '#42A5F5'
    c_process = '#81C784'
    c_output = '#78909C'
    c_feature = '#C8E6C9'
    
    # Font sizes
    fontsize_main_title = 17
    fontsize_section_title = 17
    fontsize_box_title = 14
    fontsize_box_subtext = 12
    fontsize_annotation = 12
    
    # Box dimensions
    box_w_main = 60
    box_h_main = 10
    box_w_inner = 40
    box_h_inner = 13
    
    # Central container box dimensions
    fe_top, fe_bottom = 89.5, 46.5  # Bottom moved up by 3 units
    box_w_central = 92  # Width of the central Iterative Pattern Optimization container
    box_h_central = fe_top - fe_bottom  # Height calculated from top and bottom positions
    
    # Fixed arrow connection points to big box (independent of fe_top/fe_bottom)
    big_box_top_arrow_y = 90.0  # Fixed y position where arrow connects to top of big box
    big_box_bottom_arrow_y = 46.0  # Fixed y position where arrow connects to bottom of big box (moved up by 3)
    
    # Arrow sizes - ALL arrows use these same values
    arrow_lw = 2.5  # Line width for all arrows
    arrow_color = '#555'  # Color for all arrows (same as default)
    arrow_gap = 3  # Standard gap between boxes (consistent arrow length)
    cx = 50

    # 1. Input
    input_y = 110
    add_box(cx, input_y, box_w_main, box_h_main, "Input Data", c_input, fontsize=fontsize_main_title,
            subtext="Time series: (n_samples × n_channels × n_timepoints)\nSupports variable-length sequences",
            img_path=str(TEMP_ICON_DIR / "input_icon.png"))
    input_bottom = input_y - box_h_main / 2

    # 2. Precompute Transforms - positioned to have even spacing with Input
    # Input bottom = 105, want gap of arrow_gap = 3, so precompute top = 102
    precompute_y = 102 - box_h_main / 2  # = 97
    add_box(cx, precompute_y, box_w_main, box_h_main, "Precompute Transformations", c_process, fontsize=fontsize_main_title-1,
            subtext="Apply transformations per channel:\nraw, zscore, FFT power, ...",
            img_path=str(TEMP_ICON_DIR / "transforms_icon.png"))
    precompute_bottom = precompute_y - box_h_main / 2
    precompute_top = precompute_y + box_h_main / 2
    
    # Pattern Characteristics Optimization container
    # Center the box: x position = center - width/2
    fe_box_x = cx - box_w_central / 2
    fe_box_pad = 0.5  # Padding of the box (used to calculate visual edges)
    
    # Arrows: from bottom of box to top of next box (direct connection)
    add_arrow(cx, input_bottom, cx, precompute_top)
    # Arrow to big box: end at fixed position (independent of fe_top/fe_bottom)
    add_arrow(cx, precompute_bottom, cx, big_box_top_arrow_y)
    fe_box = FancyBboxPatch((fe_box_x, fe_bottom), box_w_central, box_h_central,
                            boxstyle=f"round,pad={fe_box_pad},rounding_size=1.5",
                            facecolor=c_feature, edgecolor='#2E7D32', 
                            linewidth=2.5, zorder=0, alpha=0.6)
    ax.add_patch(fe_box)
    ax.text(50, fe_top - 1, "Iterative Pattern Optimization", fontsize=fontsize_section_title, color='#1B5E20', ha='center', va='top', fontweight='bold')
    
    # Bayesian optimization text and image side by side
    opt_text_x, opt_img_x = 20, 70
    opt_y = 81
    ax.text(opt_text_x, opt_y, "Bayesian optimization per pattern;\nstop when adding pattern no longer improves score",
            fontsize=fontsize_box_subtext, color='#555', ha='left', va='center')
    add_img(str(TEMP_ICON_DIR / "optimization_icon.png"), opt_img_x, opt_y, 24, 9)

    # Inner boxes - 2x2 grid with equal spacing
    # Calculate positions for equal horizontal and vertical spacing
    inner_box_spacing = arrow_gap  # Same spacing as arrow_gap for consistency
    # Center the grid in the available space (moved up by 3 units)
    grid_center_x = cx
    grid_center_y = 62  # Moved up by 3 from 59
    # Horizontal: boxes centered at grid_center_x ± (box_w_inner + inner_box_spacing) / 2
    col1_x = grid_center_x - (box_w_inner + inner_box_spacing) / 2
    col2_x = grid_center_x + (box_w_inner + inner_box_spacing) / 2
    # Vertical: boxes centered at grid_center_y ± (box_h_inner + inner_box_spacing) / 2
    row1_y = grid_center_y + (box_h_inner + inner_box_spacing) / 2
    row2_y = grid_center_y - (box_h_inner + inner_box_spacing) / 2

    # Box 1: Pattern Proposal (top-left)
    add_box(col1_x, row1_y, box_w_inner, box_h_inner, 
            "1. Propose Pattern Parameters", c_process, fontsize=fontsize_box_title,
            subtext="channel, transform, control points,\nsearch interval, width",
            img_path=str(TEMP_ICON_DIR / "propose_icon.png"))
    
    # Box 2: B-Spline Generation (top-right)
    add_box(col2_x, row1_y, box_w_inner, box_h_inner,
            "2. Generate Spline Pattern", c_process, fontsize=fontsize_box_title,
            subtext="Cubic spline \nwith 4 control points (c0,c1,c2,c3)",
            img_path=str(TEMP_ICON_DIR / "bspline_icon.png"))

    # Box 3: Sliding-window min-MSE feature (bottom-right)
    add_box(col2_x, row2_y, box_w_inner, box_h_inner,
            "3. Compute min-MSE Feature", c_process, fontsize=fontsize_box_title,
            subtext="Slide pattern across the search interval\ncompute MSE per window, take min\n(fallback: center if width is large)",
            img_path=str(TEMP_ICON_DIR / "sliding_mse_icon.png"))

    # Box 4: Model Evaluation (bottom-left)
    add_box(col1_x, row2_y, box_w_inner, box_h_inner,
            "4. Evaluate Model", c_process, fontsize=fontsize_box_title,
            subtext="Append new feature, evaluate\nLightGBM via cross-validation\nto get validation score",
            img_path=str(TEMP_ICON_DIR / "ml_model_icon.png"))

    # Arrows inside feature extraction (clockwise: 1→2→3→4→1)
    # All arrows are exactly arrow_gap units long for consistency
    # Calculate box edges
    box1_right = col1_x + box_w_inner / 2
    box1_bottom = row1_y - box_h_inner / 2
    box2_bottom = row1_y - box_h_inner / 2
    box3_left = col2_x - box_w_inner / 2
    box4_bottom = row2_y - box_h_inner / 2
    
    # Arrow 1→2 (horizontal, top row) - exactly arrow_gap units long
    arrow1_start_x = box1_right
    arrow1_end_x = arrow1_start_x + arrow_gap
    add_arrow(arrow1_start_x, row1_y, arrow1_end_x, row1_y)
    
    # Arrow 2→3 (vertical, right side) - exactly arrow_gap units long
    arrow2_start_y = box2_bottom
    arrow2_end_y = arrow2_start_y - arrow_gap
    add_arrow(col2_x, arrow2_start_y, col2_x, arrow2_end_y)
    
    # Arrow 3→4 (horizontal, bottom row, reversed) - exactly arrow_gap units long
    arrow3_start_x = box3_left
    arrow3_end_x = arrow3_start_x - arrow_gap
    add_arrow(arrow3_start_x, row2_y, arrow3_end_x, row2_y)
    
    # Arrow 4→1 (vertical, left side, reversed - loop back) - exactly arrow_gap units long
    # Mirror arrow 2→3: arrow 2→3 goes from box2_bottom to box3_top (downward)
    # So arrow 4→1 goes from box4_bottom to box1_bottom (upward)
    arrow4_start_y = box4_bottom
    arrow4_end_y = box1_bottom  # Go upward to box1_bottom (mirror of arrow 2→3 going to box3_top)
    add_arrow(col1_x, arrow4_start_y, col1_x, arrow4_end_y)
    loop_x = col1_x  # For text positioning
    ax.text(loop_x + 2, (row1_y + row2_y)/2, 'repeat while improving', fontsize=fontsize_annotation, color='#2E7D32', ha='left', va='center', fontweight='bold')
    # 5. Output - positioned to have even spacing with big box
    # Big box bottom = fe_bottom, want gap of arrow_gap = 3, so output top = fe_bottom - arrow_gap
    output_y = (fe_bottom - arrow_gap) - box_h_main / 2
    add_box(cx, output_y, box_w_main, box_h_main, "Final Output", c_output, fontsize=fontsize_main_title,
            subtext="Interpretable spline patterns with metadata\n+ train/test feature matrices",
            img_path=str(TEMP_ICON_DIR / "output_icon.png"))
    output_top = output_y + box_h_main / 2
    
    # Arrow from big box to output - starts at fixed position (independent of fe_top/fe_bottom)
    add_arrow(cx, big_box_bottom_arrow_y, cx, output_top)
    # Position text outside the big box, next to the arrow (to the right of center)
    text_y = (big_box_bottom_arrow_y + output_top) / 2
    ax.text(cx + 2, text_y, 'stop: adding a pattern gives no CV improvement', fontsize=fontsize_annotation, color='#C62828', ha='left', va='center', fontweight='bold')
    
    # Remove all padding and margins - don't use tight_layout as it adds padding
    output_path = str(output_path)
    # Save with no padding - bbox_inches='tight' finds content bounds, pad_inches=0 removes padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, 
                facecolor='white', edgecolor='none', transparent=False)
    plt.close()

ELSCARTICLE_FLOWCHART_DIR.mkdir(parents=True, exist_ok=True)
draw_flowchart(ELSCARTICLE_FLOWCHART_DIR / "flowchart.svg")

# Clean up temp icons
for p in TEMP_ICON_DIR.glob("*.png"):
    p.unlink()
TEMP_ICON_DIR.rmdir()
