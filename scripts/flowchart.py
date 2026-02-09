import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ICON_DIR = REPO_ROOT / "scripts" / "_flowchart_tmp_icons"

FIGSIZE = (5, 3)


def create_input_icon():
    """Create icon showing multi-channel time series input."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    t = np.linspace(0, 10, 200)
    signal = np.sin(t) + 0.5 * np.sin(3*t) + 0.3 * np.random.randn(200)
    ax.plot(t, signal, 'k-', linewidth=1.5)
    ax.fill_between(t, signal, alpha=0.3, color='#2196F3')
    ax.set_xlim(0, 10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "input_icon.png", dpi=150, bbox_inches='tight', 
                facecolor='white')
    plt.close()


def create_transforms_icon():
    """Create icon showing the 4 signal transformations."""
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.5))
    t = np.linspace(0, 2*np.pi, 100)
    signal = np.sin(t) + 0.3 * np.sin(3*t)
    zscore = (signal - signal.mean()) / signal.std()
    derivative = np.gradient(signal)
    fft_pow = np.abs(np.fft.rfft(signal)) ** 2
    
    transforms = [
        ("Raw", signal, "#4CAF50"),
        ("Z-score", zscore, "#2196F3"),
        ("Derivative", derivative, "#FF9800"),
        ("FFT Power", fft_pow, "#9C27B0"),
    ]
    for ax, (name, data, color) in zip(axes, transforms):
        ax.plot(data, color=color, linewidth=1.5)
        ax.fill_between(range(len(data)), data, alpha=0.3, color=color)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "transforms_icon.png", dpi=150, bbox_inches='tight', 
                facecolor='white')
    plt.close()


def create_segment_selection_icon():
    """Create icon showing segment selection on a signal."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    n = 140
    x = np.arange(n)
    signal = 0.25 * np.sin(np.linspace(0, 5 * np.pi, n)) + 0.08 * np.random.randn(n) + 0.5
    
    # Segment boundaries
    start, end = 45, 95
    
    # Plot signal
    ax.plot(x, signal, color="#222", lw=1.6, alpha=0.9, zorder=2)
    
    # Highlight segment
    ax.axvspan(start, end, color="#E3F2FD", alpha=0.9, zorder=0)
    ax.axvline(start, color="#1976D2", linestyle='--', lw=2, alpha=0.8)
    ax.axvline(end, color="#1976D2", linestyle='--', lw=2, alpha=0.8)
    
    # Highlight the segment portion of the signal
    ax.plot(x[start:end+1], signal[start:end+1], color="#E91E63", lw=3, zorder=3)
    
    # Labels
    ax.text((start + end) / 2, 0.95, "Selected Segment", ha="center", va="top", 
            fontsize=11, fontweight="bold", color="#1565C0")
    ax.annotate("start", xy=(start, 0.15), ha="center", va="bottom", 
                fontsize=10, color="#1976D2", fontweight="bold")
    ax.annotate("end", xy=(end, 0.15), ha="center", va="bottom", 
                fontsize=10, color="#1976D2", fontweight="bold")
    
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0.1, 1.0)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "segment_icon.png", dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()


def create_aggregation_icon():
    """Create icon showing different aggregation functions."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Sample segment data
    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    segment = 0.3 * np.sin(4 * np.pi * t) + 0.5 + 0.1 * np.random.randn(50)
    
    # Plot segment
    ax.plot(t, segment, color="#333", lw=2, alpha=0.8)
    ax.fill_between(t, segment, alpha=0.2, color='#2196F3')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.95)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "aggregation_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_optimization_icon():
    fig, ax = plt.subplots(figsize=FIGSIZE)
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
    ax.axis('off')
    ax.set_title("Bayesian optimization", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "optimization_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_propose_icon():
    """Create icon showing parameter proposal."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.random.rand(30)
    y = np.random.rand(30)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 30))
    for i in range(30):
        ax.scatter(x[i], y[i], c=[colors[i]], s=40 + i*2, alpha=0.7, 
                   edgecolors='#333', linewidth=0.5)
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
    """Create icon showing LightGBM decision tree."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    root_x, root_y = 0.5, 0.85
    root_box = FancyBboxPatch((root_x - 0.12, root_y - 0.08), 0.24, 0.16, 
                               boxstyle="round,pad=0.01", facecolor='#2196F3', 
                               edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(root_box)
    ax.text(root_x, root_y, "x < 0.5", ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=4)
    
    left_x, right_x, mid_y = 0.25, 0.75, 0.5
    left_box = FancyBboxPatch((left_x - 0.1, mid_y - 0.08), 0.2, 0.16, 
                               boxstyle="round,pad=0.01", facecolor='#4CAF50', 
                               edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(left_box)
    ax.text(left_x, mid_y, "y < 0.3", ha='center', va='center', fontsize=8, 
            fontweight='bold', color='white', zorder=4)
    
    right_box = FancyBboxPatch((right_x - 0.1, mid_y - 0.08), 0.2, 0.16, 
                                boxstyle="round,pad=0.01", facecolor='#4CAF50', 
                                edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(right_box)
    ax.text(right_x, mid_y, "z > 0.7", ha='center', va='center', fontsize=8, 
            fontweight='bold', color='white', zorder=4)
    
    leaf_y, leaf_width, leaf_height = 0.15, 0.15, 0.12
    leaf1_x, leaf2_x = 0.15, 0.35
    
    leaf1 = FancyBboxPatch((leaf1_x - leaf_width/2, leaf_y - leaf_height/2), 
                            leaf_width, leaf_height, boxstyle="round,pad=0.01", 
                            facecolor='#FF9800', edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(leaf1)
    ax.text(leaf1_x, leaf_y, "Class A", ha='center', va='center', fontsize=8, 
            fontweight='bold', color='white', zorder=4)
    
    leaf2 = FancyBboxPatch((leaf2_x - leaf_width/2, leaf_y - leaf_height/2), 
                            leaf_width, leaf_height, boxstyle="round,pad=0.01", 
                            facecolor='#FF9800', edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(leaf2)
    ax.text(leaf2_x, leaf_y, "Class B", ha='center', va='center', fontsize=8, 
            fontweight='bold', color='white', zorder=4)
    
    leaf3_x, leaf4_x = 0.65, 0.85
    leaf3 = FancyBboxPatch((leaf3_x - leaf_width/2, leaf_y - leaf_height/2), 
                            leaf_width, leaf_height, boxstyle="round,pad=0.01", 
                            facecolor='#FF9800', edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(leaf3)
    ax.text(leaf3_x, leaf_y, "Class A", ha='center', va='center', fontsize=8, 
            fontweight='bold', color='white', zorder=4)
    
    leaf4 = FancyBboxPatch((leaf4_x - leaf_width/2, leaf_y - leaf_height/2), 
                            leaf_width, leaf_height, boxstyle="round,pad=0.01", 
                            facecolor='#FF9800', edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(leaf4)
    ax.text(leaf4_x, leaf_y, "Class B", ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    
    # Draw edges
    ax.plot([root_x, left_x], [root_y - 0.08, mid_y + 0.08], 'k-', lw=2, zorder=1)
    ax.plot([root_x, right_x], [root_y - 0.08, mid_y + 0.08], 'k-', lw=2, zorder=1)
    ax.plot([left_x, leaf1_x], [mid_y - 0.08, leaf_y + leaf_height/2], 'k-', lw=1.5, zorder=1)
    ax.plot([left_x, leaf2_x], [mid_y - 0.08, leaf_y + leaf_height/2], 'k-', lw=1.5, zorder=1)
    ax.plot([right_x, leaf3_x], [mid_y - 0.08, leaf_y + leaf_height/2], 'k-', lw=1.5, zorder=1)
    ax.plot([right_x, leaf4_x], [mid_y - 0.08, leaf_y + leaf_height/2], 'k-', lw=1.5, zorder=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "ml_model_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_output_icon():
    """Create icon showing output segment features."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Show multiple features as horizontal bars
    features = ['mean(ch1, raw, [10:50])', 'max(ch2, deriv, [30:80])', 
                'slope(ch1, zscore, [5:25])']
    colors = ['#E91E63', '#00BCD4', '#8BC34A']
    values = [0.72, 0.45, 0.88]
    
    y_positions = [0.75, 0.5, 0.25]
    for i, (feat, color, val, y_pos) in enumerate(zip(features, colors, values, y_positions)):
        ax.barh(y_pos, val, height=0.18, color=color, alpha=0.8, edgecolor='#333')
        ax.text(0.02, y_pos, f'F{i+1}', fontsize=11, ha='left', va='center', fontweight='bold', color='white')
        ax.text(val + 0.03, y_pos, f'{val:.2f}', fontsize=10, ha='left', va='center')
    
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "output_icon.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# Create temporary directory and generate icons
TEMP_ICON_DIR.mkdir(parents=True, exist_ok=True)
create_input_icon()
create_transforms_icon()
create_segment_selection_icon()
create_aggregation_icon()
create_optimization_icon()
create_propose_icon()
create_ml_model_icon()
create_output_icon()


class FlowchartBox:
    W = 54
    H = 16
    TITLE_FS = 24
    SUBTEXT_FS = 22
    
    @staticmethod
    def draw(ax, x, y, title, color, subtext=None, img_path=None, add_img_func=None):
        w, h = FlowchartBox.W, FlowchartBox.H
        shadow = FancyBboxPatch((x - w/2 + 0.4, y - h/2 - 0.4), w, h, 
                                 boxstyle="round,pad=0.02,rounding_size=1", 
                                 facecolor='#00000018', edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        box = FancyBboxPatch((x - w/2, y - h/2), w, h, 
                              boxstyle="round,pad=0.02,rounding_size=1", 
                              facecolor=color, edgecolor='#333', linewidth=2, zorder=2)
        ax.add_patch(box)
        img_w = 22
        img_x = x - w/2 + img_w/2 + 3
        text_x = x + 12
        if img_path and add_img_func:
            add_img_func(img_path, img_x, y, img_w, h * 0.75)
        ax.text(text_x, y + h*0.22, title, ha='center', va='center', 
                fontsize=FlowchartBox.TITLE_FS, fontweight='bold', color='#222', zorder=4)
        ax.text(text_x, y - h*0.12, subtext, ha='center', va='center', 
                fontsize=FlowchartBox.SUBTEXT_FS, color='#444', zorder=4, linespacing=1.2)
        return y - h/2


def draw_flowchart(output_path):
    fig, ax = plt.subplots(figsize=(26, 24))
    ax.set_xlim(0, 120)
    
    # Calculate content bottom first to set proper ylim
    cx, arrow_gap = 60, 3.5
    fe_bottom = 30
    y_final_output = fe_bottom - arrow_gap - FlowchartBox.H/2
    content_bottom = y_final_output - FlowchartBox.H/2 - 1
    
    ax.set_ylim(content_bottom, 120)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def add_img(img_path, x, y, w, h):
        img = mpimg.imread(img_path)
        # Get actual axis limits to convert data coordinates to normalized axes coordinates
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        # Convert data coordinates to normalized axes coordinates (0-1)
        img_x_norm = (x - w/2 - xlim[0]) / x_range
        img_y_norm = (y - h/2 - ylim[0]) / y_range
        img_w_norm = w / x_range
        img_h_norm = h / y_range
        ax_img = ax.inset_axes([img_x_norm, img_y_norm, img_w_norm, img_h_norm], transform=ax.transAxes)
        ax_img.imshow(img)
        ax_img.axis('off')

    def add_arrow(x1, y1, x2, y2, color='#555', lw=2.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)
    
    c_input, c_process, c_output, c_feature = '#42A5F5', '#81C784', '#78909C', '#C8E6C9'
    box_w_central, fe_top = FlowchartBox.W * 2 + 8, 82

    # 1. Input
    y = 112
    FlowchartBox.draw(ax, cx, y, "Input Data", c_input, 
                      subtext="Time series: \n(univariate or multivariate)",
                      img_path=str(TEMP_ICON_DIR / "input_icon.png"), add_img_func=add_img)
    y_bottom = y - FlowchartBox.H/2

    # 2. Precompute Transformations
    y = y_bottom - arrow_gap - FlowchartBox.H/2
    FlowchartBox.draw(ax, cx, y, "Precompute Transformations", c_process, 
                      subtext="Apply 4 transforms per channel:\n"
                              "raw, z-score, derivative, FFT power", 
                      img_path=str(TEMP_ICON_DIR / "transforms_icon.png"), add_img_func=add_img)
    pre_bottom, pre_top = y - FlowchartBox.H/2, y + FlowchartBox.H/2
    add_arrow(cx, y_bottom, cx, pre_top)
    add_arrow(cx, pre_bottom, cx, fe_top)

    # Container for iterative optimization
    fe_box = FancyBboxPatch((cx - box_w_central/2, fe_bottom), box_w_central, 
                             fe_top - fe_bottom, 
                             boxstyle="round,pad=0.5,rounding_size=1.5", 
                             facecolor=c_feature, edgecolor='#2E7D32', 
                             linewidth=2.5, zorder=0, alpha=0.6)
    ax.add_patch(fe_box)
    ax.text(cx, fe_top - 1.5, "Iterative Feature Optimization",
            fontsize=24, color='#1B5E20', ha='center', va='top',
            fontweight='bold')
    
    opt_y = fe_top - 5
    ax.text(cx - 28, opt_y, 
            "Bayesian optimization per feature;\n"
            "stop when adding feature no longer improves score", 
            fontsize=22, color='#555', ha='left', va='center')
    add_img(str(TEMP_ICON_DIR / "optimization_icon.png"), cx + 25, opt_y, 30, 11)

    inner_gap = 6
    col1_x = cx - FlowchartBox.W/2 - inner_gap/2
    col2_x = cx + FlowchartBox.W/2 + inner_gap/2
    grid_y = (fe_bottom + opt_y - 6) / 2 + 1.5
    row1_y = grid_y + FlowchartBox.H/2 + inner_gap/2
    row2_y = grid_y - FlowchartBox.H/2 - inner_gap/2

    # Step 1: Propose Feature Parameters
    FlowchartBox.draw(ax, col1_x, row1_y, "1. Propose Feature Parameters", c_process, 
                      subtext="channel, transform,\nsegment boundaries (start, end)", 
                      img_path=str(TEMP_ICON_DIR / "propose_icon.png"), add_img_func=add_img)
    
    # Step 2: Extract Segment
    FlowchartBox.draw(ax, col2_x, row1_y, "2. Extract Segment", c_process, 
                      subtext="Select segment [start:end] from\ntransformed channel data", 
                      img_path=str(TEMP_ICON_DIR / "segment_icon.png"), add_img_func=add_img)
    
    # Step 3: Compute Mean Aggregation
    FlowchartBox.draw(ax, col2_x, row2_y, "3. Compute Mean", c_process, 
                      subtext="Compute mean value\nover the segment", 
                      img_path=str(TEMP_ICON_DIR / "aggregation_icon.png"), add_img_func=add_img)
    
    # Step 4: Evaluate Model
    FlowchartBox.draw(ax, col1_x, row2_y, "4. Evaluate Model", c_process, 
                      subtext="Append new feature, evaluate\nLightGBM to get validation score", 
                      img_path=str(TEMP_ICON_DIR / "ml_model_icon.png"), add_img_func=add_img)

    # Arrows between steps
    add_arrow(col1_x + FlowchartBox.W/2, row1_y, col2_x - FlowchartBox.W/2, row1_y)
    add_arrow(col2_x, row1_y - FlowchartBox.H/2, col2_x, row2_y + FlowchartBox.H/2)
    add_arrow(col2_x - FlowchartBox.W/2, row2_y, col1_x + FlowchartBox.W/2, row2_y)
    add_arrow(col1_x, row2_y + FlowchartBox.H/2, col1_x, row1_y - FlowchartBox.H/2)
    ax.text(col1_x + 2, grid_y, 'repeat while improving',
            fontsize=18, color='#2E7D32', ha='left', va='center',
            fontweight='bold')

    # 5. Output
    y = fe_bottom - arrow_gap - FlowchartBox.H/2
    FlowchartBox.draw(ax, cx, y, "Final Output", c_output, 
                      subtext="Interpretable segment-based features\n"
                              "+ train/test feature matrices", 
                      img_path=str(TEMP_ICON_DIR / "output_icon.png"), add_img_func=add_img)
    add_arrow(cx, fe_bottom, cx, y + FlowchartBox.H/2)
    ax.text(cx + 2,
            (fe_bottom + y + FlowchartBox.H/2) / 2,
            'stop: adding a feature does not improve score',
            fontsize=20, color='#C62828', ha='left', va='center',
            fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


# Generate flowchart
draw_flowchart('../elsarticle/images/flowchart.png')
