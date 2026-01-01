import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
import numpy as np
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ELSCARTICLE_FLOWCHART_DIR = REPO_ROOT / "manuscript" / "images" / "flowchart"
TEMP_ICON_DIR = REPO_ROOT / "analysis_scripts" / "_flowchart_tmp_icons"

def create_input_icon():
    fig, ax = plt.subplots(figsize=(4, 1.5))
    np.random.seed(42)
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
    fig, axes = plt.subplots(1, 4, figsize=(5, 1.2))
    t = np.linspace(0, 2*np.pi, 100)
    signal = np.sin(t)
    transforms = [
        ("Raw", signal, "#4CAF50"),
        ("Delta0", signal - signal[0], "#2196F3"),
        ("FFT pow", np.abs(np.fft.rfft(signal)) ** 2, "#FF9800"),
        ("Wavelet", np.convolve(signal, np.ones(10) / 10, mode="same"), "#9C27B0"),
    ]
    for ax, (name, data, color) in zip(axes, transforms):
        ax.plot(data, color=color, linewidth=1.5)
        ax.fill_between(range(len(data)), data, alpha=0.3, color=color)
        ax.set_title(name, fontsize=8, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "transforms_icon.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()

def create_transform_selection_icon():
    fig, ax = plt.subplots(figsize=(4, 2.5))
    transforms = ['raw', 'cumsum', 'dct', 'fft_pow', 'coif1', 'log1p', '...']
    scores = [0.82, 0.78, 0.85, 0.80, 0.65, 0.60, 0.55]
    colors = ['#4CAF50' if s >= 0.75 else '#ccc' for s in scores]
    ax.bar(transforms, scores, color=colors, edgecolor='#333', linewidth=0.5)
    ax.axhline(y=0.75, color='red', linestyle='--', linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('CV Score', fontsize=9)
    ax.set_title('Select Top-K Transforms', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "transform_selection_icon.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_bspline_icon():
    fig, ax = plt.subplots(figsize=(3, 2))
    t = np.linspace(0, 1, 120, dtype=np.float32)
    # Quadratic spline / Bezier with 3 control points (c0, c1, c2)
    c0, c1, c2 = 0.25, 0.85, 0.35
    pattern = (1 - t) ** 2 * c0 + 2 * (1 - t) * t * c1 + t ** 2 * c2
    ax.plot(t, pattern, 'r-', linewidth=3)
    ax.scatter([0.0, 0.5, 1.0], [c0, c1, c2], c='blue', s=90, zorder=5,
               marker='o')
    ax.text(0.0, c0 - 0.08, "c0", ha="center", va="top",
            fontsize=9, fontweight="bold")
    ax.text(0.5, c1 + 0.08, "c1", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    ax.text(1.0, c2 - 0.08, "c2", ha="center", va="top",
            fontsize=9, fontweight="bold")
    ax.fill_between(t, pattern - 0.06, pattern + 0.06, alpha=0.18, color='red')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "bspline_icon.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()

def create_sliding_mse_icon():
    """
    Visualize sliding-window matching: slide the pattern across a search interval,
    compute MSE per window, and take the minimum.
    """
    fig, ax = plt.subplots(figsize=(4.2, 2.2))

    np.random.seed(42)
    n = 140
    x = np.arange(n)
    signal = (0.25 * np.sin(np.linspace(0, 5 * np.pi, n)) +
              0.08 * np.random.randn(n) + 0.5)

    # Pattern (quadratic spline) to slide
    width = 28
    t = np.linspace(0, 1, width, dtype=np.float32)
    c0, c1, c2 = 0.30, 0.78, 0.38
    pattern = (1 - t) ** 2 * c0 + 2 * (1 - t) * t * c1 + t ** 2 * c2

    # Search interval (constrain overlays so windows stay fully inside)
    start, end = 38, 96
    end = min(end, n - 1)
    max_pos = max(start, end - width)

    positions = np.arange(start, max_pos + 1)
    mse = np.array(
        [np.mean((signal[pos:pos + width] - pattern) ** 2) for pos in positions],
        dtype=np.float32,
    )
    best_idx = int(np.argmin(mse))
    best_pos = int(positions[best_idx])
    best_mse = float(mse[best_idx])

    ax.axvspan(start, end, color="#C8E6C9", alpha=0.9, zorder=0)
    ax.plot(x, signal, color="#222", lw=1.6, alpha=0.9, zorder=2)
    ax.text((start + end) / 2, 0.985, "search interval",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#1B5E20")

    if max_pos > start:
        example_positions = [
            start,
            start + (max_pos - start) // 3,
            start + 2 * (max_pos - start) // 3,
        ]
    else:
        example_positions = [start]
    example_positions = [int(p) for p in example_positions if p <= max_pos]

    for i, pos in enumerate(example_positions):
        ax.plot(x[pos:pos + width], pattern, color="#E91E63", lw=2.0,
                alpha=0.25 + 0.20 * i, zorder=3)

    # Best match window (always fully inside interval)
    ax.plot(x[best_pos:best_pos + width], pattern, color="#E91E63", lw=2.9,
            alpha=0.98, zorder=4)
    ax.annotate("best match\n(min MSE)",
                xy=(best_pos + width * 0.55,
                    pattern[int(width * 0.55)]),
                xytext=(best_pos + width * 0.55, 0.10),
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.4),
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#1565C0")

    ax.set_xlim(0, n - 1)
    ax.set_ylim(0.02, 1.0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "sliding_mse_icon.png", dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_optimization_icon():
    fig, ax = plt.subplots(figsize=(5, 2.5))
    np.random.seed(42)
    pts = np.array([0.08, 0.18, 0.32, 0.5, 0.66, 0.78, 0.9])
    y = 0.55 + 0.25*np.sin(6*np.pi*pts) + 0.1*np.cos(2*np.pi*pts) + np.random.randn(len(pts))*0.03
    x = np.sort(np.unique(np.concatenate([np.linspace(0, 1, 260), pts])))
    ell, noise = 0.12, 1e-6
    K = np.exp(-0.5*((pts[:, None]-pts[None, :])/ell)**2) + noise*np.eye(len(pts))
    Ks = np.exp(-0.5*((pts[:, None]-x[None, :])/ell)**2)
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
    ax.set_title("Bayesian optimization", fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "optimization_icon.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_propose_icon():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    np.random.seed(42)
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
    plt.savefig(TEMP_ICON_DIR / "propose_icon.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()

def create_ml_model_icon():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    layers = [3, 4, 4, 2]
    layer_x = [0.2, 0.4, 0.6, 0.8]
    for l, (n_nodes, x) in enumerate(zip(layers, layer_x)):
        y_positions = np.linspace(0.2, 0.8, n_nodes)
        for y in y_positions:
            circle = plt.Circle((x, y), 0.05, color='#2196F3', ec='#333', lw=1)
            ax.add_patch(circle)
            if l < len(layers) - 1:
                next_y_positions = np.linspace(0.2, 0.8, layers[l+1])
                for ny in next_y_positions:
                    ax.plot([x+0.05, layer_x[l+1]-0.05], [y, ny], 'k-', alpha=0.3, lw=0.5)
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
        ax.text(0.5 + i*1.1, 0.22 + offset, f'P{i+1}', fontsize=9, ha='center', fontweight='bold')
    ax.set_xlim(-0.1, 3.4)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(TEMP_ICON_DIR / "output_icon.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()

print("Generating icons...")
TEMP_ICON_DIR.mkdir(parents=True, exist_ok=True)
create_input_icon()
create_transforms_icon()
create_transform_selection_icon()
create_bspline_icon()
create_sliding_mse_icon()
create_optimization_icon()
create_propose_icon()
create_ml_model_icon()
create_output_icon()

def draw_flowchart(output_path):
    fig, ax = plt.subplots(figsize=(15, 21))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 115)
    ax.axis('off')
    
    c_input = '#42A5F5'
    c_process = '#81C784'
    c_output = '#78909C'
    c_feature = '#C8E6C9'

    def add_img(img_path, x, y, w, h):
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            img_x = (x - w/2) / 100
            img_y = (y - h/2) / 115
            ax_img = ax.inset_axes([img_x, img_y, w/100, h/115], transform=ax.transAxes)
            ax_img.imshow(img)
            ax_img.axis('off')

    def add_box(x, y, w, h, title, color, fontsize=12, subtext=None, img_path=None):
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
        
        if subtext:
            ax.text(text_x, y + h*0.25, title, ha='center', va='center', 
                    fontsize=fontsize, fontweight='bold', color='#222', zorder=4)
            ax.text(text_x, y - h*0.1, subtext, ha='center', va='center', 
                    fontsize=fontsize-2, color='#444', zorder=4, linespacing=1.35)
        else:
            ax.text(text_x, y, title, ha='center', va='center', 
                    fontsize=fontsize, fontweight='bold', color='#222', zorder=4)
        
        if img_path:
            add_img(img_path, x - w/2 + 10, y, 18, h*0.8)
        return y - h/2

    def add_arrow(x1, y1, x2, y2, color='#555', lw=3.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)

    cx = 50

    # 1. Input
    y = 110
    add_box(cx, y, 52, 8, "Input Data", c_input, fontsize=14,
            subtext="Time series: (n_samples × n_channels × n_timepoints)\nSupports variable-length sequences",
            img_path=str(TEMP_ICON_DIR / "input_icon.png"))
    add_arrow(cx, y-4, cx, y-7)

    # 2. Precompute Transforms
    y = 98
    add_box(cx, y, 52, 8, "Precompute Transformations", c_process, fontsize=13,
            subtext="Apply transformations per channel:\nraw, cumsum, DCT, FFT power, wavelet, log1p, ...",
            img_path=str(TEMP_ICON_DIR / "transforms_icon.png"))
    add_arrow(cx, y-4, cx, y-7)

    # 3. Transform Selection
    y = 86
    add_box(cx, y, 52, 8, "Select Top-K Transforms", c_process, fontsize=13,
            subtext="Train LightGBM on aggregate stats per transform\nKeep best K transforms by CV score",
            img_path=str(TEMP_ICON_DIR / "transform_selection_icon.png"))
    add_arrow(cx, y-4, cx, y-7)

    # Pattern Characteristics Optimization container
    fe_top, fe_bottom = 77, 24
    fe_box = FancyBboxPatch((4, fe_bottom), 92, fe_top - fe_bottom,
                            boxstyle="round,pad=0.5,rounding_size=1.5",
                            facecolor=c_feature, edgecolor='#2E7D32', 
                            linewidth=2.5, zorder=0, alpha=0.6)
    ax.add_patch(fe_box)
    ax.text(50, fe_top - 2, "Iterative Pattern Optimization", fontsize=14, color='#1B5E20', 
            ha='center', va='top', fontweight='bold')
    ax.text(50, fe_top - 4, "(Bayesian optimization per pattern; stop when adding a pattern no longer improves CV)",
            fontsize=9, color='#555', ha='center', va='top')
    
    # Optimization image (standalone, no box)
    add_img(str(TEMP_ICON_DIR / "optimization_icon.png"), 50, 66, 28, 10)

    # Inner boxes - 2x2 grid
    row1_y, row2_y = 52, 33
    col1_x, col2_x = 25, 75
    box_w, box_h = 38, 12

    # Box 1: Pattern Proposal (top-left)
    add_box(col1_x, row1_y, box_w, box_h, 
            "1. Propose Pattern Parameters", c_process, fontsize=11,
            subtext="channel, transform, control points,\nsearch interval [start,end], width",
            img_path=str(TEMP_ICON_DIR / "propose_icon.png"))
    
    # Box 2: B-Spline Generation (top-right)
    add_box(col2_x, row1_y, box_w, box_h,
            "2. Generate Spline Pattern", c_process, fontsize=11,
            subtext="Quadratic spline (Bezier)\nwith 3 control points (c0,c1,c2)",
            img_path=str(TEMP_ICON_DIR / "bspline_icon.png"))

    # Box 3: Sliding-window min-MSE feature (bottom-right)
    add_box(col2_x, row2_y, box_w, box_h,
            "3. Compute min-MSE Feature", c_process, fontsize=11,
            subtext="Slide pattern across the search interval\ncompute MSE per window, take min\n(fallback: center if width is large)",
            img_path=str(TEMP_ICON_DIR / "sliding_mse_icon.png"))

    # Box 4: Model Evaluation (bottom-left)
    add_box(col1_x, row2_y, box_w, box_h,
            "4. Evaluate Model", c_process, fontsize=11,
            subtext="Append new feature, evaluate\nLightGBM via cross-validation\nto get validation score",
            img_path=str(TEMP_ICON_DIR / "ml_model_icon.png"))

    # Arrows inside feature extraction (clockwise: 1→2→3→4→1)
    add_arrow(col1_x + box_w/2 + 1, row1_y, col2_x - box_w/2 - 1, row1_y, '#2E7D32', lw=3)
    add_arrow(col2_x, row1_y - box_h/2 - 1, col2_x, row2_y + box_h/2 + 1, '#2E7D32', lw=3)
    add_arrow(col2_x - box_w/2 - 1, row2_y, col1_x + box_w/2 + 1, row2_y, '#2E7D32', lw=3)
    
    # Loop back arrow (from box 4 up to box 1)
    loop_x = col1_x - box_w/2 + 4
    ax.plot([loop_x, loop_x], [row2_y + box_h/2, row1_y - box_h/2], color='#2E7D32', lw=3, zorder=1)
    ax.annotate('', xy=(loop_x, row1_y - box_h/2), xytext=(loop_x, row2_y + box_h/2),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=3), zorder=1)
    ax.text(loop_x + 2, (row1_y + row2_y)/2, 'repeat while improving', fontsize=9, 
            color='#2E7D32', ha='left', va='center', fontweight='bold')

    add_arrow(cx, fe_bottom - 1, cx, 18)
    ax.text(cx + 2, (fe_bottom - 1 + 18)/2, 'stop: adding a pattern gives no CV improvement', fontsize=9,
            color='#C62828', ha='left', va='center', fontweight='bold')

    # 5. Output
    y = 12
    add_box(cx, y, 52, 8, "Final Output", c_output, fontsize=14,
            subtext="Interpretable spline patterns with metadata\nTrained model + train/test feature matrices",
            img_path=str(TEMP_ICON_DIR / "output_icon.png"))

    plt.tight_layout(pad=0.5)
    output_path = str(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.svg'), bbox_inches='tight',
                facecolor='white')
    print(f"Saved flowchart to {output_path}")


ELSCARTICLE_FLOWCHART_DIR.mkdir(parents=True, exist_ok=True)
draw_flowchart(ELSCARTICLE_FLOWCHART_DIR / "flowchart.pdf")

# Clean up temp icons
for p in TEMP_ICON_DIR.glob("*.png"):
    try:
        p.unlink()
    except OSError:
        pass
try:
    TEMP_ICON_DIR.rmdir()
except OSError:
    pass

print("Done!")
