import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import subprocess
import os

os.chdir('/Users/jwolber/Documents/patternExtraction-manuscript')

# Generate custom visualizations first
def create_transform_selection_icon():
    """Create a visual showing transform selection with bar chart"""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    transforms = ['raw', 'deriv', 'fft', 'wavelet', 'diff', 'recip', '...']
    scores = [0.82, 0.78, 0.85, 0.80, 0.65, 0.60, 0.55]
    colors = ['#4CAF50' if s >= 0.75 else '#ccc' for s in scores]
    bars = ax.bar(transforms, scores, color=colors, edgecolor='#333', linewidth=0.5)
    ax.axhline(y=0.75, color='red', linestyle='--', linewidth=1.5, label='Top-5 threshold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('CV Score', fontsize=9)
    ax.set_title('Select Top-5 Transforms', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('manuscript/images/flowchart/transform_selection_icon.png', dpi=150, bbox_inches='tight', 
                facecolor='white', transparent=False)
    plt.close()

def create_bspline_icon():
    """Create a B-spline pattern visualization with 5 control points"""
    fig, ax = plt.subplots(figsize=(3, 2))
    t = np.linspace(0, 1, 100)
    # Simulate B-spline with 5 control points
    ctrl_pts = [0.3, 0.8, 0.4, 0.9, 0.5]
    pattern = np.sin(t * 2 * np.pi) * 0.3 + 0.5
    ax.plot(t, pattern, 'r-', linewidth=3, label='B-spline')
    ax.scatter([0.1, 0.3, 0.5, 0.7, 0.9], ctrl_pts, c='blue', s=80, zorder=5, marker='o', label='5 Control Pts')
    ax.fill_between(t, pattern - 0.1, pattern + 0.1, alpha=0.2, color='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('B-Spline (5 ctrl pts)', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig('manuscript/images/flowchart/bspline_icon.png', dpi=150, bbox_inches='tight', 
                facecolor='white', transparent=False)
    plt.close()

def create_rmse_icon():
    """Create RMSE calculation visualization"""
    fig, ax = plt.subplots(figsize=(3.5, 2))
    t = np.linspace(0, 1, 50)
    signal = np.sin(t * 4 * np.pi) * 0.3 + 0.5 + np.random.randn(50) * 0.05
    pattern = np.sin(t * 4 * np.pi) * 0.3 + 0.5
    ax.plot(t, signal, 'k-', linewidth=1.5, alpha=0.7, label='Signal')
    ax.plot(t, pattern, 'r-', linewidth=2, label='Pattern')
    for i in range(0, len(t), 5):
        ax.plot([t[i], t[i]], [signal[i], pattern[i]], 'b-', alpha=0.5, linewidth=1)
    ax.set_xlim(0, 1)
    ax.axis('off')
    ax.set_title('RMSE = √mean(diff²)', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig('manuscript/images/flowchart/rmse_icon.png', dpi=150, bbox_inches='tight', 
                facecolor='white', transparent=False)
    plt.close()

def create_input_icon():
    """Create time series input visualization"""
    fig, ax = plt.subplots(figsize=(4, 1.5))
    t = np.linspace(0, 10, 200)
    signal = np.sin(t) + 0.5 * np.sin(3*t) + 0.3 * np.random.randn(200)
    ax.plot(t, signal, 'k-', linewidth=1.5)
    ax.fill_between(t, signal, alpha=0.3, color='#2196F3')
    ax.set_xlim(0, 10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('manuscript/images/flowchart/input_icon.png', dpi=150, bbox_inches='tight', 
                facecolor='white', transparent=False)
    plt.close()

def create_transforms_icon():
    """Create multi-transform visualization"""
    fig, axes = plt.subplots(1, 4, figsize=(5, 1.2))
    t = np.linspace(0, 2*np.pi, 100)
    signal = np.sin(t)
    
    transforms = [
        ('Raw', signal, '#4CAF50'),
        ('Deriv', np.gradient(signal), '#2196F3'),
        ('FFT', np.abs(np.fft.fft(signal))[:50], '#FF9800'),
        ('Wavelet', np.convolve(signal, np.ones(10)/10, mode='same'), '#9C27B0')
    ]
    
    for ax, (name, data, color) in zip(axes, transforms):
        ax.plot(data, color=color, linewidth=1.5)
        ax.fill_between(range(len(data)), data, alpha=0.3, color=color)
        ax.set_title(name, fontsize=8, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('manuscript/images/flowchart/transforms_icon.png', dpi=150, bbox_inches='tight', 
                facecolor='white', transparent=False)
    plt.close()

def create_output_icon():
    """Create output patterns visualization"""
    fig, ax = plt.subplots(figsize=(4, 2))
    
    # Draw multiple patterns
    for i, (color, offset) in enumerate(zip(['#E91E63', '#00BCD4', '#8BC34A'], [0, 0.3, 0.6])):
        t = np.linspace(0, 1, 30)
        pattern = np.sin(t * (i+2) * np.pi) * 0.15 + 0.5 + offset
        ax.plot(t + i*1.2, pattern, color=color, linewidth=3)
        ax.text(0.5 + i*1.2, 0.3 + offset, f'P{i+1}', fontsize=10, ha='center', fontweight='bold')
    
    ax.set_xlim(-0.1, 3.7)
    ax.set_ylim(0, 1.5)
    ax.axis('off')
    ax.set_title('Discovered Patterns', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('manuscript/images/flowchart/output_icon.png', dpi=150, bbox_inches='tight', 
                facecolor='white', transparent=False)
    plt.close()

def create_nsga2_icon():
    """Create NSGA-II evolutionary optimization visualization"""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    np.random.seed(42)
    
    # Draw population as scattered points (Pareto fronts)
    for gen, (color, alpha) in enumerate(zip(['#ccc', '#aaa', '#4CAF50'], [0.3, 0.5, 1.0])):
        x = np.random.rand(15) * 0.8 + gen * 0.1
        y = 1 - x + np.random.randn(15) * 0.1
        ax.scatter(x, y, c=color, s=40, alpha=alpha, edgecolors='#333', linewidth=0.5)
    
    # Draw Pareto front line
    x_front = np.linspace(0.1, 0.9, 20)
    y_front = 1 - x_front + 0.1
    ax.plot(x_front, y_front, 'r--', linewidth=2, label='Pareto Front')
    
    # Arrow showing evolution direction
    ax.annotate('', xy=(0.7, 0.5), xytext=(0.3, 0.8),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2))
    ax.text(0.5, 0.75, 'Evolution', fontsize=8, color='#2196F3', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Objective 1', fontsize=8)
    ax.set_ylabel('Objective 2', fontsize=8)
    ax.set_title('NSGA-II Population', fontsize=9, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('manuscript/images/flowchart/nsga2_icon.png', dpi=150, bbox_inches='tight', 
                facecolor='white', transparent=False)
    plt.close()

def create_backward_elim_icon():
    """Create backward elimination visualization"""
    fig, ax = plt.subplots(figsize=(4, 2))
    
    # Draw patterns with some crossed out
    patterns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    kept = [True, True, False, True, False, True, False, True]
    
    for i, (p, k) in enumerate(zip(patterns, kept)):
        x = i * 0.5 + 0.3
        color = '#4CAF50' if k else '#ccc'
        ax.add_patch(plt.Circle((x, 0.5), 0.15, color=color, ec='#333', lw=1.5))
        ax.text(x, 0.5, p, ha='center', va='center', fontsize=8, fontweight='bold')
        if not k:
            ax.plot([x-0.12, x+0.12], [0.62, 0.38], 'r-', lw=2)
            ax.plot([x-0.12, x+0.12], [0.38, 0.62], 'r-', lw=2)
    
    ax.set_xlim(0, 4.3)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('15 → N patterns (remove redundant)', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig('manuscript/images/flowchart/backward_elim_icon.png', dpi=150, bbox_inches='tight', 
                facecolor='white', transparent=False)
    plt.close()

# Generate all icons
print("Generating custom icons...")
create_transform_selection_icon()
create_bspline_icon()
create_rmse_icon()
create_input_icon()
create_transforms_icon()
create_output_icon()
create_nsga2_icon()
create_backward_elim_icon()

# Convert existing SVGs to PNGs
svgs_to_convert = ['bayesian_optimization', 'decision_tree']
for s in svgs_to_convert:
    src = f'manuscript/images/flowchart/{s}.svg'
    dst = f'manuscript/images/flowchart/{s}.png'
    if os.path.exists(src):
        subprocess.run(['rsvg-convert', '-h', '300', src, '-o', dst], check=False)

def draw_flowchart(output_path):
    fig, ax = plt.subplots(figsize=(12, 17))
    ax.set_xlim(0, 85)
    ax.set_ylim(-18, 105)
    ax.axis('off')
    
    # Gradient-like background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap='Blues', alpha=0.08, 
              extent=[0, 85, -18, 105], zorder=-1)

    # Colors
    c_input = '#42A5F5'      # Blue for input
    c_process = '#81C784'    # Green for all processing
    c_decision = '#FFD54F'   # Yellow for decision
    c_output = '#78909C'     # Gray for output

    def add_fancy_box(x, y, w, h, text, color, img_path=None, fontsize=11, bold=False, subtext=None):
        # Shadow
        shadow = FancyBboxPatch((x - w/2 + 0.4, y - h/2 - 0.4), w, h,
                                boxstyle="round,pad=0.02,rounding_size=1",
                                facecolor='#00000018', edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        
        # Main box
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=1",
                             facecolor=color, edgecolor='#333333', linewidth=1.5, zorder=2)
        ax.add_patch(box)
        
        # Embedded image on left side - adjusted position
        img_offset = 0
        if img_path and os.path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                img_w, img_h = 0.12, 0.040
                # Convert data coords to axes coords for proper positioning
                img_x = (x - w/2 + 1.5) / 85
                img_y = (y + 18 - h*0.30) / 123  # Adjusted for new ylim (-18 to 105 = 123)
                ax_img = ax.inset_axes([img_x, img_y, img_w, img_h], transform=ax.transAxes)
                ax_img.imshow(img)
                ax_img.axis('off')
                img_offset = 8  # Shift text right more for bigger image
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        # Text - shifted right if image present
        text_x = x + img_offset/2
        ax.text(text_x, y + h*0.25, text, ha='center', va='center', fontsize=fontsize, 
                fontweight='bold', color='black', zorder=4)
        
        if subtext:
            ax.text(text_x, y - h*0.15, subtext, ha='center', va='center', fontsize=fontsize-1, 
                    color='#111', zorder=4, linespacing=1.3)
        
        return y - h/2

    def add_diamond(x, y, size, text, color):
        # Shadow
        diamond_shadow = mpatches.RegularPolygon((x+0.3, y-0.3), numVertices=4, radius=size,
                                                  orientation=np.pi/4, facecolor='#00000020', 
                                                  edgecolor='none', zorder=1)
        ax.add_patch(diamond_shadow)
        
        diamond = mpatches.RegularPolygon((x, y), numVertices=4, radius=size,
                                          orientation=np.pi/4, facecolor=color, 
                                          edgecolor='#333333', linewidth=1.8, zorder=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', 
                color='#333', zorder=3)

    def add_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2, 
                                    connectionstyle='arc3,rad=0'),
                    zorder=1)

    def add_label(x, y, text, bg='white'):
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg, edgecolor='#999', alpha=0.9),
                zorder=5)

    cx = 38

    # 1. Input Data
    y = 98
    add_fancy_box(cx, y, 50, 10, "Input Data", c_input, 
                  img_path='manuscript/images/flowchart/input_icon.png',
                  fontsize=13, subtext="• Time series, spatial, or multi-channel data\n• Shape: (n_samples, n_channels, n_timepoints)\n• Supports variable-length sequences")
    add_arrow(cx, y-5, cx, y-7.5)

    # 2. Precompute Transforms
    y = 84
    add_fancy_box(cx, y, 50, 11, "Precompute Transformations", c_process,
                  img_path='manuscript/images/flowchart/transforms_icon.png',
                  fontsize=12, subtext="• 21 transform types applied to all channels\n• Includes: Raw, Derivative, FFT Power, DCT\n• Wavelets: db4, sym4, coif1, haar\n• Also: Autocorr, Reciprocal, Cumsum, Diff...")
    add_arrow(cx, y-5.5, cx, y-8)

    # 3. Transform Selection
    y = 69
    add_fancy_box(cx, y, 50, 11, "Transform Selection", c_process,
                  img_path='manuscript/images/flowchart/transform_selection_icon.png',
                  fontsize=12, subtext="• Compute aggregate stats per transform\n• Train LightGBM model for each transform\n• 3-fold CV for stable performance\n• Select Top-5 transforms by CV score")
    
    add_arrow(cx, y-5.5, cx, y-8)

    # Optimization container - contains joint pattern optimization
    loop_box = FancyBboxPatch((5, 16), 66, 44,
                              boxstyle="round,pad=0.5,rounding_size=1.5",
                              facecolor='#f5f5f5', edgecolor='#444', 
                              linewidth=2, linestyle='--', zorder=0, alpha=0.7)
    ax.add_patch(loop_box)
    ax.text(7, 58, "Joint Pattern Optimization", fontsize=13, color='black', 
            ha='left', va='bottom', fontweight='bold')

    # 4. NSGA-II Evolutionary Optimization
    y = 52
    add_fancy_box(cx, y, 50, 11, "Evolutionary Optimization (NSGA-II)", c_process,
                  img_path='manuscript/images/flowchart/nsga2_icon.png',
                  fontsize=11, subtext="• Optuna NSGA-II sampler optimizes jointly\n• 15 patterns per trial (center, width, ctrl_pts)\n• Also: series_idx, transform_type\n• Early stopping after 1000 trials w/o improvement")
    
    add_arrow(cx, y-5.5, cx, y-8)

    # 5. Pattern Matching & Evaluation
    y = 37
    add_fancy_box(cx, y, 50, 11, "Pattern Matching & Evaluation", c_process,
                  img_path='manuscript/images/flowchart/rmse_icon.png',
                  fontsize=11, subtext="• Generate 15 B-splines from control points\n• Compute RMSE features for all patterns\n• Inner 3-fold CV with LightGBM\n• Return validation score (AUC/Acc/RMSE)")
    add_arrow(cx, y-5.5, cx, y-8)

    # 6. Decision - Early stopping check
    y = 23
    add_diamond(cx, y, 5, "Early\nStop?", c_decision)

    # No path (continue optimization)
    ax.plot([cx+4, 67, 67, cx+25], [y, y, 52, 52], color='#555', lw=1.8, zorder=1)
    ax.annotate('', xy=(cx+25, 52), xytext=(67, 52),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.8), zorder=1)
    add_label(67, y+8, "No", '#c8e6c9')

    # Yes path - early stop triggered
    add_arrow(cx, y-4, cx, y-8)
    add_label(cx+4, y-6, "Yes", '#ffcdd2')

    # 7. Backward Elimination
    y = 8
    add_fancy_box(cx, y, 50, 10, "Backward Elimination", c_process,
                  img_path='manuscript/images/flowchart/backward_elim_icon.png',
                  fontsize=12, subtext="• Start with 15 optimized patterns\n• Iteratively remove least useful pattern\n• Stop when removal hurts performance\n• Yields compact final pattern set")
    add_arrow(cx, y-5, cx, y-7.5)

    # 8. Output
    y = -6
    add_fancy_box(cx, y, 50, 11, "Final Output", c_output,
                  img_path='manuscript/images/flowchart/output_icon.png',
                  fontsize=13, subtext="• List of extracted patterns with metadata\n• Trained LightGBM model\n• Train/test feature matrices")

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.svg'), bbox_inches='tight', facecolor='white')
    print(f"Saved flowchart to {output_path}")

# Run
draw_flowchart("manuscript/images/flowchart/flowchart.pdf")

# Cleanup temp files
for f in ['input_icon', 'transforms_icon', 'transform_selection_icon', 'bspline_icon', 
          'rmse_icon', 'output_icon', 'nsga2_icon', 'backward_elim_icon', 
          'bayesian_optimization', 'decision_tree']:
    path = f'manuscript/images/flowchart/{f}.png'
    if os.path.exists(path):
        os.remove(path)

print("Done!")
