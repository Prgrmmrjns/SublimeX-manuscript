import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def generate_methods_figure(output_path):
    # Use seaborn style for aesthetics
    sns.set_theme(style="whitegrid", context="paper")
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1.2])
    
    # --- Subplot 1: B-spline Generation (Left Column, Spanning both rows) ---
    ax_bezier = fig.add_subplot(gs[:, 0])
    
    t = np.linspace(0, 1, 100)
    c0, c1, c2 = 0.2, 0.9, 0.4
    pattern = (1-t)**2 * c0 + 2*(1-t)*t * c1 + t**2 * c2
    
    # Draw the skeleton (control polygon)
    ax_bezier.plot([0, 0.5, 1], [c0, c1, c2], 'k--', lw=1, alpha=0.5, label='Control Polygon')
    
    # Draw the curve
    ax_bezier.plot(t, pattern, color=sns.color_palette("rocket")[1], lw=4, label='B-spline Pattern')
    
    # Draw control points
    ax_bezier.scatter([0, 0.5, 1], [c0, c1, c2], color=sns.color_palette("rocket")[3], s=120, zorder=5, label='Control Points')
    
    # Annotations for calculation
    ax_bezier.text(0, c0 - 0.05, '$c_0$', fontsize=14, fontweight='bold', ha='center')
    ax_bezier.text(0.5, c1 + 0.03, '$c_1$', fontsize=14, fontweight='bold', ha='center')
    ax_bezier.text(1, c2 - 0.05, '$c_2$', fontsize=14, fontweight='bold', ha='center')
    
    ax_bezier.set_title('A: Quadratic B-spline Generation', fontsize=16, fontweight='bold', pad=15)
    ax_bezier.set_xlabel('Relative Time ($t$)', fontsize=13)
    ax_bezier.set_ylabel('Amplitude', fontsize=13)
    ax_bezier.set_ylim(0, 1.05)
    ax_bezier.legend(loc='upper right', frameon=True)

    # --- Subplot 2 & 3: Merged Sliding Window & MSE (Right Column) ---
    # Top part: Signal + Sliding Patterns
    ax_signal = fig.add_subplot(gs[0, 1])
    
    np.random.seed(42)
    n_points = 120
    signal_t = np.linspace(0, 10, n_points)
    # Generate a more interesting signal
    signal = np.sin(signal_t * 0.6) * 0.3 + 0.5 + np.random.normal(0, 0.03, n_points)
    
    p_width = 25
    p_t = np.linspace(0, 1, p_width)
    p_shape = (1-p_t)**2 * 0.15 + 2*(1-p_t)*p_t * 0.85 + p_t**2 * 0.25
    
    ax_signal.plot(signal, color='gray', lw=1.5, alpha=0.6, label='Input Signal')
    
    # Find best match
    search_range = range(0, n_points - p_width)
    mse_values = [np.mean((signal[o : o + p_width] - p_shape)**2) for o in search_range]
    best_offset = np.argmin(mse_values)

    # Show the whole search space as background
    search_start, search_end = min(search_range), max(search_range) + p_width
    ax_signal.axvspan(search_start, search_end, color='lightgray', alpha=0.2, label='Search Interval')

    # Sliding windows examples
    offsets = [15, 45, 75]
    colors = sns.color_palette("viridis", n_colors=len(offsets))
    
    for i, offset in enumerate(offsets):
        ax_signal.plot(range(offset, offset + p_width), p_shape, color=colors[i], lw=2.5, alpha=0.8)

    # Highlight the best match
    ax_signal.plot(range(best_offset, best_offset + p_width), p_shape, color=sns.color_palette("rocket")[1], lw=4, label='Optimal Match')
    ax_signal.axvspan(best_offset, best_offset + p_width, color=sns.color_palette("rocket")[1], alpha=0.3)
    
    ax_signal.set_title('B: Translation-Invariant Matching', fontsize=16, fontweight='bold', pad=15)
    ax_signal.set_ylabel('Amplitude', fontsize=13)
    ax_signal.set_ylim(0, 1.05)
    ax_signal.legend(loc='upper right', fontsize=10)
    ax_signal.set_xticklabels([]) # Hide x-labels for top part

    # Bottom part: MSE Response Map
    ax_mse = fig.add_subplot(gs[1, 1], sharex=ax_signal)
    
    ax_mse.plot(range(len(mse_values)), mse_values, color=sns.color_palette("mako")[2], lw=2.5, label='MSE Response Map')
    
    min_mse = mse_values[best_offset]
    ax_mse.scatter(best_offset, min_mse, color='red', s=100, zorder=5)
    
    # Connector between signal best match and MSE min
    ax_mse.annotate('Minimum MSE\n(Feature Value)', 
                    xy=(best_offset, min_mse), 
                    xytext=(best_offset + 10, min_mse + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                    fontsize=11, fontweight='bold')

    ax_mse.set_xlabel('Window Offset', fontsize=13)
    ax_mse.set_ylabel('MSE', fontsize=13)
    ax_mse.set_ylim(0, max(mse_values) * 1.2)
    ax_mse.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    output_dir = Path("manuscript/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_methods_figure(output_dir / "methods_details.png")
