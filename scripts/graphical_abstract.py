
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "remc", "remc_E003.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "elsarticle", "images", "ga_input_plots.png")
OUTPUT_RAW_PATH = os.path.join(BASE_DIR, "elsarticle", "images", "ga_raw_plot.png")

# Transform functions
def fft_power_transform(data):
    """Compute FFT power spectrum and interpolate back to original time length."""
    # Handle 1D array case
    if data.ndim == 1:
        data = data.reshape(1, -1)
        
    n_time = data.shape[-1]
    power = np.abs(np.fft.rfft(data, axis=-1)) ** 2
    n_freq = power.shape[-1]
    x_new = np.linspace(0, n_freq - 1, n_time)
    idx = np.minimum(x_new.astype(np.int32), n_freq - 2)
    frac = (x_new - idx)
    
    result = (power[..., idx] * (1 - frac) + power[..., idx + 1] * frac)
    
    if result.shape[0] == 1:
        return result.flatten()
    return result

def zscore_transform(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

def derivative_transform(data):
    return np.gradient(data)

def main():

    df = pd.read_parquet(DATA_PATH)
    channel_prefix = "H3K4me3"
    cols = [c for c in df.columns if c.startswith(channel_prefix + "_")]
    
    high_expr_df = df[df['target'] == 1]
    if len(high_expr_df) > 0:
        # Pick the sample with the highest peak in the middle (typical for H3K4me3)
        # Middle index approx 100 (200 bins total)
        mid_idx = len(cols) // 2
        mid_col = cols[mid_idx]
        sample_idx = high_expr_df[mid_col].idxmax()
        sample = df.loc[sample_idx, cols].values.astype(np.float32)
    else:
        sample = df.iloc[0][cols].values.astype(np.float32)

    # Apply transforms
    raw = sample
    zscore = zscore_transform(sample)
    derivative = derivative_transform(sample)
    fft_power = fft_power_transform(sample)
    
    # Plotting all transforms
    fig, axes = plt.subplots(4, 1, figsize=(4, 6), sharex=False)
    
    transforms = [
        ('Raw', raw, '#2c3e50'),
        ('Z-score', zscore, '#e67e22'),
        ('Derivative', derivative, '#27ae60'),
        ('FFT', fft_power, '#8e44ad')
    ]
    
    for ax, (name, data, color) in zip(axes, transforms):
        # Plot signal
        ax.plot(data, color=color, lw=2)
        
        # Remove axes
        ax.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1) # Reduce space between plots
    
    # Save with white background
    plt.savefig(OUTPUT_PATH, facecolor='white', transparent=False, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Plotting just the raw data
    fig_raw, ax_raw = plt.subplots(figsize=(4, 1.5))
    ax_raw.plot(raw, color='#2c3e50', lw=4)
    ax_raw.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_RAW_PATH, facecolor='white', transparent=False, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_raw)

if __name__ == "__main__":
    main()
