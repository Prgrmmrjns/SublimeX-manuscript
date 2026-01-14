import numpy as np
import matplotlib.pyplot as plt
from patx.core import PatternExtractor, generate_bspline_pattern
from sklearn.linear_model import LogisticRegression

def visualize_patterns():
    print("Generating synthetic data...")
    # 1. Create Synthetic Data with clear patterns
    n_samples = 100
    n_time = 100
    n_channels = 1
    
    # Class 0: Flat noise
    X0 = np.random.normal(0, 0.5, (n_samples // 2, n_channels, n_time))
    y0 = np.zeros(n_samples // 2)
    
    # Class 1: Flat noise + Bump at t=60
    X1 = np.random.normal(0, 0.5, (n_samples // 2, n_channels, n_time))
    # Add a Gaussian bump
    t = np.linspace(0, 1, n_time)
    bump = 3.0 * np.exp(-0.5 * ((t - 0.6) / 0.05)**2) 
    X1[:, 0, :] += bump
    y1 = np.ones(n_samples // 2)
    
    X = np.concatenate([X0, X1], axis=0).astype(np.float32)
    y = np.concatenate([y0, y1], axis=0)
    
    print("Fitting PatternExtractor...")
    # 2. Fit PatternExtractor
    # We use a simple wrapper to make LR compatible with patx's clone() expectation
    class LRWrapper(LogisticRegression):
        def clone(self):
            return LRWrapper(**self.get_params())
        def score(self, X, y, metric="auc"):
            return self.score(X, y) # Default accuracy or implement AUC manually if needed

    pe = PatternExtractor(
        base_model=LRWrapper(),
        n_trials=50, # Fast search
        n_control_points=5,
        show_progress=True
    )
    pe.fit(X, y)

    # 3. Visualize the best pattern
    best_pattern = pe.patterns[0]
    print(f"Best Pattern Found (Score: {pe.model.score(pe.train_features, y):.4f})")
    
    # Reconstruct pattern shape
    pat_shape = generate_bspline_pattern(best_pattern['control_points'], n_time)
    
    # Pick a sample from Class 1 (Target)
    sample_idx = n_samples - 1 # Last sample is Class 1
    sample_signal = X[sample_idx, 0, :]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Signal (Left Axis)
    plt.plot(sample_signal, 'k-', alpha=0.6, label='Sample Signal (Class 1)')
    plt.ylabel('Signal Amplitude')
    plt.xlabel('Time')
    
    # Plot Pattern (Right Axis)
    ax2 = plt.gca().twinx()
    ax2.plot(pat_shape, 'r-', linewidth=3, label='Learned Pattern (Kernel)')
    ax2.set_ylabel('Pattern Weight', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Feature value
    feature_val = np.dot(sample_signal, pat_shape)
    plt.title(f"Interpretability: Global Pattern Match\nDot Product Feature = {feature_val:.2f}")
    
    # Combine legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Note: matplotlib might split legends between axes
    # Simple workaround for legend
    plt.legend(loc='upper left')
    
    output_file = "pattern_vis_example.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_patterns()
