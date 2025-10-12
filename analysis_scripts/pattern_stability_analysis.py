import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_json_patterns(json_path):
    """Load patterns from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def compute_pattern_similarity(pattern1, pattern2):
    """Compute DTW distance between two patterns."""
    p1 = np.array(pattern1['pattern'])
    p2 = np.array(pattern2['pattern'])
    distance, _ = fastdtw(p1.reshape(-1, 1), p2.reshape(-1, 1), dist=euclidean)
    max_len = max(len(p1), len(p2))
    normalized_distance = distance / max_len
    similarity = 1 / (1 + normalized_distance)
    return similarity

def analyze_fold_patterns(fold_data):
    """Analyze patterns across folds for a single dataset."""
    if not fold_data:
        return None
    
    num_folds = len(fold_data)
    if num_folds < 2:
        return None
    
    # Collect patterns per fold
    fold_patterns = {fold_key: patterns for fold_key, patterns in fold_data.items()}
    fold_keys = sorted(fold_patterns.keys())
    
    # Compute pairwise similarities
    similarities = []
    for i, fold_i in enumerate(fold_keys):
        for j, fold_j in enumerate(fold_keys):
            if i < j:
                patterns_i = fold_patterns[fold_i]
                patterns_j = fold_patterns[fold_j]
                
                # For each pattern in fold_i, find best match in fold_j
                for p_i in patterns_i:
                    best_sim = 0
                    for p_j in patterns_j:
                        if p_i['series_idx'] == p_j['series_idx']:
                            sim = compute_pattern_similarity(p_i, p_j)
                            best_sim = max(best_sim, sim)
                    if best_sim > 0:
                        similarities.append(best_sim)
    
    # Analyze pattern positions and widths
    all_starts = defaultdict(list)
    all_widths = defaultdict(list)
    all_series = defaultdict(list)
    
    for fold_key, patterns in fold_patterns.items():
        for idx, pattern in enumerate(patterns):
            all_starts[idx].append(pattern['start'])
            all_widths[idx].append(pattern['width'])
            all_series[idx].append(pattern['series_idx'])
    
    # Compute statistics
    start_stds = [np.std(starts) if len(starts) > 1 else 0 for starts in all_starts.values()]
    width_stds = [np.std(widths) if len(widths) > 1 else 0 for widths in all_widths.values()]
    
    return {
        'mean_similarity': np.mean(similarities) if similarities else 0,
        'std_similarity': np.std(similarities) if similarities else 0,
        'min_similarity': np.min(similarities) if similarities else 0,
        'max_similarity': np.max(similarities) if similarities else 0,
        'mean_start_std': np.mean(start_stds) if start_stds else 0,
        'mean_width_std': np.mean(width_stds) if width_stds else 0,
        'num_folds': num_folds,
        'patterns_per_fold': [len(patterns) for patterns in fold_patterns.values()]
    }

def load_all_datasets():
    """Load patterns from all datasets."""
    base_path = Path('../json_files')
    results = {}
    
    # MIMIC
    mimic_path = base_path / 'mimic' / 'pattern_parameters.json'
    if mimic_path.exists():
        results['MIMIC'] = load_json_patterns(mimic_path)
        print(f"Loaded MIMIC: {len(results['MIMIC'])} folds")
    
    # Bonn EEG
    bonn_path = base_path / 'bonn_eeg' / 'pattern_parameters.json'
    if bonn_path.exists():
        results['Bonn_EEG'] = load_json_patterns(bonn_path)
        print(f"Loaded Bonn EEG: {len(results['Bonn_EEG'])} folds")
    
    # MITBIH
    mitbih_path = base_path / 'mitbih_pattern_parameters.json'
    if mitbih_path.exists():
        results['MITBIH'] = load_json_patterns(mitbih_path)
        print(f"Loaded MITBIH: {len(results['MITBIH'])} folds")
    
    # REMC (multiple cell lines)
    remc_patterns = {}
    remc_path = base_path / 'remc'
    if remc_path.exists():
        for json_file in remc_path.glob('pattern_parameters_*.json'):
            cell_line = json_file.stem.replace('pattern_parameters_', '')
            remc_patterns[cell_line] = load_json_patterns(json_file)
            print(f"Loaded REMC {cell_line}: {len(remc_patterns[cell_line])} folds")
    results['REMC'] = remc_patterns
    
    # AZT1D (multiple subjects - different structure)
    azt1d_patterns = {}
    azt1d_path = base_path / 'azt1d'
    if azt1d_path.exists():
        for json_file in azt1d_path.glob('pattern_parameters_*.json'):
            subject = json_file.stem.replace('pattern_parameters_', '')
            azt1d_patterns[subject] = load_json_patterns(json_file)
            print(f"Loaded AZT1D {subject}")
    results['AZT1D'] = azt1d_patterns
    
    return results

def analyze_all_datasets():
    """Analyze pattern stability across all datasets."""
    all_data = load_all_datasets()
    summary_results = []
    
    # MIMIC
    if 'MIMIC' in all_data and all_data['MIMIC']:
        stats = analyze_fold_patterns(all_data['MIMIC'])
        if stats:
            summary_results.append({
                'Dataset': 'MIMIC',
                'Subset': 'All',
                **stats
            })
    
    # Bonn EEG
    if 'Bonn_EEG' in all_data and all_data['Bonn_EEG']:
        stats = analyze_fold_patterns(all_data['Bonn_EEG'])
        if stats:
            summary_results.append({
                'Dataset': 'Bonn_EEG',
                'Subset': 'All',
                **stats
            })
    
    # MITBIH
    if 'MITBIH' in all_data and all_data['MITBIH']:
        stats = analyze_fold_patterns(all_data['MITBIH'])
        if stats:
            summary_results.append({
                'Dataset': 'MITBIH',
                'Subset': 'All',
                **stats
            })
    
    # REMC (per cell line)
    if 'REMC' in all_data and all_data['REMC']:
        for cell_line, fold_data in all_data['REMC'].items():
            stats = analyze_fold_patterns(fold_data)
            if stats:
                summary_results.append({
                    'Dataset': 'REMC',
                    'Subset': cell_line,
                    **stats
                })
    
    # AZT1D - different structure (single pattern set per subject, no folds)
    if 'AZT1D' in all_data and all_data['AZT1D']:
        for subject, data in all_data['AZT1D'].items():
            if 'patterns' in data:
                num_patterns = len(data['patterns'])
                summary_results.append({
                    'Dataset': 'AZT1D',
                    'Subset': subject,
                    'num_patterns': num_patterns,
                    'mean_similarity': None,  # No folds to compare
                    'std_similarity': None,
                    'min_similarity': None,
                    'max_similarity': None,
                    'mean_start_std': None,
                    'mean_width_std': None,
                    'num_folds': 1,
                    'patterns_per_fold': [num_patterns]
                })
    
    return pd.DataFrame(summary_results)

def plot_stability_heatmap(all_data, dataset_name, fold_data):
    """Create similarity heatmap between folds."""
    if not fold_data or len(fold_data) < 2:
        return None
    
    fold_keys = sorted(fold_data.keys())
    n_folds = len(fold_keys)
    
    # Create similarity matrix
    sim_matrix = np.zeros((n_folds, n_folds))
    
    for i, fold_i in enumerate(fold_keys):
        for j, fold_j in enumerate(fold_keys):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                patterns_i = fold_data[fold_i]
                patterns_j = fold_data[fold_j]
                
                sims = []
                for p_i in patterns_i:
                    for p_j in patterns_j:
                        if p_i['series_idx'] == p_j['series_idx']:
                            sim = compute_pattern_similarity(p_i, p_j)
                            sims.append(sim)
                
                avg_sim = np.mean(sims) if sims else 0
                sim_matrix[i, j] = avg_sim
                sim_matrix[j, i] = avg_sim
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, square=True, ax=ax,
                xticklabels=fold_keys, yticklabels=fold_keys)
    ax.set_title(f'{dataset_name}: Pattern Similarity Between Folds')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Fold')
    plt.tight_layout()
    
    return fig

def create_comprehensive_stability_plot(results_df):
    """Create a single comprehensive plot showing stability across all datasets."""
    # Filter to only datasets with fold comparisons
    fold_df = results_df[results_df['mean_similarity'].notna()].copy()
    
    if fold_df.empty:
        return None
    
    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data - aggregate by dataset
    dataset_stats = []
    for dataset in fold_df['Dataset'].unique():
        subset = fold_df[fold_df['Dataset'] == dataset]
        dataset_stats.append({
            'Dataset': dataset,
            'Mean Similarity': subset['mean_similarity'].mean(),
            'Std Similarity': subset['mean_similarity'].std() if len(subset) > 1 else 0,
            'Mean Start Std': subset['mean_start_std'].mean(),
            'Mean Width Std': subset['mean_width_std'].mean()
        })
    
    stats_df = pd.DataFrame(dataset_stats)
    
    # Panel 1: Pattern Similarity Across Folds
    x_pos = np.arange(len(stats_df))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars1 = ax1.bar(x_pos, stats_df['Mean Similarity'], 
                    yerr=stats_df['Std Similarity'],
                    color=colors[:len(stats_df)], alpha=0.8, 
                    capsize=5, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Pattern Similarity', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Pattern Reproducibility Across Folds', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stats_df['Dataset'], fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel 2: Position Stability
    width = 0.35
    x_pos2 = np.arange(len(stats_df))
    
    bars2 = ax2.bar(x_pos2 - width/2, stats_df['Mean Start Std'], width,
                    label='Start Position Std', color='#3498db', alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    bars3 = ax2.bar(x_pos2 + width/2, stats_df['Mean Width Std'], width,
                    label='Width Std', color='#e74c3c', alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Pattern Location Consistency', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(stats_df['Dataset'], fontsize=11)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def main():
    print("="*60)
    print("Pattern Stability and Reproducibility Analysis")
    print("="*60)
    
    # Load and analyze all datasets
    print("\nLoading datasets...")
    all_data = load_all_datasets()
    
    print("\nAnalyzing stability metrics...")
    results_df = analyze_all_datasets()
    
    # Save summary statistics
    output_path = Path('../results')
    output_path.mkdir(exist_ok=True)
    results_df.to_csv(output_path / 'pattern_stability_summary.csv', index=False)
    print(f"\nSaved summary to {output_path / 'pattern_stability_summary.csv'}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for dataset in results_df['Dataset'].unique():
        subset_df = results_df[results_df['Dataset'] == dataset]
        print(f"\n{dataset}:")
        
        # Only show similarity metrics for datasets with folds
        if subset_df['mean_similarity'].notna().any():
            print(f"  Mean Similarity: {subset_df['mean_similarity'].mean():.4f} ± {subset_df['mean_similarity'].std():.4f}")
            print(f"  Mean Start Std: {subset_df['mean_start_std'].mean():.2f} ± {subset_df['mean_start_std'].std():.2f}")
            print(f"  Mean Width Std: {subset_df['mean_width_std'].mean():.2f} ± {subset_df['mean_width_std'].std():.2f}")
        
        print(f"  Avg Patterns per Fold: {np.mean([np.mean(p) for p in subset_df['patterns_per_fold']]):.1f}")
        print(f"  Subsets analyzed: {len(subset_df)}")
    
    # Generate comprehensive visualization
    print("\nGenerating comprehensive visualization...")
    vis_output = Path('../manuscript/images')
    vis_output.mkdir(exist_ok=True, parents=True)
    
    fig = create_comprehensive_stability_plot(results_df)
    if fig:
        fig.savefig(vis_output / 'pattern_stability.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved comprehensive stability plot to {vis_output / 'pattern_stability.png'}")
    
    # Also create individual heatmaps for key datasets (for supplementary)
    print("\nGenerating individual heatmaps (for supplementary materials)...")
    heatmap_output = Path('../manuscript/images/supplementary')
    heatmap_output.mkdir(exist_ok=True, parents=True)
    
    # MIMIC heatmap
    if 'MIMIC' in all_data and all_data['MIMIC']:
        fig = plot_stability_heatmap(all_data, 'MIMIC', all_data['MIMIC'])
        if fig:
            fig.savefig(heatmap_output / 'mimic_stability_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # REMC E003 heatmap
    if 'REMC' in all_data and all_data['REMC'] and 'E003' in all_data['REMC']:
        fig = plot_stability_heatmap(all_data, 'REMC', all_data['REMC']['E003'])
        if fig:
            fig.savefig(heatmap_output / 'remc_stability_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()

