# Pattern Stability Analysis

## Overview
Analyzes first pattern reproducibility across cross-validation folds to assess patX's pattern discovery consistency.

## Scripts

### `pattern_stability_analysis.py`
Main script for analyzing and visualizing pattern stability.

**Usage:**
```bash
# Default (MITBIH)
python pattern_stability_analysis.py

# Specific datasets
python pattern_stability_analysis.py --dataset=mitbih
python pattern_stability_analysis.py --dataset=emotions
python pattern_stability_analysis.py --dataset=mimic

# AZT1D (auto-picks first subject if not specified)
python pattern_stability_analysis.py --dataset=azt1d
python pattern_stability_analysis.py --dataset=azt1d --subject=540

# REMC (auto-picks first cell line if not specified)
python pattern_stability_analysis.py --dataset=remc
python pattern_stability_analysis.py --dataset=remc --cell_line=E003
```

**Output:**
- Statistics: center/width CV, transformation distribution, Spearman correlations
- Visualization: overlaid patterns with search boundaries
- Saved figures:
  - `../manuscript/images/pattern_stability.png` (MITBIH)
  - `../manuscript/images/pattern_stability_{dataset}.png` (others)

### `aggregate_remc_stability.py`
Aggregates stability statistics across all 46 REMC cell lines.

**Usage:**
```bash
python aggregate_remc_stability.py
```

**Output:**
- Median center/width CV across all cell lines
- Aggregate Spearman correlation statistics
- Mean scores across all patterns

## Key Findings

### MITBIH (5-fold)
- **Excellent** positional stability (center CV=0.039)
- 4/5 folds converge to cumsum transformation
- Strong inter-correlations among similar transforms (ρ=0.48-0.87)

### Emotions (5-fold)
- High center variability (CV=1.095) reflects multiple optimal solutions
- Consistent width (CV=0.190)
- Diverse transformation selections (3 raw, 1 cumsum, 1 wavelet)

### MIMIC (3-fold)
- **Excellent** width consistency (CV=0.037)
- 2/3 folds achieve perfect agreement (ρ=1.000)
- Strong derivative transformation preference

### REMC (5-fold × 46 cell lines)
- Moderate consistency (median CV~0.4)
- Substantial biological variability across tissue contexts
- Strong predictive performance despite pattern diversity

## Interpretation

1. **Positional Stability**: PatX reliably identifies discriminative temporal regions
2. **Multiple Solutions**: Optimization may converge to different transformations with comparable performance
3. **Biological Context**: Pattern variability reflects domain complexity and biological diversity
4. **Clinical Relevance**: Patterns remain interpretable despite morphological variation

## Files Generated
- `pattern_stability_summary.txt`: Comprehensive findings across all datasets
- `pattern_stability_{dataset}.png`: Visualization for each dataset
- `README_stability_analysis.md`: This documentation

