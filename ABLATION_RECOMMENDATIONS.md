# Ablation Study Summary and Recommendations

## Completed Updates

### 1. Manuscript Section (main.tex)
Updated the Ablation Study subsection with findings from the completed experiments:

**B-spline degree ablation:**
- Tested degrees 3, 4, 5 (baseline), 6, 7
- Degree 5 is optimal: 0.9413±0.0043 accuracy, 344s runtime
- Degree 3 similar accuracy but 12× slower (optimization challenges)
- Higher degrees (6, 7) degrade performance

**Optimization trials ablation:**
- Tested 50, 100 (baseline), 200 trials
- 50 trials: competitive (0.9377) but 2.4× faster (142s)
- 200 trials: marginal gain (0.9414 vs 0.9413) at 1.7× cost

### 2. Ablation Script Updates (ablation_study.py)
- Reduced from 258 to 121 lines
- Added model wrappers: `LogRegWrapper` and `MLPWrapper`
- Added model ablation variants to test patX with:
  - Logistic Regression (linear model)
  - MLP (non-linear, non-tree-based)
- Script now tests 9 variants total

## Running New Ablations

To run the updated script with model ablations:

```bash
cd eval_scripts
python ablation_study.py
```

This will test:
1. Baseline (LightGBM, degree 5, 100 trials)
2. Degree variations (3, 4, 6, 7)
3. Trial variations (50, 200)
4. **NEW: Logistic Regression** (instead of LightGBM)
5. **NEW: MLP** (instead of LightGBM)

## Additional Ablation Ideas (Require core.py Modifications)

The following ablations would require modifying `eval_scripts/core.py`:

### 1. Transform Restrictions
**Goal:** Test if math transforms (FFT, wavelet, derivative, cumsum) improve over raw signal

**Modification needed:**
- Line 141 in `core.py`: `transform_types = ['raw', 'derivative', 'cumsum', 'fft_power', 'wavelet']`
- Add parameter `allowed_transforms` to `feature_extraction()`
- Test variants:
  - Raw only: `['raw']`
  - Raw + derivative: `['raw', 'derivative']`
  - No frequency domain: `['raw', 'derivative', 'cumsum']`

**Expected finding:** Multi-representation search should significantly outperform raw-only

### 2. Shift Tolerance Ablation
**Goal:** Test if flexible shift matching improves pattern discovery

**Modification needed:**
- Line 160 in `core.py`: `shift_tolerance = trial.suggest_float('shift_tolerance', 0.0, 1.0)`
- Add parameter `fixed_shift_tolerance` to `feature_extraction()`
- Test variants:
  - No shift: `shift_tolerance = 0.0` (exact position matching)
  - Fixed moderate: `shift_tolerance = 0.5`
  - Adaptive (baseline): optimize in [0.0, 1.0]

**Expected finding:** Adaptive shift should help for patterns with temporal variation

### 3. Dissimilarity Feature Ablation
**Goal:** Test contribution of relative vs absolute matching

**Modification needed:**
- Line 159: `use_relative = trial.suggest_categorical('use_relative', [False, True])`
- Add parameter `force_relative` or `force_absolute`
- Test variants:
  - Absolute only: RMSE distance
  - Relative only: Correlation-based
  - Both (baseline): optimize choice

**Expected finding:** Relative features should help with amplitude-invariant patterns

### 4. Window Size Range Ablation
**Goal:** Test if constraining pattern width improves discovery

**Modification needed:**
- Line 163: `pattern_width = trial.suggest_float('pattern_width', 2.0, min(50.0, n_time_points))`
- Add parameters `min_width`, `max_width`
- Test variants:
  - Small patterns: [2, 20]
  - Large patterns: [30, 50]
  - Flexible (baseline): [2, 50]

**Expected finding:** Optimal range is dataset-dependent (ECG may need wider windows)

### 5. Multi-Scale Patterns
**Goal:** Test if forcing patterns at different scales improves coverage

**Modification needed:**
- Add stratified pattern discovery (e.g., first 3 patterns use width [2,15], next 3 use [15,50])
- Requires refactoring the pattern search loop

**Expected finding:** May improve diversity but increase search time

## Priority Recommendations

For manuscript inclusion, prioritize:

1. **Model ablations (LogReg, MLP)** - Already implemented, just need to run
   - Tests if patX patterns generalize beyond tree-based models
   
2. **Transform restrictions** - Highest scientific value
   - Core claim is that multi-representation search is beneficial
   - Would validate this directly

3. **Shift tolerance ablation** - Medium value
   - Tests a key flexibility mechanism
   - Useful for understanding when exact vs approximate matching matters

## Dataset-Specific Suggestions

Beyond MITBIH, consider running ablations on:
- **Emotions**: Small dataset, tests if degree/trials can be reduced
- **AZT1D**: Regression task, tests if findings generalize beyond classification
- **REMC**: High-dimensional (5 channels), tests if multi-series benefits are maintained

## Implementation Note

For transforms/shift ablations, cleanest approach:
```python
# In core.py
def feature_extraction(..., allowed_transforms=None, fixed_shift=None, ...):
    transform_types = allowed_transforms or ['raw', 'derivative', 'cumsum', 'fft_power', 'wavelet']
    ...
    if fixed_shift is not None:
        shift_tolerance = fixed_shift
    else:
        shift_tolerance = trial.suggest_float('shift_tolerance', 0.0, 1.0)
```

This preserves backward compatibility while enabling ablation studies.

