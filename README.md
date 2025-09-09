# Pattern Extraction and Machine Learning Pipeline

This repository contains a pattern extraction and machine learning pipeline for time series and spatial data. The code identifies important patterns and uses them as features for machine learning tasks. The found patterns can be visualized and localized and thus offer superior explainability. 
During feature extraction we optimize the start and end positions of a pattern as well as the polynomial function with which we define the pattern. We then create a feature by calculating the mean squared error of the given pattern against the sample values. During optimization we optimize for the specified evaluation metric and so we obtain highly predictive, localizable and explainable patterns. We further do hyperparameter tuning and backward elimination. This three step process (feature extraction, hyperparameter tuning and backward elimination) is done iteratively until no improvement can be found anymore. 

## Setup

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Prgrmmrjns/patX
cd patX
```

2. Create a virtual environment (recommended):
```bash
# Using venv
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

To run the pipeline:

```bash
python main.py
```

## Configuration

You can configure the dataset and task type in the `params.py` file:

```python
DATASET = 'REMC'  # Options: 'REMC', 'MITBIH', 'D1NAMO'
CELL_LINE = 'E003'
METRIC = 'auc'   # Options: 'auc', 'accuracy', 'rmse'
TASK_TYPE = 'classification'  # Options: 'classification', 'regression'
USE_MULTIPLE_SERIES = True  # Whether to use multiple time series
```

## Supported Datasets

- **REMC**: Histone modification data with multiple series (cell lines E003 and E004)
- **MITBIH**: ECG data for arrhythmia classification

## Main Components

- **Pattern Optimization**: Finds optimal patterns in time-series data
- **Feature Extraction**: Converts time-series data into features
- **Feature Selection**: Selects the most informative features
- **Model Training**: Trains and tunes machine learning models

## Project Structure

- `patternextraction.py`: Contains the `PatternOptimizer` class
- `main.py`: Script to run the full pipeline
- `params.py`: Configuration parameters
- `model.py`: Define your model params here
- `data_processing.py`: Code to preprocess the datasets (train test split, etc.)
- `preprcess_mitbih.py`: Preprocess and bin the MITBIH dataset
- `remc.ipynb`: Code to run entire remc cell lines. For that please store all cell lines locally as .csv files
- `datasets/`: Directory containing example datasets

## Example Usage

```python
from patx import PatternOptimizer
# Initialize the pattern optimizer
optimizer = PatternOptimizer(
    input_data,
    y_train, 
    model=model, 
    max_n_trials=MAX_N_TRIALS,
    show_progress=SHOW_PROGRESS,
    test_size=VAL_SIZE,
    optuna_no_improvement_rounds=OPTUNA_NO_IMPROVEMENT_ROUNDS,
    n_jobs=N_JOBS,
    dataset=DATASET,  
    multiple_series=len(TIME_SERIES_IDENTIFIERS) > 0,
    X_test_data=test_data,
    polynomial_degree=POLYNOMIAL_DEGREE,
    metric=METRIC,
    val_size=VAL_SIZE
)

# Run feature extraction
patex_result = optimizer.automatic_feature_extraction()

# Extract results
patterns = patex_result['patterns']
train_features = patex_result['features']
X_train = patex_result['X_train']
X_val = patex_result['X_val']
y_train = patex_result['y_train']
y_val = patex_result['y_val']
X_test = patex_result['X_test']  
best_model = patex_result['best_model']  

# Use the best model for predictions
if METRIC == 'auc':
    train_preds = model.predict_proba(X_train)[:, 1]
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
else:
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

# Calculate performance metrics
if METRIC == 'auc':
    train_score = roc_auc_score(y_train, train_preds)
    val_score = roc_auc_score(y_val, val_preds)
    test_score = roc_auc_score(y_test, test_preds)
elif METRIC == 'accuracy':
    train_score = accuracy_score(y_train, train_preds)
    val_score = accuracy_score(y_val, val_preds)
    test_score = accuracy_score(y_test, test_preds)
else:
    train_score = np.sqrt(mean_squared_error(y_train, train_preds))
    val_score = np.sqrt(mean_squared_error(y_val, val_preds))
    test_score = np.sqrt(mean_squared_error(y_test, test_preds))

# Print final results
print(f"\nFinal model performance:")
print(f"Train {METRIC}: {train_score:.4f}")
print(f"Val {METRIC}: {val_score:.4f}")
print(f"Test {METRIC}: {test_score:.4f}")

# Save pattern parameters to JSON
optimizer.save_parameters_to_json(DATASET)

# Visualize all patterns
all_indices = list(range(len(patterns)))
optimizer.visualize_patterns(all_indices, DATASET)

```

## Dataset Sources

* The Mitbih dataset is available from: https://www.kaggle.com/datasets/mondejar/mitbih-database 
* The REMC dataset is taken from https://gitlab.gwdg.de/MedBioinf/generegulation/patternchrome 

## To Dos

* More datasets (MIMIC IV)
* More explainability functions
* Baseline comparison to other feature engieering algorithms and DL techniques
* Package development