# PatX Manuscript Repository

This repository contains the evaluation scripts and manuscript for the PatX paper.

## Structure

- `patx/` - PatX package source (published to [PyPI](https://pypi.org/project/patx/))
- `eval_scripts/` - Evaluation pipelines for all datasets
- `manuscript/` - LaTeX manuscript source
- `processed_datasets/` - Preprocessed data artifacts
- `results/` - CSV results from experiments

## Installation

```bash
pip install patx
```

Or for development:
```bash
pip install -e patx/
```

## Running Evaluations

```bash
cd eval_scripts
python mitbih.py      # ECG arrhythmia classification
python emotions.py    # EEG emotion classification
python remc.py        # Gene expression prediction
python azt1d.py       # Glucose forecasting
python pamap2.py      # Activity recognition
python mimic.py       # ARDS prediction
python svd.py         # Voice pathology detection
```

## PatX Features

See `patx/README.md` for full documentation. Key features:

- **Custom transforms**: `TRANSFORMS.register('name', func)`
- **Custom distance metrics**: `DISTANCES.register('name', func)`
- **Multiple ML backends**: LightGBM, XGBoost, RandomForest, or any sklearn model
- **Discovery modes**: Joint optimization or iterative greedy search
