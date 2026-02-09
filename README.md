# SublimeX Manuscript Repository

This repository contains the evaluation pipelines, analysis scripts, and manuscript
source for the SublimeX paper: *"SublimeX: Supervised bottom-up localized
multi-representative feature eXtraction for time series and spatial data"*.

---

## Reproducing Results

Follow these steps to reproduce the paper’s experiments and figures.

### 1. Environment setup

From the repository root:

```bash
git clone https://github.com/Prgrmmrjns/patternExtraction-manuscript.git
cd patternExtraction-manuscript

conda create -n sublimex python=3.13
conda activate sublimex
pip install -r requirements.txt
```

### 2. Download datasets

Several datasets exceed GitHub’s 100 MB limit and must be downloaded separately.
Place each dataset in the path listed under **Target location**.

| Dataset | Approx. size | Source | Target location |
|---------|--------------|--------|------------------|
| **PAMAP2** | ~1.5 GB | [UCI ML Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | `datasets/pamap2/Protocol/` |
| **MIMIC-IV** | ~571 MB | [PhysioNet](https://physionet.org/content/mimiciv/) (credentialing required) | `datasets/mimic/` (e.g. `final_df.csv`) |
| **MIT-BIH** | ~100 MB | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) | `datasets/mitbih/` |
| **SVD** | ~3 GB | [Saarbrücken Voice Database](https://stimmdb.coli.uni-saarland.de) | `datasets/SVD/raw/` |
| **AZT1D** | ~50 MB | [Zenodo](https://zenodo.org/records/15094234) or [Mendeley](https://data.mendeley.com/datasets/gk9m674wcx/1) | `datasets/azt1d/CGM Records/` |
| **REMC** | ~2 GB | [Roadmap Epigenomics](https://gitlab.gwdg.de/MedBioinf/generegulation/patternchrome/-/tree/main/datasets/Binned_sequencing_data) | Convert .RData to .parquet and place in `datasets/remc/` |
| **Emotions** | ~10 MB | [Kaggle](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) | `datasets/emotions/` |

For MIMIC-IV, complete [PhysioNet credentialing](https://physionet.org/settings/credentialing/) and follow their export instructions to produce `final_df.csv` (or equivalent) in `datasets/mimic/`.

### 3. Preprocess data

From the repository root, run the preprocessing pipeline. It expects raw data in the locations above and writes processed files under `datasets/` (e.g. `emotions_processed.csv`, `mimic_processed.parquet`, `pamap2.parquet`, `SVD/svd.parquet`, `azt1d_all_patients.parquet`). MIT-BIH and REMC use pre-existing processed files in their dataset folders.

```bash
python scripts/preprocess.py
```

This also writes `elsarticle/tables/dataset_characteristics.tex`.

### 4. Run main evaluation

The main evaluation compares SublimeX with baselines (TSFRESH, CATCH22, MiniRocket, RDST, CNN) on all datasets. **Run from the `scripts/` directory** so result paths resolve correctly.

```bash
cd scripts
python main_eval.py
```

Output: `results/main_eval.csv`, and SublimeX parameters under `parameters/<dataset>/`.

### 5. Run ablation study

Ablation over SublimeX variants (aggregates, pattern search, decision tree, trials, etc.). Also run from `scripts/`.

```bash
cd scripts
python ablation_study.py
```

Output: `results/ablation_study.csv`.

### 6. Generate manuscript tables and figures

From `scripts/`:

```bash
# LaTeX tables (main results + ablation)
python generate_tables.py
```

Output: `elsarticle/tables/results_table.tex`, `elsarticle/tables/ablation_results.tex`.

From the repository root (these scripts use pathlib and work from any cwd):

```bash
# Domain interpretation figure (SHAP on REMC E003 and AZT1D)
python scripts/domain_interpretation.py

# Methodology flowchart
python scripts/flowchart.py

# Performance stability analysis
python scripts/performance_analysis.py

# Incremental feature analysis (optional)
python scripts/incremental_features.py
```

Figures are written to `elsarticle/images/`; optional CSVs to `results/`.

### 7. Build the manuscript

Compile the LaTeX source in `elsarticle/` (e.g. `pdflatex` or your usual workflow):

```bash
cd elsarticle
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

---

## Repository structure

```
├── elsarticle/              # LaTeX manuscript (Elsevier style)
│   ├── main.tex
│   ├── bibliography.bib
│   ├── images/
│   └── tables/
├── scripts/                 # Evaluation and analysis
│   ├── main_eval.py         # SublimeX + baselines
│   ├── ablation_study.py    # Ablation over SublimeX variants
│   ├── preprocess.py        # Dataset preprocessing and loaders
│   ├── generate_tables.py   # LaTeX tables from result CSVs
│   ├── domain_interpretation.py
│   ├── flowchart.py
│   ├── performance_analysis.py
│   ├── incremental_features.py
│   ├── core.py              # SublimeX core
│   └── model.py             # LightGBM wrapper
├── datasets/                # Raw and processed data (see .gitignore)
├── results/                 # main_eval.csv, ablation_study.csv, etc.
├── parameters/              # Extracted SublimeX parameters (JSON)
└── requirements.txt
```

---

## Datasets summary

| Dataset | Task | Samples | Channels | Length | Metric |
|---------|------|---------|----------|--------|--------|
| AZT1D | Glucose forecasting | ~12k | 3 | 24 | RMSE |
| MIT-BIH | Arrhythmia classification | ~12k | 1 | 100 | Accuracy |
| Emotions | Emotion classification | 2,048 | 5 | 254 | AUC |
| REMC | Gene expression | 18,421 | 5 | 200 | AUC |
| MIMIC-IV | Mortality prediction | 6,439 | 12 | 24 | AUC |
| PAMAP2 | Activity recognition | ~4k | 51 | 100 | Accuracy |
| SVD | Voice pathology | 1,988 | 1 | 1,000 | Accuracy |

---

## Baselines

SublimeX is compared with:

| Method | Description |
|--------|-------------|
| **TSFRESH** | Statistical time series features |
| **CATCH22** | 22 canonical time series features |
| **MiniRocket** | Random convolutional kernels |
| **RDST** | Random Dilated Shapelet Transform |
| **CNN** | 1D convolutional neural network |

---

## Preprocessed data (skip preprocessing)

If you want to skip preprocessing and only run evaluation, you need the processed
artifacts. Large ones are not in the repo; contact the authors for access if needed:

- `datasets/pamap2/pamap2.parquet` (364 MB)
- `datasets/mimic/final_df.csv` (571 MB) — then run preprocessing to produce
  `mimic_processed.parquet` used by the pipeline.

---

## Citation

If you use this code or the SublimeX method, please cite:

```bibtex
@article{wolber2025sublimex,
  title={SublimeX: Supervised bottom-up localized multi-representative feature
         eXtraction for time series and spatial data},
  author={Wolber, J.C. and Paul, N.B. and Sellin, J. and Muecke, M. and Schuppert, A.},
  journal={Pattern Recognition},
  year={2025}
}
```

---

## SublimeX package

The SublimeX Python package is available on PyPI:

```bash
pip install sublimex
```

Documentation and source: https://github.com/Prgrmmrjns/sublimex

---

## License and contact

This project is licensed under the MIT License (see LICENSE).

- Jonas Chanrithy Wolber — jwolber@ukaachen.de  
- Institute of Digitalization and General Medicine, RWTH Aachen University
