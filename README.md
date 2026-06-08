# SublimeX Manuscript Repository
---

## Quick reproduction (paper numbers)

With **preprocessed data** in `datasets/` and **saved features** in `parameters/` (committed or downloaded), regenerate tables and figures in minutes:

```bash
python -m venv .venv && source .venv/bin/activate   # or: conda create -n sublimex python=3.11
pip install -r requirements.txt

python scripts/generate_tables.py
python scripts/flowchart.py
python scripts/feature_analysis.py
python scripts/domain_interpretation.py
python scripts/incremental_features.py
```

This re-runs LightGBM/CNN/MiniRocket scoring and plotting using stored SublimeX segment definitions. Set `LOAD_FEATURES = False` in `scripts/config.py` only if you intend to re-run full Optuna feature search (days of compute; results may differ slightly from parallel TPE).

---

## Full pipeline

| Step | Command | Outputs |
|------|---------|---------|
| 1. Setup | `pip install -r requirements.txt` | Python env |
| 2. Data | Download raw data (see below), then `python scripts/preprocess.py` | `datasets/*` processed files; `elsarticle/dataset_characteristics.tex` |
| 3. Benchmarks | `python scripts/main_eval.py` | `results/main_eval.csv`, `parameters/<dataset>/fold*.json` |
| 4. Ablations | `python scripts/ablation_study.py` | `results/ablation_study.csv`, `parameters/ablation/` |
| 5. Tables | `python scripts/generate_tables.py` | `elsarticle/results_table.tex`, `elsarticle/ablation_results.tex` |
| 6. Figures | see below | `elsarticle/*.eps`, `*.png`, … |
| 7. PDF | `cd elsarticle && latexmk -pdf main.tex` | `elsarticle/main.pdf` |

**Figure scripts** (run from repo root; order does not matter):

```bash
python scripts/flowchart.py              # Figure 1 — methodology flowchart
python scripts/feature_analysis.py       # Figure 2 — performance stability
python scripts/incremental_features.py   # Figure 3 — incremental feature analysis
python scripts/domain_interpretation.py  # Figure 4 — domain interpretation (SHAP)
python scripts/graphical_abstract.py     # graphical abstract (optional)
```

All analysis scripts use absolute paths via `scripts/config.py` and may be invoked from **any working directory**.

---

## Environment

- Python **3.11+**
- **Hardware** (paper): MacBook Pro M2, 12 cores, 32 GB RAM
- Thread caps are set in `main_eval.py` for reproducible library baselines (`OMP_NUM_THREADS=1`, etc.)

```bash
pip install -r requirements.txt
```

---

## Datasets

Large raw files are **not** in git (see `.gitignore`). Download sources match `supplementary_data/supplementary.tex`.

| Dataset | Approx. | Source | Place under |
|---------|---------|--------|-------------|
| **AZT1D** | ~50 MB | [Zenodo](https://zenodo.org/records/15094234) | `datasets/azt1d/CGM Records/` |
| **MIT-BIH** | ~100 MB | [PhysioNet mitdb](https://physionet.org/content/mitdb/1.0.0/) | `datasets/mitbih/` |
| **Emotions** | ~10 MB | [Kaggle EEG emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) | `datasets/emotions/` |
| **REMC** | ~2 GB | [PatternChrome bins](https://gitlab.gwdg.de/MedBioinf/generegulation/patternchrome/-/tree/main/datasets/Binned_sequencing_data) | `datasets/remc/*.parquet` |
| **MIMIC-IV** | ~571 MB | [PhysioNet](https://physionet.org/content/mimiciv/) (credentialing) | `datasets/mimic/final_df.csv` |
| **PAMAP2** | ~1.5 GB | [UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | `datasets/pamap2/Protocol/` |
| **SVD** | ~3 GB | [Saarbrücken Voice DB](https://stimmdb.coli.uni-saarland.de) | `datasets/SVD/raw/` |

Then:

```bash
python scripts/preprocess.py
```

Metadata and channel definitions: `datasets/metadata.json`.

---

## Evaluation settings (main text)

Aligned with `elsarticle/main.tex` Methods:

- **SublimeX**: 300 Optuna trials per feature (TPE); mean over segment; four views (raw, z-score, derivative, FFT power); LightGBM depth 5; inner 50/50 search split (seed 42); stop when no strict validation improvement.
- **Baselines**: tsfresh (`MinimalFCParameters`), catch22, MiniRocket (`random_state=42`), RDST (`max_shapelets=1000`), 1D CNN — all on the same outer splits; feature-based methods use the same LightGBM readout.
- **Splits**: 5-fold CV (most sets); PAMAP2 8-fold LOSO; REMC 5-fold × 56 cell lines; AZT1D temporal 80/20 per patient (pooled SublimeX discovery on training windows only).
- **Hyperparameters**: Supplementary Table S1 (`supplementary_data/supplementary.tex`).

---

## Key paths

| Path | Role |
|------|------|
| `scripts/config.py` | Paths, `LOAD_FEATURES`, `K_FOLDS` |
| `parameters/<dataset>/fold<N>.json` | Saved SublimeX features (main evaluation) |
| `parameters/ablation/` | Ablation feature JSONs |
| `results/main_eval.csv` | Benchmark scores, times, feature counts |
| `results/ablation_study.csv` | Ablation scores |
| `elsarticle/results_table.tex` | Table~3 |
| `elsarticle/ablation_results.tex` | Table~4 |
| `elsarticle/flowchart.eps` | Figure~1 |
| `elsarticle/feature_analysis.eps` | Figure~2 |
| `elsarticle/incremental_features.png` | Figure~3 |
| `elsarticle/domain_interpretation.png` | Figure~4 |

---

## Reproducibility notes

| Component | Bit-exact? | Notes |
|-----------|------------|--------|
| Saved `parameters/*.json` | Yes | Re-load with `LOAD_FEATURES=True` |
| LightGBM on fixed features | Yes | `deterministic=True` for SublimeX readout |
| Optuna feature search | No | Parallel TPE can vary; use committed JSON for paper features |
| CNN / MiniRocket | Approx. | GPU/MPS vs CPU may differ slightly |

**Ablation baseline** rows are taken from `main_eval.csv` (SublimeX), not re-fit in `ablation_study.py`. Delete `results/*.csv` to force a full re-run.

---

## Citation

```bibtex
@article{wolber2025sublimex,
  title={Interpretable supervised feature extraction for time series and spatial data},
  author={Wolber, J.C. and Paul, N.B. and Sellin, J. and Samadi, M. E. and Muecke, M. and Schuppert, A.},
  journal={Pattern Recognition},
  year={2025}
}
```
