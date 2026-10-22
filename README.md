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
| 2. Data | Download raw data (see below), then `python scripts/preprocess.py` | `datasets/*` processed files; `elsarticle/clean/dataset_characteristics.tex` |
| 3. Benchmarks | `python scripts/main_eval.py` | `results/main_eval.csv`, `parameters/<dataset>/fold*.json` |
| 4. Ablations | `python scripts/ablation_study.py` | `results/ablation_study.csv`, `parameters/ablation/` |
| 5. Tables | `python scripts/generate_tables.py` | `elsarticle/clean/results_table.tex`, `elsarticle/clean/ablation_results.tex` |
| 6. Figures | see below | `elsarticle/clean/*.eps`, `*.png`, … |
| 7. PDF | `cd elsarticle/clean && latexmk -pdf main.tex` | `elsarticle/clean/main.pdf` |

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

Git tracks `scripts/`, `results/`, `parameters/`, `requirements.txt`, and `datasets/metadata.json`. Large raw files, `elsarticle/`, and a local `sublimex/` checkout are gitignored. Processing notes are in `elsarticle/clean/supplementary/supplementary.tex` (local manuscript; not on GitHub).

| Dataset | Approx. | Source | Place under |
|---------|---------|--------|-------------|
| **AZT1D** | ~50 MB | [Zenodo](https://zenodo.org/records/15094234) | `datasets/azt1d/CGM Records/Subject N/Subject N.csv` |
| **MIT-BIH** | ~100 MB | [PhysioNet mitdb](https://physionet.org/content/mitdb/1.0.0/) | `datasets/mitbih/{record}.csv` and `{record}annotations.txt` |
| **Emotions** | ~10 MB | [Kaggle EEG emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) | `datasets/emotions/emotions.csv` |
| **REMC** | ~2 GB | [PatternChrome](https://gitlab.gwdg.de/MedBioinf/generegulation/patternchrome) | `datasets/remc/remc_<cell>.parquet` (loaded as-is) |
| **MIMIC-IV** | ~571 MB | Preselected `final_df.csv`, not a raw [PhysioNet](https://physionet.org/content/mimiciv/) extract | `datasets/mimic/final_df.csv` |
| **PAMAP2** | ~1.5 GB | [UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | `datasets/pamap2/Protocol/subject10N.dat` |
| **SVD** | ~3 GB | [Saarbrücken Voice DB](https://stimmdb.coli.uni-saarland.de) | `datasets/SVD/raw/<diagnosis>/` with `overview.csv` and phrase wavs |

Then:

```bash
python scripts/preprocess.py
```

That command rewrites every processed file. The AZT1D parquet behind Table 2 omits subjects 14 and 23; a fresh run keeps them, so the sample count changes. MIT-BIH is rebuilt from the record CSVs and annotation texts, not a raw WFDB drop. MIMIC-IV is not rebuilt from PhysioNet: `preprocess.py` only reshapes a preselected `final_df.csv` (cohort, variables, and hourly alignment are already fixed in that file). REMC parquets are used as released by PatternChrome.

Metadata and channel definitions: `datasets/metadata.json`.

---

## Evaluation settings (main text)

Aligned with `elsarticle/clean/main.tex` Methods:

- **SublimeX**: 300 Optuna trials per feature (TPE); mean over segment; four views (raw, z-score, derivative, FFT power); LightGBM depth 5; inner 50/50 search split (seed 42); stop when no strict validation improvement.
- **Baselines**: tsfresh (`MinimalFCParameters`), catch22, MiniRocket (`random_state=42`), RDST (`max_shapelets=1000`), 1D CNN — all on the same outer splits; feature-based methods use the same LightGBM readout.
- **Splits**: 5-fold CV (most sets); PAMAP2 8-fold LOSO; REMC 5-fold × 56 cell lines; AZT1D temporal 80/20 per patient (pooled SublimeX discovery on training windows only).
- **Hyperparameters**: Supplementary Table S1 (`elsarticle/clean/supplementary/supplementary.tex`).
- **Package**: `pip install -r requirements.txt` installs `sublimex==0.1.2`. If a checkout exists at `sublimex/`, scripts put that directory on `sys.path` first. The manuscript imports `SublimeX`, `mean_objective`, `LightGBMModel`, `TRANSFORMS`, and `extract_feature`.

---

## Key paths

| Path | Role |
|------|------|
| `scripts/config.py` | Paths, `LOAD_FEATURES`, `K_FOLDS` |
| `parameters/<dataset>/fold<N>.json` | Saved SublimeX features (main evaluation) |
| `parameters/ablation/` | Ablation feature JSONs |
| `results/main_eval.csv` | Benchmark scores, times, feature counts |
| `results/ablation_study.csv` | Ablation scores |
| `elsarticle/clean/results_table.tex` | Table~3 |
| `elsarticle/clean/ablation_results.tex` | Table~4 |
| `elsarticle/clean/flowchart.png` | Figure~1 |
| `elsarticle/clean/feature_analysis.eps` | Figure~2 |
| `elsarticle/clean/incremental_features.png` | Figure~3 |
| `elsarticle/clean/domain_interpretation.png` | Figure~4 |

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
  title={Interpretable supervised feature extraction for multivariate biomedical sequences},
  author={Wolber, J.C. and Paul, N.B. and Sellin, J. and Samadi, M. E. and Muecke, M. and Schuppert, A.},
  journal={Pattern Recognition},
  year={2025}
}
```
