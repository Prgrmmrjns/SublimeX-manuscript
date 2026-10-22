## 2026-10-22
* **Update**: Pushed manuscript `bcbf830` and sublimex `20a43ba` to GitHub. The clean manuscript stays gitignored.
* **Update**: README notes that a full preprocess rewrites processed files and would add AZT1D subjects 14 and 23, which the paper parquet omits.
* **Update**: Preprocessing now rebuilds MIT-BIH beats from the record CSVs and keeps AZT1D subjects 14 and 23 (glucose column name; empty carb sum). README and gitignore match the tracked tree; the sublimex checkout clone URL is the real repo.
* **Update**: QUANT probe on SVD, MIMIC-IV, and AZT1D finished (not added to the paper). Regenerated Figure 4 with reversed SHAP y labels.
## 2026-10-19
* **Update**: Marked manuscript rebuilt (49 pages). Repaired three latexdiff breaks that stopped the build: the 9,996 split, deleted figure filenames, and the TSFEL accent.
* **Update**: Dropped the REMC sentence listing the five histone modifications. The main PDF is 35 pages.
* **Update**: Removed the MIT-BIH sentence on the post-R derivative versus raw voltage. The main PDF is still 36 pages; only reference 45 is on the last page.
* **Update**: Section 3.5 drops one sentence from REMC and one from AZT1D, restores the first MIT-BIH interpretation, and puts the full Figure 4 panel caption back. The main PDF is 36 pages.
* **Update**: Shortened the MIT-BIH SHAP subsection and the Figure 4 caption, and dropped the dataset URL notes from the bibliography. The main PDF is 35 pages.* **Update**: The R2 letter now describes the supplementary PAMAP2 table as a test-only comparison on the same leave-one-subject-out subjects.* **Update**: The supplementary PAMAP2 table note now states that both searches use the same leave-one-subject-out test subject.* **Update**: The supplementary PAMAP2 table now lines up subject-held-out and default LOSO test accuracy per subject, with both means.* **Update**: Section 2.7 now cites the supplementary PAMAP2 subject-held-out test accuracy ($0.840\pm0.072$ versus $0.860\pm0.081$).* **Update**: Printed the eight PAMAP2 subject-held-out folds in the supplementary text. The open PDF was built before that subsection existed.
* **Update**: Restored the first-submission opening and three-paradigms paragraphs of the Introduction (clean and marked); kept the bridge sentence, the SublimeX/showcase sentences, and the Objectives.
* **Update**: Addressed the review findings: REMC runtime is now median (min–max) in Table 3, the AZT1D gap is 0.22 throughout, Section 4.2 says "traceable", and the response letter quotes now match the manuscript. Reverted synonym-only rewording in Sections 2.2, 2.4, 3.2, 3.3, and 3.5 so the marked PDF shows only reviewer-driven changes.
* **Update**: Removed the contributions list from Related Work in both manuscripts (it repeated the Objectives). The letter now points Reviewers #9 and #11 to the objectives in Section 1.1.
* **Update**: Rewrote Related Work as four bridged paragraphs (feature libraries, localized methods, deep models, SublimeX's position) and dropped the Pattern Search and ablation detail. The response letter now gives RDST as about 3,000 features, not 3,000–10,000.
* **Update**: Response letter now quotes the introduction's showcase sentence on the seven biomedical tasks.
* **Update**: Introduction now states SublimeX on aligned one-dimensional channels indexed by time or genomic position, without the 2D/3D disclaimer. The response letter quotes match.
* **Update**: Dropped LightGBM from the keyword line. The first submission has no keywords.
* **Update**: Shortened the abstract. SHAP is general, with no arrhythmia abbreviations, and SublimeX is described as extracting a minimal, traceable feature set that domain knowledge can interpret.
## 2026-10-15
* **Update**: Removed the UEA archive subsection and its two tables from the clean and marked manuscripts.
* **Update**: Refit the nine single-worker UEA datasets with parallel Optuna. Mean accuracy is 0.572; both manuscript tables use that run.
* **Update**: SublimeX on the 26 UEA equal-length sets (default split) finished with mean accuracy 0.556. Both manuscripts now have that table and the published bake-off means.
* **Update**: Added `scripts/mark_manuscript.py` and regenerated the marked manuscript from the first submission against the current clean file.
* **Update**: Dropped “we do not read these panels as a diagnosis” from Section 3.5. The Discussion now says the SHAP localization can point to the underlying pathology without being a diagnosis.
* **Update**: Section 3.5's MIT-BIH clinical note is one paragraph on ECG derivative 118–142. The five-rhythm definitions were removed.
* **Update**: Section 3.5 now explains the post-R ECG derivative clinically for the five MIT-BIH rhythms. The Wilcoxon method moved from a Results subsection into Evaluation; Results only report the outcomes.
* **Update**: Figure 4 x-axis labels are 5.5 pt, and the gaps between the nine panels are tighter.
* **Update**: Figure 4 panel I is now a beeswarm of mean absolute SHAP across the five MIT-BIH classes, matching panels C and F. Section 3.5 and the caption no longer describe it as a per-class table.
* **Update**: Figure 4's MIT-BIH row now shows all five classes. Mean |SHAP| is averaged across classes; the leading feature remains ECG derivative 118–142 (class means 0.03, 0.35, 5.38, 1.52, −0.54). The figure is taller and the panel fonts are smaller. Class counts are unchanged.
* **Update**: Figure 4 now has a third SHAP row for MIT-BIH (fold 1, RBBB class). The leading feature is the ECG derivative 118–142 (about 47–114 ms after the R peak; means 5.38 vs −0.54 for RBBB vs PVC). Section 3.5, the Discussion, and the response letter add this case and keep SHAP off the remaining datasets.
* **Update**: Figure 1's extract box now uses the Raw green (`#4CAF50`) with a white segment background and P-wave means −0.13 and 0.14, matching the append and output icons.
* **Update**: Rebuilt Table 3 from the rerun CSV. Marked REMC feature cell is 12.8±3.1. Figure 2 was redrawn from saved features, which do not include the E122 refit.
* **Update**: Restored the hardware line to MacBook Pro M2 (12 cores, 32 GB RAM). Shortened Reviewer #6's Wilcoxon and REMC runtime replies so they cite only the manuscript.
* **Update**: Added Section 3.6 on the two-sided exact Wilcoxon test, the five-fold p-value floor, and per-unit pairing for REMC, AZT1D, and PAMAP2. Reviewer #6's reply points there.
* **Update**: Rewrote the Reviewer #5 Table 1 reply so it describes the Explanation type column and the weakened traceable-feature-descriptions claim.
* **Update**: Replaced Table 1's High/Medium/Low ratings with an Explanation type column in the clean and marked manuscripts.
* **Update**: Stopped the InceptionTime rerun. Added LSTM to Table 3 and Section 2.6; the response now declines HIVE-COTE 2.0 and InceptionTime on compute cost, with six baselines treated as sufficient.
* **Update**: Dropped the 15 InceptionTime rows and restarted that baseline on every dataset with batch 512 and length-140 pooling. LSTM rows kept.
* **Update**: Sped up InceptionTime (shuffle once per epoch, batch 512). Dropped the partial MIT-BIH fold so all five folds use the same loop.
* **Update**: Resumed overnight extra baselines after the run died at MIT-BIH InceptionTime fold 1 (347 rows already in `results/extra_baselines.csv`). LSTM is done; InceptionTime continues, then HIVE-COTE 2.0.
## 2026-10-14
* **Update**: Started overnight extra baselines (`scripts/extra_baselines.py`): LSTM, then InceptionTime, then HIVE-COTE 2.0, on all seven benchmarks. Results append to `results/extra_baselines.csv`. Emotions LSTM and InceptionTime are reused.
* **Update**: Removed the GroupShuffleSplit option from SublimeX. PAMAP2 subject-held-out search stays in the supplementary only, because it did not beat the default 50/50 split (0.840 vs 0.860).
* **Update**: Folded Experimental Setup back into Methods (2.5--2.7). Reviewer #16 is answered by keeping datasets, baselines, and evaluation there; section numbers in the R2 letter follow.
* **Update**: Merged classification and regression back into Results 4.1; AZT1D (RMSE) is the last table row and its own paragraph. Reviewer #16 reply and later section numbers updated.
* **Update**: Marked manuscript uses the same 4.2 cm margins as the clean PDF (51 pages with the tracked text).
* **Update**: Removed the algorithm float; Section 2.3 states the loop next to Figure 1. Margins set to 4.2 cm so the clean PDF is 34 pages (2.5 cm margins had widened the line and cut it to 29).
* **Update**: Split `elsarticle/` into `clean/` (current manuscript), `marked/` (latexdiff against the first submission), and `first_submission/` (round-1 manuscript; original bundle in `repository/`).
* **Update**: Added `elsarticle/clean/highlights.txt` (5 Elsevier highlights, each under 85 characters).
* **Update**: Added Reviewer #6's 8-point report (`PR-D-26-04476.docx`) to `response_to_reviewers_r2.tex`, added Supplementary Table S1, fixed PAMAP2 baseline significance in `generate_tables.py`, standardized `MIT-BIH`/`MIMIC-IV` labels across figures/text, added keywords and split citation clusters in `main.tex`, and cleaned up scripts/logs.
* **Update**: Redrew Figure 4 at text width with 8 pt labels; manuscript builds to 28 pages, so both red markers are filled in the R2 letter.
## 2026-10-13
* **Update**: Filled PAMAP2 subject-held-out results (val 0.917, test 0.840) and inference medians into main.tex, the supplementary, and the R2 letter.
## 2026-10-12
* **Update**: Bibliography refreshed (7 verified 2023–2025 refs incl. SAST/Pattern Recognition, arXiv entries replaced, 5 old refs dropped; 55 cited); margins 4.8 cm → 2.5 cm to get under the 35-page limit (was 39 pages).
* **Update**: R2 response letter expanded to full reviewer texts with detailed answers; PAMAP2/inference/refs numbers left as red \todo markers; ablation conclusion softened in main.tex.
* **Update**: Started PR-D-26-04476R1 revision: retitled to multivariate biomedical sequences; added GroupShuffleSplit to SublimeX.fit; fixed Wilcoxon/REMC stats, Emotions 2-channel table, tsfresh=10; restructured Methods vs Experimental Setup; drafted R2 response letter. PAMAP2 grouped search left for user to run locally.
