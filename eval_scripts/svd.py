import os
import json
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from patx import feature_extraction, LightGBMModelWrapper
from tsfresh_utils import run_tsfresh
from catch22_utils import run_catch22
from cnn import run_cnn
from params import *
warnings.filterwarnings("ignore")

CNN_ONLY = False
N_MFCC = 13
N_FRAMES = 100
VOWELS = ["a_n", "i_n", "u_n"]
VOICE_FEATS = ["f0_mean", "f0_std", "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3", "hnr"]


def load_svd():
    df = pd.read_parquet("../processed_datasets/svd/svd.parquet")
    # For PATX: Treat each MFCC coefficient of each vowel as a separate channel (39 channels total)
    mfcc_channels = []
    for v in VOWELS:
        # Load each MFCC coefficient as a channel
        for i in range(N_MFCC):
            # Select every N_MFCC-th column starting from i
            # Actually, preprocess stores them as v_mfcc0, v_mfcc1... where 0..99 are frames for coef 0?
            # Let's check preprocess_svd.py:
            # mfcc_resampled shape: (N_MFCC, N_FRAMES)
            # Flattened: [mfcc0_t0, mfcc0_t1... mfcc0_t99, mfcc1_t0...]
            # So v_mfcc0..v_mfcc99 corresponds to coef 0 over time
            # v_mfcc100..v_mfcc199 corresponds to coef 1 over time
            cols = [f"{v}_mfcc{i * N_FRAMES + t}" for t in range(N_FRAMES)]
            mfcc_channels.append(df[cols].astype(np.float32))
            
    mfcc_cols = [c for c in df.columns if "_mfcc" in c]
    X_mfcc = df[mfcc_cols].astype(np.float32)
    voice_cols = [f"{v}_{vf}" for v in VOWELS for vf in VOICE_FEATS]
    X_voice = df[voice_cols].astype(np.float32)
    X_flat = pd.concat([X_mfcc, X_voice], axis=1)
    return mfcc_channels, X_mfcc, X_voice, X_flat, df["target"].astype(int)


mfcc_channels, X_mfcc, X_voice, X_flat, y = load_svd()
splits = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(X_flat, y))
results = []
all_patterns = {}

print(f"Running SVD with {3 * N_MFCC} MFCC channels (39) x {N_FRAMES} frames + {len(VOWELS)*len(VOICE_FEATS)} voice features")
for fold, (train_idx, test_idx) in enumerate(splits, 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{K_FOLDS}")
    print(f"{'='*60}")
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train_flat = X_flat.iloc[train_idx]
    X_test_flat = X_flat.iloc[test_idx]
    X_train_mfcc = X_mfcc.iloc[train_idx]
    X_test_mfcc = X_mfcc.iloc[test_idx]
    voice_train = X_voice.iloc[train_idx].values
    voice_test = X_voice.iloc[test_idx].values
    # Use 39 channels for PATX
    mfcc_train = [ch.iloc[train_idx] for ch in mfcc_channels]
    mfcc_test = [ch.iloc[test_idx] for ch in mfcc_channels]

    if not CNN_ONLY:
        t0 = time.time()
        res = feature_extraction(
            mfcc_train, y_train.values, mfcc_test, metric="auc",
            n_trials=N_TRIALS, n_control_points=N_CONTROL_POINTS, n_patterns=N_PATTERNS,
            n_transforms=N_TRANSFORMS, max_samples=MAX_SAMPLES, inner_k_folds=INNER_K_FOLDS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE, val_size=VAL_SIZE,
            show_progress=SHOW_PROGRESS, n_workers=N_WORKERS,
            initial_features=(voice_train, voice_test)
        )
        preds_proba = res["model"].predict_proba(res["test_features"])
        preds = preds_proba[:, 1] if preds_proba.ndim > 1 else preds_proba
        auc = roc_auc_score(y_test, preds)
        elapsed = time.time() - t0
        print(f"PATX: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
        results.append({"approach": "PATX", "fold": fold, "score": auc, "processing_time": elapsed, "n_features": len(res["patterns"])})
        all_patterns[f"fold_{fold}"] = res["patterns"]

        t0 = time.time()
        test_feat, train_feat = run_tsfresh(X_train_flat.values, X_test_flat.values)
        tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper("classification", n_classes=2)
        model.fit(tr_f, y_tr, val_f, y_val)
        preds_proba = model.predict_proba(test_feat)
        preds = preds_proba[:, 1] if preds_proba.ndim > 1 else preds_proba
        auc = roc_auc_score(y_test, preds)
        elapsed = time.time() - t0
        print(f"TSFRESH: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({"approach": "TSFRESH", "fold": fold, "score": auc, "processing_time": elapsed, "n_features": train_feat.shape[1]})

        t0 = time.time()
        test_feat, train_feat = run_catch22(X_train_flat.values, X_test_flat.values)
        tr_f, val_f, y_tr, y_val = train_test_split(train_feat, y_train.values, test_size=VAL_SIZE, random_state=42, stratify=y_train.values)
        model = LightGBMModelWrapper("classification", n_classes=2)
        model.fit(tr_f, y_tr, val_f, y_val)
        preds_proba = model.predict_proba(test_feat)
        preds = preds_proba[:, 1] if preds_proba.ndim > 1 else preds_proba
        auc = roc_auc_score(y_test, preds)
        elapsed = time.time() - t0
        print(f"CATCH22: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={train_feat.shape[1]}")
        results.append({"approach": "CATCH22", "fold": fold, "score": auc, "processing_time": elapsed, "n_features": train_feat.shape[1]})

    t0 = time.time()
    # Use only MFCCs for CNN, structured as (N, 3900) which CNN reshapes to (N, 39, 100)
    preds = run_cnn(X_train_mfcc.values, y_train.values, X_test_mfcc.values, task_type="classification", metric="auc", num_classes=2, epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE)
    auc = roc_auc_score(y_test, preds)
    elapsed = time.time() - t0
    print(f"CNN: AUC={auc:.4f}, Time={elapsed:.1f}s, Features={X_train_mfcc.shape[1]}")
    results.append({"approach": "CNN", "fold": fold, "score": auc, "processing_time": elapsed, "n_features": X_train_mfcc.shape[1]})

df_res = pd.DataFrame(results)
print("\n" + "=" * 60)
print("SUMMARY RESULTS (Mean ± Std)")
print("=" * 60)
for app in ["PATX", "TSFRESH", "CATCH22", "CNN"]:
    app_res = df_res[df_res["approach"] == app]
    if len(app_res) == 0:
        continue
    scores = app_res["score"].values
    times = app_res["processing_time"].values
    feats = app_res["n_features"].values
    print(f"{app:8}: AUC={np.mean(scores):.4f}±{np.std(scores):.4f}, Time={np.mean(times):.1f}±{np.std(times):.1f}s, Features={np.mean(feats):.1f}±{np.std(feats):.1f}")

df_res.to_csv("../results/svd.csv", index=False)

if not CNN_ONLY:
    os.makedirs("../json_files/svd", exist_ok=True)
    serializable = {}
    for fold_key, patterns in all_patterns.items():
        serializable_patterns = []
        for pattern in patterns:
            sp = {}
            for k, v in pattern.items():
                if k == "pattern":
                    continue
                if isinstance(v, np.ndarray):
                    sp[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    sp[k] = v.item()
                else:
                    sp[k] = v
            serializable_patterns.append(sp)
        serializable[fold_key] = serializable_patterns
    with open("../json_files/svd/pattern_parameters.json", "w") as f:
        json.dump(serializable, f, indent=2)
