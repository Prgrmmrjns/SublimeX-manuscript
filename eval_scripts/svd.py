import os
import json
import time
import warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patx"))
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from patx_runner import *
from tsfresh_utils import eval_tsfresh
from catch22_utils import eval_catch22
from cnn import eval_cnn
warnings.filterwarnings("ignore")

CHANNELS = [f"{v}_{p}" for v in ["a", "i", "u"] for p in ["n", "h", "l"]]


def load_svd():
    path = Path("../processed_datasets/svd") / "svd.parquet"
    df = pd.read_parquet(path)

    def _to_float32(frame):
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        return numeric.fillna(0).astype(np.float32)

    audio_channels, all_cols = [], []
    for ch in CHANNELS:
        cols = sorted([c for c in df.columns if c.startswith(f"{ch}_")], key=lambda x: int(x.split('_')[-1]))
        audio_channels.append(_to_float32(df[cols]))
        all_cols.extend(cols)
    X_audio = _to_float32(df[all_cols])
    return audio_channels, X_audio, df["target"].astype(int)


results = []
all_patterns = {}

audio_channels, X_audio, y = load_svd()
splits = list(StratifiedKFold(K_FOLDS, shuffle=True, random_state=42).split(X_audio, y))
for fold, (train_idx, test_idx) in enumerate(splits, 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{K_FOLDS}")
    print(f"{'='*60}")
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train_audio = X_audio.iloc[train_idx]
    X_test_audio = X_audio.iloc[test_idx]
    audio_train = [ch.iloc[train_idx] for ch in audio_channels]
    audio_test = [ch.iloc[test_idx] for ch in audio_channels]

    t0 = time.time()
    res = run_patx(audio_train, y_train.values, audio_test, metric="accuracy")
    preds_proba = res["model"].predict_proba(res["test_features"])
    preds = (preds_proba[:, 1] if preds_proba.ndim > 1 else preds_proba) >= 0.5
    acc = accuracy_score(y_test, preds)
    elapsed = time.time() - t0
    print(f"PATX: ACC={acc:.4f}, Time={elapsed:.1f}s, Features={len(res['patterns'])}")
    results.append({"approach": "PATX", "fold": fold, "score": acc, "processing_time": elapsed, "n_features": len(res["patterns"])})
    all_patterns[f"fold_{fold}"] = res["patterns"]

    acc, elapsed, n_feat = eval_tsfresh(X_train_audio.values, X_test_audio.values, y_train.values, y_test.values, metric="accuracy", val_size=VAL_SIZE, n_classes=2)
    print(f"TSFRESH: ACC={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({"approach": "TSFRESH", "fold": fold, "score": acc, "processing_time": elapsed, "n_features": n_feat})

    acc, elapsed, n_feat = eval_catch22(X_train_audio.values, X_test_audio.values, y_train.values, y_test.values, metric="accuracy", n_classes=2)
    print(f"CATCH22: ACC={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({"approach": "CATCH22", "fold": fold, "score": acc, "processing_time": elapsed, "n_features": n_feat})

    acc, elapsed, n_feat = eval_cnn(X_train_audio.values, X_test_audio.values, y_train.values, y_test.values, task_type="classification", metric="accuracy", num_classes=2)
    print(f"CNN: ACC={acc:.4f}, Time={elapsed:.1f}s, Features={n_feat}")
    results.append({"approach": "CNN", "fold": fold, "score": acc, "processing_time": elapsed, "n_features": n_feat})

df_res = pd.DataFrame(results)
print("\n" + "=" * 60)
print("SUMMARY RESULTS (Mean ± Std, accuracy)")
print("=" * 60)
for app in ["PATX", "TSFRESH", "CATCH22", "CNN"]:
    app_res = df_res[df_res["approach"] == app]
    scores = app_res["score"].values
    times = app_res["processing_time"].values
    feats = app_res["n_features"].values
    print(f"{app:8}: ACC={np.mean(scores):.4f}±{np.std(scores):.4f}, Time={np.mean(times):.1f}±{np.std(times):.1f}s, Features={np.mean(feats):.1f}±{np.std(feats):.1f}")

df_res.to_csv("../results/svd.csv", index=False)

os.makedirs("../json_files/svd", exist_ok=True)
serializable = {}
for fold_key, patterns in all_patterns.items():
    serializable_patterns = []
    for pattern in patterns:
        sp = {}
        for k, v in pattern.items():
            if k == "pattern":
                continue
            if isinstance(v, list):
                sp[k] = [x.item() if hasattr(x, 'item') else x for x in v]
            elif isinstance(v, np.ndarray):
                sp[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                sp[k] = v.item()
            else:
                sp[k] = v
        serializable_patterns.append(sp)
    serializable[fold_key] = serializable_patterns
with open("../json_files/svd/pattern_parameters.json", "w") as f:
    json.dump(serializable, f, indent=2)
