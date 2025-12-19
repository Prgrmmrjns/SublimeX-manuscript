import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_datasets"
OUT = ROOT / "manuscript" / "images" / "supplementary"


def _cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda s: int(s.rsplit("_", 1)[-1]) if s.rsplit("_", 1)[-1].isdigit() else s)


def load(dataset):
    if dataset == "mitbih":
        p = PROC / "mitbih" / "mitbih_processed.csv"
        if not p.exists():
            return
        df = pd.read_csv(p)
        cols = sorted([c for c in df.columns if c.isdigit()], key=int)[:100]
        y = df.iloc[0][cols].to_numpy() if cols else df.iloc[0].to_numpy()[:100]
        x = np.arange(len(y)) - len(y) // 2
        return {"ECG": y}, x, "Sample (relative to R-peak)", "Amplitude"

    if dataset == "emotions":
        p = PROC / "emotions" / "emotions.csv"
        if not p.exists():
            return
        df = pd.read_csv(p)
        chans = ["AF3", "F7", "F3", "FC5", "T7"]
        d = {ch: df.iloc[0][_cols(df, f"{ch}_")][:254].to_numpy() for ch in chans if _cols(df, f"{ch}_")}
        if not d:
            y = df.iloc[0].to_numpy()[:254]
            d = {"EEG": y}
        x = np.arange(len(next(iter(d.values()))))
        return d, x, "Frequency bin", "Spectral power"

    if dataset == "remc":
        p = PROC / "remc" / "E003.parquet"
        if not p.exists():
            return
        df = pd.read_parquet(p)
        s = df[df["target"] == 1].iloc[0]
        marks = ["H3K4me3", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]
        d = {}
        for m in marks:
            cols = _cols(df, f"{m}_")
            if cols:
                d[m] = s[cols].to_numpy()
        y0 = next(iter(d.values()))
        x = np.linspace(-100, 100, len(y0))
        return d, x, "Distance from TSS (bins)", "ChIP-seq signal"

    if dataset == "mimic":
        p = PROC / "mimic" / "mimic_processed.csv"
        if not p.exists():
            return
        df = pd.read_csv(p)
        names = [
            "resp_rate",
            "heart_rate",
            "o2_sat",
            "bp_sys",
            "temp",
            "fio2",
            "creatinine",
            "platelets",
            "lactate",
            "pao2_fio2",
        ]
        d = {}
        for n in names:
            cols = _cols(df, f"{n}_") or _cols(df, f"{n}_hour_")
            if cols:
                d[n] = df.iloc[0][cols].to_numpy()[:24]
        if not d:
            y = df.select_dtypes(include=[np.number]).iloc[0].to_numpy()[:24]
            d = {"signal": y}
        x = np.arange(len(next(iter(d.values()))))
        return d, x, "Time (hours)", "Value"

    if dataset == "pamap2":
        p = PROC / "pamap2" / "pamap2.parquet"
        if not p.exists():
            return
        df = pd.read_parquet(p)
        s = df.iloc[0]
        keys = ["heart_rate", "hand_acc16_x", "hand_acc16_y", "hand_acc16_z"]
        d = {}
        for k in keys:
            cols = _cols(df, f"{k}_")
            if cols:
                d[k] = s[cols].to_numpy()[:100]
        if not d:
            y = df.select_dtypes(include=[np.number]).iloc[0].to_numpy()[:100]
            d = {"signal": y}
        x = np.arange(len(next(iter(d.values()))))
        return d, x, "Time", "Value"

    if dataset == "svd":
        p = PROC / "svd" / "svd.parquet"
        if not p.exists():
            return
        df = pd.read_parquet(p)
        s = df[df["target"] == 0].iloc[0]
        d = {}
        for ch in ["a_n", "a_h", "a_l"]:
            cols = _cols(df, f"{ch}_")
            if cols:
                d[ch] = s[cols].to_numpy()
        x = np.arange(len(next(iter(d.values()))))
        return d, x, "Sample", "Amplitude"


def plot(dataset):
    out = load(dataset)
    if not out:
        print(f"Skip {dataset}: missing processed file")
        return
    d, x, xl, yl = out
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(d))))
    for i, (k, y) in enumerate(d.items()):
        ax.plot(x, y, lw=1.8 if dataset != "svd" else 0.9, alpha=0.9, color=colors[i], label=k)
    if dataset == "remc":
        ax.axvline(0, color="black", ls="--", lw=1.4, alpha=0.6)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if len(d) > 1:
        ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{dataset}_sample.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {dataset}_sample.png")


def main():
    for ds in ["mitbih", "emotions", "remc", "mimic", "pamap2", "svd"]:
        plot(ds)


if __name__ == "__main__":
    main()
