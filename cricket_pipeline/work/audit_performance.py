"""Step 1: Performance Audit.

Reads the held-out test predictions saved by the final-ensemble run and
computes:
  - LogLoss, Brier, AUC, accuracy, ECE
  - 10-bin reliability curve (predicted bin -> observed win rate)
  - Accuracy by confidence bucket (|p - 0.5| * 2)
  - Weakest segments by competition tier and year

Run from project root:
    .venv/Scripts/python -m cricket_pipeline.work.audit_performance

Designed to be paste-back friendly — output is plain text tables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

RUNS = Path(__file__).resolve().parent / "runs"
SOURCES = {
    "T20": RUNS / "final_t20_test_preds.csv",
    "ODI": RUNS / "final_odi_test_preds.csv",
    "ALL": RUNS / "final_all_test_preds.csv",
}

CONF_BUCKETS = [(0.00, 0.10), (0.10, 0.20), (0.20, 0.30),
                (0.30, 0.50), (0.50, 1.00)]   # |p-0.5|*2 ranges
CAL_BINS = 10


def headline(df: pd.DataFrame) -> dict:
    y = df["y_t1_wins"].astype(int).to_numpy()
    p = df["pred_p_t1"].astype(float).to_numpy()
    pred = (p >= 0.5).astype(int)
    out = {
        "n":       len(df),
        "acc":     accuracy_score(y, pred),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "brier":   brier_score_loss(y, p),
        "auc":     roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
    }
    out["ece"] = expected_calibration_error(y, p, bins=CAL_BINS)
    return out


def expected_calibration_error(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any(): continue
        ece += abs(y[mask].mean() - p[mask].mean()) * mask.sum() / len(y)
    return float(ece)


def reliability_table(df: pd.DataFrame, bins=CAL_BINS) -> pd.DataFrame:
    y = df["y_t1_wins"].astype(int).to_numpy()
    p = df["pred_p_t1"].astype(float).to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append({
            "bin":        f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n":          n,
            "mean_pred":  float(p[mask].mean()) if n else float("nan"),
            "obs_freq":   float(y[mask].mean()) if n else float("nan"),
            "gap_pp":     (float(y[mask].mean() - p[mask].mean()) * 100) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def confidence_bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    p = df["pred_p_t1"].astype(float).to_numpy()
    y = df["y_t1_wins"].astype(int).to_numpy()
    pred = (p >= 0.5).astype(int)
    conf = np.abs(p - 0.5) * 2
    correct = (pred == y).astype(int)
    rows = []
    for lo, hi in CONF_BUCKETS:
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        n = int(mask.sum())
        rows.append({
            "conf_band":  f"{lo:.2f}-{hi:.2f}",
            "n":          n,
            "share":      n / len(df) if len(df) else 0,
            "acc":        float(correct[mask].mean()) if n else float("nan"),
            "mean_p_fav": float(np.maximum(p, 1 - p)[mask].mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def segment_table(df: pd.DataFrame, key: str, min_n: int = 30) -> pd.DataFrame:
    rows = []
    for v, sub in df.groupby(key):
        if len(sub) < min_n: continue
        y = sub["y_t1_wins"].astype(int).to_numpy()
        p = sub["pred_p_t1"].astype(float).to_numpy()
        rows.append({
            key:        v,
            "n":        len(sub),
            "acc":      accuracy_score(y, (p >= 0.5).astype(int)),
            "brier":    brier_score_loss(y, p),
            "ece":      expected_calibration_error(y, p, bins=5),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("brier", ascending=False)
    return out


def fmt_dict(d: dict) -> str:
    return " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in d.items()
    )


def main():
    print("=" * 78)
    print("PERFORMANCE AUDIT  (held-out test set, current production ensemble)")
    print("=" * 78)
    for label, path in SOURCES.items():
        if not path.exists():
            print(f"\n[{label}]  MISSING: {path}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"\n[{label}]  EMPTY")
            continue
        print(f"\n[{label}]  {path.name}  ({len(df):,} rows)")
        print(f"  headline:        {fmt_dict(headline(df))}")

        print("\n  reliability (10 bins):")
        rel = reliability_table(df).to_string(index=False, float_format="%.4f")
        print("    " + rel.replace("\n", "\n    "))

        print("\n  confidence buckets:  (conf = |p-0.5|*2)")
        cb = confidence_bucket_table(df).to_string(index=False, float_format="%.4f")
        print("    " + cb.replace("\n", "\n    "))

    # Segment analysis on the combined set
    df_all = pd.read_csv(SOURCES["ALL"])
    for key in ("competition", "tier", "year"):
        if key not in df_all.columns: continue
        print(f"\nWeakest segments by {key} (sorted by Brier desc, n>=30):")
        seg = segment_table(df_all, key).head(8)
        if seg.empty:
            print("  (no segments meet n>=30)")
        else:
            print("  " + seg.to_string(index=False, float_format="%.4f").replace("\n", "\n  "))

    print("\n" + "=" * 78)
    print("Paste the above output back into chat for analysis.")
    print("=" * 78)


if __name__ == "__main__":
    main()
