"""Small focused hyperparameter sweep for the match-winner model.

Uses the same train/calib/test split as eval.py. Sweeps a curated grid
to avoid burning days on Optuna. Reports the best config by test_acc_cal,
test_brier_cal, and test_auc.

Usage:
  python -m cricket_pipeline.work.sweep --formats T20,IT20 --tag t20_sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .eval import RUNS_DIR, train_eval
from .features_v2 import CATEGORICAL, NUMERIC


GRID = {
    "num_leaves":       [15, 31, 63],
    "min_data_in_leaf": [20, 50, 100],
    "learning_rate":    [0.03, 0.05],
    "feature_fraction": [0.7, 0.9],
    "lambda_l2":        [0.0, 1.0, 5.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default=None)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--cat", action="store_true",
                    help="Include categorical features (default: numeric only)")
    ap.add_argument("--max-runs", type=int, default=200)
    args = ap.parse_args()

    fmts = [f.strip() for f in args.formats.split(",")] if args.formats else None
    feat_cols = NUMERIC + CATEGORICAL if args.cat else list(NUMERIC)
    cats = CATEGORICAL if args.cat else []

    keys, values = zip(*GRID.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    if len(combos) > args.max_runs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(combos), args.max_runs, replace=False)
        combos = [combos[i] for i in sorted(idx)]
    print(f"Sweeping {len(combos)} configs.")

    rows = []
    t0 = time.time()
    for i, params in enumerate(combos):
        tag = f"{args.tag}_{i:03d}"
        try:
            m = train_eval(formats=fmts, feature_cols=feat_cols, cat_cols=cats,
                           model_tag=tag, extra_params=params,
                           num_boost_round=2000, early_stopping=80)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            continue
        rows.append({
            **params,
            "tag":             tag,
            "acc_raw":         m["test_acc_raw"],
            "acc_cal":         m["test_acc_cal"],
            "logloss_cal":     m["test_logloss_cal"],
            "brier_cal":       m["test_brier_cal"],
            "auc":             m["test_auc"],
            "ece_cal":         m["test_ece_cal"],
            "best_iter":       m["best_iter"],
            "high_conf_acc":   m.get("test_acc_high_conf"),
            "high_conf_share": m.get("test_high_conf_share"),
        })
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(combos) - i - 1)
        print(f"  [{i+1}/{len(combos)}] acc_cal={m['test_acc_cal']:.4f} "
              f"auc={m['test_auc']:.4f} brier={m['test_brier_cal']:.4f}  "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    df = pd.DataFrame(rows)
    csv_path = RUNS_DIR / f"{args.tag}_grid.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- TOP 10 by acc_cal ---")
    print(df.sort_values("acc_cal", ascending=False).head(10).to_string(index=False))
    print("\n--- TOP 10 by brier_cal (lower better) ---")
    print(df.sort_values("brier_cal", ascending=True).head(10).to_string(index=False))
    print("\n--- TOP 10 by auc ---")
    print(df.sort_values("auc", ascending=False).head(10).to_string(index=False))
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
