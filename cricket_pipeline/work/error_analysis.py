"""Error analysis on high-confidence misclassifications.

Train current best (no_cat + players + margin Elo), score test set, dump:
  1. confusion matrix
  2. confidence × correctness reliability table
  3. high-confidence wrongs (top 30 by |p - 0.5|)
  4. patterns: by competition, by format, by margin (close vs blowout)
  5. % of upsets (lower-Elo-team won) the model caught
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .eval import RUNS_DIR, fit_isotonic, lgb_params, time_split, ece_bins
from .features_v2 import NUMERIC, PLAYER_NUMERIC, build_features_with_players


def main(formats=("T20", "IT20"), out_tag="t20_err"):
    df = build_features_with_players(format_filter=list(formats))
    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    feat = NUMERIC + PLAYER_NUMERIC

    ds = lgb.Dataset(train[feat], label=train["y_t1_wins"].astype(int), free_raw_data=False)
    vs = lgb.Dataset(calib[feat], label=calib["y_t1_wins"].astype(int), reference=ds, free_raw_data=False)
    b  = lgb.train(lgb_params(), ds, num_boost_round=2000, valid_sets=[vs],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pt_calib = b.predict(calib[feat])
    pt_test  = b.predict(test[feat])
    iso = fit_isotonic(pt_calib, calib["y_t1_wins"].astype(int).to_numpy())
    pt_test_cal = iso.transform(pt_test)

    test = test.copy()
    test["pred_p_t1"] = pt_test_cal
    test["pred"]       = (pt_test_cal >= 0.5).astype(int)
    test["correct"]    = (test["pred"] == test["y_t1_wins"]).astype(int)
    test["confidence"] = (np.abs(pt_test_cal - 0.5) * 2)   # 0..1

    # 1. confusion matrix
    cm = pd.crosstab(test["y_t1_wins"], test["pred"], rownames=["actual"], colnames=["pred"], margins=True)

    # 2. reliability by confidence bin
    test["conf_bin"] = pd.cut(test["confidence"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
                              labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"], right=False)
    rel = test.groupby("conf_bin", observed=True).agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        avg_p=("pred_p_t1", lambda s: float(s.where(s >= 0.5, 1 - s).mean())),
    ).reset_index()

    # 3. high-confidence wrongs
    wrongs = test[test["correct"] == 0].copy()
    wrongs = wrongs.sort_values("confidence", ascending=False).head(30)
    pretty = wrongs[["match_id", "format", "competition", "start_date",
                     "team_home", "team_away", "venue", "winner",
                     "pred_p_t1", "elo_diff_pre", "h2h_t1_winpct"]]

    # 4. by competition
    by_comp = test.groupby("competition", observed=True).agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        avg_conf=("confidence", "mean"),
    ).reset_index().sort_values("n", ascending=False).head(20)

    # 5. by format
    by_fmt = test.groupby("format", observed=True).agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
    ).reset_index()

    # 6. upset detection
    test["elo_favored_t1"] = (test["elo_diff_pre"] >= 0).astype(int)
    test["upset"]          = (test["elo_favored_t1"] != test["y_t1_wins"]).astype(int)
    upset_caught_rate = float(((test["pred"] == test["y_t1_wins"]) & (test["upset"] == 1)).sum() /
                               max(test["upset"].sum(), 1))
    upset_share       = float(test["upset"].mean())

    out = {
        "n_test":             int(len(test)),
        "test_acc":           float(test["correct"].mean()),
        "logloss":            float(log_loss(test["y_t1_wins"], np.clip(pt_test_cal, 1e-7, 1-1e-7))),
        "brier":              float(brier_score_loss(test["y_t1_wins"], pt_test_cal)),
        "ece":                float(ece_bins(test["y_t1_wins"].to_numpy(), pt_test_cal)["ece"]),
        "confusion_matrix":   cm.to_string(),
        "reliability_table":  rel.to_string(index=False),
        "high_conf_wrongs":   pretty.to_string(index=False),
        "by_competition":     by_comp.to_string(index=False),
        "by_format":          by_fmt.to_string(index=False),
        "upset_share":        upset_share,
        "upset_catch_rate":   upset_caught_rate,
    }

    out_path = RUNS_DIR / f"{out_tag}_analysis.txt"
    with open(out_path, "w") as f:
        for k, v in out.items():
            f.write(f"=== {k} ===\n")
            if isinstance(v, str):
                f.write(v + "\n\n")
            else:
                f.write(f"{v}\n\n")
    print(out_path)
    print(f"\nTest acc: {out['test_acc']:.4f}  logloss: {out['logloss']:.4f}  brier: {out['brier']:.4f}  ece: {out['ece']:.4f}")
    print(f"Upsets: {upset_share:.1%} of test, model caught {upset_caught_rate:.1%}")
    print()
    print("=== Reliability ===")
    print(rel.to_string(index=False))
    print()
    print("=== By format ===")
    print(by_fmt.to_string(index=False))
    print()
    print("=== Top 10 high-conf wrongs ===")
    print(pretty.head(10).to_string(index=False))


if __name__ == "__main__":
    import sys
    fmts = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("T20", "IT20")
    tag  = sys.argv[2] if len(sys.argv) > 2 else "t20_err"
    main(formats=fmts, out_tag=tag)
