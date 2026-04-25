"""Final evaluation: train the best model (stacked ensemble) on T20+IT20,
ODI, and all-formats. Save the test predictions, calibration table, and a
summary. Used to verify the user's main calibration target ("when model
says 70%, it should be right ~70%").
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .ensemble import (_cat_pred, _lgb_pred, _lr_pred, _xgb_pred, evaluate)
from .eval import RUNS_DIR, ece_bins, time_split
from .features_v2 import (CATEGORICAL, NUMERIC, PLAYER_NUMERIC,
                           build_features_with_players)


def calibration_table(y, p, edges=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001)):
    """A standard reliability table. Symmetrize so 0.30 == 0.70 from t2 view."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "mean_p": None, "obs_t1_winrate": None,
                         "abs_gap": None})
            continue
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "n": int(mask.sum()),
            "mean_p":         float(p[mask].mean()),
            "obs_t1_winrate": float(y[mask].mean()),
            "abs_gap":        float(abs(p[mask].mean() - y[mask].mean())),
        })
    return pd.DataFrame(rows)


def hi_conf_table(y, p):
    """Table over predicted-winner-confidence bins (folds 0.30 ↔ 0.70 together)."""
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(int)
    conf = np.abs(p - 0.5) * 2  # 0..1
    edges = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            rows.append({"conf_lo": lo, "conf_hi": hi, "n": 0,
                         "mean_p_winner": None, "acc": None})
            continue
        # mean predicted prob for the side we picked = max(p, 1-p)
        mp = float(np.maximum(p[mask], 1 - p[mask]).mean())
        rows.append({
            "conf_lo": lo, "conf_hi": hi, "n": int(mask.sum()),
            "mean_p_winner": mp,
            "acc": float(correct[mask].mean()),
        })
    return pd.DataFrame(rows)


def run_one(formats, tag):
    print(f"\n========== {tag}  formats={formats} ==========")
    df = build_features_with_players(format_filter=list(formats) if formats else None)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    feat_num = NUMERIC + PLAYER_NUMERIC
    feat_cat = CATEGORICAL
    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()
    yte = test["y_t1_wins"].astype(int).to_numpy()
    print(f"  train={len(train)}  calib={len(calib)}  test={len(test)}")
    print(f"  test dates: {test['start_date'].min()} -> {test['start_date'].max()}")

    print("  base learners ...")
    lgb_n_ca, lgb_n_te = _lgb_pred(train, calib, test, feat_num, [], ytr, yca)
    lgb_c_ca, lgb_c_te = _lgb_pred(train, calib, test, feat_num + feat_cat, feat_cat, ytr, yca)
    xgb_ca, xgb_te     = _xgb_pred(train, calib, test, feat_num, ytr, yca)
    cat_ca, cat_te     = _cat_pred(train, calib, test, feat_num, feat_cat, ytr, yca)
    lr_ca, lr_te       = _lr_pred(train, calib, test, feat_num, ytr, yca)

    Xst_ca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, cat_ca, lr_ca])
    Xst_te = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, cat_te, lr_te])

    from sklearn.linear_model import LogisticRegression
    stk = LogisticRegression(C=10.0)
    stk.fit(Xst_ca, yca)
    p_te = stk.predict_proba(Xst_te)[:, 1]

    metrics = evaluate("stack_lr", yte, p_te)
    print(f"\n  ENSEMBLE TEST METRICS:")
    for k, v in metrics.items():
        print(f"    {k:<12} {v}")

    cal_tab = calibration_table(yte, p_te)
    print(f"\n  REL TABLE (for t1 prob; symmetric):")
    print(cal_tab.to_string(index=False))

    hc_tab = hi_conf_table(yte, p_te)
    print(f"\n  CONFIDENCE TABLE (winner-side):")
    print(hc_tab.to_string(index=False))

    # high-confidence slice
    hi_mask = (np.abs(p_te - 0.5) * 2 >= 0.4)   # winner prob >= 0.7
    if hi_mask.sum():
        from sklearn.metrics import accuracy_score as acc_score
        print(f"\n  Hi-conf (winner-prob>=0.70): n={int(hi_mask.sum())}  acc={acc_score(yte[hi_mask], (p_te[hi_mask] >= 0.5).astype(int)):.4f}")

    # save
    out = {"tag": tag, "formats": list(formats) if formats else None,
           "metrics": metrics,
           "n_train": int(len(train)), "n_calib": int(len(calib)), "n_test": int(len(test)),
           "test_dates": [str(test["start_date"].min()), str(test["start_date"].max())]}
    (RUNS_DIR / f"final_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    cal_tab.to_csv(RUNS_DIR / f"final_{tag}_cal_table.csv", index=False)
    hc_tab.to_csv(RUNS_DIR / f"final_{tag}_conf_table.csv", index=False)

    # save a CSV of test predictions
    test_out = test[["match_id", "format", "competition", "start_date",
                     "team_home", "team_away", "venue", "winner",
                     "y_t1_wins", "elo_diff_pre"]].copy()
    test_out["pred_p_t1"] = p_te
    test_out["pred_t1_wins"] = (p_te >= 0.5).astype(int)
    test_out["correct"] = (test_out["pred_t1_wins"] == test_out["y_t1_wins"]).astype(int)
    test_out.to_csv(RUNS_DIR / f"final_{tag}_test_preds.csv", index=False)

    return metrics


def main():
    out = {}
    out["t20"]  = run_one(["T20", "IT20"], "t20")
    out["odi"]  = run_one(["ODI"],         "odi")
    out["all"]  = run_one(None,            "all")
    print("\n\n=== FINAL SUMMARY ===")
    for k, m in out.items():
        print(f"  {k:<6} acc={m['acc']:.4f}  logloss={m['logloss']:.4f}  brier={m['brier']:.4f}  auc={m['auc']:.4f}  ece={m['ece']:.4f}")
    Path(RUNS_DIR / "final_summary.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
