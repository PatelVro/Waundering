"""Cycle 9 — toss-conditional venue features (24-month windowed).

A/B test: does adding venue_*_24mo features (recent venue dynamics) lift
accuracy on T20 / ODI on top of the existing all-time venue features?

Run:
    python -m cricket_pipeline.work.step5_venue_window_experiment --fmt T20,IT20 --tag t20
    python -m cricket_pipeline.work.step5_venue_window_experiment --fmt ODI       --tag odi
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler

from .ensemble import (DEFAULT_RECENCY_HL_DAYS, _lgb_pred, _lr_pred, _xgb_pred,
                        recency_weights)
from .eval import RUNS_DIR, ece_bins, lgb_params, time_split
from .features_v2 import (CATEGORICAL, NUMERIC, PLAYER_NUMERIC,
                           build_features_with_players)


WINDOW_FEATS = [
    "venue_n_prior_24mo",
    "venue_avg_first_24mo",
    "venue_bat1_winrate_24mo",
    "venue_toss_winner_winpct_24mo",
    "venue_bat_first_pct_24mo",
]


def _evaluate(name, y, p):
    return {
        "config":  name,
        "n":       int(len(y)),
        "acc":     float(accuracy_score(y, (p >= 0.5).astype(int))),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "brier":   float(brier_score_loss(y, p)),
        "auc":     float(roc_auc_score(y, p)),
        "ece":     float(ece_bins(y, p)["ece"]),
    }


def run(formats, tag):
    print(f"Loading features (formats={formats}) …")
    df = build_features_with_players(format_filter=(formats if formats != ["all"] else None))
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()
    yte = test["y_t1_wins"].astype(int).to_numpy()
    w   = recency_weights(train["start_date"], DEFAULT_RECENCY_HL_DAYS)
    print(f"  rows  train={len(train):,}  calib={len(calib):,}  test={len(test):,}")

    # confirm new feats are present
    missing = [c for c in WINDOW_FEATS if c not in train.columns]
    if missing:
        print(f"  ERROR: missing window cols: {missing}"); return
    n_nonnull = train[WINDOW_FEATS].notna().sum().to_dict()
    print(f"  window-feat non-null counts (train): {n_nonnull}")

    rows = []
    for label, extra in (("control (all-time only)", []),
                          ("with 24mo windows",       WINDOW_FEATS)):
        feat_num = NUMERIC + PLAYER_NUMERIC + extra
        feat_cat = CATEGORICAL
        print(f"\n=== {label}  ({len(feat_num)} numeric feats) ===")

        t0 = time.time()
        lgb_n_ca, lgb_n_te = _lgb_pred(train, calib, test, feat_num, [], ytr, yca, weights=w)
        lgb_c_ca, lgb_c_te = _lgb_pred(train, calib, test, feat_num + feat_cat, feat_cat, ytr, yca, weights=w)
        xgb_ca,    xgb_te  = _xgb_pred(train, calib, test, feat_num, ytr, yca, weights=w)
        lr_ca,     lr_te   = _lr_pred(train, calib, test, feat_num, ytr, yca, weights=w)
        Xca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, lr_ca])
        Xte = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, lr_te])
        stk = LogisticRegression(C=10.0).fit(Xca, yca)
        p_te = stk.predict_proba(Xte)[:, 1]
        print(f"  trained {time.time()-t0:.0f}s")

        m = _evaluate(label, yte, p_te)
        rows.append(m)
        print(f"  → acc={m['acc']*100:.2f}%  brier={m['brier']:.4f}  "
              f"ece={m['ece']*100:.2f}%  auc={m['auc']:.3f}")

    res = pd.DataFrame(rows)
    if rows:
        ctrl, exp = rows[0], rows[1]
        for r in rows:
            r["d_acc_pp"] = (r["acc"]   - ctrl["acc"])   * 100
            r["d_brier"]  =  r["brier"] - ctrl["brier"]
            r["d_ece_pp"] = (r["ece"]   - ctrl["ece"])   * 100
    res = pd.DataFrame(rows)

    out = RUNS_DIR / f"{tag}_venue_window.csv"
    res.to_csv(out, index=False)
    print("\n=== Comparison ===")
    print(res[["config","acc","d_acc_pp","brier","d_brier","ece","d_ece_pp","auc"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="T20,IT20")
    ap.add_argument("--tag", default="t20")
    args = ap.parse_args()
    fmts = [s.strip() for s in args.fmt.split(",") if s.strip()]
    run(fmts, args.tag)
