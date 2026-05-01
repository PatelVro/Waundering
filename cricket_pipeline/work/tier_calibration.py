"""Per-tier isotonic calibration refit.

Step-1 error analysis found Tier-1 (IPL/T20WC/franchise/Test-bilaterals)
calibration is OFF in the 60-80% confidence band:
   model says 85%   actually right 94%  → under-confident
   model says 92%   actually right 98%  → under-confident

A single isotonic regression fit on the global calib slice can't fix this
because tier-2 (associate/qualifier) and tier-2-main (women's main) have
different miscalibration shapes that pull the global fit.

This module fits SEPARATE isotonic stages per tier on the calib slice,
then applies the matching stage at predict time. ECE should drop
materially on tier-1 without touching accuracy.

Run:
    python -m cricket_pipeline.work.tier_calibration --fmt T20,IT20 --tag t20
    python -m cricket_pipeline.work.tier_calibration --fmt ODI       --tag odi
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                              roc_auc_score)

from .ensemble import (DEFAULT_RECENCY_HL_DAYS, _cat_pred, _lgb_pred, _lr_pred,
                        _xgb_pred, recency_weights)
from .error_analysis_v2 import classify_tier
from .eval import RUNS_DIR, ece_bins, time_split
from .features_v2 import (CATEGORICAL, NUMERIC, PLAYER_NUMERIC,
                           build_features_with_players)


CAL_DIR = RUNS_DIR / "calibrators"
CAL_DIR.mkdir(parents=True, exist_ok=True)


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


def fit_per_tier_isotonic(p_calib: np.ndarray, y_calib: np.ndarray,
                           tier_calib: np.ndarray) -> dict:
    """Fit one IsotonicRegression per tier present in calib. Falls back to
    global isotonic for tiers with too few samples (<30)."""
    out = {"_global": IsotonicRegression(out_of_bounds="clip").fit(p_calib, y_calib)}
    for tier in pd.Series(tier_calib).unique():
        mask = (tier_calib == tier)
        if mask.sum() < 30:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_calib[mask], y_calib[mask])
        out[tier] = iso
    return out


def apply_per_tier_isotonic(calibrators: dict, p: np.ndarray,
                              tier: np.ndarray) -> np.ndarray:
    out = np.empty_like(p, dtype=float)
    for i, t in enumerate(tier):
        iso = calibrators.get(t) or calibrators["_global"]
        out[i] = iso.transform([p[i]])[0]
    return out


def run(formats, tag):
    print(f"Loading features (formats={formats}) …")
    df = build_features_with_players(format_filter=(formats if formats != ["all"] else None))
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    feat_num = NUMERIC + PLAYER_NUMERIC
    feat_cat = CATEGORICAL
    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()
    yte = test["y_t1_wins"].astype(int).to_numpy()
    w   = recency_weights(train["start_date"], DEFAULT_RECENCY_HL_DAYS)
    print(f"  rows  train={len(train):,}  calib={len(calib):,}  test={len(test):,}")

    # Train production ensemble (matches predict_match.py shape — no fast path)
    print("\nTraining ensemble (LGBM-num + LGBM-cat + XGB + CatBoost + LR + LR meta) …")
    t0 = time.time()
    lgb_n_ca, lgb_n_te = _lgb_pred(train, calib, test, feat_num, [], ytr, yca, weights=w)
    lgb_c_ca, lgb_c_te = _lgb_pred(train, calib, test, feat_num + feat_cat, feat_cat, ytr, yca, weights=w)
    xgb_ca,    xgb_te  = _xgb_pred(train, calib, test, feat_num, ytr, yca, weights=w)
    cat_ca,    cat_te  = _cat_pred(train, calib, test, feat_num, feat_cat, ytr, yca, weights=w)
    lr_ca,     lr_te   = _lr_pred(train, calib, test, feat_num, ytr, yca, weights=w)
    Xca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, cat_ca, lr_ca])
    Xte = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, cat_te, lr_te])
    stk = LogisticRegression(C=10.0).fit(Xca, yca)
    p_ca = stk.predict_proba(Xca)[:, 1]
    p_te = stk.predict_proba(Xte)[:, 1]
    print(f"  trained {time.time()-t0:.0f}s")

    # Compute tier per row
    calib_tier = calib.apply(lambda r: classify_tier(r.get("competition"),
                                                      r.get("team_home"),
                                                      r.get("team_away")), axis=1).to_numpy()
    test_tier  = test.apply(lambda r: classify_tier(r.get("competition"),
                                                      r.get("team_home"),
                                                      r.get("team_away")), axis=1).to_numpy()
    print(f"  calib tier distribution: {pd.Series(calib_tier).value_counts().to_dict()}")
    print(f"  test  tier distribution: {pd.Series(test_tier).value_counts().to_dict()}")

    # Three calibration variants:
    # (a) raw (uncalibrated)
    # (b) global isotonic (current production)
    # (c) per-tier isotonic
    iso_global = IsotonicRegression(out_of_bounds="clip").fit(p_ca, yca)
    p_te_iso_g = iso_global.transform(p_te)

    iso_per_tier = fit_per_tier_isotonic(p_ca, yca, calib_tier)
    p_te_iso_t   = apply_per_tier_isotonic(iso_per_tier, p_te, test_tier)

    # Per-tier metrics for each variant
    rows = []
    for variant_name, p_v in (("raw",          p_te),
                                ("global_iso",   p_te_iso_g),
                                ("per_tier_iso", p_te_iso_t)):
        rows.append({"variant": variant_name, "tier": "ALL",
                     **_evaluate(variant_name, yte, p_v)})
        for tier in pd.Series(test_tier).unique():
            mask = (test_tier == tier)
            if mask.sum() < 20: continue
            rows.append({"variant": variant_name, "tier": tier,
                         **_evaluate(variant_name + "@" + tier, yte[mask], p_v[mask])})
    res = pd.DataFrame(rows)

    # Display ECE comparison table by (variant, tier)
    pivot = res.pivot_table(index="tier", columns="variant", values="ece").round(4)
    pivot["Δ vs global"] = (pivot["per_tier_iso"] - pivot["global_iso"]).round(4)
    print("\n=== ECE by (tier × variant) ===")
    print(pivot.to_string())

    # Accuracy should be unchanged (calibration doesn't move 0.5 threshold much)
    acc_pivot = res.pivot_table(index="tier", columns="variant", values="acc").round(4)
    print("\n=== Accuracy by (tier × variant) ===")
    print(acc_pivot.to_string())

    # Persist the per-tier calibrators for production use
    cal_path = CAL_DIR / f"{tag}_per_tier_iso.joblib"
    joblib.dump({"calibrators": iso_per_tier,
                  "global":      iso_global,
                  "format":      formats,
                  "tag":         tag}, cal_path)
    print(f"\nSaved → {cal_path}")

    # CSV for the run log
    out_csv = RUNS_DIR / f"{tag}_per_tier_calibration.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="T20,IT20")
    ap.add_argument("--tag", default="t20")
    args = ap.parse_args()
    fmts = [s.strip() for s in args.fmt.split(",") if s.strip()]
    run(fmts, args.tag)
