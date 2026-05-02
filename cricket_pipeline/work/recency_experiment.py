"""Recency-weighted training experiment.

Hypothesis (from Cycle-5 error analysis):
   ODI tier-1 accuracy declines year-over-year (2023=74%, 2024=66%, 2025=64%).
   Stale Elo + flat sample weights mean the model anchors on older patterns.
   Adding `sample_weight = exp(-Δ_days / half_life)` should make the model
   track recent dynamics faster and lift recent-test accuracy.

How this is structured
----------------------
1. Load the same time-based train/calib/test split as the production ensemble.
2. For each candidate half-life (in days; `0` = no weighting / control), train:
     - LGBM-numeric    (fast, dominant base learner)
     - LGBM-with-cats  (catches franchise IDs)
     - XGBoost-numeric (different bias)
   Then stack via LR meta-learner trained on calib.
3. Report per-half-life: test acc, Brier, log-loss, ECE, AUC.
4. Save a single CSV + a markdown summary for the progress log.

Run:
    python -m cricket_pipeline.work.recency_experiment --fmt T20,IT20 --tag t20
    python -m cricket_pipeline.work.recency_experiment --fmt ODI       --tag odi

Half-life sweep defaults to: none, 720, 540, 365, 270 days.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler

from .eval import RUNS_DIR, ece_bins, lgb_params, time_split
from .features_v2 import (CATEGORICAL, NUMERIC, PLAYER_NUMERIC,
                           build_features_with_players)


# ---------- recency weighting ----------

def recency_weights(dates: pd.Series, half_life_days: float | None,
                     ref_date: pd.Timestamp | None = None) -> np.ndarray:
    """Exponential decay sample weights. Most-recent training row → weight 1.0,
    one half-life back → 0.5, two → 0.25, etc.

    `half_life_days = None` (or 0/<=0) returns uniform 1.0 weights — the control.
    """
    if not half_life_days or half_life_days <= 0:
        return np.ones(len(dates), dtype=float)
    d = pd.to_datetime(dates)
    ref = ref_date if ref_date is not None else d.max()
    delta = (ref - d).dt.days.astype(float).clip(lower=0).to_numpy()
    return np.exp(-np.log(2.0) * delta / float(half_life_days))


# ---------- base learners that accept sample_weight ----------

def _lgb_pred_w(train, calib, test, feat, cats, ytr, yca, weights, seed=42):
    ds = lgb.Dataset(train[feat], label=ytr, weight=weights,
                     categorical_feature=cats, free_raw_data=False)
    vs = lgb.Dataset(calib[feat], label=yca, categorical_feature=cats,
                     reference=ds, free_raw_data=False)
    params = lgb_params() | {"seed": seed, "feature_fraction_seed": seed,
                              "bagging_seed": seed}
    b = lgb.train(params, ds, num_boost_round=2500, valid_sets=[vs],
                   callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
    return b.predict(calib[feat]), b.predict(test[feat])


def _xgb_pred_w(train, calib, test, feat, ytr, yca, weights, seed=42):
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=2500, learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, eval_metric="logloss",
        early_stopping_rounds=60, random_state=seed, tree_method="hist",
        n_jobs=-1, verbosity=0,
    )
    m.fit(train[feat], ytr, sample_weight=weights,
           eval_set=[(calib[feat], yca)], verbose=False)
    return m.predict_proba(calib[feat])[:, 1], m.predict_proba(test[feat])[:, 1]


def _lr_pred_w(train, calib, test, feat, ytr, yca, weights):
    sc = StandardScaler()
    Xtr = sc.fit_transform(train[feat].fillna(train[feat].median()))
    Xca = sc.transform(calib[feat].fillna(train[feat].median()))
    Xte = sc.transform(test[feat].fillna(train[feat].median()))
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(Xtr, ytr, sample_weight=weights)
    return lr.predict_proba(Xca)[:, 1], lr.predict_proba(Xte)[:, 1]


# ---------- experiment ----------

def evaluate(name, y, p):
    return {
        "config":  name,
        "n":       int(len(y)),
        "acc":     float(accuracy_score(y, (p >= 0.5).astype(int))),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "brier":   float(brier_score_loss(y, p)),
        "auc":     float(roc_auc_score(y, p)),
        "ece":     float(ece_bins(y, p)["ece"]),
    }


def run(formats: list[str], half_lives: list[float | None], tag: str):
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
    print(f"  rows  train={len(train):,}  calib={len(calib):,}  test={len(test):,}")
    print(f"  train date range: {train['start_date'].min().date()} → {train['start_date'].max().date()}")

    rows = []
    for hl in half_lives:
        label = "uniform (control)" if not hl else f"hl={int(hl)}d"
        print(f"\n=== {label} ===")
        w = recency_weights(train["start_date"], hl)
        # show weight distribution sanity
        print(f"  weight summary: min={w.min():.4f}  median={np.median(w):.4f}  max={w.max():.4f}")
        # effective sample size
        ess = (w.sum() ** 2) / (w ** 2).sum()
        print(f"  effective sample size: {ess:.0f} / {len(w):,}  ({ess/len(w)*100:.1f}%)")

        t0 = time.time()
        print("  training LGBM-num …")
        lgb_n_ca, lgb_n_te = _lgb_pred_w(train, calib, test, feat_num, [], ytr, yca, w)
        print("  training LGBM-cat …")
        lgb_c_ca, lgb_c_te = _lgb_pred_w(train, calib, test, feat_num + feat_cat, feat_cat, ytr, yca, w)
        print("  training XGB …")
        xgb_ca, xgb_te = _xgb_pred_w(train, calib, test, feat_num, ytr, yca, w)
        print("  training LR …")
        lr_ca,  lr_te  = _lr_pred_w(train, calib, test, feat_num, ytr, yca, w)
        print(f"  base learners trained in {time.time()-t0:.1f}s")

        # Stack via LR on calib (calib weights left uniform — calibration target)
        Xca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, lr_ca])
        Xte = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, lr_te])
        stk = LogisticRegression(C=10.0).fit(Xca, yca)
        p_te = stk.predict_proba(Xte)[:, 1]

        m = evaluate(label, yte, p_te)
        rows.append(m)
        print(f"  → acc={m['acc']*100:.2f}%  brier={m['brier']:.4f}  "
              f"ece={m['ece']*100:.2f}%  auc={m['auc']:.3f}")

    res = pd.DataFrame(rows)
    out_csv = RUNS_DIR / f"{tag}_recency.csv"
    res.to_csv(out_csv, index=False)

    # Compare to control
    if rows:
        ctrl = next((r for r in rows if "uniform" in r["config"]), rows[0])
        for r in rows:
            r["d_acc_pp"]   = (r["acc"]   - ctrl["acc"])   * 100
            r["d_brier"]    =  r["brier"] - ctrl["brier"]
            r["d_ece_pp"]   = (r["ece"]   - ctrl["ece"])   * 100

    res2 = pd.DataFrame(rows)
    print("\n=== Recency comparison ===")
    cols = ["config", "acc", "d_acc_pp", "brier", "d_brier", "ece", "d_ece_pp", "auc"]
    cols = [c for c in cols if c in res2.columns]
    print(res2[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved → {out_csv}")
    return res2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt",   default="T20,IT20",
                    help="Comma-separated formats, or 'all'.")
    ap.add_argument("--tag",   default="t20")
    ap.add_argument("--hl",    default="none,720,540,365,270",
                    help="Comma-separated half-life days; 'none' = no weighting.")
    args = ap.parse_args()

    fmts = [f.strip() for f in args.fmt.split(",") if f.strip()]
    half_lives = []
    for tok in args.hl.split(","):
        tok = tok.strip().lower()
        if tok in ("none", "0", "", "control"): half_lives.append(None)
        else: half_lives.append(float(tok))
    run(fmts, half_lives, args.tag)
