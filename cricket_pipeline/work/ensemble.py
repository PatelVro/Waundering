"""Stacked ensemble: LightGBM + XGBoost + CatBoost + LogReg.

Each base learner uses the SAME train/calib/test split. We then:
  - blend by simple average
  - blend by logistic regression (stacking) trained on calib

Reports per-base and ensemble metrics.
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


def recency_weights(dates: pd.Series, half_life_days: float | None = None,
                     ref_date: pd.Timestamp | None = None) -> np.ndarray:
    """Exponential decay sample weights. half_life_days=None → uniform 1.0.

    Validated in cricket_pipeline/work/recency_experiment.py:
       T20:  hl=720d  → +1.15pp acc (74.78% → 75.93%)
       ODI:  hl=720d  → +2.70pp acc (66.44% → 69.14%) AND -1.6pp ECE
    """
    if not half_life_days or half_life_days <= 0:
        return np.ones(len(dates), dtype=float)
    d = pd.to_datetime(dates)
    ref = ref_date if ref_date is not None else d.max()
    delta = (ref - d).dt.days.astype(float).clip(lower=0).to_numpy()
    return np.exp(-np.log(2.0) * delta / float(half_life_days))


# Default production half-life — see Cycle 7 in progress_log.md
DEFAULT_RECENCY_HL_DAYS = 720


def _lgb_pred(train, calib, test, feat, cats, ytr, yca, seeds=(0, 7, 42),
              weights: np.ndarray | None = None):
    ds = lgb.Dataset(train[feat], label=ytr, weight=weights,
                     categorical_feature=cats, free_raw_data=False)
    vs = lgb.Dataset(calib[feat], label=yca, categorical_feature=cats, reference=ds, free_raw_data=False)
    pca, pte = [], []
    for s in seeds:
        p = lgb_params() | {"seed": s, "feature_fraction_seed": s, "bagging_seed": s}
        b = lgb.train(p, ds, num_boost_round=2500, valid_sets=[vs],
                      callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
        pca.append(b.predict(calib[feat]))
        pte.append(b.predict(test[feat]))
    return np.mean(pca, axis=0), np.mean(pte, axis=0)


def _xgb_pred(train, calib, test, feat_num, ytr, yca, seeds=(0, 7, 42),
              weights: np.ndarray | None = None):
    import xgboost as xgb
    pca, pte = [], []
    for s in seeds:
        m = xgb.XGBClassifier(
            n_estimators=2500, learning_rate=0.05, max_depth=5,
            min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, eval_metric="logloss",
            early_stopping_rounds=60, random_state=s, tree_method="hist",
            n_jobs=-1, verbosity=0,
        )
        m.fit(train[feat_num], ytr, sample_weight=weights,
               eval_set=[(calib[feat_num], yca)], verbose=False)
        pca.append(m.predict_proba(calib[feat_num])[:, 1])
        pte.append(m.predict_proba(test[feat_num])[:, 1])
    return np.mean(pca, axis=0), np.mean(pte, axis=0)


def _cat_pred(train, calib, test, feat_num, feat_cat, ytr, yca, seeds=(0, 7, 42),
              weights: np.ndarray | None = None):
    from catboost import CatBoostClassifier, Pool
    feat = feat_num + feat_cat
    pca, pte = [], []
    for s in seeds:
        m = CatBoostClassifier(
            iterations=2500, learning_rate=0.05, depth=6,
            l2_leaf_reg=3.0, loss_function="Logloss", eval_metric="Logloss",
            early_stopping_rounds=80, random_seed=s, allow_writing_files=False,
            verbose=False,
        )
        train_pool = Pool(train[feat], label=ytr, weight=weights, cat_features=feat_cat)
        calib_pool = Pool(calib[feat], label=yca, cat_features=feat_cat)
        m.fit(train_pool, eval_set=calib_pool, use_best_model=True)
        pca.append(m.predict_proba(calib[feat])[:, 1])
        pte.append(m.predict_proba(test[feat])[:, 1])
    return np.mean(pca, axis=0), np.mean(pte, axis=0)


def _lr_pred(train, calib, test, feat_num, ytr, yca,
             weights: np.ndarray | None = None):
    sc = StandardScaler()
    med = train[feat_num].median(numeric_only=True)
    # Belt-and-braces: if a feature is entirely NaN in train, median is NaN too.
    # Fill those columns with 0.0 so LR never sees NaN.
    med = med.fillna(0.0)
    Xtr = sc.fit_transform(train[feat_num].fillna(med).fillna(0.0))
    Xca = sc.transform(calib[feat_num].fillna(med).fillna(0.0))
    Xte = sc.transform(test[feat_num].fillna(med).fillna(0.0))
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(Xtr, ytr, sample_weight=weights)
    return lr.predict_proba(Xca)[:, 1], lr.predict_proba(Xte)[:, 1]


def evaluate(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "name":          name,
        "n":             int(len(y)),
        "acc":           float(accuracy_score(y, (p >= 0.5).astype(int))),
        "logloss":       float(log_loss(y, np.clip(p, 1e-7, 1-1e-7))),
        "brier":         float(brier_score_loss(y, p)),
        "auc":           float(roc_auc_score(y, p)),
        "ece":           float(ece_bins(y, p)["ece"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default=None)
    ap.add_argument("--tag", default="ens")
    args = ap.parse_args()
    fmts = [f.strip() for f in args.formats.split(",")] if args.formats else None

    print(f"Building features (formats={fmts}) ...")
    df = build_features_with_players(format_filter=fmts)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    feat_num = NUMERIC + PLAYER_NUMERIC
    feat_cat = CATEGORICAL
    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()
    yte = test["y_t1_wins"].astype(int).to_numpy()
    print(f"sizes: train={len(train)}  calib={len(calib)}  test={len(test)}")

    base_calib = {}
    base_test  = {}

    t0 = time.time()
    print("\n[1/4] LGBM (numeric only) ...")
    pca, pte = _lgb_pred(train, calib, test, feat_num, [], ytr, yca)
    base_calib["lgbm_num"] = pca; base_test["lgbm_num"] = pte
    print(f"  {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[2/4] LGBM (with cats) ...")
    pca, pte = _lgb_pred(train, calib, test, feat_num + feat_cat, feat_cat, ytr, yca)
    base_calib["lgbm_cat"] = pca; base_test["lgbm_cat"] = pte
    print(f"  {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[3/4] XGBoost (numeric only) ...")
    pca, pte = _xgb_pred(train, calib, test, feat_num, ytr, yca)
    base_calib["xgb"] = pca; base_test["xgb"] = pte
    print(f"  {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[4/4] CatBoost (with cats) ...")
    pca, pte = _cat_pred(train, calib, test, feat_num, feat_cat, ytr, yca)
    base_calib["cat"] = pca; base_test["cat"] = pte
    print(f"  {time.time()-t0:.1f}s")

    t0 = time.time()
    print("[bonus] LogReg (numeric only) ...")
    pca, pte = _lr_pred(train, calib, test, feat_num, ytr, yca)
    base_calib["lr"] = pca; base_test["lr"] = pte
    print(f"  {time.time()-t0:.1f}s")

    rows = []
    for nm in base_test:
        rows.append(evaluate(nm, yte, base_test[nm]))

    # simple average of all
    p_avg = np.mean(list(base_test.values()), axis=0)
    rows.append(evaluate("ensemble_avg", yte, p_avg))

    # average without LR (often weaker)
    p_avg_no_lr = np.mean([base_test[k] for k in base_test if k != "lr"], axis=0)
    rows.append(evaluate("ensemble_no_lr", yte, p_avg_no_lr))

    # stacking: LR on top of base predictions, fitted on calib
    Xst_ca = np.column_stack(list(base_calib.values()))
    Xst_te = np.column_stack([base_test[k] for k in base_calib])
    stk = LogisticRegression(C=10.0)
    stk.fit(Xst_ca, yca)
    p_stk = stk.predict_proba(Xst_te)[:, 1]
    rows.append(evaluate("stack_lr", yte, p_stk))

    # weighted by calib log-loss
    w = []
    for nm in base_test:
        ll = log_loss(yca, np.clip(base_calib[nm], 1e-7, 1-1e-7))
        w.append(1.0 / max(ll, 1e-3))
    w = np.array(w); w = w / w.sum()
    p_wavg = np.average(np.column_stack(list(base_test.values())), axis=1, weights=w)
    rows.append(evaluate("weighted_avg", yte, p_wavg))

    res = pd.DataFrame(rows)
    print("\n=== Per-model results ===")
    print(res.to_string(index=False))

    out = RUNS_DIR / f"{args.tag}_ensemble.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
