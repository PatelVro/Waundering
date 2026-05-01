"""ODI-specific stacked-ensemble + per-format hyperparameter tuning.

Why a separate harness:
   ODI tier-1 sits at ~66% accuracy vs T20 tier-1 at ~78%.
   Step-1 error analysis showed subcontinent home wins dominate the
   high-confidence misses (BAN×2 home wins vs NZ, SL×2 home wins vs IND,
   BAN home win vs PAK, ZIM home win vs PAK). Conclusion: home-country
   advantage matters far more in ODI than T20.

What this module adds:
   1. ODI-only training corpus (filtered) with recency weighting (hl=720d
      from Cycle 7 — already validated +2.7pp on ODI).
   2. Extra ODI features: t1_is_home / t2_is_home / is_neutral_venue
      (already computed in features_v2.build_features() but excluded from
      the cross-format NUMERIC list since they regressed T20).
   3. Optuna hyperparameter search on LGBM-num (the dominant base learner)
      for ODI-specific best params.
   4. Side-by-side: control (cross-format defaults, no recency) vs
      ODI-tuned + recency + home-advantage features.

Run:
   python -m cricket_pipeline.work.odi_model --search       # Optuna search
   python -m cricket_pipeline.work.odi_model --eval         # full ensemble eval at best params
   python -m cricket_pipeline.work.odi_model --search --eval

Best params persisted to runs/odi_best_params.json so the production
predict_match path can load them.
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

from .ensemble import (DEFAULT_RECENCY_HL_DAYS, _cat_pred, _lgb_pred, _lr_pred,
                        _xgb_pred, recency_weights)
from .eval import RUNS_DIR, ece_bins, lgb_params, time_split
from .features_v2 import (CATEGORICAL, NUMERIC, PLAYER_NUMERIC,
                           build_features_with_players)


# ---------- ODI feature additions ----------

# Home-advantage features that exist in features_v2 but were excluded from the
# cross-format NUMERIC list because they hurt T20 (Cycle 1a). For ODI, the
# Step-1 error analysis says they should be a real signal.
ODI_EXTRA_NUMERIC = [
    "t1_is_home",
    "t2_is_home",
    "is_neutral_venue",
]

ODI_PARAMS_PATH = RUNS_DIR / "odi_best_params.json"


def odi_feature_set() -> list[str]:
    """Numeric feature columns for ODI training."""
    return NUMERIC + PLAYER_NUMERIC + ODI_EXTRA_NUMERIC


# ---------- per-format LGBM params (loaded from Optuna search if available) ----------

def odi_lgb_params() -> dict:
    """Best LGBM params for ODI (loaded from disk if Optuna has run, else
    the cross-format defaults from eval.lgb_params)."""
    if ODI_PARAMS_PATH.exists():
        try:
            saved = json.loads(ODI_PARAMS_PATH.read_text())
            base = lgb_params()
            base.update(saved.get("params", {}))
            return base
        except Exception:
            pass
    return lgb_params()


# ---------- evaluation helper ----------

def _evaluate(name: str, y, p) -> dict:
    return {
        "config":  name,
        "n":       int(len(y)),
        "acc":     float(accuracy_score(y, (p >= 0.5).astype(int))),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "brier":   float(brier_score_loss(y, p)),
        "auc":     float(roc_auc_score(y, p)),
        "ece":     float(ece_bins(y, p)["ece"]),
    }


# ---------- LGBM with custom params + weights ----------

def _lgb_pred_with_params(train, calib, test, feat, cats, ytr, yca, weights,
                           params: dict, seeds=(0, 7, 42)):
    ds = lgb.Dataset(train[feat], label=ytr, weight=weights,
                     categorical_feature=cats, free_raw_data=False)
    vs = lgb.Dataset(calib[feat], label=yca, categorical_feature=cats,
                     reference=ds, free_raw_data=False)
    pca, pte = [], []
    for s in seeds:
        p = {**params, "seed": s, "feature_fraction_seed": s, "bagging_seed": s}
        b = lgb.train(p, ds, num_boost_round=2500, valid_sets=[vs],
                       callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
        pca.append(b.predict(calib[feat]))
        pte.append(b.predict(test[feat]))
    return np.mean(pca, axis=0), np.mean(pte, axis=0)


# ---------- Optuna search ----------

def search(n_trials: int = 50, seed: int = 0) -> dict:
    """Optuna search on LGBM-num with recency weights + home-advantage feats."""
    try:
        import optuna
    except ImportError:
        print("optuna not installed; skipping. install with: pip install optuna")
        return {}

    print("Loading ODI features ...")
    df = build_features_with_players(format_filter=["ODI"])
    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    feat = odi_feature_set()
    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()
    yte = test["y_t1_wins"].astype(int).to_numpy()
    w   = recency_weights(train["start_date"], DEFAULT_RECENCY_HL_DAYS)
    print(f"  rows  train={len(train)}  calib={len(calib)}  test={len(test)}  "
          f"feats={len(feat)}  ess={(w.sum()**2)/(w**2).sum():.0f}")

    def objective(trial: "optuna.trial.Trial") -> float:
        p = lgb_params()
        p.update({
            "learning_rate":       trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "num_leaves":          trial.suggest_int("num_leaves", 15, 95),
            "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 10, 100),
            "feature_fraction":    trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction":    trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq":        trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1":           trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
            "lambda_l2":           trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
            "min_gain_to_split":   trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        })
        # Train with single seed for speed inside the search loop
        ds = lgb.Dataset(train[feat], label=ytr, weight=w, free_raw_data=False)
        vs = lgb.Dataset(calib[feat], label=yca, reference=ds, free_raw_data=False)
        b  = lgb.train({**p, "seed": seed},
                        ds, num_boost_round=1500, valid_sets=[vs],
                        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        p_te = b.predict(test[feat])
        # Optimize log-loss (correlated with both acc + calibration)
        return float(log_loss(yte, np.clip(p_te, 1e-7, 1 - 1e-7)))

    study = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n=== Optuna best ===")
    print(f"  best logloss: {study.best_value:.4f}")
    print(f"  best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Persist
    ODI_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ODI_PARAMS_PATH.write_text(json.dumps({
        "params": study.best_params,
        "best_logloss": study.best_value,
        "n_trials": n_trials,
        "feature_set": feat,
        "recency_hl_days": DEFAULT_RECENCY_HL_DAYS,
    }, indent=2))
    print(f"\nSaved → {ODI_PARAMS_PATH}")
    return study.best_params


# ---------- full ensemble eval at best params ----------

def evaluate(use_home_feats: bool = True, use_recency: bool = True,
             use_odi_params: bool = True, tag: str = "odi_tuned"):
    print("Loading ODI features ...")
    df = build_features_with_players(format_filter=["ODI"])
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    feat_num = NUMERIC + PLAYER_NUMERIC + (ODI_EXTRA_NUMERIC if use_home_feats else [])
    feat_cat = CATEGORICAL
    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()
    yte = test["y_t1_wins"].astype(int).to_numpy()
    w   = recency_weights(train["start_date"], DEFAULT_RECENCY_HL_DAYS) if use_recency else None
    params = odi_lgb_params() if use_odi_params else lgb_params()

    print(f"  config: use_home={use_home_feats} recency={use_recency} odi_params={use_odi_params}")
    print(f"  rows  train={len(train)}  calib={len(calib)}  test={len(test)}  feats={len(feat_num)}")

    # LGBM-num (with custom params if requested)
    print("  LGBM-num ...")
    lgb_n_ca, lgb_n_te = _lgb_pred_with_params(
        train, calib, test, feat_num, [], ytr, yca, w, params)
    # LGBM-cat (cross-format defaults — categorical handling doesn't change w/ params much)
    print("  LGBM-cat ...")
    lgb_c_ca, lgb_c_te = _lgb_pred(train, calib, test, feat_num + feat_cat, feat_cat, ytr, yca, weights=w)
    # XGB
    print("  XGB ...")
    xgb_ca, xgb_te = _xgb_pred(train, calib, test, feat_num, ytr, yca, weights=w)
    # CatBoost
    print("  CatBoost ...")
    cat_ca, cat_te = _cat_pred(train, calib, test, feat_num, feat_cat, ytr, yca, weights=w)
    # LR
    print("  LR ...")
    lr_ca, lr_te = _lr_pred(train, calib, test, feat_num, ytr, yca, weights=w)

    # Stack
    Xca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, cat_ca, lr_ca])
    Xte = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, cat_te, lr_te])
    stk = LogisticRegression(C=10.0).fit(Xca, yca)
    p_te = stk.predict_proba(Xte)[:, 1]

    rows = []
    rows.append(_evaluate("lgbm_num",  yte, lgb_n_te))
    rows.append(_evaluate("lgbm_cat",  yte, lgb_c_te))
    rows.append(_evaluate("xgb",       yte, xgb_te))
    rows.append(_evaluate("cat",       yte, cat_te))
    rows.append(_evaluate("lr",        yte, lr_te))
    rows.append(_evaluate("ensemble",  yte, p_te))

    res = pd.DataFrame(rows)
    out_csv = RUNS_DIR / f"{tag}.csv"
    res.to_csv(out_csv, index=False)
    print("\n=== ODI ensemble metrics ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved → {out_csv}")
    return res


# ---------- A/B comparison ----------

def ab_compare():
    """Train control (no home, no recency, default params) vs full ODI build."""
    print("\n##### A/B comparison: ODI control vs ODI-tuned #####\n")
    a = evaluate(use_home_feats=False, use_recency=False, use_odi_params=False, tag="odi_control")
    b = evaluate(use_home_feats=True,  use_recency=True,  use_odi_params=True,  tag="odi_tuned")
    a["build"] = "control"
    b["build"] = "tuned"
    cmp = pd.concat([a, b])
    print("\n=== A/B summary (ensemble row only) ===")
    only = cmp[cmp["config"] == "ensemble"][["build","acc","brier","ece","auc"]]
    print(only.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    delta = b[b["config"] == "ensemble"].iloc[0]["acc"] - a[a["config"] == "ensemble"].iloc[0]["acc"]
    print(f"\nensemble Δacc = {delta*100:+.2f}pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search",   action="store_true", help="run Optuna search")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--eval",     action="store_true", help="evaluate full ensemble at best params")
    ap.add_argument("--ab",       action="store_true", help="A/B control vs tuned (slow)")
    args = ap.parse_args()

    if args.search:
        search(n_trials=args.n_trials)
    if args.eval:
        evaluate()
    if args.ab:
        ab_compare()
    if not (args.search or args.eval or args.ab):
        ap.print_help()


if __name__ == "__main__":
    main()
