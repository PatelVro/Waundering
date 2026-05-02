"""Canonical eval harness for the cricket match-winner model.

Time-based train / val(=calibration) / test split. Reports:
  - test accuracy
  - test log-loss + Brier (raw and isotonic-calibrated)
  - ECE (expected calibration error, 10 bins)
  - ROC AUC
  - sanity baselines: always-team1, majority class, higher-form, higher-elo

Usage:
  python -m cricket_pipeline.work.eval --formats T20,IT20
  python -m cricket_pipeline.work.eval --formats T20,IT20 --model-tag baseline
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                              roc_auc_score)

from .features_v2 import (CATEGORICAL, NUMERIC, PLAYER_NUMERIC, build_features,
                          build_features_with_players)


WORK_DIR    = Path(__file__).resolve().parent
RUNS_DIR    = WORK_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)


@dataclass
class SplitDates:
    train_end: str        # exclusive
    test_start: str       # inclusive (= train_end + calib_window)


def time_split(df: pd.DataFrame,
               test_frac: float = 0.15,
               calib_frac: float = 0.10,
               test_cutoff: str | None = None,
               calib_cutoff: str | None = None,
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitDates]:
    df = df.sort_values("start_date").reset_index(drop=True)
    if test_cutoff:
        # explicit time-based cutoff: train < calib_cutoff <= calib < test_cutoff <= test
        cc = pd.to_datetime(calib_cutoff or test_cutoff)
        tc = pd.to_datetime(test_cutoff)
        train = df[df["start_date"] <  cc].copy()
        calib = df[(df["start_date"] >= cc) & (df["start_date"] < tc)].copy()
        test  = df[df["start_date"] >= tc].copy()
    else:
        n = len(df)
        n_test  = int(round(n * test_frac))
        n_calib = int(round(n * calib_frac))
        n_train = n - n_test - n_calib
        train = df.iloc[:n_train].copy()
        calib = df.iloc[n_train:n_train + n_calib].copy()
        test  = df.iloc[n_train + n_calib:].copy()
    sd = SplitDates(
        train_end  = str(train["start_date"].max().date()) if not train.empty else "",
        test_start = str(test["start_date"].min().date())  if not test.empty  else "",
    )
    return train, calib, test, sd


def ece_bins(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    out = []
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            out.append({"bin": b, "lo": bins[b], "hi": bins[b+1], "n": 0,
                        "p_mean": None, "y_mean": None})
            continue
        pm = float(p[mask].mean()); ym = float(y_true[mask].mean())
        out.append({"bin": b, "lo": float(bins[b]), "hi": float(bins[b+1]),
                    "n": cnt, "p_mean": pm, "y_mean": ym})
        ece += (cnt / n) * abs(pm - ym)
    return {"ece": float(ece), "bins": out}


def baselines(test: pd.DataFrame) -> dict:
    y = test["y_t1_wins"].astype(int).to_numpy()
    out = {
        "always_t1":    float((y == 1).mean()),
        "always_t2":    float((y == 0).mean()),
        "majority":     float(max((y == 1).mean(), (y == 0).mean())),
    }
    # higher-form picker (last10)
    if "t1_last10" in test.columns:
        f1 = test["t1_last10"].fillna(0.5); f2 = test["t2_last10"].fillna(0.5)
        pred = (f1 >= f2).astype(int)
        out["higher_last10"] = float(accuracy_score(y, pred.to_numpy()))
    # higher-elo picker
    if "elo_diff_pre" in test.columns:
        pred = (test["elo_diff_pre"].fillna(0) >= 0).astype(int)
        out["higher_elo"] = float(accuracy_score(y, pred.to_numpy()))
    # higher form + elo
    if {"elo_diff_pre", "form_diff_10"}.issubset(test.columns):
        score = test["elo_diff_pre"].fillna(0) * 0.7 + test["form_diff_10"].fillna(0) * 200
        pred = (score >= 0).astype(int)
        out["elo_plus_form"] = float(accuracy_score(y, pred.to_numpy()))
    return out


def fit_isotonic(p_calib: np.ndarray, y_calib: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_calib, y_calib)
    return iso


class _PlattCalibrator:
    """Logistic regression on the raw probability."""
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e6)
    def fit(self, p, y):
        self.lr.fit(p.reshape(-1, 1), y)
        return self
    def transform(self, p):
        return self.lr.predict_proba(p.reshape(-1, 1))[:, 1]


def fit_platt(p_calib: np.ndarray, y_calib: np.ndarray) -> _PlattCalibrator:
    return _PlattCalibrator().fit(p_calib, y_calib)


class _BetaCalibrator:
    """Beta calibration (Kull et al. 2017): logistic on (log p, log(1-p))."""
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e6)
    def fit(self, p, y):
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        self.lr.fit(X, y); return self
    def transform(self, p):
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        return self.lr.predict_proba(X)[:, 1]


def fit_beta(p_calib, y_calib): return _BetaCalibrator().fit(p_calib, y_calib)


def lgb_params() -> dict:
    return {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq":     5,
        # Wave 1, Agent 46: deterministic mode disables thread-race
        # accumulation so two runs with the same data + seed produce
        # byte-identical predictions. ~10-15% slower but the
        # reproducibility win outweighs the speed loss.
        "deterministic":    True,
        "force_col_wise":   True,
        "num_threads":      1,
        "verbose":          -1,
    }


def train_eval(formats: list[str] | None,
               feature_cols: list[str],
               cat_cols: list[str],
               model_tag: str = "v2_baseline",
               num_boost_round: int = 1500,
               early_stopping: int = 50,
               extra_params: dict | None = None,
               with_players: bool = False) -> dict:
    t0 = time.time()
    if with_players:
        df = build_features_with_players(format_filter=formats)
    else:
        df = build_features(format_filter=formats)
    if df.empty:
        raise RuntimeError("Empty feature frame.")

    print(f"Built features for {len(df):,} matches in {time.time()-t0:.1f}s "
          f"(formats={formats or 'all'}, dropped_unfinished={df.attrs.get('dropped_unfinished')})")

    # set up category dtype (LightGBM expects category dtype on cat cols)
    for c in cat_cols:
        df[c] = df[c].astype("category")

    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    print(f"Split dates: train <= {sd.train_end} | calib ... | test >= {sd.test_start}")
    print(f"Sizes: train={len(train):,}  calib={len(calib):,}  test={len(test):,}")

    Xtr, ytr = train[feature_cols], train["y_t1_wins"].astype(int)
    Xca, yca = calib[feature_cols], calib["y_t1_wins"].astype(int)
    Xte, yte = test[feature_cols],  test["y_t1_wins"].astype(int)

    train_set = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(Xca, label=yca, categorical_feature=cat_cols, reference=train_set,
                            free_raw_data=False)

    params = lgb_params()
    if extra_params:
        params.update(extra_params)

    booster = lgb.train(
        params, train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
    )
    best_iter = booster.best_iteration

    # raw test
    p_test_raw = booster.predict(Xte, num_iteration=best_iter)
    p_calib_raw = booster.predict(Xca, num_iteration=best_iter)
    iso = fit_isotonic(p_calib_raw, yca.to_numpy())
    p_test_cal = iso.transform(p_test_raw)

    yte_np = yte.to_numpy()
    metrics = {
        "model_tag":            model_tag,
        "formats":              formats,
        "n_train":              int(len(train)),
        "n_calib":              int(len(calib)),
        "n_test":               int(len(test)),
        "best_iter":            int(best_iter),
        "n_features":           len(feature_cols),
        "test_acc_raw":         float(accuracy_score(yte_np, (p_test_raw >= 0.5).astype(int))),
        "test_acc_cal":         float(accuracy_score(yte_np, (p_test_cal >= 0.5).astype(int))),
        "test_logloss_raw":     float(log_loss(yte_np, np.clip(p_test_raw, 1e-7, 1-1e-7))),
        "test_logloss_cal":     float(log_loss(yte_np, np.clip(p_test_cal, 1e-7, 1-1e-7))),
        "test_brier_raw":       float(brier_score_loss(yte_np, p_test_raw)),
        "test_brier_cal":       float(brier_score_loss(yte_np, p_test_cal)),
        "test_auc":             float(roc_auc_score(yte_np, p_test_cal)),
        "test_ece_raw":         ece_bins(yte_np, p_test_raw)["ece"],
        "test_ece_cal":         ece_bins(yte_np, p_test_cal)["ece"],
        "split":                asdict(sd),
        "baselines":            baselines(test),
    }

    # high-confidence slice
    hi = p_test_cal >= 0.70
    lo = p_test_cal <= 0.30
    hc_mask = hi | lo
    if hc_mask.sum() >= 20:
        # for low-conf "team2 favored" we measure as 1 - p
        pred_hc = (p_test_cal[hc_mask] >= 0.5).astype(int)
        metrics["test_acc_high_conf"]   = float(accuracy_score(yte_np[hc_mask], pred_hc))
        metrics["test_n_high_conf"]     = int(hc_mask.sum())
        metrics["test_high_conf_share"] = float(hc_mask.mean())

    # feature importance
    imp = pd.DataFrame({
        "feature":    feature_cols,
        "gain":       booster.feature_importance(importance_type="gain"),
        "split":      booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    metrics["top10_features_by_gain"] = imp.head(10).to_dict(orient="records")

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        if k in {"baselines", "top10_features_by_gain", "split"}:
            continue
        print(f"  {k:<24} {v}")
    print("  baselines:")
    for k, v in metrics["baselines"].items():
        print(f"    {k:<18} {v:.4f}")
    print("  top10 features (gain):")
    for row in metrics["top10_features_by_gain"]:
        print(f"    {row['feature']:<28} gain={row['gain']:.0f} splits={row['split']}")

    # persist
    out_path = RUNS_DIR / f"{model_tag}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved → {out_path}")

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default=None,
                    help="Comma-separated formats (e.g. 'T20,IT20'). Default: all.")
    ap.add_argument("--model-tag", default="v2_baseline")
    ap.add_argument("--features", default="all",
                    help="'all' (default), 'no_cat' (drop categorical id features), "
                         "'no_cat_players' (no_cat + player aggregates), "
                         "or comma list of features to use.")
    ap.add_argument("--players", action="store_true",
                    help="Add team-aggregate player features (uses balls table).")
    args = ap.parse_args()

    fmts = [f.strip() for f in args.formats.split(",")] if args.formats else None

    use_players = args.players or args.features == "no_cat_players"

    if args.features == "all":
        feat_cols = NUMERIC + CATEGORICAL + (PLAYER_NUMERIC if use_players else [])
        cats = CATEGORICAL
    elif args.features in ("no_cat", "no_cat_players"):
        feat_cols = list(NUMERIC) + (PLAYER_NUMERIC if use_players else [])
        cats = []
    else:
        feat_cols = [f.strip() for f in args.features.split(",")]
        cats = [c for c in CATEGORICAL if c in feat_cols]

    train_eval(formats=fmts, feature_cols=feat_cols, cat_cols=cats,
               model_tag=args.model_tag, with_players=use_players)


if __name__ == "__main__":
    main()
