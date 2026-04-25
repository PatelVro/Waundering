"""Train ball-outcome models (runs multiclass + wicket binary) with LightGBM."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from . import calibrate as C
from . import features as F

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
RUNS_PATH    = MODEL_DIR / "runs.lgb"
WICKET_PATH  = MODEL_DIR / "wicket.lgb"
META_PATH    = MODEL_DIR / "meta.json"

RUN_BUCKETS = [0, 1, 2, 3, 4, 5, 6]            # 5 = "other / 5+ except 6"


def _matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[F.NUMERIC + F.CATEGORICAL].copy()


def _runs_params() -> dict:
    return {
        "objective":     "multiclass",
        "num_class":     len(RUN_BUCKETS),
        "metric":        "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves":    127,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq":  5,
        "verbose":       -1,
    }


def _wicket_params() -> dict:
    return {
        "objective":     "binary",
        "metric":        "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves":    127,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq":  5,
        "is_unbalance":  True,
        "verbose":       -1,
    }


def _train_one(X_tr, y_tr, X_te, y_te, params, num_round=600):
    train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=F.CATEGORICAL)
    valid = lgb.Dataset(X_te, label=y_te, categorical_feature=F.CATEGORICAL, reference=train)
    return lgb.train(
        params, train,
        num_boost_round=num_round,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )


def train(format_filter: str | None = "IT20", limit: int | None = None) -> dict:
    print(f"Loading features (format={format_filter}, limit={limit}) …")
    df = F.build(format_filter=format_filter, limit=limit)
    if df.empty:
        raise RuntimeError("No rows. Run `pipeline cricsheet` and `pipeline views` first.")

    print(f"  rows: {len(df):,}")
    # 70 / 10 / 20 — train / calibration / test, all by date
    train_df, holdout_df = F.split_by_date(df, test_frac=0.30)
    calib_df, test_df    = F.split_by_date(holdout_df, test_frac=2 / 3)
    print(f"  train: {len(train_df):,}   calib: {len(calib_df):,}   test: {len(test_df):,}")

    X_tr, X_ca, X_te = _matrix(train_df), _matrix(calib_df), _matrix(test_df)

    print("Training runs model …")
    runs = _train_one(X_tr, train_df["y_runs_bucket"], X_te, test_df["y_runs_bucket"],
                      _runs_params())

    print("Training wicket model …")
    wkt = _train_one(X_tr, train_df["y_wicket"], X_te, test_df["y_wicket"],
                     _wicket_params())

    print("Fitting calibrators on holdout …")
    runs_calib_raw = runs.predict(X_ca)
    wkt_calib_raw  = wkt.predict(X_ca)
    runs_isos  = C.fit_multiclass(runs_calib_raw, calib_df["y_runs_bucket"].to_numpy(),
                                  n_classes=len(RUN_BUCKETS))
    wicket_iso = C.fit_binary(wkt_calib_raw, calib_df["y_wicket"].to_numpy())
    C.save(runs_isos, wicket_iso, n_calib=len(calib_df))

    runs_pred_raw = runs.predict(X_te)
    wkt_pred_raw  = wkt.predict(X_te)
    runs_pred = C.transform_multiclass(runs_isos, runs_pred_raw)
    wkt_pred  = C.transform_binary(wicket_iso, wkt_pred_raw)

    metrics = {
        "rows_train":             int(len(train_df)),
        "rows_calib":             int(len(calib_df)),
        "rows_test":              int(len(test_df)),
        "runs_logloss_raw":       float(log_loss(test_df["y_runs_bucket"], runs_pred_raw,
                                                 labels=list(range(len(RUN_BUCKETS))))),
        "runs_logloss_calib":     float(log_loss(test_df["y_runs_bucket"], runs_pred,
                                                 labels=list(range(len(RUN_BUCKETS))))),
        "runs_top1":              float(accuracy_score(test_df["y_runs_bucket"],
                                                       np.argmax(runs_pred, axis=1))),
        "wicket_logloss_raw":     float(log_loss(test_df["y_wicket"], wkt_pred_raw)),
        "wicket_logloss_calib":   float(log_loss(test_df["y_wicket"], wkt_pred)),
        "wicket_auc":             float(roc_auc_score(test_df["y_wicket"], wkt_pred)),
        "format_filter":          format_filter,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    runs.save_model(str(RUNS_PATH))
    wkt.save_model(str(WICKET_PATH))
    META_PATH.write_text(json.dumps({
        "metrics": metrics,
        "categorical": F.CATEGORICAL,
        "numeric":     F.NUMERIC,
        "run_buckets": RUN_BUCKETS,
    }, indent=2))

    print("\n=== metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<18} {v}")
    return metrics
