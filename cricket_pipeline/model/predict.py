"""Load trained models and score a single ball state."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import calibrate as C
from . import features as F
from .train import META_PATH, RUNS_PATH, WICKET_PATH, RUN_BUCKETS


@lru_cache(maxsize=1)
def _load() -> tuple[lgb.Booster, lgb.Booster, dict, object]:
    if not RUNS_PATH.exists() or not WICKET_PATH.exists():
        raise RuntimeError("Models not found. Run `pipeline model train` first.")
    runs = lgb.Booster(model_file=str(RUNS_PATH))
    wkt  = lgb.Booster(model_file=str(WICKET_PATH))
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    cal  = C.load()  # tuple (runs_isos, wicket_iso) or None
    return runs, wkt, meta, cal


def _row_to_df(state: dict) -> pd.DataFrame:
    row = {col: state.get(col) for col in F.NUMERIC + F.CATEGORICAL}
    df = pd.DataFrame([row])
    for col in F.CATEGORICAL:
        df[col] = df[col].astype("category")
    return df[F.NUMERIC + F.CATEGORICAL]


def predict_ball(state: dict) -> dict:
    """Score one ball state. Returns calibrated probabilities when calibrators
    are available (they are after `pipeline model train`)."""
    runs, wkt, _, cal = _load()
    X = _row_to_df(state)
    rp = runs.predict(X)
    wp = wkt.predict(X)
    if cal is not None:
        runs_isos, wicket_iso = cal
        rp = C.transform_multiclass(runs_isos, rp)
        wp = C.transform_binary(wicket_iso, wp)
    runs_probs = {bucket: float(rp[0][i]) for i, bucket in enumerate(RUN_BUCKETS)}
    expected = sum(b * p for b, p in runs_probs.items() if b != 5) + 5 * runs_probs.get(5, 0)
    return {
        "runs_probs":    runs_probs,
        "wicket_prob":   float(wp[0]),
        "expected_runs": expected,
        "calibrated":    cal is not None,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of ball states. Returns the same df with added columns."""
    runs, wkt, _, cal = _load()
    for col in F.CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("category")
    X = df[F.NUMERIC + F.CATEGORICAL]
    rp = runs.predict(X)
    wp = wkt.predict(X)
    if cal is not None:
        runs_isos, wicket_iso = cal
        rp = C.transform_multiclass(runs_isos, rp)
        wp = C.transform_binary(wicket_iso, wp)
    out = df.copy()
    for i, b in enumerate(RUN_BUCKETS):
        out[f"p_runs_{b}"] = rp[:, i]
    out["p_wicket"]      = wp
    out["expected_runs"] = sum(b * out[f"p_runs_{b}"] for b in RUN_BUCKETS)
    return out
