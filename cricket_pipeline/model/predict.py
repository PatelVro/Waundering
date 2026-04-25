"""Load trained models and score a single ball state."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as F
from .train import META_PATH, RUNS_PATH, WICKET_PATH, RUN_BUCKETS


@lru_cache(maxsize=1)
def _load() -> tuple[lgb.Booster, lgb.Booster, dict]:
    if not RUNS_PATH.exists() or not WICKET_PATH.exists():
        raise RuntimeError("Models not found. Run `pipeline model train` first.")
    runs = lgb.Booster(model_file=str(RUNS_PATH))
    wkt  = lgb.Booster(model_file=str(WICKET_PATH))
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return runs, wkt, meta


def _row_to_df(state: dict) -> pd.DataFrame:
    row = {col: state.get(col) for col in F.NUMERIC + F.CATEGORICAL}
    df = pd.DataFrame([row])
    for col in F.CATEGORICAL:
        df[col] = df[col].astype("category")
    return df[F.NUMERIC + F.CATEGORICAL]


def predict_ball(state: dict) -> dict:
    """Score one ball state.

    Required keys (others default to None):
      format, venue, batter, bowler, batter_hand, bowler_type, phase
      innings_no, over_no, ball_in_over, runs_so_far, wickets_so_far,
      deliveries_so_far, legal_balls_left, current_run_rate, required_run_rate
      batter_sr, batter_avg, bowler_econ, bowler_avg, ...
    Returns:
      {"runs_probs": {0: p, 1: p, 2: p, 3: p, 4: p, 5: p, 6: p},
       "wicket_prob": float,
       "expected_runs": float}
    """
    runs, wkt, _ = _load()
    X = _row_to_df(state)
    rp = runs.predict(X)[0]
    wp = float(wkt.predict(X)[0])
    runs_probs = {bucket: float(rp[i]) for i, bucket in enumerate(RUN_BUCKETS)}
    expected = sum(b * p for b, p in runs_probs.items() if b != 5) + 5 * runs_probs.get(5, 0)
    return {
        "runs_probs":    runs_probs,
        "wicket_prob":   wp,
        "expected_runs": expected,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of ball states. Returns the same df with added columns."""
    runs, wkt, _ = _load()
    for col in F.CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("category")
    X = df[F.NUMERIC + F.CATEGORICAL]
    rp = runs.predict(X)
    wp = wkt.predict(X)
    out = df.copy()
    for i, b in enumerate(RUN_BUCKETS):
        out[f"p_runs_{b}"] = rp[:, i]
    out["p_wicket"]      = wp
    out["expected_runs"] = sum(b * out[f"p_runs_{b}"] for b in RUN_BUCKETS)
    return out
