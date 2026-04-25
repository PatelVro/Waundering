"""Monte Carlo innings / match simulator.

Given a starting ball state, sample outcomes ball-by-ball using the trained
runs + wicket models. We do `n_sim` parallel rollouts per call to get a
distribution of final scores and (when a target is set) a win probability.

Caveats / refinements left as TODOs:
  * Strike rotation is approximated — a 1 or 3 swaps the striker pointer
    but the *batter's* feature row stays the same per ball (we don't update
    batter_sr etc. mid-innings).
  * Extras (wides, no-balls) are folded into the same RUN_BUCKETS via the
    legal-ball framing of the model.
  * Player-out updates: when a wicket falls we keep the same striker label —
    in reality you'd swap in the next batter from a queue. To extend, pass
    `batting_order` and update on dismissal.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from . import calibrate as C
from .predict import _load
from . import features as F
from .train import RUN_BUCKETS

RUN_VALUES = np.array(RUN_BUCKETS, dtype=int)   # [0,1,2,3,4,5,6]
RNG = np.random.default_rng(42)


def _state_to_array(state: dict) -> np.ndarray:
    import pandas as pd
    df = pd.DataFrame([{c: state.get(c) for c in F.NUMERIC + F.CATEGORICAL}])
    for col in F.CATEGORICAL:
        df[col] = df[col].astype("category")
    return df[F.NUMERIC + F.CATEGORICAL]


def _predict(states_df) -> tuple[np.ndarray, np.ndarray]:
    runs, wkt, _, cal = _load()
    rp = runs.predict(states_df)
    wp = wkt.predict(states_df)
    if cal is not None:
        runs_isos, wicket_iso = cal
        rp = C.transform_multiclass(runs_isos, rp)
        wp = C.transform_binary(wicket_iso, wp)
    return rp, wp


def simulate_innings(
    state: dict,
    n_sim: int = 5000,
    max_wickets: int = 10,
    seed: int | None = None,
) -> dict:
    """Roll the innings forward from `state` until done. Returns a dict with
    the score distribution and (if `state['target']`) a chase win probability.
    """
    import pandas as pd

    rng = np.random.default_rng(seed) if seed is not None else RNG
    target = state.get("target")
    finals = np.zeros(n_sim, dtype=int)
    won    = np.zeros(n_sim, dtype=bool)

    runs_arr     = np.full(n_sim, int(state.get("runs_so_far") or 0))
    wkts_arr     = np.full(n_sim, int(state.get("wickets_so_far") or 0))
    balls_arr    = np.full(n_sim, int(state.get("legal_balls_left") or 120))
    deliv_arr    = np.full(n_sim, int(state.get("deliveries_so_far") or 0))

    base_row = {c: state.get(c) for c in F.NUMERIC + F.CATEGORICAL}

    while True:
        active = (balls_arr > 0) & (wkts_arr < max_wickets)
        if target is not None:
            active &= runs_arr < target
        if not active.any():
            break

        # Build a batched dataframe of the active simulations
        idx = np.flatnonzero(active)
        rows = []
        for i in idx:
            r = dict(base_row)
            r.update({
                "runs_so_far":       int(runs_arr[i]),
                "wickets_so_far":    int(wkts_arr[i]),
                "deliveries_so_far": int(deliv_arr[i]),
                "legal_balls_left":  int(balls_arr[i]),
                "current_run_rate":  6.0 * runs_arr[i] / max(deliv_arr[i], 1),
                "required_run_rate": (
                    6.0 * (target - runs_arr[i]) / max(balls_arr[i], 1)
                    if target is not None else float("nan")
                ),
            })
            rows.append(r)
        df = pd.DataFrame(rows)
        for col in F.CATEGORICAL:
            df[col] = df[col].astype("category")
        df = df[F.NUMERIC + F.CATEGORICAL]

        rp, wp = _predict(df)

        is_wkt = rng.random(len(idx)) < wp
        # cumulative for runs
        cdf = rp.cumsum(axis=1)
        u = rng.random(len(idx))[:, None]
        runs_outcome = (u < cdf).argmax(axis=1)
        runs_outcome = RUN_VALUES[runs_outcome]
        runs_outcome[is_wkt] = 0  # treat wicket-balls as zero runs

        runs_arr[idx]  += runs_outcome
        wkts_arr[idx]  += is_wkt.astype(int)
        balls_arr[idx] -= 1
        deliv_arr[idx] += 1

    finals = runs_arr
    if target is not None:
        won = finals >= target

    return {
        "n_sim":    n_sim,
        "mean":     float(finals.mean()),
        "p10":      int(np.percentile(finals, 10)),
        "p50":      int(np.percentile(finals, 50)),
        "p90":      int(np.percentile(finals, 90)),
        "histogram": _histogram(finals.tolist()),
        "win_prob": float(won.mean()) if target is not None else None,
        "target":   target,
    }


def _histogram(scores: Iterable[int], buckets: int = 10) -> dict:
    if not scores:
        return {}
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return {f"{lo}": len(list(scores))}
    width = max((hi - lo) // buckets, 1)
    counts: Counter = Counter()
    for s in scores:
        bucket = lo + ((s - lo) // width) * width
        counts[bucket] += 1
    return {f"{k}-{k+width-1}": counts[k] for k in sorted(counts)}
