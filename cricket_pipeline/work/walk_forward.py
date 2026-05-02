"""Walk-forward backtest harness.

Round-50 audit (Wave 1, Agents 10/22/34/50) flagged that prior cycle gains
were likely 0.5-2pp inflated by leakage and Optuna-on-test:

  - `time_split()` puts the most recent 10% in calib and 15% in test, but
    Optuna's objective scans the **test set** to pick hyperparameters.
  - `bet_engine` and prediction history both report metrics on the same
    rolling slice across cycles, so improvements partly fit that slice.
  - The LR meta-learner trains on calib outputs and is then evaluated on
    test — but calib and test share temporal structure (consecutive 25% of
    history), so the meta-learner sees the test distribution implicitly.

This harness fixes those by giving every claim a strict walk-forward stance:

  1. **Cutoff windows.** Provide an explicit `train_cutoff` and a list of
     evaluation windows. For window W=[start, end), train uses only matches
     with `start_date < start`; calib uses a slice immediately before W;
     meta-test (if requested) uses a disjoint slice; W itself is the test.

  2. **Season awareness.** `seasons_to_exclude` removes the predicted
     season's prior matches from train, so IPL-2024 predictions don't
     leak via Elo/form built from IPL-2024 group stage.

  3. **Disjoint meta-test.** Hyperparameter tuning gets a meta-test slice
     that's disjoint from final test — kills the "Optuna picks the test
     winner" failure mode.

  4. **Reproducibility.** Each run logs feature-set hash, git SHA, seeds,
     library versions to `runs/walk_forward/{tag}.jsonl`. Re-runs with the
     same config produce byte-identical predictions.

This module is a HARNESS — it doesn't define a model. Pass any callable
`fit_predict(train, calib, test, **kw) -> np.ndarray` and it'll handle
the splits and metric collection.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .eval import RUNS_DIR, baselines, ece_bins
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                              roc_auc_score)


WF_DIR = RUNS_DIR / "walk_forward"
WF_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Window dataclass
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardWindow:
    """One evaluation window in a walk-forward sweep.

    Boundaries are chronological:
      train     = matches with start_date < train_cutoff
      calib     = matches in [calib_start, calib_end)
      meta_test = matches in [meta_test_start, meta_test_end) (optional)
      test      = matches in [test_start, test_end)

    All datetimes are UTC dates (YYYY-MM-DD). `None` for meta-test means
    the harness will not produce a held-out slice for hyperparameter
    selection — caller is responsible for not peeking at test.
    """
    train_cutoff: str
    calib_start: str
    calib_end: str
    test_start: str
    test_end: str
    meta_test_start: str | None = None
    meta_test_end: str | None = None
    label: str = ""

    def slice(self, df: pd.DataFrame, exclude_seasons: Iterable[str] | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
        df = df.copy()
        df["start_date"] = pd.to_datetime(df["start_date"])

        train_cut = pd.to_datetime(self.train_cutoff)
        c0 = pd.to_datetime(self.calib_start)
        c1 = pd.to_datetime(self.calib_end)
        t0 = pd.to_datetime(self.test_start)
        t1 = pd.to_datetime(self.test_end)

        train = df[df["start_date"] < train_cut].copy()
        calib = df[(df["start_date"] >= c0) & (df["start_date"] < c1)].copy()
        test  = df[(df["start_date"] >= t0) & (df["start_date"] < t1)].copy()

        meta = None
        if self.meta_test_start and self.meta_test_end:
            m0 = pd.to_datetime(self.meta_test_start)
            m1 = pd.to_datetime(self.meta_test_end)
            meta = df[(df["start_date"] >= m0) & (df["start_date"] < m1)].copy()

        # Season-aware exclusion: drop train rows whose season matches the
        # season(s) we're predicting. Ensures no in-tournament leakage.
        if exclude_seasons:
            seasons = set(str(s) for s in exclude_seasons)
            if "season" in train.columns:
                train = train[~train["season"].astype(str).isin(seasons)].copy()

        # Sanity: train must be temporally before everything else
        if not train.empty and not test.empty:
            assert train["start_date"].max() < test["start_date"].min(), \
                f"train leaks into test: {train['start_date'].max()} >= {test['start_date'].min()}"

        return train, calib, meta, test


# ---------------------------------------------------------------------------
# Reproducibility manifest
# ---------------------------------------------------------------------------

def _git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return default


def _data_hash(df: pd.DataFrame) -> str:
    """Stable hash over (row count, column list, first/last start_date,
    summary stats of the label column). Captures enough to detect ingest
    drift without serialising the whole frame."""
    try:
        cols = sorted(df.columns.tolist())
        sd_min = str(df["start_date"].min()) if "start_date" in df.columns and not df.empty else ""
        sd_max = str(df["start_date"].max()) if "start_date" in df.columns and not df.empty else ""
        y_pos = int((df.get("y_t1_wins", pd.Series(dtype=int)) == 1).sum())
        y_neg = int((df.get("y_t1_wins", pd.Series(dtype=int)) == 0).sum())
        signature = f"{len(df)}|{','.join(cols)}|{sd_min}|{sd_max}|{y_pos}|{y_neg}"
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unhashable"


def make_manifest(*, tag: str, df: pd.DataFrame, seeds: dict,
                  hyperparams: dict | None = None,
                  feature_columns: list[str] | None = None) -> dict:
    """Build a reproducibility manifest for a model artifact.

    Write this alongside any saved model so a future engineer can
    answer 'was this trained on the same data with the same seeds?'.
    """
    return {
        "tag":             tag,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "git_sha":         _git_sha(),
        "data_hash":       _data_hash(df),
        "row_count":       int(len(df)),
        "seeds":           seeds,
        "hyperparams":     hyperparams or {},
        "feature_columns": feature_columns or [],
        "platform":        platform.platform(),
        "python_version":  sys.version.split()[0],
        "library_versions": _library_versions(),
    }


def _library_versions() -> dict[str, str]:
    versions = {}
    for mod_name in ("numpy", "pandas", "lightgbm", "xgboost",
                      "catboost", "sklearn", "duckdb", "torch"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not-installed"
        except Exception as e:
            # OSError (Windows DLL load), RuntimeError (CUDA init), etc.
            # Don't let a broken library break manifest generation.
            versions[mod_name] = f"import-failed: {type(e).__name__}"
    return versions


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int = 42) -> dict:
    """Set seeds at every relevant random source. Returns the seed table
    so callers can persist it in a manifest.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    seeds = {"python_random": seed, "numpy": seed, "pythonhashseed": seed}
    try:
        import torch  # noqa: F401
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        seeds["torch"] = seed
    except (ImportError, OSError, RuntimeError):
        # OSError on Windows when MSVC redist is missing; RuntimeError on
        # CUDA init failures. Determinism for non-torch sources still holds.
        pass
    return seeds


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true: np.ndarray, p: np.ndarray, *,
                          name: str = "") -> dict:
    p = np.asarray(p, dtype=float)
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)
    ece = ece_bins(y_true, p)["ece"]
    return {
        "name":     name,
        "n":        int(len(y_true)),
        "acc":      float(accuracy_score(y_true, (p >= 0.5).astype(int))),
        "logloss":  float(log_loss(y_true, p_clipped)),
        "brier":    float(brier_score_loss(y_true, p)),
        "auc":      float(roc_auc_score(y_true, p)) if len(set(y_true)) > 1 else float("nan"),
        "ece":      float(ece),
    }


def evaluate_baselines(test: pd.DataFrame) -> dict:
    """Wrap eval.baselines so the harness always reports model-vs-floor."""
    return baselines(test)


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    tag: str
    window: WalkForwardWindow
    metrics: dict
    baselines: dict
    n_train: int
    n_calib: int
    n_meta_test: int | None
    n_test: int
    manifest: dict
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["window"] = asdict(self.window)
        return d


def run_window(
    df: pd.DataFrame,
    window: WalkForwardWindow,
    fit_predict: Callable[..., np.ndarray],
    *,
    tag: str,
    seed: int = 42,
    exclude_seasons: Iterable[str] | None = None,
    use_meta_test: bool = True,
    save_jsonl: bool = True,
    extras: dict | None = None,
) -> WalkForwardResult:
    """Execute a single walk-forward window end-to-end.

    `fit_predict` signature:  fit_predict(train, calib, test, *,
                                          meta_test=None, seed=int, **kw) -> np.ndarray
    Returns predicted P(t1_wins) on `test` (only). The harness logs
    metrics + baselines + manifest as JSONL.
    """
    seeds = set_global_seeds(seed)
    train, calib, meta, test = window.slice(df, exclude_seasons=exclude_seasons)

    if test.empty:
        raise ValueError(f"walk_forward: test slice empty for window {window.label or window.test_start}")
    if train.empty:
        raise ValueError("walk_forward: train slice empty (train_cutoff too early?)")

    p_test = fit_predict(train, calib, test,
                          meta_test=meta if use_meta_test else None,
                          seed=seed)
    p_test = np.asarray(p_test, dtype=float).ravel()
    if len(p_test) != len(test):
        raise ValueError(f"fit_predict returned {len(p_test)} preds; test has {len(test)} rows")

    y = test["y_t1_wins"].astype(int).to_numpy()
    metrics = evaluate_predictions(y, p_test, name=tag)
    base = evaluate_baselines(test)

    manifest = make_manifest(
        tag=tag,
        df=df,
        seeds=seeds,
        feature_columns=sorted(c for c in df.columns if c not in ("y_t1_wins", "match_id")),
    )

    result = WalkForwardResult(
        tag=tag, window=window, metrics=metrics, baselines=base,
        n_train=len(train), n_calib=len(calib),
        n_meta_test=(len(meta) if meta is not None else None),
        n_test=len(test), manifest=manifest,
        extras=extras or {},
    )

    if save_jsonl:
        log_path = WF_DIR / f"{tag}.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), default=str) + "\n")

    return result


def run_sweep(
    df: pd.DataFrame,
    windows: list[WalkForwardWindow],
    fit_predict: Callable[..., np.ndarray],
    *,
    tag: str,
    seed: int = 42,
    exclude_seasons_for: Callable[[WalkForwardWindow], list[str]] | None = None,
    use_meta_test: bool = True,
) -> list[WalkForwardResult]:
    """Run a sequence of walk-forward windows. Returns one result per window."""
    out: list[WalkForwardResult] = []
    for w in windows:
        seasons = exclude_seasons_for(w) if exclude_seasons_for else None
        result = run_window(df, w, fit_predict, tag=tag, seed=seed,
                              exclude_seasons=seasons, use_meta_test=use_meta_test)
        out.append(result)
    return out


def quarterly_windows(start_year: int, end_year: int, calib_days: int = 30,
                       meta_test_days: int = 14) -> list[WalkForwardWindow]:
    """Helper: generate quarterly walk-forward windows.

    For each quarter Q in [start_year, end_year]:
       train     = all matches before (Q.start - calib_days - meta_test_days)
       meta-test = (Q.start - calib_days - meta_test_days, Q.start - calib_days)
       calib     = (Q.start - calib_days, Q.start)
       test      = (Q.start, Q.end)
    """
    out: list[WalkForwardWindow] = []
    for year in range(start_year, end_year + 1):
        for q in range(4):
            test_start = pd.Timestamp(year=year, month=q * 3 + 1, day=1)
            test_end   = test_start + pd.DateOffset(months=3)
            calib_end   = test_start
            calib_start = calib_end - pd.Timedelta(days=calib_days)
            mt_end      = calib_start
            mt_start    = mt_end - pd.Timedelta(days=meta_test_days)
            train_cut   = mt_start
            out.append(WalkForwardWindow(
                train_cutoff   = train_cut.strftime("%Y-%m-%d"),
                meta_test_start= mt_start.strftime("%Y-%m-%d"),
                meta_test_end  = mt_end.strftime("%Y-%m-%d"),
                calib_start    = calib_start.strftime("%Y-%m-%d"),
                calib_end      = calib_end.strftime("%Y-%m-%d"),
                test_start     = test_start.strftime("%Y-%m-%d"),
                test_end       = test_end.strftime("%Y-%m-%d"),
                label          = f"{year}Q{q+1}",
            ))
    return out
