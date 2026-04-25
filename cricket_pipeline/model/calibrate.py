"""Probability calibration via isotonic regression.

LightGBM probabilities are typically *uncalibrated* — the model can be
confident in the wrong direction or systematically biased. We fit an
isotonic mapping on a held-out calibration slice (the most recent 10% of
the training set) so downstream consumers (the simulator's chase win
probability, the expected-runs scalar) reflect actual frequencies.

For the multiclass runs model we calibrate each class independently and
then renormalise — a standard "one-vs-rest" calibration scheme.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

CALIB_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
RUNS_CALIB_PATH   = CALIB_DIR / "runs_calibrators.joblib"
WICKET_CALIB_PATH = CALIB_DIR / "wicket_calibrator.joblib"
CALIB_META_PATH   = CALIB_DIR / "calibration_meta.json"


def fit_binary(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(probs, labels)
    return iso


def fit_multiclass(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> list[IsotonicRegression]:
    """One isotonic per class (one-vs-rest)."""
    return [fit_binary(probs[:, k], (labels == k).astype(int)) for k in range(n_classes)]


def transform_binary(iso: IsotonicRegression, probs: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(probs), 1e-6, 1 - 1e-6)


def transform_multiclass(isos: list[IsotonicRegression], probs: np.ndarray) -> np.ndarray:
    out = np.column_stack([transform_binary(isos[k], probs[:, k]) for k in range(len(isos))])
    out = np.clip(out, 1e-6, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def save(runs_isos: list, wicket_iso: IsotonicRegression, n_calib: int) -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(runs_isos, RUNS_CALIB_PATH)
    joblib.dump(wicket_iso, WICKET_CALIB_PATH)
    CALIB_META_PATH.write_text(json.dumps({"calibration_rows": n_calib}, indent=2))


def load() -> tuple[list, IsotonicRegression] | None:
    if not RUNS_CALIB_PATH.exists() or not WICKET_CALIB_PATH.exists():
        return None
    return joblib.load(RUNS_CALIB_PATH), joblib.load(WICKET_CALIB_PATH)
