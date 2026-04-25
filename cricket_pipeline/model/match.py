"""Match-outcome model — direct binary classifier on team-vs-team features.

The ball-level model is interesting but fragile for top-line "who wins?"
questions because it has to roll up 240 noisy ball predictions through a
simulator with strike-rotation approximations.

This module trains a LightGBM directly on `v_match_features` to predict
P(home team wins). One row per match, ~12 features (form, h2h, venue, toss,
rest), isotonic-calibrated.

CLI:
    pipeline match-train  --fmt T20
    pipeline match-predict --home X --away Y --venue Z [--toss-winner X --toss-decision bat]
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from . import calibrate as C
from ..db import connect, install_views

MODEL_DIR     = Path(__file__).resolve().parent.parent / "data" / "models"
MATCH_PATH    = MODEL_DIR / "match.lgb"
MATCH_META    = MODEL_DIR / "match_meta.json"
MATCH_CALIB   = MODEL_DIR / "match_calibrator.joblib"

CATEGORICAL = ["format", "team_home", "team_away", "venue"]
NUMERIC = [
    "toss_winner_is_home", "toss_decision_is_bat",
    "home_last5", "home_last10", "away_last5", "away_last10",
    "home_days_rest", "away_days_rest",
    "h2h_home_winpct", "h2h_meetings",
    "venue_avg_first_innings", "venue_toss_winner_won_pct", "venue_bat_first_pct",
]


def _normalise_formats(format_filter: str | list | None) -> list[str] | None:
    """Accept a comma-string ('T20,IT20'), a list, or None (= all)."""
    if format_filter is None:
        return None
    if isinstance(format_filter, str):
        return [f.strip() for f in format_filter.split(",") if f.strip()]
    return list(format_filter)


def build_features(format_filter: str | list | None = None) -> pd.DataFrame:
    install_views()
    con = connect()
    sql = "SELECT * FROM v_match_features"
    fmts = _normalise_formats(format_filter)
    if fmts:
        in_clause = ",".join(f"'{f}'" for f in fmts)
        sql += f" WHERE format IN ({in_clause})"
    df = con.execute(sql).df()
    con.close()
    if df.empty:
        return df
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def _split_by_date(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("start_date").reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def _params() -> dict:
    return {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq":     5,
        "verbose":          -1,
    }


def train(format_filter: str | list | None = "T20,IT20",
          device: str = "auto") -> dict:
    from .train import _resolve_device
    dev = _resolve_device(device)
    fmts = _normalise_formats(format_filter)
    print(f"Loading match-level features (formats={fmts or 'all'}, device={dev}) …")
    df = build_features(format_filter=fmts)
    if df.empty:
        raise RuntimeError("No matches. Run `pipeline cricsheet` and `pipeline views` first.")

    df = df.dropna(subset=["y_home_wins"])
    print(f"  rows: {len(df):,}")

    train_df, holdout_df = _split_by_date(df, test_frac=0.30)
    calib_df, test_df    = _split_by_date(holdout_df, test_frac=2 / 3)
    print(f"  train: {len(train_df):,}   calib: {len(calib_df):,}   test: {len(test_df):,}")

    feats = NUMERIC + CATEGORICAL
    Xtr = train_df[feats]; Xca = calib_df[feats]; Xte = test_df[feats]
    ytr = train_df["y_home_wins"].astype(int)
    yca = calib_df["y_home_wins"].astype(int)
    yte = test_df["y_home_wins"].astype(int)

    train_set = lgb.Dataset(Xtr, label=ytr, categorical_feature=CATEGORICAL)
    valid_set = lgb.Dataset(Xte, label=yte, categorical_feature=CATEGORICAL, reference=train_set)
    print(f"Training match-outcome model on {dev} …")
    params = _params()
    if dev != "cpu":
        params = {**params, "device": dev}
    booster = lgb.train(
        params, train_set,
        num_boost_round=600,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )

    raw_calib = booster.predict(Xca)
    iso = C.fit_binary(raw_calib, yca.to_numpy())
    joblib.dump(iso, MATCH_CALIB)

    raw_test = booster.predict(Xte)
    cal_test = C.transform_binary(iso, raw_test)

    metrics = {
        "rows_train":       int(len(train_df)),
        "rows_calib":       int(len(calib_df)),
        "rows_test":        int(len(test_df)),
        "logloss_raw":      float(log_loss(yte, raw_test)),
        "logloss_calib":    float(log_loss(yte, cal_test)),
        "brier_raw":        float(brier_score_loss(yte, raw_test)),
        "brier_calib":      float(brier_score_loss(yte, cal_test)),
        "auc":              float(roc_auc_score(yte, cal_test)),
        "accuracy":         float(accuracy_score(yte, (cal_test > 0.5).astype(int))),
        "format_filter":    fmts,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MATCH_PATH))
    MATCH_META.write_text(json.dumps({
        "metrics":      metrics,
        "categorical":  CATEGORICAL,
        "numeric":      NUMERIC,
        "feature_order": feats,
    }, indent=2))

    print("\n=== match-model metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<18} {v}")
    return metrics


_loaded: dict | None = None


def _load() -> dict:
    global _loaded
    if _loaded is not None:
        return _loaded
    if not MATCH_PATH.exists():
        raise RuntimeError("Match model not found. Run `pipeline match-train` first.")
    booster = lgb.Booster(model_file=str(MATCH_PATH))
    meta = json.loads(MATCH_META.read_text())
    iso = joblib.load(MATCH_CALIB) if MATCH_CALIB.exists() else None
    _loaded = {"booster": booster, "meta": meta, "iso": iso}
    return _loaded


def _venue_lookup(venue: str, format_: str) -> dict:
    con = connect()
    row = con.execute(
        """SELECT avg_first_innings, toss_winner_won_pct, bat_first_pct
           FROM v_venue_profile WHERE venue = ? AND format = ?""",
        [venue, format_],
    ).fetchone()
    con.close()
    if not row:
        return {"venue_avg_first_innings": None,
                "venue_toss_winner_won_pct": None,
                "venue_bat_first_pct": None}
    return {"venue_avg_first_innings": row[0],
            "venue_toss_winner_won_pct": row[1],
            "venue_bat_first_pct": row[2]}


def _team_form(team: str, format_: str, ref_date: str | None) -> dict:
    """Last-5 / last-10 win-pct + days since last match, as of ref_date."""
    con = connect()
    rows = con.execute(
        """SELECT start_date,
                  CASE WHEN winner = ? THEN 1 ELSE 0 END AS won
           FROM matches
           WHERE format = ?
             AND (team_home = ? OR team_away = ?)
             AND winner IS NOT NULL
             AND (CAST(? AS DATE) IS NULL OR start_date < CAST(? AS DATE))
           ORDER BY start_date DESC
           LIMIT 10""",
        [team, format_, team, team, ref_date, ref_date],
    ).fetchall()
    con.close()
    if not rows:
        return {"last5": None, "last10": None, "days_rest": None}
    last5_rows  = rows[:5]
    last5_wp    = sum(r[1] for r in last5_rows)  / max(len(last5_rows), 1)
    last10_wp   = sum(r[1] for r in rows)        / len(rows)
    from datetime import datetime, date as _date
    last_played = rows[0][0]
    if isinstance(last_played, str):
        last_played = datetime.strptime(last_played, "%Y-%m-%d").date()
    if ref_date:
        rd = datetime.strptime(ref_date, "%Y-%m-%d").date() if isinstance(ref_date, str) else ref_date
        days_rest = (rd - last_played).days
    else:
        days_rest = (_date.today() - last_played).days
    return {"last5": last5_wp, "last10": last10_wp, "days_rest": days_rest}


def _h2h(home: str, away: str, format_: str, ref_date: str | None) -> dict:
    con = connect()
    rows = con.execute(
        """SELECT
              SUM(CASE WHEN winner = ? THEN 1.0 ELSE 0.0 END) /
                NULLIF(COUNT(*), 0)                          AS home_winpct,
              COUNT(*)                                       AS meetings
           FROM matches
           WHERE format = ?
             AND winner IS NOT NULL
             AND ((team_home = ? AND team_away = ?)
               OR (team_home = ? AND team_away = ?))
             AND (CAST(? AS DATE) IS NULL OR start_date < CAST(? AS DATE))""",
        [home, format_, home, away, away, home, ref_date, ref_date],
    ).fetchone()
    con.close()
    return {"h2h_home_winpct": row_or(rows, 0), "h2h_meetings": row_or(rows, 1)}


def row_or(row, idx, default=None):
    if not row:
        return default
    v = row[idx]
    return v if v is not None else default


def predict_match(
    home: str, away: str, venue: str,
    format_: str = "T20",
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    ref_date: str | None = None,
) -> dict:
    L = _load()
    booster = L["booster"]; iso = L["iso"]; meta = L["meta"]

    home_form = _team_form(home, format_, ref_date)
    away_form = _team_form(away, format_, ref_date)
    h2h       = _h2h(home, away, format_, ref_date)
    venue_p   = _venue_lookup(venue, format_)

    state = {
        "format":             format_,
        "team_home":          home,
        "team_away":          away,
        "venue":              venue,
        "toss_winner_is_home": 1 if toss_winner == home else (0 if toss_winner else 0),
        "toss_decision_is_bat": 1 if (toss_decision and toss_decision.lower().startswith("bat")) else 0,
        "home_last5":         home_form["last5"],
        "home_last10":        home_form["last10"],
        "away_last5":         away_form["last5"],
        "away_last10":        away_form["last10"],
        "home_days_rest":     home_form["days_rest"],
        "away_days_rest":     away_form["days_rest"],
        "h2h_home_winpct":    h2h["h2h_home_winpct"],
        "h2h_meetings":       h2h["h2h_meetings"] or 0,
        "venue_avg_first_innings":   venue_p["venue_avg_first_innings"],
        "venue_toss_winner_won_pct": venue_p["venue_toss_winner_won_pct"],
        "venue_bat_first_pct":       venue_p["venue_bat_first_pct"],
    }

    feats = meta["feature_order"]
    df = pd.DataFrame([{k: state.get(k) for k in feats}])
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")

    raw = float(booster.predict(df[feats])[0])
    cal = float(C.transform_binary(iso, np.array([raw]))[0]) if iso is not None else raw

    # If toss isn't known, marginalise over (home/away × bat/field) — 4 cases
    # weighted equally. The model has only one toss_winner_is_home flag, so we
    # just blend two predictions with the flag flipped.
    if toss_winner is None:
        df_alt = df.copy()
        df_alt["toss_winner_is_home"] = 1 - df_alt["toss_winner_is_home"]
        raw_alt = float(booster.predict(df_alt[feats])[0])
        cal_alt = float(C.transform_binary(iso, np.array([raw_alt]))[0]) if iso is not None else raw_alt
        cal = 0.5 * (cal + cal_alt)
        raw = 0.5 * (raw + raw_alt)

    return {
        "p_home_wins":        cal,
        "p_away_wins":        1 - cal,
        "raw_p_home_wins":    raw,
        "favored":            home if cal >= 0.5 else away,
        "edge_pct":           round(abs(cal - 0.5) * 200, 1),
        "calibrated":         iso is not None,
        "input_features":     {k: state[k] for k in feats},
        "model":              "match",
    }


def _form_prior(home_last10: float | None, away_last10: float | None) -> float:
    """Bradley-Terry-ish form prior: P(home wins) ≈ home / (home + away).
    Falls back to 0.5 if either is missing."""
    if home_last10 is None or away_last10 is None:
        return 0.5
    if home_last10 + away_last10 == 0:
        return 0.5
    return home_last10 / (home_last10 + away_last10)


def predict_match_ensemble(
    home: str, away: str, venue: str,
    format_: str = "T20",
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    ref_date: str | None = None,
    weights: dict | None = None,
) -> dict:
    """Ensemble: 60% match-model + 25% form prior + 15% h2h prior.

    The match model learns interactions; the priors anchor it on raw signal
    that's directly observable. Final prediction is a convex combination.

    Override `weights` like {"match": 0.5, "form": 0.3, "h2h": 0.2}.
    """
    w = weights or {"match": 0.60, "form": 0.25, "h2h": 0.15}
    s = sum(w.values())
    w = {k: v / s for k, v in w.items()}    # normalise

    match_out = predict_match(home, away, venue, format_, toss_winner, toss_decision, ref_date)
    p_match = match_out["p_home_wins"]

    feats = match_out["input_features"]
    p_form = _form_prior(feats.get("home_last10"), feats.get("away_last10"))
    h2h_v  = feats.get("h2h_home_winpct")
    p_h2h  = h2h_v if h2h_v is not None else 0.5

    p_ens = w["match"] * p_match + w["form"] * p_form + w["h2h"] * p_h2h

    return {
        "p_home_wins":   p_ens,
        "p_away_wins":   1 - p_ens,
        "favored":       home if p_ens >= 0.5 else away,
        "edge_pct":      round(abs(p_ens - 0.5) * 200, 1),
        "components": {
            "match_model":   p_match,
            "form_prior":    p_form,
            "h2h_prior":     p_h2h,
        },
        "weights":       w,
        "input_features": feats,
        "model":         "ensemble",
    }
