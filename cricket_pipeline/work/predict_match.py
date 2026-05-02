"""Predict an upcoming match using the trained stacked ensemble.

Steps:
  1. Load all matches in DB.
  2. Insert a synthetic "future" match row for the upcoming fixture (with today's date,
     no winner). This forces the feature pipeline to compute pre-match features for it.
  3. Use each team's most-recent match XI as a proxy lineup for the new match
     (insert into match_xi).
  4. Build features. Train ensemble on all matches that have winner != None
     and start_date < the future match.
  5. Predict the future match's row.
  6. Print a probability + headline.

Usage:
  python -m cricket_pipeline.work.predict_match \\
      --home "Rajasthan Royals" --away "Sunrisers Hyderabad" \\
      --venue "Sawai Mansingh Stadium, Jaipur" --format T20 --date 2026-04-25
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..db import connect, install_views
from .ensemble import (DEFAULT_RECENCY_HL_DAYS, _cat_pred, _lgb_pred, _lr_pred,
                        _xgb_pred, recency_weights)
from .eval import lgb_params
from .features_v2 import (CATEGORICAL, NUMERIC, PLAYER_NUMERIC,
                           build_features_with_players)


SYN_MATCH_ID = "PREDICTION_FUTURE_MATCH"


def _patch_player_features_for_future(df: pd.DataFrame, home: str, away: str, date: str,
                                       top_bat: int = 7, top_bowl: int = 5) -> None:
    """For the synthetic future match row, compute player aggregates by looking
    up each XI member's MOST RECENT v_batter_history / v_bowler_history row
    strictly before `date` (DuckDB-side query). Mutates df in place."""
    con = connect()
    install_views()
    cutoff = pd.to_datetime(date)

    def _team_aggs(team: str) -> dict:
        # XI from the synthetic match
        players = [r[0] for r in con.execute("""
            SELECT player FROM match_xi
            WHERE match_id = ? AND team = ?
        """, [SYN_MATCH_ID, team]).fetchall()]
        if not players:
            return {}
        # batting career as-of date (latest row strictly before cutoff)
        bat = con.execute(f"""
            WITH ranked AS (
                SELECT batter, career_sr, form_sr, career_avg, career_balls,
                       ROW_NUMBER() OVER (PARTITION BY batter ORDER BY start_date DESC, match_id DESC) AS rk
                FROM v_batter_history
                WHERE batter IN ({','.join(['?']*len(players))})
                  AND start_date < CAST(? AS DATE)
            ),
            top_n AS (
                SELECT * FROM ranked WHERE rk = 1
                ORDER BY COALESCE(career_balls, 0) DESC
                LIMIT {int(top_bat)}
            )
            SELECT AVG(career_sr), AVG(form_sr), AVG(career_avg), AVG(career_balls)
            FROM top_n
        """, players + [date]).fetchone()
        bowl = con.execute(f"""
            WITH ranked AS (
                SELECT bowler, career_econ, career_avg, career_balls,
                       ROW_NUMBER() OVER (PARTITION BY bowler ORDER BY start_date DESC, match_id DESC) AS rk
                FROM v_bowler_history
                WHERE bowler IN ({','.join(['?']*len(players))})
                  AND start_date < CAST(? AS DATE)
            ),
            top_n AS (
                SELECT * FROM ranked WHERE rk = 1
                ORDER BY COALESCE(career_balls, 0) DESC
                LIMIT {int(top_bowl)}
            )
            SELECT AVG(career_econ), AVG(career_avg), AVG(career_balls)
            FROM top_n
        """, players + [date]).fetchone()
        # fetchone() returns None when the windowed query has no rows; treat
        # that as "no data" instead of crashing on tuple-indexing.
        bat = bat or (None, None, None, None)
        bowl = bowl or (None, None, None)
        return {
            "bat_career_sr":   bat[0], "bat_form_sr": bat[1], "bat_career_avg": bat[2],
            "bowl_career_econ": bowl[0], "bowl_career_avg": bowl[1],
        }

    h = _team_aggs(home); a = _team_aggs(away)
    con.close()

    mask = df["match_id"] == SYN_MATCH_ID
    for k in ("bat_career_sr", "bat_form_sr", "bat_career_avg",
              "bowl_career_econ", "bowl_career_avg"):
        df.loc[mask, f"t1_{k}"] = h.get(k)
        df.loc[mask, f"t2_{k}"] = a.get(k)
    if h.get("bat_career_sr") is not None and a.get("bat_career_sr") is not None:
        df.loc[mask, "diff_bat_career_sr"]   = h["bat_career_sr"] - a["bat_career_sr"]
        df.loc[mask, "diff_bat_form_sr"]     = (h.get("bat_form_sr") or 0) - (a.get("bat_form_sr") or 0)
        df.loc[mask, "diff_bowl_career_econ"] = (h.get("bowl_career_econ") or 0) - (a.get("bowl_career_econ") or 0)


def _seed_synthetic_match(home: str, away: str, venue: str, fmt: str,
                          date: str, toss_winner: str | None,
                          toss_decision: str | None,
                          xi_home: list[str] | None = None,
                          xi_away: list[str] | None = None) -> dict:
    """Insert a synthetic match row into matches and lineup rows into match_xi.
    Idempotent — wipes old rows for SYN_MATCH_ID first.

    If xi_home / xi_away are provided (announced XIs), use them directly.
    Otherwise fall back to each team's most recent prior XI as a proxy.
    Returns metadata about which path was used for each side.
    """
    con = connect()
    install_views()
    con.execute("DELETE FROM matches  WHERE match_id = ?", [SYN_MATCH_ID])
    con.execute("DELETE FROM match_xi WHERE match_id = ?", [SYN_MATCH_ID])

    # Insert the upcoming fixture
    con.execute("""
        INSERT INTO matches (match_id, format, competition, season, start_date,
                             venue, country, team_home, team_away,
                             toss_winner, toss_decision, winner)
        VALUES (?, ?, NULL, NULL, CAST(? AS DATE), ?, NULL, ?, ?, ?, ?, NULL)
    """, [SYN_MATCH_ID, fmt, date, venue, home, away, toss_winner, toss_decision])

    # Per-side: if explicit XI supplied → use it; else fall back to proxy
    explicit = {home: xi_home, away: xi_away}
    used = {}
    for team in (home, away):
        if explicit[team] and len(explicit[team]) >= 5:
            xi = list(dict.fromkeys(explicit[team]))   # dedupe, preserve order
            for p in xi:
                con.execute("""
                    INSERT INTO match_xi (start_date, team_home, team_away, venue, team, player, match_id)
                    VALUES (CAST(? AS DATE), ?, ?, ?, ?, ?, ?)
                """, [date, home, away, venue, team, p, SYN_MATCH_ID])
            used[team] = {"source": "announced", "n": len(xi)}
            print(f"  XI for {team}: {len(xi)} players (announced)")
            continue

        # Proxy: each team's MOST RECENT match XI before the prediction date
        recent = con.execute("""
            SELECT match_id, MAX(start_date) AS dt
            FROM match_xi
            WHERE team = ? AND start_date < CAST(? AS DATE)
              AND match_id IS NOT NULL
            GROUP BY match_id
            ORDER BY dt DESC
            LIMIT 1
        """, [team, date]).fetchone()
        if not recent:
            print(f"  WARN: no prior XI found for {team}; player-aggregate features will be empty.", file=sys.stderr)
            used[team] = {"source": "missing", "n": 0}
            continue
        prior_match_id = recent[0]
        rows = con.execute("""
            SELECT player FROM match_xi
            WHERE match_id = ? AND team = ?
        """, [prior_match_id, team]).fetchall()
        for (p,) in rows:
            con.execute("""
                INSERT INTO match_xi (start_date, team_home, team_away, venue, team, player, match_id)
                VALUES (CAST(? AS DATE), ?, ?, ?, ?, ?, ?)
            """, [date, home, away, venue, team, p, SYN_MATCH_ID])
        used[team] = {"source": "proxy", "n": len(rows), "from_match_id": prior_match_id}
        print(f"  XI for {team}: {len(rows)} players (proxy from {prior_match_id})")

    con.close()
    return used


def _train_predict(future_row: pd.Series, df: pd.DataFrame, fast: bool = False) -> dict:
    # Train on all rows EXCEPT the future, and EXCEPT any rows with start_date >= future date.
    # `df` is the full feature frame including the future row.
    fut_date = future_row["start_date"]
    train_pool = df[(df["start_date"] < fut_date) & df["y_t1_wins"].notna()].copy()
    if train_pool.empty:
        raise ValueError(
            f"Insufficient training history: no completed matches with winner before {fut_date}. "
            "Check that the cricsheet ingest has populated `matches.winner` for prior fixtures."
        )
    # split off a calibration slice from the most recent 10% of training
    train_pool = train_pool.sort_values("start_date").reset_index(drop=True)
    n = len(train_pool); n_calib = max(int(round(n * 0.10)), 200)
    if n <= n_calib:
        raise ValueError(
            f"Training pool too small: n={n}, but need at least {n_calib} for calibration. "
            "Provide more historical data or reduce the calibration size."
        )
    train = train_pool.iloc[:n - n_calib].copy()
    calib = train_pool.iloc[n - n_calib:].copy()
    print(f"  Training: train={len(train)}  calib={len(calib)}  (date < {fut_date.date()})")

    # Per-format feature set:
    #   ODI  → home-advantage extras (Cycle 8: subcontinent home wins dominate misses)
    #   T20  → weather extras (Cycle 12: dew + temp + humidity lift +0.26pp, ODI hurt)
    fmt = future_row.get("format", "")
    is_odi = (str(fmt).upper() == "ODI")
    is_t20 = str(fmt).upper() in ("T20", "IT20")
    odi_extras: list[str] = []
    t20_extras: list[str] = []
    if is_odi:
        try:
            from .odi_model import ODI_EXTRA_NUMERIC
            odi_extras = [c for c in ODI_EXTRA_NUMERIC if c in train.columns]
        except Exception:
            pass
    if is_t20:
        try:
            from .features_v2 import WEATHER_NUMERIC
            t20_extras = [c for c in WEATHER_NUMERIC if c in train.columns]
        except Exception:
            pass
    feat_num = NUMERIC + PLAYER_NUMERIC + odi_extras + t20_extras
    feat_cat = CATEGORICAL
    for c in feat_cat:
        train[c] = train[c].astype("category"); calib[c] = calib[c].astype("category")
    if is_odi and odi_extras:
        print(f"  ODI build: +{len(odi_extras)} home-advantage features ({', '.join(odi_extras)})")
    if is_t20 and t20_extras:
        print(f"  T20 build: +{len(t20_extras)} weather features ({', '.join(t20_extras)})")

    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()

    # Recency-weighted training (Cycle 7): half-Kelly toward 24-month half-life.
    # Lifts T20 +1.15pp acc and ODI +2.7pp acc with better Brier/ECE on ODI.
    w = recency_weights(train["start_date"], DEFAULT_RECENCY_HL_DAYS)
    if w.sum() < 1e-6:
        raise ValueError(
            f"All training weights collapsed to ~0 (sum={w.sum():.2e}). "
            "Future date may be too far past the recency horizon, or training "
            "set is empty after filtering. Check fut_date and recency half-life."
        )
    # Guard against div-by-zero when all weights collapse to a single non-zero value
    w_sq_sum = float((w ** 2).sum())
    ess = (w.sum() ** 2) / max(w_sq_sum, 1e-12)
    print(f"  Recency weighting: hl={DEFAULT_RECENCY_HL_DAYS}d  effective N={ess:.0f}/{len(w)}")

    fut_df = future_row.to_frame().T.copy()
    # restore numeric dtypes (Series.to_frame().T re-types as object)
    for c in feat_num:
        if c in fut_df.columns:
            fut_df[c] = pd.to_numeric(fut_df[c], errors="coerce")
    for c in feat_cat:
        if c in fut_df.columns:
            fut_df[c] = fut_df[c].astype(str).astype("category")

    # Route LGBM-num through ODI-tuned params when format is ODI
    if is_odi:
        try:
            from .odi_model import _lgb_pred_with_params, odi_lgb_params
            _lgb_num = lambda *a, **kw: _lgb_pred_with_params(*a, **kw, params=odi_lgb_params())
            print(f"  ODI build: using Optuna-tuned LGBM-num params")
        except Exception:
            _lgb_num = _lgb_pred
    else:
        _lgb_num = _lgb_pred

    if fast:
        # Fast path: 1 seed of LGBM (numeric+cats) + 1 seed of XGB + LR.
        # No CatBoost. ~15-30s vs ~3-15min for the full ensemble.
        print("  Training FAST base learners (LGBM-num, LGBM-cat, XGB, LR) ...")
        seeds = (42,)
        lgb_n_ca, lgb_n_te = _lgb_num(train, calib, fut_df, feat_num, [], ytr, yca, seeds=seeds, weights=w)
        lgb_c_ca, lgb_c_te = _lgb_pred(train, calib, fut_df, feat_num + feat_cat, feat_cat, ytr, yca, seeds=seeds, weights=w)
        xgb_ca, xgb_te     = _xgb_pred(train, calib, fut_df, feat_num, ytr, yca, seeds=seeds, weights=w)
        lr_ca, lr_te       = _lr_pred(train, calib, fut_df, feat_num, ytr, yca, weights=w)
        Xst_ca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, lr_ca])
        Xst_te = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, lr_te])
    else:
        print("  Training base learners (LGBM-num, LGBM-cat, XGB, CatBoost, LR) ...")
        lgb_n_ca, lgb_n_te = _lgb_num(train, calib, fut_df, feat_num, [], ytr, yca, weights=w)
        lgb_c_ca, lgb_c_te = _lgb_pred(train, calib, fut_df, feat_num + feat_cat, feat_cat, ytr, yca, weights=w)
        xgb_ca, xgb_te     = _xgb_pred(train, calib, fut_df, feat_num, ytr, yca, weights=w)
        cat_ca, cat_te     = _cat_pred(train, calib, fut_df, feat_num, feat_cat, ytr, yca, weights=w)
        lr_ca, lr_te       = _lr_pred(train, calib, fut_df, feat_num, ytr, yca, weights=w)
        Xst_ca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, cat_ca, lr_ca])
        Xst_te = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, cat_te, lr_te])

    from sklearn.linear_model import LogisticRegression
    stk = LogisticRegression(C=10.0)
    stk.fit(Xst_ca, yca)
    p_ens_raw = float(stk.predict_proba(Xst_te)[0, 1])

    # Per-tier isotonic calibration (Wave 1, Agent 31). When a saved tier
    # calibrator exists for this format, apply it to the ensemble output.
    # Tier-1 was under-confident in the 60-80% band by ~9pp; tier-2-main
    # ECE was ~28%. Per-tier isotonic collapses both. Falls back to raw
    # probability if no calibrator is found, so unknown formats still ship.
    p_ens = _maybe_apply_tier_calibrator(p_ens_raw, future_row, fmt)

    return {
        "lgbm_num":      float(lgb_n_te[0]),
        "lgbm_cat":      float(lgb_c_te[0]),
        "xgb":           float(xgb_te[0]),
        "cat":           (None if fast else float(cat_te[0])),
        "lr":            float(lr_te[0]),
        "ensemble_raw":  p_ens_raw,
        "ensemble":      p_ens,
    }


_TIER_CALIBRATOR_CACHE: dict[str, object] = {}


def _maybe_apply_tier_calibrator(p: float, future_row: pd.Series, fmt: str) -> float:
    """Look up a per-tier isotonic calibrator artifact and apply it.

    Returns the calibrated probability, or `p` unchanged if the artifact
    isn't present. Cached across calls — calibrator load is ~10 KB.
    """
    fmt_norm = (str(fmt) or "").upper()
    tag = "t20" if fmt_norm in ("T20", "IT20") else ("odi" if fmt_norm == "ODI" else None)
    if tag is None:
        return p
    cal_path = Path(__file__).resolve().parent / "runs" / "calibrators" / f"{tag}_per_tier_iso.joblib"
    if not cal_path.exists():
        return p
    try:
        if tag not in _TIER_CALIBRATOR_CACHE:
            import joblib
            _TIER_CALIBRATOR_CACHE[tag] = joblib.load(cal_path)
        artifact = _TIER_CALIBRATOR_CACHE[tag]
        # Lazy import to keep error_analysis_v2 out of cold-start path
        from .error_analysis_v2 import classify_tier
        tier = classify_tier(future_row.get("competition"),
                              future_row.get("team_home"),
                              future_row.get("team_away"))
        calibrators = artifact.get("calibrators", {})
        iso = calibrators.get(tier) or calibrators.get("_global")
        if iso is None:
            return p
        return float(iso.transform([p])[0])
    except Exception as e:
        # Calibration is a quality lift, not a hard requirement. Log and
        # fall back to raw probability rather than crashing the prediction.
        print(f"  WARN: tier calibrator load/apply failed ({e}); using raw probability", file=sys.stderr)
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home",     required=True)
    ap.add_argument("--away",     required=True)
    ap.add_argument("--venue",    required=True)
    ap.add_argument("--format",   default="T20", dest="fmt")
    ap.add_argument("--date",     required=True, help="YYYY-MM-DD")
    ap.add_argument("--toss-winner",   default=None)
    ap.add_argument("--toss-decision", default=None, choices=[None, "bat", "field"])
    ap.add_argument("--save",          default=None,
                    help="Write prediction to this JSON file (e.g. 'RR_vs_SRH_25_April_2026.json')")
    ap.add_argument("--fast",          action="store_true",
                    help="Use the lighter ensemble (no CatBoost, single seed). 30s vs 5-15min.")
    ap.add_argument("--xi-home",       default=None,
                    help="Comma-separated XI for the home team (overrides proxy lookup).")
    ap.add_argument("--xi-away",       default=None,
                    help="Comma-separated XI for the away team (overrides proxy lookup).")
    ap.add_argument("--force",         action="store_true",
                    help="Overwrite the --save file even if it already exists.")
    args = ap.parse_args()

    # Strict ISO date parse — pandas auto-corrects '2026-02-30' to '2026-03-02'
    # silently, which then poisons SQL date filters. Reject malformed inputs.
    try:
        parsed_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"ERROR: --date {args.date!r} is not a valid YYYY-MM-DD date: {e}", file=sys.stderr)
        sys.exit(2)
    if not (1990 < parsed_date.year < 2100):
        print(f"ERROR: --date year {parsed_date.year} outside reasonable range [1990, 2100]", file=sys.stderr)
        sys.exit(2)

    xi_home = [s.strip() for s in (args.xi_home or "").split(",") if s.strip()] or None
    xi_away = [s.strip() for s in (args.xi_away or "").split(",") if s.strip()] or None

    # XI length sanity check — cricket teams are 11 + reserves; reject obvious
    # garbage (1-player, 50-player). Lower bound 5 keeps the "proxy XI fallback"
    # path open since the helper accepts >=5 players.
    for label, xi in (("home", xi_home), ("away", xi_away)):
        if xi is not None and not (5 <= len(xi) <= 15):
            print(f"ERROR: --xi-{label} must be 5-15 players, got {len(xi)}", file=sys.stderr)
            sys.exit(2)

    if args.save and not args.force and Path(args.save).exists():
        print(f"Prediction already exists at {args.save} (use --force to overwrite). Exiting.")
        sys.exit(0)

    print(f"\n=== Predicting {args.home} vs {args.away} on {args.date} ===")
    print(f"Venue: {args.venue}")
    print(f"Format: {args.fmt}")
    if args.toss_winner:
        print(f"Toss: {args.toss_winner} chose to {args.toss_decision}")

    # Step 1: insert synthetic match + XI (announced if available, else proxy).
    # Wrap the rest of the function in try/finally so a crash (during feature
    # build, training, etc.) still cleans up the synthetic rows. Without this,
    # SYN_MATCH_ID lingers in `matches` with no winner and pollutes downstream
    # Elo / feature recomputation runs.
    xi_meta = _seed_synthetic_match(args.home, args.away, args.venue, args.fmt, args.date,
                                     args.toss_winner, args.toss_decision,
                                     xi_home=xi_home, xi_away=xi_away)
    try:
        return _run_prediction(args, xi_meta)
    finally:
        try:
            con = connect()
            con.execute("DELETE FROM matches  WHERE match_id = ?", [SYN_MATCH_ID])
            con.execute("DELETE FROM match_xi WHERE match_id = ?", [SYN_MATCH_ID])
            con.close()
        except Exception as cleanup_err:
            print(f"  WARN: synthetic-match cleanup failed: {cleanup_err}", file=sys.stderr)


def _run_prediction(args, xi_meta):
    # Step 2: build features (now includes the synthetic row)
    print("\nBuilding features ...")
    df = build_features_with_players(format_filter=None, keep_unfinished=True)
    fut = df[df["match_id"] == SYN_MATCH_ID]
    if fut.empty:
        print("ERROR: synthetic match row not found in features.", file=sys.stderr)
        sys.exit(1)
    print(f"  Synthetic match feature row found ({len(df)} matches in feature frame).")

    # Player aggregates won't compute via v_batter_history for the synthetic match
    # (no balls). Patch them by using each player's most-recent history row strictly
    # before `args.date`.
    _patch_player_features_for_future(df, args.home, args.away, args.date)
    fut_row = df[df["match_id"] == SYN_MATCH_ID].iloc[0]
    print()

    # echo a few key features so the user can sanity check
    print("Pre-match feature snapshot (team1 = home perspective):")
    for k in ["t1_elo_pre", "t2_elo_pre", "elo_diff_pre",
              "t1_last5", "t2_last5", "t1_last10", "t2_last10",
              "h2h_t1_winpct", "h2h_n_prior",
              "venue_avg_first", "venue_bat1_winrate", "venue_toss_winner_winpct",
              "t1_bat_career_avg", "t2_bat_career_avg",
              "t1_bat_form_sr", "t2_bat_form_sr",
              "t1_bowl_career_econ", "t2_bowl_career_econ"]:
        v = fut_row.get(k, np.nan)
        if pd.isna(v):
            print(f"  {k:<28} NA")
        else:
            print(f"  {k:<28} {float(v):.3f}")
    print()

    # Step 3-5: train + predict
    result = _train_predict(fut_row, df, fast=args.fast)

    print("\n=== PREDICTION ===")
    p_home = result["ensemble"]
    p_away = 1 - p_home
    fav, p_fav = (args.home, p_home) if p_home >= 0.5 else (args.away, p_away)
    print(f"  P({args.home} wins) = {p_home:.3f}  ({p_home*100:.1f}%)")
    print(f"  P({args.away} wins) = {p_away:.3f}  ({p_away*100:.1f}%)")
    print(f"  Favored: {fav}  ({p_fav*100:.1f}%)")
    print()
    print("  Per-base-learner P(home wins):")
    for k in ["lgbm_num", "lgbm_cat", "xgb", "cat", "lr", "ensemble"]:
        v = result.get(k)
        print(f"    {k:<10} {'(skipped)' if v is None else f'{v:.3f}'}")

    # confidence label
    edge = abs(p_home - 0.5) * 2
    if edge < 0.2:    label = "low confidence (toss-up)"
    elif edge < 0.4:  label = "lean"
    elif edge < 0.6:  label = "favorite"
    else:             label = "strong favorite"
    print(f"\n  Confidence: {label}  (edge = {edge*100:.1f}%)")

    # synthetic-row cleanup is handled by main()'s try/finally; do not
    # duplicate it here so a save-failure still exits cleanly.

    # save prediction to file (JSON), with odds + totals decoration
    if args.save:
        try:
            from .odds_features import attach_odds_to_prediction
        except Exception:
            attach_odds_to_prediction = lambda p: p
        try:
            from .totals_model import predict_total, over_under_probs
        except Exception:
            predict_total = None
        try:
            from .top_batsman import predict_team_top_scorer
        except Exception:
            predict_team_top_scorer = None
        out = {
            "match": {
                "home":   args.home,
                "away":   args.away,
                "venue":  args.venue,
                "format": args.fmt,
                "date":   args.date,
                "toss_winner":   args.toss_winner,
                "toss_decision": args.toss_decision,
            },
            "prediction": {
                "p_home_wins":     p_home,
                "p_away_wins":     p_away,
                "favored":         fav,
                "favored_pct":     round(p_fav * 100, 1),
                "edge_pct":        round(edge * 100, 1),
                "confidence_label": label,
            },
            "base_learners": {k: (None if v is None else round(v, 4))
                              for k, v in result.items()},
            "features": {k: (None if pd.isna(fut_row.get(k, np.nan)) else float(fut_row[k]))
                         for k in ["t1_elo_pre", "t2_elo_pre", "elo_diff_pre",
                                   "t1_last5", "t2_last5", "t1_last10", "t2_last10",
                                   "h2h_t1_winpct", "h2h_n_prior",
                                   "venue_avg_first", "venue_bat1_winrate",
                                   "venue_toss_winner_winpct",
                                   "t1_bat_career_avg", "t2_bat_career_avg",
                                   "t1_bat_form_sr", "t2_bat_form_sr",
                                   "t1_bowl_career_econ", "t2_bowl_career_econ"]
                         if k in fut_row.index},
            "model": {
                "type":         "stacked_lr_ensemble",
                "base_models":  ["lgbm_num", "lgbm_cat", "xgb", "cat", "lr"],
                "trained_on":   "all CricSheet matches with start_date < " + args.date,
            },
            "xi": {
                args.home: xi_meta.get(args.home, {}),
                args.away: xi_meta.get(args.away, {}),
                "any_announced": any((xi_meta.get(t) or {}).get("source") == "announced"
                                      for t in (args.home, args.away)),
                "all_announced": all((xi_meta.get(t) or {}).get("source") == "announced"
                                      for t in (args.home, args.away)),
            },
        }
        # Totals: predict 1st-innings score for each batting-first scenario
        if predict_total is not None:
            try:
                # try both batting orders since toss may not be known
                t1_bat = predict_total(args.home, args.away, args.venue, args.fmt, args.date)
                t2_bat = predict_total(args.away, args.home, args.venue, args.fmt, args.date)
                # prefer the scenario consistent with toss when known
                if args.toss_winner and args.toss_decision:
                    bat_first = (args.toss_winner if args.toss_decision == "bat"
                                 else (args.away if args.toss_winner == args.home else args.home))
                else:
                    bat_first = None
                out["totals"] = {
                    "first_innings_p10":     None,
                    "first_innings_p50":     None,
                    "first_innings_p90":     None,
                    "scenarios": {
                        f"{args.home}_bat_first": t1_bat,
                        f"{args.away}_bat_first": t2_bat,
                    },
                    "bat_first_assumed":     bat_first,
                }
                # featured scenario = consistent with assumed bat-first; else mean of both
                feat = (t1_bat if bat_first == args.home else
                        t2_bat if bat_first == args.away else
                        {q: (t1_bat[q] + t2_bat[q]) / 2 for q in ("p10","p50","p90","spread")})
                out["totals"]["first_innings_p10"] = feat["p10"]
                out["totals"]["first_innings_p50"] = feat["p50"]
                out["totals"]["first_innings_p90"] = feat["p90"]
                # standard market lines around the median
                med = feat["p50"]
                lines = [round(med - 20), round(med - 10), round(med),
                         round(med + 10), round(med + 20)]
                out["totals"]["over_under_lines"] = [over_under_probs(feat, line=L) for L in lines]
            except Exception as e:
                print(f"  (totals decoration skipped: {e})")
        # Top-scorer per team
        if predict_team_top_scorer is not None:
            try:
                out["top_scorer"] = {
                    args.home: predict_team_top_scorer(args.home, args.away,
                                                       args.venue, args.fmt, args.date)[:5],
                    args.away: predict_team_top_scorer(args.away, args.home,
                                                       args.venue, args.fmt, args.date)[:5],
                }
            except Exception as e:
                print(f"  (top-scorer decoration skipped: {e})")
        # Decorate with bookmaker odds + edge if we have any odds for this match
        try:
            out = attach_odds_to_prediction(out)
        except Exception as e:
            print(f"  (odds decoration skipped: {e})")
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved prediction → {out_path}")
        mvb = out.get("model_vs_book") or {}
        if mvb:
            print(f"  Book consensus: P(home)={out['odds']['h2h']['consensus']['p_home']*100:.1f}%  "
                  f"edge_home={mvb['edge_home_pp']:+.1f}pp  "
                  f"edge_away={mvb['edge_away_pp']:+.1f}pp  "
                  f"value_bet={mvb['value_bet']}")


if __name__ == "__main__":
    main()
