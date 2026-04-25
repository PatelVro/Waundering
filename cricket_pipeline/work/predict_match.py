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
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..db import connect, install_views
from .ensemble import _cat_pred, _lgb_pred, _lr_pred, _xgb_pred
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
                          toss_decision: str | None) -> None:
    """Insert a synthetic match row into matches and lineup rows into match_xi.
    Idempotent — wipes old rows for SYN_MATCH_ID first.
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

    # Lineup proxy: each team's MOST RECENT match XI before the prediction date
    for team in (home, away):
        # find that team's most recent prior match in match_xi (for the same team name)
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
            continue
        prior_match_id = recent[0]
        # Copy that XI as-is into the synthetic match
        rows = con.execute("""
            SELECT player FROM match_xi
            WHERE match_id = ? AND team = ?
        """, [prior_match_id, team]).fetchall()
        for (p,) in rows:
            con.execute("""
                INSERT INTO match_xi (start_date, team_home, team_away, venue, team, player, match_id)
                VALUES (CAST(? AS DATE), ?, ?, ?, ?, ?, ?)
            """, [date, home, away, venue, team, p, SYN_MATCH_ID])
        print(f"  Lineup proxy for {team}: {len(rows)} players from {prior_match_id}")

    con.close()


def _train_predict(future_row: pd.Series, df: pd.DataFrame) -> dict:
    # Train on all rows EXCEPT the future, and EXCEPT any rows with start_date >= future date.
    # `df` is the full feature frame including the future row.
    fut_date = future_row["start_date"]
    train_pool = df[(df["start_date"] < fut_date) & df["y_t1_wins"].notna()].copy()
    # split off a calibration slice from the most recent 10% of training
    train_pool = train_pool.sort_values("start_date").reset_index(drop=True)
    n = len(train_pool); n_calib = max(int(round(n * 0.10)), 200)
    train = train_pool.iloc[:n - n_calib].copy()
    calib = train_pool.iloc[n - n_calib:].copy()
    print(f"  Training: train={len(train)}  calib={len(calib)}  (date < {fut_date.date()})")

    feat_num = NUMERIC + PLAYER_NUMERIC
    feat_cat = CATEGORICAL
    for c in feat_cat:
        train[c] = train[c].astype("category"); calib[c] = calib[c].astype("category")

    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()

    fut_df = future_row.to_frame().T.copy()
    # restore numeric dtypes (Series.to_frame().T re-types as object)
    for c in feat_num:
        if c in fut_df.columns:
            fut_df[c] = pd.to_numeric(fut_df[c], errors="coerce")
    for c in feat_cat:
        if c in fut_df.columns:
            fut_df[c] = fut_df[c].astype(str).astype("category")

    print("  Training base learners (LGBM-num, LGBM-cat, XGB, CatBoost, LR) ...")
    lgb_n_ca, lgb_n_te = _lgb_pred(train, calib, fut_df, feat_num, [], ytr, yca)
    lgb_c_ca, lgb_c_te = _lgb_pred(train, calib, fut_df, feat_num + feat_cat, feat_cat, ytr, yca)
    xgb_ca, xgb_te     = _xgb_pred(train, calib, fut_df, feat_num, ytr, yca)
    cat_ca, cat_te     = _cat_pred(train, calib, fut_df, feat_num, feat_cat, ytr, yca)
    lr_ca, lr_te       = _lr_pred(train, calib, fut_df, feat_num, ytr, yca)

    Xst_ca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, cat_ca, lr_ca])
    Xst_te = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, cat_te, lr_te])

    from sklearn.linear_model import LogisticRegression
    stk = LogisticRegression(C=10.0)
    stk.fit(Xst_ca, yca)
    p_ens = float(stk.predict_proba(Xst_te)[0, 1])

    return {
        "lgbm_num": float(lgb_n_te[0]),
        "lgbm_cat": float(lgb_c_te[0]),
        "xgb":      float(xgb_te[0]),
        "cat":      float(cat_te[0]),
        "lr":       float(lr_te[0]),
        "ensemble": p_ens,
    }


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
    args = ap.parse_args()

    print(f"\n=== Predicting {args.home} vs {args.away} on {args.date} ===")
    print(f"Venue: {args.venue}")
    print(f"Format: {args.fmt}")
    if args.toss_winner:
        print(f"Toss: {args.toss_winner} chose to {args.toss_decision}")

    # Step 1: insert synthetic match + XI proxy
    _seed_synthetic_match(args.home, args.away, args.venue, args.fmt, args.date,
                          args.toss_winner, args.toss_decision)

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
    result = _train_predict(fut_row, df)

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
        print(f"    {k:<10} {result[k]:.3f}")

    # confidence label
    edge = abs(p_home - 0.5) * 2
    if edge < 0.2:    label = "low confidence (toss-up)"
    elif edge < 0.4:  label = "lean"
    elif edge < 0.6:  label = "favorite"
    else:             label = "strong favorite"
    print(f"\n  Confidence: {label}  (edge = {edge*100:.1f}%)")

    # cleanup synthetic rows
    con = connect()
    con.execute("DELETE FROM matches  WHERE match_id = ?", [SYN_MATCH_ID])
    con.execute("DELETE FROM match_xi WHERE match_id = ?", [SYN_MATCH_ID])
    con.close()

    # save prediction to file (JSON)
    if args.save:
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
            "base_learners": {k: round(v, 4) for k, v in result.items()},
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
        }
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved prediction → {out_path}")


if __name__ == "__main__":
    main()
