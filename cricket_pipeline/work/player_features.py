"""Per-match team-aggregate player features.

For each (match, team) compute:
  - team_n_batters
  - team_avg_career_sr        (mean over team's batters who batted in this match)
  - team_avg_form_sr          (mean over team's batters; rolling last-10)
  - team_avg_career_avg
  - team_n_bowlers
  - team_avg_career_econ
  - team_avg_career_bowl_avg

These use v_batter_history / v_bowler_history (already leakage-safe — they
exclude the current match's contribution).

Important caveat: this requires knowing the XIs (which we get from balls).
For training/eval on past matches that's fine. For prediction on a future
match, the caller must pass the announced XIs separately.
"""
from __future__ import annotations

import pandas as pd

from ..db import connect, install_views


def compute_team_player_features(top_bat: int = 7, top_bowl: int = 5) -> pd.DataFrame:
    """Team-level player aggregates per match, using the announced XI from match_xi.

    Removes leakage from "batters who actually batted" which depends on outcome.
    The XI (announced at toss) is fixed pre-match.

    For each (match, team)'s 11 in match_xi:
      - Rank by prior career_balls (batting), keep top `top_bat` as the batting unit.
      - Rank by prior career_balls (bowling), keep top `top_bowl` as the bowling unit.
      - Average career_sr / form_sr / career_avg over the batting unit.
      - Average career_econ / career_avg over the bowling unit.
    """
    install_views()
    con = connect()

    bat_sql = f"""
    WITH xi_with_bat AS (
        SELECT
            x.match_id, x.team, x.player,
            vb.career_sr, vb.form_sr, vb.career_avg, vb.career_balls
        FROM match_xi x
        LEFT JOIN v_batter_history vb
          ON vb.batter   = x.player
         AND vb.match_id = x.match_id
        WHERE x.match_id IS NOT NULL
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY match_id, team
                   ORDER BY COALESCE(career_balls, 0) DESC
               ) AS rk
        FROM xi_with_bat
    )
    SELECT
        match_id, team,
        AVG(career_sr)        AS team_bat_career_sr,
        AVG(form_sr)          AS team_bat_form_sr,
        AVG(career_avg)       AS team_bat_career_avg,
        AVG(career_balls)     AS team_bat_career_balls_avg,
        COUNT(*)              AS team_n_batters_used
    FROM ranked
    WHERE rk <= {int(top_bat)}
    GROUP BY match_id, team
    """
    bat = con.execute(bat_sql).df()

    bowl_sql = f"""
    WITH xi_with_bowl AS (
        SELECT
            x.match_id, x.team, x.player,
            vw.career_econ, vw.career_avg, vw.career_balls
        FROM match_xi x
        LEFT JOIN v_bowler_history vw
          ON vw.bowler   = x.player
         AND vw.match_id = x.match_id
        WHERE x.match_id IS NOT NULL
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY match_id, team
                   ORDER BY COALESCE(career_balls, 0) DESC
               ) AS rk
        FROM xi_with_bowl
    )
    SELECT
        match_id, team,
        AVG(career_econ)      AS team_bowl_career_econ,
        AVG(career_avg)       AS team_bowl_career_avg,
        AVG(career_balls)     AS team_bowl_career_balls_avg,
        COUNT(*)              AS team_n_bowlers_used
    FROM ranked
    WHERE rk <= {int(top_bowl)}
    GROUP BY match_id, team
    """
    bowl = con.execute(bowl_sql).df()
    con.close()

    out = bat.merge(bowl, on=["match_id", "team"], how="outer")
    return out


PLAYER_FEATURE_NUMERIC_T1 = [
    "t1_bat_career_sr", "t1_bat_form_sr", "t1_bat_career_avg",
    "t1_bowl_career_econ", "t1_bowl_career_avg",
    "t1_n_batters", "t1_n_bowlers",
]
PLAYER_FEATURE_NUMERIC_T2 = [
    "t2_bat_career_sr", "t2_bat_form_sr", "t2_bat_career_avg",
    "t2_bowl_career_econ", "t2_bowl_career_avg",
    "t2_n_batters", "t2_n_bowlers",
]
PLAYER_FEATURE_DIFFS = [
    "diff_bat_career_sr", "diff_bat_form_sr", "diff_bowl_career_econ",
]


def attach_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the t1_/t2_ player-aggregate features to a match-feature frame."""
    pf = compute_team_player_features()
    rename1 = {
        "team":                       "team_home",
        "team_bat_career_sr":         "t1_bat_career_sr",
        "team_bat_form_sr":           "t1_bat_form_sr",
        "team_bat_career_avg":        "t1_bat_career_avg",
        "team_bowl_career_econ":      "t1_bowl_career_econ",
        "team_bowl_career_avg":       "t1_bowl_career_avg",
        "team_n_batters_used":        "t1_n_batters_used",
        "team_n_bowlers_used":        "t1_n_bowlers_used",
    }
    rename2 = {
        "team":                       "team_away",
        "team_bat_career_sr":         "t2_bat_career_sr",
        "team_bat_form_sr":           "t2_bat_form_sr",
        "team_bat_career_avg":        "t2_bat_career_avg",
        "team_bowl_career_econ":      "t2_bowl_career_econ",
        "team_bowl_career_avg":       "t2_bowl_career_avg",
        "team_n_batters_used":        "t2_n_batters_used",
        "team_n_bowlers_used":        "t2_n_bowlers_used",
    }
    base_cols = ["match_id", "team",
                 "team_bat_career_sr", "team_bat_form_sr", "team_bat_career_avg",
                 "team_bowl_career_econ", "team_bowl_career_avg",
                 "team_n_batters_used", "team_n_bowlers_used"]
    pf1 = pf[base_cols].rename(columns=rename1)
    pf2 = pf[base_cols].rename(columns=rename2)

    df = df.merge(pf1, on=["match_id", "team_home"], how="left")
    df = df.merge(pf2, on=["match_id", "team_away"], how="left")

    df["diff_bat_career_sr"]   = df["t1_bat_career_sr"]   - df["t2_bat_career_sr"]
    df["diff_bat_form_sr"]     = df["t1_bat_form_sr"]     - df["t2_bat_form_sr"]
    df["diff_bowl_career_econ"] = df["t1_bowl_career_econ"] - df["t2_bowl_career_econ"]
    return df


__all__ = [
    "compute_team_player_features",
    "attach_player_features",
    "PLAYER_FEATURE_NUMERIC_T1",
    "PLAYER_FEATURE_NUMERIC_T2",
    "PLAYER_FEATURE_DIFFS",
]
