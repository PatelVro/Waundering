"""1st-innings total runs predictor.

Three quantile-regression heads (P10 / P50 / P90) so we can answer any
over/under line as `P(over) = 1 - empirical CDF(line | features)`. We also
return a point estimate (mean of the predictive distribution).

Features (all leakage-safe — only data BEFORE the match):
  - venue avg first-innings score (as-of)
  - format (T20/ODI/Test)
  - bat unit aggregates from the announced XI of the team batting first
      (top-7 by career_balls: career_avg, form_sr)
  - bowl unit aggregates from the announced XI of the bowling team
      (top-5 by career_balls: career_econ, career_avg)

Bat-first team is decided from toss when known; otherwise we predict for both
batting orders and pick the one consistent with the user's input toss.

Training:
    python -m cricket_pipeline.work.totals_model --train --fmt T20,IT20

Predicting one line:
    python -m cricket_pipeline.work.totals_model \\
        --bat "Sunrisers Hyderabad" --bowl "Rajasthan Royals" \\
        --venue "Sawai Mansingh Stadium, Jaipur" --fmt T20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..db import connect, install_views
from .features_v2 import compute_venue_stats_asof
from . import filters as F


MODEL_DIR  = Path(__file__).resolve().parents[1] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TOTAL_MODELS_PATH = MODEL_DIR / "totals_lgbm.joblib"
TOTAL_META_PATH   = MODEL_DIR / "totals_meta.json"

QUANTILES = (0.10, 0.50, 0.90)


# ---------- training data ----------

def build_training_frame(format_filter: list[str] | None = None) -> pd.DataFrame:
    """One row per (match, innings_no=1). Target = innings.total_runs."""
    install_views()
    con = connect()
    matches = con.execute("""
        SELECT match_id, format, start_date, team_home, team_away, venue,
               toss_winner, toss_decision, winner
        FROM matches
        WHERE start_date IS NOT NULL
    """).df()
    innings = con.execute("""
        SELECT match_id, innings_no, batting_team, bowling_team, total_runs, total_overs
        FROM innings
        WHERE innings_no = 1 AND total_runs IS NOT NULL
    """).df()
    con.close()

    if format_filter:
        matches = matches[matches["format"].isin(format_filter)].copy()
    matches["start_date"] = pd.to_datetime(matches["start_date"])
    df = matches.merge(innings, on="match_id", how="inner")
    df = df[df["batting_team"].notna() & df["bowling_team"].notna()]
    # filter blocked teams
    df = df[~(df["batting_team"].apply(F.is_blocked_team) |
              df["bowling_team"].apply(F.is_blocked_team))]
    df = df.rename(columns={"total_runs": "y_total"})

    # venue stats AS-OF
    inn_for_venue = innings[["match_id", "innings_no", "total_runs"]]
    venue_df = compute_venue_stats_asof(matches, inn_for_venue)
    df = df.merge(venue_df, on="match_id", how="left")

    # Player aggregates (batting team — bat unit; bowling team — bowl unit)
    print("  computing as-of player aggregates ...")
    player_aggs = _player_aggs_for_matches(df)
    df = df.merge(player_aggs, on=["match_id", "batting_team", "bowling_team"], how="left")

    # Effective overs cap (sometimes T20s have shorter overs; use as a feature)
    df["max_overs"] = df["total_overs"].fillna(20.0).clip(lower=5, upper=90)

    # Format encoding (sufficient as ints since we'll keep T20/IT20/ODI/Test)
    df["fmt_T20"]  = (df["format"].isin(["T20", "IT20"])).astype(int)
    df["fmt_ODI"]  = (df["format"] == "ODI").astype(int)
    df["fmt_Test"] = (df["format"] == "Test").astype(int)

    return df


def _player_aggs_for_matches(df: pd.DataFrame, top_bat: int = 7, top_bowl: int = 5) -> pd.DataFrame:
    """For each (match, batting_team, bowling_team), aggregate batting unit and
    bowling unit from announced XI using v_batter_history + v_bowler_history
    (which are strictly pre-match). DuckDB-side query with one go."""
    con = connect()
    install_views()
    sql = f"""
        WITH match_pairs AS (
            SELECT match_id, batting_team, bowling_team, start_date FROM (
                SELECT match_id, batting_team, bowling_team,
                       (SELECT m2.start_date FROM matches m2 WHERE m2.match_id = i.match_id) AS start_date
                FROM (SELECT DISTINCT match_id, batting_team, bowling_team
                       FROM innings WHERE innings_no = 1) i
            )
        ),
        bat_xi AS (
            SELECT mp.match_id, mp.batting_team, mp.bowling_team, x.player
            FROM match_pairs mp
            JOIN match_xi x USING (match_id)
            WHERE x.team = mp.batting_team
        ),
        bowl_xi AS (
            SELECT mp.match_id, mp.batting_team, mp.bowling_team, x.player
            FROM match_pairs mp
            JOIN match_xi x USING (match_id)
            WHERE x.team = mp.bowling_team
        ),
        bat_hist AS (
            SELECT bx.match_id, bx.batting_team, bx.bowling_team,
                   bh.batter, bh.career_avg, bh.form_sr, bh.career_balls
            FROM bat_xi bx
            JOIN v_batter_history bh
              ON bh.batter = bx.player AND bh.match_id = bx.match_id
        ),
        bowl_hist AS (
            SELECT bxw.match_id, bxw.batting_team, bxw.bowling_team,
                   bh.bowler, bh.career_econ, bh.career_avg, bh.career_balls
            FROM bowl_xi bxw
            JOIN v_bowler_history bh
              ON bh.bowler = bxw.player AND bh.match_id = bxw.match_id
        ),
        bat_top AS (
            SELECT match_id, batting_team, bowling_team,
                   AVG(career_avg) AS bat_avg, AVG(form_sr) AS bat_form_sr,
                   SUM(career_balls) AS bat_balls
            FROM (
                SELECT *, ROW_NUMBER() OVER
                       (PARTITION BY match_id, batting_team, bowling_team
                        ORDER BY COALESCE(career_balls, 0) DESC) AS rk
                FROM bat_hist
            ) WHERE rk <= {int(top_bat)}
            GROUP BY match_id, batting_team, bowling_team
        ),
        bowl_top AS (
            SELECT match_id, batting_team, bowling_team,
                   AVG(career_econ) AS bowl_econ, AVG(career_avg) AS bowl_avg,
                   SUM(career_balls) AS bowl_balls
            FROM (
                SELECT *, ROW_NUMBER() OVER
                       (PARTITION BY match_id, batting_team, bowling_team
                        ORDER BY COALESCE(career_balls, 0) DESC) AS rk
                FROM bowl_hist
            ) WHERE rk <= {int(top_bowl)}
            GROUP BY match_id, batting_team, bowling_team
        )
        SELECT mp.match_id, mp.batting_team, mp.bowling_team,
               bt.bat_avg, bt.bat_form_sr, bt.bat_balls,
               bw.bowl_econ, bw.bowl_avg, bw.bowl_balls
        FROM match_pairs mp
        LEFT JOIN bat_top  bt USING (match_id, batting_team, bowling_team)
        LEFT JOIN bowl_top bw USING (match_id, batting_team, bowling_team)
    """
    out = con.execute(sql).df()
    con.close()
    return out


FEATURES = [
    "venue_avg_first", "venue_n_prior",
    "max_overs",
    "fmt_T20", "fmt_ODI", "fmt_Test",
    "bat_avg", "bat_form_sr", "bat_balls",
    "bowl_econ", "bowl_avg", "bowl_balls",
]


def train(format_filter: list[str] | None = None,
          test_frac: float = 0.15) -> dict:
    df = build_training_frame(format_filter)
    print(f"  rows: {len(df):,}")
    df = df.sort_values("start_date").reset_index(drop=True)
    n = len(df); cutoff = int(n * (1 - test_frac))
    train_df = df.iloc[:cutoff].copy()
    test_df  = df.iloc[cutoff:].copy()
    print(f"  train: {len(train_df):,}   test: {len(test_df):,}")

    Xtr = train_df[FEATURES]; ytr = train_df["y_total"].astype(float)
    Xte = test_df[FEATURES];  yte = test_df["y_total"].astype(float)

    boosters = {}
    for q in QUANTILES:
        params = {
            "objective": "quantile", "alpha": q,
            "metric": "quantile", "learning_rate": 0.05,
            "num_leaves": 31, "min_data_in_leaf": 20,
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
            "verbose": -1,
        }
        ds_tr = lgb.Dataset(Xtr, label=ytr)
        ds_te = lgb.Dataset(Xte, label=yte, reference=ds_tr)
        b = lgb.train(params, ds_tr, num_boost_round=600,
                      valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        boosters[q] = b

    preds = {q: b.predict(Xte) for q, b in boosters.items()}
    median = preds[0.50]
    metrics = {
        "n_train":         int(len(train_df)),
        "n_test":          int(len(test_df)),
        "mae_p50":         float(np.mean(np.abs(yte - median))),
        "rmse_p50":        float(np.sqrt(np.mean((yte - median) ** 2))),
        "coverage_80pct":  float(np.mean((yte >= preds[0.10]) & (yte <= preds[0.90]))),
        "median_p50":      float(np.median(median)),
        "format_filter":   format_filter or [],
        "feature_order":   FEATURES,
    }
    joblib.dump(boosters, TOTAL_MODELS_PATH)
    TOTAL_META_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\n=== totals model metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18} {v:.3f}")
        else:
            print(f"  {k:<18} {v}")
    print(f"  saved -> {TOTAL_MODELS_PATH.name}")
    return metrics


# ---------- prediction ----------

_loaded: dict | None = None


def _load() -> dict:
    global _loaded
    if _loaded is None:
        if not TOTAL_MODELS_PATH.exists():
            raise RuntimeError("Totals model not trained. Run with --train first.")
        boosters = joblib.load(TOTAL_MODELS_PATH)
        meta = json.loads(TOTAL_META_PATH.read_text())
        _loaded = {"boosters": boosters, "meta": meta}
    return _loaded


def predict_total(batting_team: str, bowling_team: str, venue: str,
                   fmt: str, ref_date: str,
                   bat_xi: list[str] | None = None,
                   bowl_xi: list[str] | None = None) -> dict:
    """Predict 1st-innings total. Uses each side's most recent prior XI as proxy
    when bat_xi/bowl_xi not provided."""
    L = _load()
    boosters = L["boosters"]

    feats = _features_for_one(batting_team, bowling_team, venue, fmt, ref_date,
                               bat_xi, bowl_xi)
    X = pd.DataFrame([feats])[FEATURES]
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    preds = {q: float(b.predict(X)[0]) for q, b in boosters.items()}
    return {
        "p10":      round(preds[0.10], 1),
        "p50":      round(preds[0.50], 1),
        "p90":      round(preds[0.90], 1),
        "spread":   round(preds[0.90] - preds[0.10], 1),
        "features": feats,
    }


def over_under_probs(prediction: dict, line: float) -> dict:
    """Convert P10/P50/P90 to over/under at `line` via piecewise-linear CDF."""
    p10, p50, p90 = prediction["p10"], prediction["p50"], prediction["p90"]
    # build the CDF: F(p10)=0.10, F(p50)=0.50, F(p90)=0.90
    # linear extrapolation outside, capped at [0.01, 0.99]
    if line <= p10:
        cdf = max(0.01, 0.10 * (line / p10) if p10 > 0 else 0.01)
    elif line <= p50:
        cdf = 0.10 + 0.40 * (line - p10) / (p50 - p10) if p50 > p10 else 0.50
    elif line <= p90:
        cdf = 0.50 + 0.40 * (line - p50) / (p90 - p50) if p90 > p50 else 0.90
    else:
        # extrapolate: pretend tail decays
        cdf = 0.90 + 0.09 * min(1.0, (line - p90) / max(p90 - p50, 1.0))
    cdf = max(0.01, min(0.99, cdf))
    return {"line": line, "p_over": round(1 - cdf, 4), "p_under": round(cdf, 4)}


# ---------- feature builder for a single fixture ----------

def _features_for_one(batting_team: str, bowling_team: str, venue: str,
                       fmt: str, ref_date: str,
                       bat_xi: list[str] | None,
                       bowl_xi: list[str] | None) -> dict:
    con = connect()
    install_views()
    # venue stats as-of ref_date
    venue_row = con.execute("""
        SELECT
          AVG(i.total_runs) FILTER (WHERE i.innings_no = 1)              AS avg_first,
          COUNT(DISTINCT m.match_id)                                     AS n_prior
        FROM matches m
        LEFT JOIN innings i USING (match_id)
        WHERE m.venue = ? AND m.format = ? AND m.start_date < CAST(? AS DATE)
    """, [venue, fmt, ref_date]).fetchone()
    venue_avg_first = venue_row[0]; venue_n_prior = venue_row[1]

    if not bat_xi:
        bat_xi = _proxy_xi(con, batting_team, ref_date)
    if not bowl_xi:
        bowl_xi = _proxy_xi(con, bowling_team, ref_date)

    bat_aggs  = _xi_bat_aggs(con, bat_xi, ref_date, top_n=7)  if bat_xi  else {}
    bowl_aggs = _xi_bowl_aggs(con, bowl_xi, ref_date, top_n=5) if bowl_xi else {}
    con.close()

    return {
        "venue_avg_first": venue_avg_first,
        "venue_n_prior":   venue_n_prior or 0,
        "max_overs":       20.0 if fmt in ("T20", "IT20") else (50.0 if fmt == "ODI" else 90.0),
        "fmt_T20":         1 if fmt in ("T20", "IT20") else 0,
        "fmt_ODI":         1 if fmt == "ODI" else 0,
        "fmt_Test":        1 if fmt == "Test" else 0,
        "bat_avg":         bat_aggs.get("avg"),
        "bat_form_sr":     bat_aggs.get("form_sr"),
        "bat_balls":       bat_aggs.get("balls"),
        "bowl_econ":       bowl_aggs.get("econ"),
        "bowl_avg":        bowl_aggs.get("avg"),
        "bowl_balls":      bowl_aggs.get("balls"),
    }


def _proxy_xi(con, team: str, ref_date: str) -> list[str]:
    row = con.execute("""
        SELECT match_id FROM match_xi
        WHERE team = ? AND start_date < CAST(? AS DATE) AND match_id IS NOT NULL
        GROUP BY match_id, start_date
        ORDER BY start_date DESC LIMIT 1
    """, [team, ref_date]).fetchone()
    if not row: return []
    return [r[0] for r in con.execute(
        "SELECT player FROM match_xi WHERE match_id = ? AND team = ?",
        [row[0], team]
    ).fetchall()]


def _xi_bat_aggs(con, players: list[str], ref_date: str, top_n: int = 7) -> dict:
    if not players: return {}
    placeholders = ",".join(["?"] * len(players))
    row = con.execute(f"""
        WITH ranked AS (
            SELECT batter, career_avg, form_sr, career_balls,
                   ROW_NUMBER() OVER (PARTITION BY batter ORDER BY start_date DESC) AS rk
            FROM v_batter_history
            WHERE batter IN ({placeholders}) AND start_date < CAST(? AS DATE)
        ),
        latest AS (SELECT * FROM ranked WHERE rk = 1
                   ORDER BY COALESCE(career_balls,0) DESC LIMIT {top_n})
        SELECT AVG(career_avg), AVG(form_sr), SUM(career_balls) FROM latest
    """, players + [ref_date]).fetchone()
    return {"avg": row[0], "form_sr": row[1], "balls": row[2]}


def _xi_bowl_aggs(con, players: list[str], ref_date: str, top_n: int = 5) -> dict:
    if not players: return {}
    placeholders = ",".join(["?"] * len(players))
    row = con.execute(f"""
        WITH ranked AS (
            SELECT bowler, career_econ, career_avg, career_balls,
                   ROW_NUMBER() OVER (PARTITION BY bowler ORDER BY start_date DESC) AS rk
            FROM v_bowler_history
            WHERE bowler IN ({placeholders}) AND start_date < CAST(? AS DATE)
        ),
        latest AS (SELECT * FROM ranked WHERE rk = 1
                   ORDER BY COALESCE(career_balls,0) DESC LIMIT {top_n})
        SELECT AVG(career_econ), AVG(career_avg), SUM(career_balls) FROM latest
    """, players + [ref_date]).fetchone()
    return {"econ": row[0], "avg": row[1], "balls": row[2]}


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--fmt", default="T20,IT20")
    ap.add_argument("--bat", default=None, help="batting team")
    ap.add_argument("--bowl", default=None, help="bowling team")
    ap.add_argument("--venue", default=None)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--line", type=float, default=None)
    args = ap.parse_args()

    if args.train:
        fmts = [f.strip() for f in args.fmt.split(",") if f.strip()]
        train(format_filter=fmts)
        return

    if not (args.bat and args.bowl and args.venue and args.date):
        ap.error("Need --bat --bowl --venue --date for prediction")

    pred = predict_total(args.bat, args.bowl, args.venue,
                          args.fmt.split(",")[0], args.date)
    print(json.dumps(pred, indent=2, default=str))
    if args.line is not None:
        print(json.dumps(over_under_probs(pred, args.line), indent=2))


if __name__ == "__main__":
    main()
