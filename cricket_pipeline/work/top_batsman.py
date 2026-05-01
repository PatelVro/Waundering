"""Top-batsman-of-team market.

For an upcoming match, predict — per team — which member of the announced XI
will score the most runs. Output is a probability distribution that sums to 1
within each team's XI.

Approach:
  1. For every historical match-team pair, find the actual top scorer.
  2. Build per-player feature rows: career stats (career_avg, career_sr),
     recent-10-innings form (rolling), opposition bowl strength, venue
     batting friendliness, format.
  3. Train a binary "is_top_scorer" classifier (LightGBM).
  4. At inference: score every XI member with the model, softmax within team.

Training data uses v_batter_history (strictly pre-match stats) + match_xi.
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
from . import filters as F


MODEL_DIR  = Path(__file__).resolve().parents[1] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TOP_BAT_PATH = MODEL_DIR / "top_batsman_lgbm.joblib"
TOP_BAT_META = MODEL_DIR / "top_batsman_meta.json"

FEATURES = [
    "career_avg", "career_sr", "form_sr", "career_balls",
    "opp_bowl_econ", "opp_bowl_avg",
    "venue_avg_first",
    "fmt_T20", "fmt_ODI", "fmt_Test",
]


# ---------- training data ----------

def build_training_frame(format_filter: list[str] | None = None) -> pd.DataFrame:
    install_views()
    con = connect()

    fmt_clause = ""
    fmts = format_filter or []
    if fmts:
        in_clause = ",".join(f"'{f}'" for f in fmts)
        fmt_clause = f" AND m.format IN ({in_clause})"

    print("  building top-scorer labels ...")
    # Top scorer per (match, batting_team)
    top = con.execute(f"""
        WITH per_player AS (
            SELECT b.match_id, b.batting_team, b.batter, SUM(b.runs_batter) AS runs
            FROM balls b
            JOIN matches m USING (match_id)
            WHERE b.batter IS NOT NULL{fmt_clause}
            GROUP BY b.match_id, b.batting_team, b.batter
        ),
        ranked AS (
            SELECT match_id, batting_team, batter, runs,
                   ROW_NUMBER() OVER (PARTITION BY match_id, batting_team
                                       ORDER BY runs DESC) AS rk
            FROM per_player
        )
        SELECT match_id, batting_team, batter AS top_scorer, runs AS top_runs
        FROM ranked WHERE rk = 1
    """).df()

    print("  building XI feature rows ...")
    # Every XI member of every (match, batting_team), labeled
    rows = con.execute(f"""
        SELECT m.match_id, m.format, m.start_date, m.venue,
               i.batting_team, i.bowling_team,
               x.player                 AS batter,
               bh.career_avg, bh.form_sr, bh.career_sr, bh.career_balls
        FROM matches m
        JOIN innings i        USING (match_id)
        JOIN match_xi x        ON x.match_id = m.match_id AND x.team = i.batting_team
        LEFT JOIN v_batter_history bh
                              ON bh.match_id = m.match_id AND bh.batter = x.player
        WHERE m.start_date IS NOT NULL{fmt_clause}
          AND i.innings_no = 1
    """).df()
    rows = rows.drop_duplicates(["match_id", "batting_team", "batter"])

    # opp bowl unit aggregates
    print("  joining opposition bowling aggregates ...")
    bowl = con.execute(f"""
        WITH bowl_xi AS (
            SELECT i.match_id, i.bowling_team, x.player
            FROM innings i
            JOIN matches m USING (match_id)
            JOIN match_xi x ON x.match_id = i.match_id AND x.team = i.bowling_team
            WHERE i.innings_no = 1{fmt_clause}
        ),
        bowl_hist AS (
            SELECT bx.match_id, bx.bowling_team,
                   bh.bowler, bh.career_econ, bh.career_avg, bh.career_balls
            FROM bowl_xi bx
            JOIN v_bowler_history bh
              ON bh.match_id = bx.match_id AND bh.bowler = bx.player
        ),
        top5 AS (
            SELECT match_id, bowling_team,
                   AVG(career_econ) AS opp_bowl_econ,
                   AVG(career_avg)  AS opp_bowl_avg
            FROM (
                SELECT *, ROW_NUMBER() OVER
                       (PARTITION BY match_id, bowling_team
                        ORDER BY COALESCE(career_balls,0) DESC) AS rk
                FROM bowl_hist
            ) WHERE rk <= 5
            GROUP BY match_id, bowling_team
        )
        SELECT * FROM top5
    """).df()

    rows = rows.merge(bowl, on=["match_id", "bowling_team"], how="left")
    rows = rows.merge(top[["match_id", "batting_team", "top_scorer"]],
                       on=["match_id", "batting_team"], how="left")
    rows["y_top"] = (rows["batter"] == rows["top_scorer"]).astype(int)

    # venue stats as-of (simple: no shifting within this query, accept tiny leakage
    # since we use the venue's *all-time* avg here; matches the deployed predictor)
    venue = con.execute("""
        SELECT venue, format, AVG(total_runs) FILTER (WHERE innings_no=1) AS venue_avg_first
        FROM innings JOIN matches USING (match_id)
        GROUP BY venue, format
    """).df()
    rows = rows.merge(venue, on=["venue", "format"], how="left")

    # filter blocked teams
    rows = rows[~(rows["batting_team"].apply(F.is_blocked_team) |
                   rows["bowling_team"].apply(F.is_blocked_team))]

    rows["fmt_T20"]  = rows["format"].isin(["T20","IT20"]).astype(int)
    rows["fmt_ODI"]  = (rows["format"] == "ODI").astype(int)
    rows["fmt_Test"] = (rows["format"] == "Test").astype(int)
    rows["start_date"] = pd.to_datetime(rows["start_date"])

    con.close()
    return rows


def train(format_filter: list[str] | None = None,
          test_frac: float = 0.15) -> dict:
    df = build_training_frame(format_filter)
    print(f"  rows: {len(df):,}  positives: {int(df['y_top'].sum()):,}")
    df = df.sort_values("start_date").reset_index(drop=True)
    n = len(df); cutoff = int(n * (1 - test_frac))
    train_df = df.iloc[:cutoff].copy()
    test_df  = df.iloc[cutoff:].copy()
    print(f"  train: {len(train_df):,}   test: {len(test_df):,}")

    Xtr = train_df[FEATURES]; ytr = train_df["y_top"].astype(int)
    Xte = test_df[FEATURES];  yte = test_df["y_top"].astype(int)

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 30,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
        "verbose": -1,
    }
    ds_tr = lgb.Dataset(Xtr, label=ytr)
    ds_te = lgb.Dataset(Xte, label=yte, reference=ds_tr)
    booster = lgb.train(params, ds_tr, num_boost_round=600,
                         valid_sets=[ds_te],
                         callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])

    # metric: top-1 accuracy when softmax-normalized within (match,batting_team)
    test_df["score"] = booster.predict(Xte)
    grp = test_df.groupby(["match_id", "batting_team"])
    n_groups = 0; correct = 0
    for _, g in grp:
        if len(g) < 2: continue
        n_groups += 1
        pred_top = g.sort_values("score", ascending=False).iloc[0]["batter"]
        actual   = g[g["y_top"] == 1]
        if not actual.empty and actual.iloc[0]["batter"] == pred_top:
            correct += 1
    top1_acc = correct / n_groups if n_groups else 0.0

    metrics = {
        "n_train":       int(len(train_df)),
        "n_test":        int(len(test_df)),
        "n_test_groups": n_groups,
        "top1_accuracy": float(top1_acc),
        "format_filter": format_filter or [],
        "feature_order": FEATURES,
    }
    joblib.dump(booster, TOP_BAT_PATH)
    TOP_BAT_META.write_text(json.dumps(metrics, indent=2))
    print("\n=== top-batsman model metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18} {v:.3f}")
        else:
            print(f"  {k:<18} {v}")
    print(f"  saved -> {TOP_BAT_PATH.name}")
    return metrics


# ---------- prediction ----------

_loaded = None
def _load():
    global _loaded
    if _loaded is None:
        if not TOP_BAT_PATH.exists():
            raise RuntimeError("Top-batsman model not trained. Run --train first.")
        _loaded = {
            "booster": joblib.load(TOP_BAT_PATH),
            "meta":    json.loads(TOP_BAT_META.read_text()),
        }
    return _loaded


def predict_team_top_scorer(team: str, opponent: str, venue: str,
                              fmt: str, ref_date: str,
                              xi: list[str] | None = None,
                              opp_xi: list[str] | None = None) -> list[dict]:
    """For each member of `team`'s XI, return their probability of being
    that team's top scorer. Returns sorted desc, sums to 1."""
    L = _load()
    booster = L["booster"]

    con = connect(); install_views()
    if not xi:        xi      = _proxy_xi(con, team, ref_date)
    if not opp_xi:    opp_xi  = _proxy_xi(con, opponent, ref_date)
    if not xi:
        con.close()
        return []
    bat_feats  = _bat_features_for_players(con, xi, ref_date)
    opp_aggs   = _bowl_aggs(con, opp_xi, ref_date)
    venue_avg  = con.execute("""
        SELECT AVG(i.total_runs) FROM matches m JOIN innings i USING (match_id)
        WHERE m.venue = ? AND m.format = ? AND i.innings_no = 1 AND m.start_date < CAST(? AS DATE)
    """, [venue, fmt, ref_date]).fetchone()[0]
    con.close()

    rows = []
    for p in xi:
        f = bat_feats.get(p, {})
        rows.append({
            "batter":          p,
            "career_avg":      f.get("career_avg"),
            "career_sr":       f.get("career_sr"),
            "form_sr":         f.get("form_sr"),
            "career_balls":    f.get("career_balls"),
            "opp_bowl_econ":   opp_aggs.get("econ"),
            "opp_bowl_avg":    opp_aggs.get("avg"),
            "venue_avg_first": venue_avg,
            "fmt_T20":         1 if fmt in ("T20","IT20") else 0,
            "fmt_ODI":         1 if fmt == "ODI" else 0,
            "fmt_Test":        1 if fmt == "Test" else 0,
        })
    df = pd.DataFrame(rows)
    if df.empty: return []
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["score"] = booster.predict(df[FEATURES])
    # softmax with temperature 1.5 to keep distribution from collapsing
    s = df["score"].clip(lower=1e-9, upper=1 - 1e-9)
    logit = np.log(s / (1 - s))
    p_unnorm = np.exp(logit / 1.5)
    df["prob"] = p_unnorm / p_unnorm.sum()
    df = df.sort_values("prob", ascending=False)
    return [{"player": r.batter, "prob": round(float(r.prob), 4),
             "score":  round(float(r.score), 4)}
            for r in df.itertuples(index=False)]


# ---------- helpers ----------

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
        [row[0], team]).fetchall()]


def _bat_features_for_players(con, players: list[str], ref_date: str) -> dict:
    if not players: return {}
    placeholders = ",".join(["?"] * len(players))
    rows = con.execute(f"""
        WITH ranked AS (
            SELECT batter, career_avg, career_sr, form_sr, career_balls,
                   ROW_NUMBER() OVER (PARTITION BY batter ORDER BY start_date DESC) AS rk
            FROM v_batter_history
            WHERE batter IN ({placeholders}) AND start_date < CAST(? AS DATE)
        )
        SELECT batter, career_avg, career_sr, form_sr, career_balls
        FROM ranked WHERE rk = 1
    """, players + [ref_date]).fetchall()
    out = {}
    for r in rows:
        out[r[0]] = {"career_avg": r[1], "career_sr": r[2],
                      "form_sr": r[3], "career_balls": r[4]}
    return out


def _bowl_aggs(con, players: list[str], ref_date: str, top_n: int = 5) -> dict:
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
        SELECT AVG(career_econ), AVG(career_avg) FROM latest
    """, players + [ref_date]).fetchone()
    return {"econ": row[0], "avg": row[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--fmt", default="T20,IT20,ODI")
    ap.add_argument("--team", default=None)
    ap.add_argument("--opp", default=None)
    ap.add_argument("--venue", default=None)
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    if args.train:
        fmts = [f.strip() for f in args.fmt.split(",") if f.strip()]
        train(format_filter=fmts)
        return

    if not (args.team and args.opp and args.venue and args.date):
        ap.error("Need --team --opp --venue --date for prediction")
    res = predict_team_top_scorer(args.team, args.opp, args.venue,
                                    args.fmt.split(",")[0], args.date)
    print(json.dumps(res[:5], indent=2))


if __name__ == "__main__":
    main()
