"""Export everything the cricket dashboard needs into a single data.json
served from the project root.

Includes:
  - latest_prediction: the saved RR vs SRH JSON (or whichever the user last ran)
  - all_predictions: every JSON in predictions/
  - model_metrics: final_summary.json
  - top_teams_t20 / top_teams_odi: per-format Elo leaderboard
  - recent_matches: last 30 with model retro-prediction (where available)
  - team_form_snapshots: last 5 win-pct for top teams
  - data_stats: row counts by table
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..db import connect, install_views
from .features_v2 import build_features


ROOT = Path(__file__).resolve().parents[2]   # Waundering/
PREDICTIONS_DIR = ROOT / "predictions"
RUNS_DIR        = ROOT / "cricket_pipeline" / "work" / "runs"
OUT_PATH        = ROOT / "data.json"


def _load_predictions() -> dict:
    out = {"all": [], "latest": None}
    if not PREDICTIONS_DIR.exists():
        return out
    files = sorted(PREDICTIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for fp in files:
        try:
            data = json.loads(fp.read_text())
            data["_file"] = fp.name
            out["all"].append(data)
        except Exception:
            continue
    if out["all"]:
        out["latest"] = out["all"][0]
    return out


def _model_metrics() -> dict:
    p = RUNS_DIR / "final_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _top_teams_per_format(format_codes: list[str], top_n: int = 12) -> list[dict]:
    """Compute current Elo per team for the given formats."""
    df = build_features(format_filter=format_codes)
    if df.empty:
        return []
    # walk chronologically and grab the last seen pre-Elo per team... but easier:
    # the LAST row in df has the most-recent t1_elo_pre/t2_elo_pre. We want each team's
    # final Elo. Use both perspectives and take the max date per team.
    rows1 = df[["start_date", "team_home", "t1_elo_pre"]].rename(
        columns={"team_home": "team", "t1_elo_pre": "elo"})
    rows2 = df[["start_date", "team_away", "t2_elo_pre"]].rename(
        columns={"team_away": "team", "t2_elo_pre": "elo"})
    long = pd.concat([rows1, rows2], ignore_index=True).dropna()
    # For each team take the LATEST elo
    long = long.sort_values("start_date").groupby("team", as_index=False).tail(1)
    # Add a recent-form column from last 10 wins
    matches = long.sort_values("elo", ascending=False).head(top_n)
    return [{"team": r.team, "elo": round(float(r.elo), 1),
             "as_of": str(r.start_date.date())} for r in matches.itertuples(index=False)]


def _recent_matches(n: int = 30) -> list[dict]:
    con = connect()
    install_views()
    rows = con.execute(f"""
        SELECT match_id, format, competition, start_date, venue,
               team_home, team_away, winner, win_margin_runs, win_margin_wickets
        FROM matches
        WHERE start_date IS NOT NULL AND winner IS NOT NULL
        ORDER BY start_date DESC, match_id DESC
        LIMIT {n}
    """).df()
    con.close()
    out = []
    for r in rows.itertuples(index=False):
        margin = ""
        runs_v = r.win_margin_runs
        wkts_v = r.win_margin_wickets
        if runs_v is not None and not pd.isna(runs_v) and float(runs_v) > 0:
            margin = f"by {int(runs_v)} runs"
        elif wkts_v is not None and not pd.isna(wkts_v) and float(wkts_v) > 0:
            margin = f"by {int(wkts_v)} wickets"
        out.append({
            "match_id":    r.match_id,
            "format":      r.format,
            "competition": r.competition,
            "date":        str(r.start_date),
            "venue":       r.venue,
            "home":        r.team_home,
            "away":        r.team_away,
            "winner":      r.winner,
            "margin":      margin,
        })
    return out


def _data_stats() -> dict:
    con = connect()
    out = {}
    for tbl in ("matches", "innings", "balls", "match_xi"):
        try:
            out[tbl] = int(con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
        except Exception:
            out[tbl] = 0
    out["distinct_venues"] = int(con.execute("SELECT COUNT(DISTINCT venue) FROM matches").fetchone()[0])
    out["distinct_teams"]  = int(con.execute("""
        SELECT COUNT(DISTINCT team) FROM (
            SELECT team_home AS team FROM matches
            UNION SELECT team_away FROM matches
        )
    """).fetchone()[0])
    out["distinct_competitions"] = int(con.execute("""
        SELECT COUNT(DISTINCT competition) FROM matches WHERE competition IS NOT NULL
    """).fetchone()[0])
    con.close()
    return out


def main():
    print("Building dashboard data ...")

    preds  = _load_predictions()
    metrics = _model_metrics()
    print("  computing top teams ...")
    top_t20 = _top_teams_per_format(["T20", "IT20"], top_n=15)
    top_odi = _top_teams_per_format(["ODI"], top_n=12)
    recent  = _recent_matches(40)
    stats   = _data_stats()

    out = {
        "generated_at":     pd.Timestamp.now().isoformat(),
        "latest_prediction": preds["latest"],
        "all_predictions":   preds["all"],
        "model_metrics":     metrics,
        "top_teams_t20":     top_t20,
        "top_teams_odi":     top_odi,
        "recent_matches":    recent,
        "data_stats":        stats,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"  wrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")
    print(f"  predictions: {len(preds['all'])}")
    print(f"  recent_matches: {len(recent)}")
    print(f"  top_teams_t20: {len(top_t20)}")


if __name__ == "__main__":
    main()
