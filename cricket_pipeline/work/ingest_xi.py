"""Extract the announced XI for each cached match into `match_xi` table.

CricSheet JSONs include `info.players[team_name] = [player1, ...]` — the
match's announced playing XI. This is fixed pre-match (announced at toss),
so unlike "batters who batted" (which leaks the outcome), the XI is the
clean lineup signal.

This walks every cached zip, re-parses each match JSON, and writes
(match_id, team, player) into a `match_xi` table. Idempotent — uses INSERT
OR REPLACE on PK (match_id, team, player).
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

from .. import config
from ..db import connect
from ..ingest.cricsheet import _iter_match_jsons


def _xi_rows(m: dict) -> list[dict]:
    info = m.get("info", {}) or {}
    dates = info.get("dates") or [""]
    teams = info.get("teams") or []
    venue = info.get("venue") or ""
    start_date = dates[0]
    t1 = teams[0] if len(teams) > 0 else ""
    t2 = teams[1] if len(teams) > 1 else ""
    players = info.get("players") or {}
    out = []
    for team, names in players.items():
        if not isinstance(names, list):
            continue
        for nm in names:
            if not nm:
                continue
            out.append({
                "start_date": start_date,
                "team_home":  t1,
                "team_away":  t2,
                "venue":      venue,
                "team":       team,
                "player":     nm,
            })
    return out


def _ensure_table(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS match_xi (
            start_date DATE,
            team_home  VARCHAR,
            team_away  VARCHAR,
            venue      VARCHAR,
            team       VARCHAR,
            player     VARCHAR,
            match_id   VARCHAR,
            PRIMARY KEY (start_date, team_home, team_away, venue, team, player)
        )
    """)


def ingest_all_cached() -> dict:
    cache = Path(config.CACHE_DIR)
    zips = sorted(cache.glob("*.zip"))
    if not zips:
        print(f"No zips in {cache}", file=sys.stderr)
        return {"zips": 0, "rows": 0}

    con = connect()
    # drop & recreate to re-run cleanly
    con.execute("DROP TABLE IF EXISTS match_xi")
    _ensure_table(con)

    total_rows = 0
    total_matches = 0
    for zp in zips:
        t0 = time.time()
        rows = []
        n_matches = 0
        for m in _iter_match_jsons(zp):
            xi = _xi_rows(m)
            if xi:
                rows.extend(xi); n_matches += 1
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows).drop_duplicates(
                subset=["start_date", "team_home", "team_away", "venue", "team", "player"]
            )
            con.register("xi_in", df)
            con.execute("""
                INSERT OR REPLACE INTO match_xi
                SELECT start_date, team_home, team_away, venue, team, player, NULL AS match_id
                FROM xi_in
            """)
            con.unregister("xi_in")
        total_rows += len(rows)
        total_matches += n_matches
        print(f"  {zp.name:<40} matches={n_matches:>5}  xi_rows={len(rows):>6}  {time.time()-t0:.1f}s")

    # Backfill match_id by joining to matches on (start_date, teams, venue).
    # Some matches in matches table won't have an XI (older / missing data).
    con.execute("""
        UPDATE match_xi x
        SET match_id = m.match_id
        FROM matches m
        WHERE x.start_date = m.start_date
          AND x.team_home  = m.team_home
          AND x.team_away  = m.team_away
          AND COALESCE(x.venue, '')  = COALESCE(m.venue, '')
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_match_xi_match ON match_xi(match_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_match_xi_team  ON match_xi(team)")
    n_xi    = con.execute("SELECT COUNT(*) FROM match_xi").fetchone()[0]
    n_match = con.execute("SELECT COUNT(DISTINCT match_id) FROM match_xi WHERE match_id IS NOT NULL").fetchone()[0]
    con.close()
    print(f"\nFinal: {n_xi:,} xi_rows across {n_match:,} matches.")
    return {"zips": len(zips), "rows": total_rows, "matches": total_matches}


if __name__ == "__main__":
    ingest_all_cached()
