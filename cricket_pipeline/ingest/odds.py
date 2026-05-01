"""Bookmaker odds ingestion.

Default provider: **The Odds API** (https://the-odds-api.com).
- Free tier: 500 requests / month. Paid plans available.
- Set `THE_ODDS_API_KEY` env var to enable.
- Without a key: this module is a no-op (the rest of the pipeline keeps
  working; we just won't have odds-driven features).

Pluggable: any provider that returns
  [{external_id, commence_time, home_team, away_team, bookmakers:[
       {key, last_update, markets:[{key, outcomes:[{name, price, point?}]}]}
  ]}]
can be wired in.

Cricket sport keys we care about (from /v4/sports?all=true):
  cricket_ipl, cricket_t20i, cricket_odi, cricket_test_match,
  cricket_big_bash, cricket_caribbean_premier_league, cricket_psl
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

from ..db import connect


CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "odds"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "uk,eu,us,au"
DEFAULT_MARKETS = "h2h,totals,spreads"      # extend later for player props
DEFAULT_SPORT_KEYS = (
    "cricket_ipl",
    "cricket_t20i",
    "cricket_odi",
    "cricket_test_match",
    "cricket_big_bash",
    "cricket_caribbean_premier_league",
    "cricket_international_t20",     # alt name
)

# Pakistani tournaments are filtered project-wide (per user request)
from ..work.filters import BLOCKED_SLUG_PATTERN, is_blocked_match  # type: ignore


def _api_key() -> str | None:
    return os.environ.get("THE_ODDS_API_KEY") or None


def list_sports() -> list[dict]:
    """List all sports the provider exposes (cricket + others). Costs 1 req."""
    key = _api_key()
    if not key: return []
    r = requests.get(f"{API_BASE}/sports", params={"apiKey": key, "all": "false"}, timeout=20)
    r.raise_for_status()
    return [s for s in r.json() if s.get("group") == "Cricket" or "cricket" in s.get("key","")]


def fetch_odds(sport_key: str,
               regions: str = DEFAULT_REGIONS,
               markets: str = DEFAULT_MARKETS,
               cache: bool = True) -> list[dict]:
    """Fetch odds for one sport key. Returns the API's event list.
    Caches the raw response under cache/odds/{sport}_{ts}.json."""
    key = _api_key()
    if not key:
        return []
    r = requests.get(
        f"{API_BASE}/sports/{sport_key}/odds",
        params={"apiKey": key, "regions": regions, "markets": markets,
                "oddsFormat": "decimal", "dateFormat": "iso"},
        timeout=30,
    )
    if r.status_code == 429:
        # rate limited or quota exhausted
        raise RuntimeError("the-odds-api rate-limited (429)")
    if r.status_code == 422:
        # sport not in season / no odds available
        return []
    r.raise_for_status()
    data = r.json()
    if cache:
        ts = int(time.time())
        (CACHE_DIR / f"{sport_key}_{ts}.json").write_text(json.dumps(data, indent=2))
    return data


def _store_event(con, snapshot_at: datetime, sport_key: str, evt: dict) -> int:
    """Store one event's odds rows. Returns rows inserted.

    Team names are canonicalized so 'Royal Challengers Bangalore' (Odds API) joins
    cleanly to 'Royal Challengers Bengaluru' (CricSheet)."""
    from ..work.team_aliases import canonicalize
    home = canonicalize(evt.get("home_team")); away = canonicalize(evt.get("away_team"))
    if is_blocked_match(home, away):
        return 0
    n = 0
    ext_id = evt.get("id"); commence = evt.get("commence_time")
    # try to resolve to our internal match_id (best-effort, by team names + commence date)
    match_id = _resolve_match_id(con, home, away, commence)

    for bk in evt.get("bookmakers") or []:
        bookmaker = bk.get("key") or bk.get("title")
        for mkt in bk.get("markets") or []:
            market = mkt.get("key")
            for o in mkt.get("outcomes") or []:
                sel = canonicalize(o.get("name"))
                price = o.get("price")
                line = o.get("point")
                try:
                    con.execute("""
                        INSERT OR REPLACE INTO odds_snapshot
                        (snapshot_at, sport_key, external_id, match_id, commence_time,
                         home_team, away_team, bookmaker, market, selection,
                         decimal_odds, line, raw_json)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, [
                        snapshot_at, sport_key, ext_id, match_id, commence,
                        home, away, bookmaker, market, sel,
                        float(price) if price is not None else None,
                        float(line) if line is not None else None,
                        json.dumps(o),
                    ])
                    n += 1
                except Exception:
                    pass
    return n


def _resolve_match_id(con, home: str | None, away: str | None,
                       commence_iso: str | None) -> str | None:
    """Best-effort join from API event → our matches table."""
    if not (home and away and commence_iso):
        return None
    try:
        commence = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    except Exception:
        return None
    d = commence.date().isoformat()
    row = con.execute("""
        SELECT match_id FROM matches
        WHERE start_date = CAST(? AS DATE)
          AND ((team_home = ? AND team_away = ?)
            OR (team_home = ? AND team_away = ?))
        ORDER BY match_id
        LIMIT 1
    """, [d, home, away, away, home]).fetchone()
    return row[0] if row else None


def fetch_and_store(sport_keys: Iterable[str] = DEFAULT_SPORT_KEYS) -> dict:
    """Pull odds for every sport key, persist to odds_snapshot. Returns counts."""
    if not _api_key():
        return {"status": "no_api_key", "rows": 0, "events": 0, "sport_keys": list(sport_keys)}
    snapshot_at = datetime.now(timezone.utc)
    con = connect()
    rows = 0; events = 0; per_sport: dict[str, int] = {}
    for sk in sport_keys:
        try:
            data = fetch_odds(sk)
        except Exception as e:
            per_sport[sk] = -1
            print(f"odds: {sk} fetch failed: {e}")
            continue
        per_sport[sk] = len(data)
        for evt in data:
            events += 1
            rows += _store_event(con, snapshot_at, sk, evt)
    con.close()
    return {
        "snapshot_at": snapshot_at.isoformat(),
        "events":      events,
        "rows":        rows,
        "per_sport":   per_sport,
    }


def latest_for_event(external_id: str, market: str = "h2h") -> list[dict]:
    """Pull the most recent odds snapshot for an event+market."""
    con = connect()
    rows = con.execute("""
        SELECT snapshot_at, bookmaker, selection, decimal_odds, line
        FROM odds_snapshot
        WHERE external_id = ? AND market = ?
          AND snapshot_at = (SELECT MAX(snapshot_at) FROM odds_snapshot
                             WHERE external_id = ? AND market = ?)
        ORDER BY bookmaker, selection
    """, [external_id, market, external_id, market]).fetchall()
    con.close()
    return [{"snapshot_at": r[0], "bookmaker": r[1], "selection": r[2],
             "decimal_odds": r[3], "line": r[4]} for r in rows]


def latest_for_match(home: str, away: str, market: str = "h2h",
                      commence_within_days: int = 5) -> list[dict]:
    """Pull most recent odds for a match by team names. Useful when the
    external_id isn't known (e.g. our internal upcoming-match id)."""
    con = connect()
    rows = con.execute(f"""
        SELECT snapshot_at, bookmaker, selection, decimal_odds, line, external_id, commence_time
        FROM odds_snapshot
        WHERE market = ?
          AND ((home_team = ? AND away_team = ?)
            OR (home_team = ? AND away_team = ?))
        ORDER BY snapshot_at DESC, bookmaker
        LIMIT 200
    """, [market, home, away, away, home]).fetchall()
    con.close()
    if not rows: return []
    # keep the latest snapshot only
    latest = rows[0][0]
    return [{"snapshot_at": r[0], "bookmaker": r[1], "selection": r[2],
             "decimal_odds": r[3], "line": r[4], "external_id": r[5],
             "commence_time": r[6]} for r in rows if r[0] == latest]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-sports", action="store_true")
    ap.add_argument("--sport", default=None, help="single sport_key")
    ap.add_argument("--all", action="store_true", help="fetch + store all default sports")
    args = ap.parse_args()
    if args.list_sports:
        for s in list_sports(): print(s)
    elif args.sport:
        print(json.dumps(fetch_odds(args.sport), indent=2)[:2000])
    elif args.all:
        print(json.dumps(fetch_and_store(), indent=2))
    else:
        ap.print_help()
