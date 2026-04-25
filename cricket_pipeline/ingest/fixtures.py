"""Upcoming and live match fixtures via CricAPI (free tier).

CricAPI's `/cricScore` endpoint returns matches across formats/leagues with
status, venue, and teams. Set `CRICAPI_KEY` in env (free tier at cricapi.com).
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
import json

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect


def _cache_path(endpoint: str) -> Path:
    h = hashlib.sha1(f"{endpoint}|{date.today()}".encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"cricapi_{endpoint}_{h}.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _get(endpoint: str, params: dict | None = None) -> dict:
    if not config.CRICAPI_KEY:
        raise RuntimeError("Set CRICAPI_KEY in your environment first.")
    cache = _cache_path(endpoint)
    if cache.exists():
        return json.loads(cache.read_text())
    p = {"apikey": config.CRICAPI_KEY, **(params or {})}
    r = requests.get(f"{config.CRICAPI_BASE}/{endpoint}", params=p, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    return data


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def fetch_scores() -> int:
    data = _get("cricScore")
    items = data.get("data") or []
    rows = []
    for m in items:
        teams = m.get("teamInfo") or [{}, {}]
        rows.append({
            "fixture_id": m.get("id") or m.get("matchId"),
            "format":     m.get("matchType") or m.get("format"),
            "competition": m.get("series") or m.get("name"),
            "start_date": _parse_date(m.get("dateTimeGMT") or m.get("date")),
            "venue":      m.get("venue"),
            "city":       None,
            "country":    None,
            "team_home":  (teams[0].get("name") if len(teams) > 0 else None),
            "team_away":  (teams[1].get("name") if len(teams) > 1 else None),
            "status":     m.get("status"),
            "source":     "cricapi",
        })
    rows = [r for r in rows if r["fixture_id"]]
    if not rows:
        return 0
    con = connect()
    con.executemany(
        """INSERT OR REPLACE INTO fixtures
           (fixture_id, format, competition, start_date, venue, city, country,
            team_home, team_away, status, source)
           VALUES ($fixture_id, $format, $competition, $start_date, $venue,
                   $city, $country, $team_home, $team_away, $status, $source)""",
        rows,
    )
    con.close()
    return len(rows)
