"""Cricbuzz live match-state fetcher (best-effort, unofficial endpoint).

Cricbuzz exposes JSON-ish endpoints that power their live scorecards. They
are not officially documented, the path occasionally changes, and there is
no SLA. Use only for personal/research projects and back off aggressively.

Stores one row per fetch into `live_state` so you can replay how the match
unfolded.
"""

from __future__ import annotations

import json
from datetime import datetime

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (cricket_pipeline live state)",
    "Accept": "application/json",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def fetch_match_json(match_id: str) -> dict | None:
    url = f"{config.CRICBUZZ_MATCH_API}/{match_id}"
    r = requests.get(url, headers=_HEADERS, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def _extract(data: dict) -> dict:
    """Best-effort field extraction; tolerant to schema drift."""
    mh = data.get("matchHeader") or {}
    ms = data.get("miniscore") or {}
    bs = ms.get("batsmanStriker") or {}
    bn = ms.get("batsmanNonStriker") or {}
    bw = ms.get("bowlerStriker") or {}
    return {
        "status":        mh.get("status") or ms.get("status"),
        "score":         f"{ms.get('batTeamScore', '')}/{ms.get('batTeamWkts', '')}".strip("/"),
        "overs":         str(ms.get("overs") or ""),
        "striker":       bs.get("batName"),
        "striker_runs":  bs.get("batRuns"),
        "striker_balls": bs.get("batBalls"),
        "non_striker":   bn.get("batName"),
        "bowler":        bw.get("bowlName"),
        "last_ball":     ms.get("lastWicket") or ms.get("recentOvsStats"),
    }


def snapshot(match_id: str) -> bool:
    data = fetch_match_json(match_id)
    if not data:
        return False
    extracted = _extract(data)
    con = connect()
    con.execute(
        """INSERT OR REPLACE INTO live_state
           (match_id, fetched_at, status, score, overs,
            striker, striker_runs, striker_balls, non_striker,
            bowler, last_ball, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            match_id, datetime.utcnow(),
            extracted["status"], extracted["score"], extracted["overs"],
            extracted["striker"], extracted["striker_runs"], extracted["striker_balls"],
            extracted["non_striker"], extracted["bowler"], extracted["last_ball"],
            json.dumps(data)[:500_000],
        ],
    )
    con.close()
    return True


def snapshot_many(match_ids: list[str]) -> int:
    n = 0
    for mid in match_ids:
        try:
            if snapshot(mid):
                n += 1
        except Exception as e:
            print(f"cricbuzz failed for {mid}: {e}")
    return n
