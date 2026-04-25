"""Backfill `players.country` from cached CricSheet zips — no Cricinfo needed.

CricSheet match JSON has `info.players` mapping each team to its players.
For *international* matches the team name *is* the country, so we can fill
`players.country` for free. For each player we pick the most-recent
international team they've played for as their canonical country.

This avoids the Cricinfo scrape for the most common metadata field.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .. import config
from ..db import connect
from .cricsheet import _iter_match_jsons, download_zip

# Recognised international team names. Anything else is treated as a
# domestic / franchise team and skipped for country attribution.
_INTERNATIONAL = {
    "Australia", "England", "India", "Pakistan", "South Africa", "New Zealand",
    "Sri Lanka", "West Indies", "Bangladesh", "Zimbabwe", "Afghanistan",
    "Ireland", "Scotland", "Netherlands", "United Arab Emirates", "Nepal",
    "Oman", "Hong Kong", "Papua New Guinea", "United States of America",
    "Canada", "Namibia", "Kenya", "Bermuda", "Jersey", "Guernsey",
}


def _walk(zip_paths: list[Path]) -> dict[str, str]:
    """Return {player_name: country}, preferring the most-recent attribution."""
    # We track (date, team) per player; the latest team wins.
    latest: dict[str, tuple[str, str]] = {}  # name -> (iso_date, team)
    for zp in zip_paths:
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp):
            for m in tqdm(_iter_match_jsons(zp), desc=zp.name):
                info = m.get("info", {}) or {}
                dates = info.get("dates") or []
                date = dates[0] if dates else ""
                players = info.get("players") or {}
                for team, names in players.items():
                    if team not in _INTERNATIONAL:
                        continue
                    for name in names or []:
                        prev = latest.get(name)
                        if prev is None or date > prev[0]:
                            latest[name] = (date, team)
    return {name: team for name, (_, team) in latest.items()}


def backfill(datasets: list[str] | None = None) -> int:
    datasets = datasets or ["all_json"]
    paths = []
    for d in datasets:
        try:
            paths.append(download_zip(d))
        except Exception as e:
            print(f"  skip {d}: {e}")
    mapping = _walk(paths)
    if not mapping:
        return 0
    con = connect()
    n = 0
    for name, country in mapping.items():
        updated = con.execute(
            "UPDATE players SET country = COALESCE(country, ?) WHERE name = ?",
            [country, name],
        ).rowcount or 0
        n += updated
    con.close()
    return n
