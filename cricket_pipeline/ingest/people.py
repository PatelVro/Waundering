"""CricSheet people registry — cross-provider player identifiers.

Downloads https://cricsheet.org/register/people.csv and upserts into
`players`. The registry links CricSheet's internal player id to Cricinfo,
Cricbuzz, BCCI, NVPlay, Opta, Pulse, and others — the foundation for joining
data across sources.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import requests

from .. import config
from ..db import connect


def download(force: bool = False) -> Path:
    dest = config.CACHE_DIR / "people.csv"
    if dest.exists() and not force:
        return dest
    print(f"Downloading {config.CRICSHEET_PEOPLE_CSV}")
    r = requests.get(config.CRICSHEET_PEOPLE_CSV, timeout=60)
    r.raise_for_status()
    dest.write_text(r.text, encoding="utf-8")
    return dest


def ingest(force: bool = False) -> int:
    path = download(force=force)
    rows = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            pid = r.get("identifier") or r.get("key_cricsheet")
            if not pid:
                continue
            rows.append({
                "player_id":    pid,
                "name":         r.get("name") or r.get("unique_name"),
                "country":      None,
                "batting_hand": None,
                "bowling_type": None,
            })
    if not rows:
        return 0
    con = connect()
    con.executemany(
        """INSERT OR REPLACE INTO players
           (player_id, name, country, batting_hand, bowling_type)
           VALUES ($player_id, $name, $country, $batting_hand, $bowling_type)""",
        rows,
    )
    con.close()
    return len(rows)
