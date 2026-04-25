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
                "unique_name":  r.get("unique_name"),
                "key_cricinfo": r.get("key_cricinfo"),
                "key_cricbuzz": r.get("key_cricbuzz"),
                "key_bcci":     r.get("key_bcci"),
                "key_opta":     r.get("key_opta"),
                "key_nvplay":   r.get("key_nvplay"),
                "key_pulse":    r.get("key_pulse"),
            })
    if not rows:
        return 0
    con = connect()
    con.executemany(
        """INSERT INTO players (
              player_id, name, unique_name,
              key_cricinfo, key_cricbuzz, key_bcci, key_opta, key_nvplay, key_pulse
           ) VALUES (
              $player_id, $name, $unique_name,
              $key_cricinfo, $key_cricbuzz, $key_bcci, $key_opta, $key_nvplay, $key_pulse
           )
           ON CONFLICT (player_id) DO UPDATE SET
              name         = COALESCE(excluded.name, players.name),
              unique_name  = COALESCE(excluded.unique_name, players.unique_name),
              key_cricinfo = COALESCE(excluded.key_cricinfo, players.key_cricinfo),
              key_cricbuzz = COALESCE(excluded.key_cricbuzz, players.key_cricbuzz),
              key_bcci     = COALESCE(excluded.key_bcci, players.key_bcci),
              key_opta     = COALESCE(excluded.key_opta, players.key_opta),
              key_nvplay   = COALESCE(excluded.key_nvplay, players.key_nvplay),
              key_pulse    = COALESCE(excluded.key_pulse, players.key_pulse)""",
        rows,
    )
    con.close()
    return len(rows)
