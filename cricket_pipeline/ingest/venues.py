"""Geocode venues using OpenStreetMap Nominatim.

Nominatim is free but strict: 1 req/sec max and must send a descriptive
User-Agent. Responses are cached to disk forever.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect


def _headers() -> dict:
    contact = os.environ.get("STATSGURU_CONTACT", "cricket_pipeline@example.com")
    return {"User-Agent": f"cricket_pipeline/0.1 ({contact})"}


def _cache_path(query: str) -> Path:
    h = hashlib.sha1(query.encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"nominatim_{h}.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _search(query: str) -> list[dict]:
    cache = _cache_path(query)
    if cache.exists():
        return json.loads(cache.read_text())
    time.sleep(config.NOMINATIM_SLEEP_SECONDS)
    r = requests.get(
        config.NOMINATIM_BASE,
        params={"q": query, "format": "json", "limit": 1},
        headers=_headers(),
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    return data


def geocode_venue(venue: str, city: str | None = None, country: str | None = None) -> tuple[float, float] | None:
    candidates = [
        ", ".join(x for x in (venue, city, country) if x),
        ", ".join(x for x in (venue, country) if x),
        venue,
    ]
    for q in dict.fromkeys(candidates):
        try:
            hits = _search(q)
        except Exception as e:
            print(f"nominatim error for '{q}': {e}")
            continue
        if hits:
            h = hits[0]
            return float(h["lat"]), float(h["lon"])
    return None


def enrich_from_matches(limit: int | None = None) -> int:
    con = connect()
    rows = con.execute(
        """SELECT DISTINCT m.venue, m.city, m.country
           FROM matches m
           LEFT JOIN venues v ON v.venue = m.venue
           WHERE m.venue IS NOT NULL AND v.lat IS NULL
           LIMIT COALESCE(?, 1000000)""",
        [limit],
    ).fetchall()
    count = 0
    for venue, city, country in rows:
        coords = geocode_venue(venue, city, country)
        if not coords:
            continue
        lat, lon = coords
        con.execute(
            """INSERT OR REPLACE INTO venues (venue, city, country, lat, lon, boundary_m, notes)
               VALUES (?, ?, ?, ?, ?, NULL, NULL)""",
            [venue, city, country, lat, lon],
        )
        count += 1
    con.close()
    return count
