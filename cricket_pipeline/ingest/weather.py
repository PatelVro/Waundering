"""Visual Crossing weather fetcher — historical daily weather per venue.

Set VISUAL_CROSSING_KEY in env. Free tier: 1000 records/day.
Historical data goes back decades, which is ideal for backfilling past matches.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect


def _cache_path(location: str, day: date) -> Path:
    key = hashlib.sha1(f"{location}|{day}".encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"weather_{key}.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def fetch_day(location: str, day: date) -> dict | None:
    if not config.VISUAL_CROSSING_KEY:
        raise RuntimeError("Set VISUAL_CROSSING_KEY in your environment first.")
    cache = _cache_path(location, day)
    if cache.exists():
        return json.loads(cache.read_text())
    url = (
        f"{config.VISUAL_CROSSING_BASE}/{requests.utils.quote(location)}/"
        f"{day.isoformat()}/{day.isoformat()}"
    )
    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": config.VISUAL_CROSSING_KEY,
        "contentType": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    return data


def _extract_day(data: dict) -> dict | None:
    days = data.get("days") or []
    if not days:
        return None
    d = days[0]
    return {
        "temp_c":    d.get("temp"),
        "humidity":  d.get("humidity"),
        "dew_point": d.get("dew"),
        "wind_kmh":  d.get("windspeed"),
        "cloud_pct": d.get("cloudcover"),
        "precip_mm": d.get("precip"),
    }


def store_weather(venue: str, location: str, day: date) -> bool:
    data = fetch_day(location, day)
    if not data:
        return False
    extracted = _extract_day(data)
    if not extracted:
        return False
    con = connect()
    con.execute(
        """INSERT OR REPLACE INTO weather_daily
           (venue, date, temp_c, humidity, dew_point,
            wind_kmh, cloud_pct, precip_mm, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'visualcrossing')""",
        [
            venue, day,
            extracted["temp_c"], extracted["humidity"], extracted["dew_point"],
            extracted["wind_kmh"], extracted["cloud_pct"], extracted["precip_mm"],
        ],
    )
    con.close()
    return True


def backfill_from_matches(limit: int | None = None) -> int:
    """Fetch weather for every (venue, match_date) pair already in the DB."""
    con = connect()
    rows = con.execute(
        """SELECT DISTINCT venue, city, country, start_date
           FROM matches
           WHERE venue IS NOT NULL AND start_date IS NOT NULL
           ORDER BY start_date DESC
           LIMIT COALESCE(?, 1000000)""",
        [limit],
    ).fetchall()
    con.close()
    ok = 0
    for venue, city, country, start_date in rows:
        loc = ", ".join(x for x in (city, country) if x) or venue
        try:
            if store_weather(venue, loc, start_date):
                ok += 1
        except Exception as e:
            print(f"weather failed for {venue} {start_date}: {e}")
    return ok
