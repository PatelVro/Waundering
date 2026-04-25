"""OpenWeatherMap fetcher — live conditions and 5-day forecast for geocoded venues.

Uses the free tier's /weather and /forecast endpoints. Needs `OPENWEATHER_KEY`
and a geocoded venue (lat/lon must already exist in `venues` — see
ingest/venues.py).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect


def _cache_path(endpoint: str, lat: float, lon: float) -> Path:
    key = hashlib.sha1(f"{endpoint}|{lat:.4f}|{lon:.4f}|{date.today()}".encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"owm_{endpoint}_{key}.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch(endpoint: str, lat: float, lon: float) -> dict:
    if not config.OPENWEATHER_KEY:
        raise RuntimeError("Set OPENWEATHER_KEY in your environment first.")
    cache = _cache_path(endpoint, lat, lon)
    if cache.exists():
        return json.loads(cache.read_text())
    r = requests.get(
        f"{config.OPENWEATHER_BASE}/{endpoint}",
        params={"lat": lat, "lon": lon, "appid": config.OPENWEATHER_KEY, "units": "metric"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    return data


def _store(venue: str, day: date, d: dict) -> None:
    con = connect()
    con.execute(
        """INSERT OR REPLACE INTO weather_daily
           (venue, date, temp_c, humidity, dew_point,
            wind_kmh, cloud_pct, precip_mm, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'openweather')""",
        [venue, day, d.get("temp_c"), d.get("humidity"), d.get("dew_point"),
         d.get("wind_kmh"), d.get("cloud_pct"), d.get("precip_mm")],
    )
    con.close()


def _from_current(d: dict) -> dict:
    main = d.get("main", {}) or {}
    wind = d.get("wind", {}) or {}
    clouds = d.get("clouds", {}) or {}
    rain = (d.get("rain", {}) or {}).get("1h", 0.0)
    return {
        "temp_c":    main.get("temp"),
        "humidity":  main.get("humidity"),
        "dew_point": None,
        "wind_kmh":  (wind.get("speed") or 0.0) * 3.6,
        "cloud_pct": clouds.get("all"),
        "precip_mm": rain,
    }


def fetch_current(venue: str, lat: float, lon: float) -> bool:
    data = _fetch("weather", lat, lon)
    _store(venue, date.today(), _from_current(data))
    return True


def fetch_forecast(venue: str, lat: float, lon: float) -> int:
    """Store one aggregated row per forecast day (up to 5 days out)."""
    data = _fetch("forecast", lat, lon)
    by_day: dict[date, list[dict]] = {}
    for entry in data.get("list", []):
        ts = datetime.fromtimestamp(entry["dt"], tz=timezone.utc).date()
        by_day.setdefault(ts, []).append(_from_current(entry))
    count = 0
    for day, entries in by_day.items():
        agg = {
            "temp_c":    _avg(entries, "temp_c"),
            "humidity":  _avg(entries, "humidity"),
            "dew_point": None,
            "wind_kmh":  _avg(entries, "wind_kmh"),
            "cloud_pct": _avg(entries, "cloud_pct"),
            "precip_mm": sum((e.get("precip_mm") or 0) for e in entries),
        }
        _store(venue, day, agg)
        count += 1
    return count


def _avg(entries: list[dict], key: str) -> float | None:
    vals = [e.get(key) for e in entries if e.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def fetch_all_venues(limit: int | None = None, include_forecast: bool = True) -> int:
    con = connect()
    rows = con.execute(
        """SELECT venue, lat, lon FROM venues
           WHERE lat IS NOT NULL AND lon IS NOT NULL
           LIMIT COALESCE(?, 1000000)""",
        [limit],
    ).fetchall()
    con.close()
    ok = 0
    for venue, lat, lon in rows:
        try:
            fetch_current(venue, lat, lon)
            if include_forecast:
                fetch_forecast(venue, lat, lon)
            ok += 1
        except Exception as e:
            print(f"owm failed for {venue}: {e}")
    return ok
