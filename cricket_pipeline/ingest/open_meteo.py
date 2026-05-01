"""Open-Meteo weather fetcher — no API key required.

Free for non-commercial use; daily aggregates go back to 1940 via
archive-api.open-meteo.com. Forecast endpoint at api.open-meteo.com.

Stores into the existing `weather_daily` table with source='open-meteo'
so the same downstream feature builder works regardless of which provider
backfilled a given (venue, date).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect


_HIST_URL = "https://archive-api.open-meteo.com/v1/archive"
_FCST_URL = "https://api.open-meteo.com/v1/forecast"
_GEO_URL  = "https://geocoding-api.open-meteo.com/v1/search"

_CACHE_DIR = config.CACHE_DIR / "open_meteo"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(kind: str, lat: float, lon: float, day: date) -> Path:
    h = hashlib.sha1(f"{kind}|{lat:.4f}|{lon:.4f}|{day}".encode()).hexdigest()[:16]
    return _CACHE_DIR / f"om_{kind}_{h}.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=1, max=10))
def _geocode(query: str) -> tuple[float, float] | None:
    """Convert a venue/location string to (lat, lon) via Open-Meteo's geocoding."""
    if not query: return None
    cache = _CACHE_DIR / f"geo_{hashlib.sha1(query.encode()).hexdigest()[:16]}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("latitude") and cached.get("longitude"):
            return (cached["latitude"], cached["longitude"])
    r = requests.get(_GEO_URL, params={"name": query, "count": 1, "format": "json"}, timeout=15)
    r.raise_for_status()
    data = r.json() or {}
    results = data.get("results") or []
    if not results: return None
    res = results[0]
    cache.write_text(json.dumps(res))
    return (res["latitude"], res["longitude"])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=1, max=10))
def _fetch_one(kind: str, lat: float, lon: float, day: date) -> dict | None:
    cache = _cache_path(kind, lat, lon, day)
    if cache.exists():
        return json.loads(cache.read_text())
    if kind == "hist":
        url = _HIST_URL
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": day.isoformat(), "end_date": day.isoformat(),
            "daily": ",".join([
                "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                "relative_humidity_2m_mean", "dew_point_2m_mean",
                "wind_speed_10m_max", "cloud_cover_mean", "precipitation_sum",
            ]),
            "timezone": "GMT",
        }
    else:
        url = _FCST_URL
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": day.isoformat(), "end_date": day.isoformat(),
            "daily": ",".join([
                "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                "relative_humidity_2m_mean", "dew_point_2m_mean",
                "wind_speed_10m_max", "cloud_cover_mean", "precipitation_sum",
            ]),
            "timezone": "GMT",
        }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code == 400:
        # bad request (e.g., date out of range) — cache empty result so we don't retry
        cache.write_text("{}")
        return {}
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    return data


def _extract_day(data: dict) -> dict | None:
    daily = data.get("daily") or {}
    if not daily.get("time"): return None
    return {
        "temp_c":    (daily.get("temperature_2m_mean") or [None])[0],
        "humidity":  (daily.get("relative_humidity_2m_mean") or [None])[0],
        "dew_point": (daily.get("dew_point_2m_mean") or [None])[0],
        "wind_kmh":  (daily.get("wind_speed_10m_max") or [None])[0],
        "cloud_pct": (daily.get("cloud_cover_mean") or [None])[0],
        "precip_mm": (daily.get("precipitation_sum") or [None])[0],
    }


def store_weather(venue: str, lat: float, lon: float, day: date,
                   forecast: bool = False) -> bool:
    kind = "fcst" if forecast else "hist"
    try:
        data = _fetch_one(kind, lat, lon, day)
    except Exception as e:
        print(f"open-meteo {kind} failed for {venue} {day}: {e}")
        return False
    if not data:
        return False
    extracted = _extract_day(data)
    if not extracted: return False
    con = connect()
    con.execute("""
        INSERT OR REPLACE INTO weather_daily
           (venue, date, temp_c, humidity, dew_point,
            wind_kmh, cloud_pct, precip_mm, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [venue, day,
          extracted["temp_c"], extracted["humidity"], extracted["dew_point"],
          extracted["wind_kmh"], extracted["cloud_pct"], extracted["precip_mm"],
          "open-meteo-" + kind])
    con.close()
    return True


def backfill_from_matches(limit: int | None = None,
                           max_age_days: int | None = 365) -> dict:
    """Backfill historical weather for recent matches in the DB. Looks up
    venue→(lat,lon) via the existing `venues` table when available, falls
    back to geocoding the venue name otherwise."""
    con = connect()
    rows = con.execute("""
        SELECT m.venue, v.lat, v.lon, m.city, m.country, m.start_date
        FROM matches m
        LEFT JOIN venues v ON v.venue = m.venue
        WHERE m.venue IS NOT NULL AND m.start_date IS NOT NULL
          AND (CAST(? AS INTEGER) IS NULL
               OR m.start_date >= CAST(? AS DATE) - INTERVAL (CAST(? AS INTEGER)) DAY)
        ORDER BY m.start_date DESC
        LIMIT COALESCE(?, 1000000)
    """, [max_age_days, datetime.now(timezone.utc).date(), max_age_days, limit]).fetchall()
    con.close()

    seen = set()
    n_geo = 0; n_ok = 0; n_skip = 0; n_fail = 0
    for venue, lat, lon, city, country, start_date in rows:
        key = (venue, start_date)
        if key in seen: continue
        seen.add(key)
        if lat is None or lon is None:
            q = ", ".join(x for x in (city, country) if x) or venue
            try:
                ll = _geocode(q) or _geocode(venue)
            except Exception as e:
                print(f"geocode failed for {q}: {e}"); ll = None
            if not ll:
                n_skip += 1
                continue
            lat, lon = ll
            n_geo += 1
        try:
            if store_weather(venue, lat, lon, start_date, forecast=False):
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:
            print(f"weather failed for {venue} {start_date}: {e}")
            n_fail += 1
        time.sleep(0.05)   # be polite to the free endpoint
    return {"ok": n_ok, "geocoded": n_geo, "skipped": n_skip, "failed": n_fail}


def _venue_to_query(venue: str) -> list[str]:
    """Open-Meteo's geocoder finds cities, not stadium names. Build a list of
    increasingly-loose query strings to try."""
    qs = []
    if not venue: return qs
    qs.append(venue)
    # 'Stadium, City' → try just 'City'
    if "," in venue:
        parts = [p.strip() for p in venue.split(",")]
        for p in reversed(parts):       # innermost (city) first
            if p and p not in qs: qs.append(p)
    # also try the LAST whitespace-token (often the city for "X Stadium Y")
    last = venue.split()[-1] if venue.split() else None
    if last and last not in qs: qs.append(last)
    return qs


def fetch_forecast(venue: str, day: date,
                    city: str | None = None, country: str | None = None) -> dict | None:
    """Fetch forecast weather for an upcoming match. Resolves venue → lat/lon
    via the venues table or geocodes (city-name first, then venue tokens)."""
    con = connect()
    row = con.execute("SELECT lat, lon, city, country FROM venues WHERE venue = ? LIMIT 1",
                       [venue]).fetchone()
    con.close()
    if row and row[0] is not None and row[1] is not None:
        lat, lon = row[0], row[1]
    else:
        # Build query candidates: provided city/country first, then venue tokens
        candidates = []
        if city and country: candidates.append(f"{city}, {country}")
        if city:             candidates.append(city)
        if row and row[2] and row[3]: candidates.append(f"{row[2]}, {row[3]}")
        if row and row[2]:            candidates.append(row[2])
        candidates.extend(_venue_to_query(venue))
        ll = None
        for q in candidates:
            try:
                ll = _geocode(q)
                if ll: break
            except Exception: continue
        if not ll: return None
        lat, lon = ll
    if not store_weather(venue, lat, lon, day, forecast=True):
        return None
    con = connect()
    out = con.execute(
        "SELECT temp_c, humidity, dew_point, wind_kmh, cloud_pct, precip_mm "
        "FROM weather_daily WHERE venue = ? AND date = ? LIMIT 1",
        [venue, day]).fetchone()
    con.close()
    if not out: return None
    return {
        "temp_c": out[0], "humidity": out[1], "dew_point": out[2],
        "wind_kmh": out[3], "cloud_pct": out[4], "precip_mm": out[5],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="historical backfill from matches table")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-age-days", type=int, default=365,
                    help="only backfill matches in the last N days")
    args = ap.parse_args()
    if args.backfill:
        out = backfill_from_matches(limit=args.limit, max_age_days=args.max_age_days)
        print(json.dumps(out, indent=2))
