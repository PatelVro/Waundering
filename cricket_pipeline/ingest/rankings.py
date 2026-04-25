"""ICC rankings scraper (men's Test / ODI / T20I + team).

The ICC ranking pages render a hero row plus a table. Scraping is brittle —
if the page layout changes, update `_parse_player_table` / `_parse_team_table`.
Responses are cached; set ICC_SLEEP to be a good citizen.
"""

from __future__ import annotations

import hashlib
import time
from datetime import date
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect

Format = Literal["test", "odi", "t20i"]
Category = Literal["batting", "bowling", "allrounder"]


def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"icc_{h}.html"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch(url: str) -> str:
    cache = _cache_path(url)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    time.sleep(config.ICC_SLEEP_SECONDS)
    r = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (cricket_pipeline research)"},
        timeout=30,
    )
    r.raise_for_status()
    cache.write_text(r.text, encoding="utf-8")
    return r.text


def _to_int(s: str) -> int | None:
    try:
        return int("".join(c for c in s if c.isdigit()))
    except ValueError:
        return None


def _parse_player_table(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows = []

    hero = soup.select_one(".rankings-block__banner") or soup.select_one(".rankings-block__hero")
    if hero:
        name = hero.select_one(".rankings-block__banner--name, .rankings-block__hero--name")
        country = hero.select_one(".rankings-block__banner--nationality, .rankings-block__hero--nationality")
        rating = hero.select_one(".rankings-block__banner--rating, .rankings-block__hero--rating")
        if name:
            rows.append({
                "rank":    1,
                "name":    name.get_text(strip=True),
                "country": country.get_text(strip=True) if country else None,
                "rating":  _to_int(rating.get_text(strip=True)) if rating else None,
            })

    for tr in soup.select("tr.table-body, tbody tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(cells) < 4:
            continue
        rank, name, country, rating = cells[0], cells[1], cells[2], cells[3]
        if not rank.isdigit():
            continue
        rows.append({
            "rank":    int(rank),
            "name":    name,
            "country": country,
            "rating":  _to_int(rating),
        })
    return rows


def _parse_team_table(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("tr.table-body, tbody tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(cells) < 4:
            continue
        rank = cells[0]
        if not rank.isdigit():
            continue
        rows.append({
            "rank":    int(rank),
            "name":    cells[1],
            "country": cells[1],
            "rating":  _to_int(cells[-1]),
        })
    return rows


def fetch_player_rankings(fmt: Format, category: Category) -> list[dict]:
    url = f"{config.ICC_RANKINGS_BASE}/{fmt}/{category}"
    html = _fetch(url)
    return _parse_player_table(html)


def fetch_team_rankings(fmt: Format) -> list[dict]:
    url = f"{config.ICC_TEAM_RANKINGS_BASE}/{fmt}"
    html = _fetch(url)
    return _parse_team_table(html)


def store(rows: list[dict], fmt: str, category: str, snapshot: date | None = None) -> int:
    if not rows:
        return 0
    snapshot = snapshot or date.today()
    con = connect()
    payload = [{
        "snapshot_date": snapshot,
        "format":        fmt,
        "category":      category,
        "rank":          r["rank"],
        "name":          r["name"],
        "country":       r.get("country"),
        "rating":        r.get("rating"),
    } for r in rows]
    con.executemany(
        """INSERT OR REPLACE INTO rankings
           (snapshot_date, format, category, rank, name, country, rating)
           VALUES ($snapshot_date, $format, $category, $rank, $name, $country, $rating)""",
        payload,
    )
    con.close()
    return len(payload)


def ingest_all() -> dict:
    results = {}
    for fmt in ("test", "odi", "t20i"):
        for cat in ("batting", "bowling", "allrounder"):
            try:
                rows = fetch_player_rankings(fmt, cat)
                results[f"{fmt}_{cat}"] = store(rows, fmt, cat)
            except Exception as e:
                results[f"{fmt}_{cat}"] = f"error: {e}"
        try:
            rows = fetch_team_rankings(fmt)
            results[f"{fmt}_team"] = store(rows, fmt, "team")
        except Exception as e:
            results[f"{fmt}_team"] = f"error: {e}"
    return results
