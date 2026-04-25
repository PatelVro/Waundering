"""Polite Statsguru scraper for player career splits.

Statsguru has no official API — we scrape the HTML tables. Be kind: set a
descriptive User-Agent via STATSGURU_CONTACT env var, cache responses, and
sleep between requests. Only scrape for personal / research use.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect

FORMAT_CLASS = {"test": 1, "odi": 2, "t20i": 3, "ipl": 6}
STAT_TYPE = {"batting": "batting", "bowling": "bowling"}


def _headers() -> dict:
    contact = os.environ.get("STATSGURU_CONTACT", "")
    ua = config.STATSGURU_USER_AGENT
    if contact:
        ua = ua.replace("set via STATSGURU_CONTACT env var", contact)
    return {"User-Agent": ua, "Accept": "text/html"}


def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"statsguru_{h}.html"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _fetch(url: str) -> str:
    cache = _cache_path(url)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    time.sleep(config.STATSGURU_SLEEP_SECONDS)
    r = requests.get(url, headers=_headers(), timeout=30)
    r.raise_for_status()
    cache.write_text(r.text, encoding="utf-8")
    return r.text


def build_url(
    stat: Literal["batting", "bowling"],
    fmt: Literal["test", "odi", "t20i", "ipl"],
    filters: dict | None = None,
) -> str:
    """Minimal query builder. Pass extra filters in the Statsguru URL format.

    Example filters: {"orderby": "runs", "qualmin1": "1000", "qualval1": "runs"}
    """
    params = {
        "class": FORMAT_CLASS[fmt],
        "type": STAT_TYPE[stat],
        "template": "results",
    }
    if filters:
        params.update(filters)
    q = ";".join(f"{k}={v}" for k, v in params.items())
    return f"{config.STATSGURU_BASE}?{q}"


def parse_table(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="engineTable")
    if not table:
        return []
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr", class_="data1"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _to_int(x):
    try:
        return int(str(x).replace(",", "").replace("-", "0") or 0)
    except ValueError:
        return None


def _to_float(x):
    try:
        return float(str(x).replace("-", "nan") or "nan")
    except ValueError:
        return None


def fetch_player_overall(
    stat: Literal["batting", "bowling"],
    fmt: Literal["test", "odi", "t20i", "ipl"],
) -> list[dict]:
    url = build_url(stat, fmt, filters={"orderby": "matches"})
    html = _fetch(url)
    return parse_table(html)


# Statsguru opposition codes (subset — extend as needed).
OPPOSITION_CODES = {
    "India": 6, "Australia": 2, "England": 1, "South Africa": 3,
    "West Indies": 4, "New Zealand": 5, "Pakistan": 7, "Sri Lanka": 8,
    "Zimbabwe": 9, "Bangladesh": 25, "Afghanistan": 40, "Ireland": 29,
}


def fetch_split(
    stat: Literal["batting", "bowling"],
    fmt: Literal["test", "odi", "t20i", "ipl"],
    groupby: Literal["year", "ground", "opposition", "season"] | None = None,
    opposition: str | None = None,
    ground_id: int | None = None,
    year: int | None = None,
) -> list[dict]:
    """Fetch a Statsguru table grouped by the chosen dimension.

    Example: fetch T20I batting grouped by year, for England only:
        fetch_split("batting", "t20i", groupby="year", opposition="England")
    """
    filters: dict = {"orderby": "matches"}
    if groupby:
        filters["groupby"] = groupby
    if opposition and opposition in OPPOSITION_CODES:
        filters["opposition"] = OPPOSITION_CODES[opposition]
    if ground_id:
        filters["ground"] = ground_id
    if year:
        filters["spanmin1"] = f"01+Jan+{year}"
        filters["spanmax1"] = f"31+Dec+{year}"
        filters["spanval1"] = "span"
    url = build_url(stat, fmt, filters=filters)
    html = _fetch(url)
    return parse_table(html)


def store_splits(rows: list[dict], fmt: str, split_type: str = "overall", split_key: str = "all"):
    if not rows:
        return 0
    con = connect()
    payload = []
    for r in rows:
        payload.append({
            "player_name": r.get("Player") or r.get("Name"),
            "format":      fmt,
            "split_type":  split_type,
            "split_key":   split_key,
            "matches":     _to_int(r.get("Mat")),
            "innings":     _to_int(r.get("Inns")),
            "runs":        _to_int(r.get("Runs")),
            "balls":       _to_int(r.get("BF")) or _to_int(r.get("Balls")),
            "avg":         _to_float(r.get("Ave")),
            "sr":          _to_float(r.get("SR")),
            "hs":          _to_int((r.get("HS") or "").replace("*", "")),
            "hundreds":    _to_int(r.get("100")),
            "fifties":     _to_int(r.get("50")),
            "wickets":     _to_int(r.get("Wkts")),
            "bbi":         r.get("BBI"),
            "econ":        _to_float(r.get("Econ")),
            "bowl_avg":    _to_float(r.get("Ave")) if "Wkts" in r else None,
            "bowl_sr":     _to_float(r.get("SR")) if "Wkts" in r else None,
        })
    con.executemany(
        """INSERT OR REPLACE INTO player_splits
            (player_name, format, split_type, split_key,
             matches, innings, runs, balls, avg, sr, hs,
             hundreds, fifties, wickets, bbi, econ, bowl_avg, bowl_sr)
           VALUES ($player_name, $format, $split_type, $split_key,
             $matches, $innings, $runs, $balls, $avg, $sr, $hs,
             $hundreds, $fifties, $wickets, $bbi, $econ, $bowl_avg, $bowl_sr)""",
        payload,
    )
    con.close()
    return len(payload)
