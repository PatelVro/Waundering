"""ESPNCricinfo player profile enricher.

Uses `key_cricinfo` from people.csv to construct profile URLs and parse:
batting hand, bowling style, role, country, date of birth. The page layout
shifts occasionally — extraction tries JSON-LD first, then a label/value
walk over the structured info block, and falls back to plain-text regex.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect

LABELS = {
    "Full Name":        "full_name",
    "Born":             "born",
    "Age":              "age",
    "Batting Style":    "batting_hand",
    "Bowling Style":    "bowling_type",
    "Playing Role":     "role",
    "Role":             "role",
    "Major teams":      "teams",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "x"


def _headers() -> dict:
    contact = os.environ.get("STATSGURU_CONTACT", "research")
    return {
        "User-Agent": f"Mozilla/5.0 (cricket_pipeline; contact: {contact})",
        "Accept": "text/html,application/xhtml+xml",
    }


def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"cricinfo_{h}.html"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch(url: str) -> str:
    cache = _cache_path(url)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    time.sleep(config.CRICINFO_SLEEP_SECONDS)
    r = requests.get(url, headers=_headers(), timeout=30)
    r.raise_for_status()
    cache.write_text(r.text, encoding="utf-8")
    return r.text


def profile_url(name: str, key_cricinfo: str) -> str:
    return f"{config.CRICINFO_PROFILE_BASE}/{_slug(name)}-{key_cricinfo}"


def _parse_jsonld(soup: BeautifulSoup) -> dict:
    out: dict = {}
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if item.get("@type") == "Person":
                out["full_name"] = item.get("name") or out.get("full_name")
                out["dob"] = item.get("birthDate") or out.get("dob")
                bp = item.get("birthPlace")
                if isinstance(bp, dict):
                    out["country"] = (bp.get("addressCountry") or out.get("country"))
    return out


def _parse_info_block(soup: BeautifulSoup) -> dict:
    out: dict = {}
    text = soup.get_text("\n", strip=True)
    for label, key in LABELS.items():
        m = re.search(rf"{re.escape(label)}\s*[\n:]\s*(.+)", text)
        if m:
            out[key] = m.group(1).split("\n")[0].strip()
    return out


def _parse_dob(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s.split(",")[0].strip() if "," in s and "-" not in s else s, fmt).date()
        except ValueError:
            continue
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", s)
    if m:
        return date(int(m.group(1)), 1, 1)
    return None


def _normalise_country(s: str | None) -> str | None:
    if not s:
        return None
    s = s.split(",")[-1].strip()
    return s or None


def parse_profile(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    data = _parse_jsonld(soup)
    data.update({k: v for k, v in _parse_info_block(soup).items() if v and not data.get(k)})
    return {
        "full_name":    data.get("full_name"),
        "dob":          _parse_dob(data.get("dob") or data.get("born")),
        "country":      _normalise_country(data.get("country") or data.get("born")),
        "role":         data.get("role"),
        "batting_hand": data.get("batting_hand"),
        "bowling_type": data.get("bowling_type"),
    }


def enrich_player(player_id: str, name: str, key_cricinfo: str) -> bool:
    url = profile_url(name, key_cricinfo)
    try:
        html = _fetch(url)
    except Exception as e:
        print(f"  cricinfo fetch failed for {name}: {e}")
        return False
    parsed = parse_profile(html)
    if not any(parsed.values()):
        return False
    con = connect()
    con.execute(
        """UPDATE players SET
              dob          = COALESCE(?, dob),
              role         = COALESCE(?, role),
              batting_hand = COALESCE(?, batting_hand),
              bowling_type = COALESCE(?, bowling_type),
              country      = COALESCE(?, country),
              profile_url  = ?,
              enriched_at  = CURRENT_TIMESTAMP
           WHERE player_id = ?""",
        [parsed["dob"], parsed["role"], parsed["batting_hand"],
         parsed["bowling_type"], parsed["country"], url, player_id],
    )
    con.close()
    return True


def enrich_all(limit: int | None = None, only_active: bool = True) -> int:
    con = connect()
    sql = """SELECT player_id, name, key_cricinfo
             FROM players
             WHERE key_cricinfo IS NOT NULL AND enriched_at IS NULL"""
    if only_active:
        sql += """ AND name IN (
                     SELECT DISTINCT batter   FROM balls
                     UNION SELECT DISTINCT bowler FROM balls
                  )"""
    sql += " LIMIT COALESCE(?, 1000000)"
    rows = con.execute(sql, [limit]).fetchall()
    con.close()
    n = 0
    for player_id, name, key in rows:
        if enrich_player(player_id, name, key):
            n += 1
    return n
