"""Wikipedia venue enrichment via the MediaWiki API.

Pulls a venue's wikitext, regex-greps the infobox for capacity, ends, and
established date, and writes them onto the `venues` row. Free, no key
required, but be polite — cache aggressively and batch.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect

_HEADERS = {"User-Agent": "cricket_pipeline/0.1 (research; contact env STATSGURU_CONTACT)"}


def _cache_path(title: str) -> Path:
    h = hashlib.sha1(title.encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"wiki_{h}.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def fetch_wikitext(title: str) -> str | None:
    cache = _cache_path(title)
    if cache.exists():
        data = json.loads(cache.read_text())
        return data.get("wikitext")
    r = requests.get(
        config.WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "titles": title,
            "redirects": 1,
        },
        headers=_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    pages = (data.get("query") or {}).get("pages") or {}
    text = None
    for _, page in pages.items():
        revs = page.get("revisions") or []
        if revs:
            text = (revs[0].get("slots") or {}).get("main", {}).get("*")
            break
    cache.write_text(json.dumps({"wikitext": text or ""}))
    return text


_NUM = re.compile(r"[\d,]+")


def _parse_capacity(text: str) -> int | None:
    m = re.search(r"\|\s*capacity\s*=\s*([^\n|]+)", text, re.I)
    if not m:
        return None
    nums = _NUM.findall(m.group(1))
    if not nums:
        return None
    try:
        return int(nums[0].replace(",", ""))
    except ValueError:
        return None


def _parse_ends(text: str) -> str | None:
    m = re.search(r"\|\s*end1\s*=\s*([^\n|]+).*?\|\s*end2\s*=\s*([^\n|]+)", text, re.I | re.S)
    if not m:
        return None
    e1, e2 = m.group(1).strip(), m.group(2).strip()
    e1 = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", e1)
    e2 = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", e2)
    return f"{e1} / {e2}".strip(" /")


def _parse_established(text: str) -> int | None:
    for key in ("establishment", "established", "opened", "first_used"):
        m = re.search(rf"\|\s*{key}\s*=\s*([^\n|]+)", text, re.I)
        if m:
            yr = re.search(r"\b(1[7-9]\d{2}|20\d{2})\b", m.group(1))
            if yr:
                return int(yr.group(1))
    return None


def enrich(title: str, venue: str) -> bool:
    text = fetch_wikitext(title)
    if not text:
        return False
    capacity = _parse_capacity(text)
    ends     = _parse_ends(text)
    estd     = _parse_established(text)
    if capacity is None and ends is None and estd is None:
        return False
    con = connect()
    con.execute(
        """UPDATE venues SET
              capacity   = COALESCE(?, capacity),
              ends       = COALESCE(?, ends),
              established= COALESCE(?, established)
           WHERE venue = ?""",
        [capacity, ends, estd, venue],
    )
    con.close()
    return True


def enrich_all(limit: int | None = None) -> int:
    con = connect()
    rows = con.execute(
        """SELECT venue FROM venues
           WHERE capacity IS NULL OR ends IS NULL OR established IS NULL
           LIMIT COALESCE(?, 1000000)""",
        [limit],
    ).fetchall()
    con.close()
    n = 0
    for (venue,) in rows:
        try:
            if enrich(venue, venue):
                n += 1
        except Exception as e:
            print(f"wiki failed for {venue}: {e}")
    return n
