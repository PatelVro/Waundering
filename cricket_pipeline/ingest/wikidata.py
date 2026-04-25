"""Wikidata SPARQL player enrichment.

Wikidata has structured data on thousands of cricketers — height, dominant
hand, debut year, country of citizenship, sometimes Cricinfo IDs. We fetch
the lot in batches via SPARQL and merge by either `key_cricinfo` (preferred)
or fuzzy name match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect

# Q5375 = cricket. P641 = sport. P106 = occupation; we keep it open.
SPARQL = """
SELECT ?player ?playerLabel ?dob ?height ?countryLabel ?cricinfoId ?debut WHERE {
  ?player wdt:P641 wd:Q5375 .
  OPTIONAL { ?player wdt:P569  ?dob . }
  OPTIONAL { ?player wdt:P2048 ?height . }
  OPTIONAL { ?player wdt:P27   ?country . }
  OPTIONAL { ?player wdt:P2697 ?cricinfoId . }
  OPTIONAL { ?player wdt:P2031 ?debut . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d OFFSET %d
"""


def _cache_path(query: str) -> Path:
    h = hashlib.sha1(query.encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"wikidata_{h}.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
def _query(sparql: str) -> dict:
    cache = _cache_path(sparql)
    if cache.exists():
        return json.loads(cache.read_text())
    r = requests.get(
        config.WIKIDATA_SPARQL,
        params={"query": sparql, "format": "json"},
        headers={
            "User-Agent": "cricket_pipeline/0.1 (research)",
            "Accept": "application/sparql-results+json",
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    return data


def _val(b, key: str) -> str | None:
    v = (b.get(key) or {}).get("value")
    return v if v else None


def _qid(b, key: str) -> str | None:
    v = _val(b, key)
    return v.rsplit("/", 1)[-1] if v else None


def _to_int(s: str | None) -> int | None:
    try:
        return int(float(s)) if s else None
    except (TypeError, ValueError):
        return None


def _year_from_iso(s: str | None) -> int | None:
    return int(s[:4]) if s and len(s) >= 4 and s[:4].isdigit() else None


def fetch_all(batch: int = 1000, max_offset: int = 10_000) -> list[dict]:
    rows = []
    offset = 0
    while offset < max_offset:
        data = _query(SPARQL % (batch, offset))
        bindings = (data.get("results") or {}).get("bindings") or []
        if not bindings:
            break
        for b in bindings:
            rows.append({
                "qid":        _qid(b, "player"),
                "name":       _val(b, "playerLabel"),
                "dob":        (_val(b, "dob") or "")[:10] or None,
                "height_cm":  _to_int(_val(b, "height")),
                "country":    _val(b, "countryLabel"),
                "cricinfo":   _val(b, "cricinfoId"),
                "debut_year": _year_from_iso(_val(b, "debut")),
            })
        if len(bindings) < batch:
            break
        offset += batch
    return rows


def merge() -> int:
    rows = fetch_all()
    if not rows:
        return 0
    con = connect()
    matched = 0
    for r in rows:
        # Prefer match on key_cricinfo; fall back to exact name.
        if r["cricinfo"]:
            updated = con.execute(
                """UPDATE players SET
                      dob          = COALESCE(dob, TRY_CAST(? AS DATE)),
                      height_cm    = COALESCE(height_cm, ?),
                      country      = COALESCE(country, ?),
                      debut_year   = COALESCE(debut_year, ?),
                      key_wikidata = COALESCE(key_wikidata, ?)
                   WHERE key_cricinfo = ?""",
                [r["dob"], r["height_cm"], r["country"], r["debut_year"], r["qid"], r["cricinfo"]],
            ).rowcount or 0
            matched += updated
            if updated:
                continue
        if r["name"]:
            updated = con.execute(
                """UPDATE players SET
                      dob          = COALESCE(dob, TRY_CAST(? AS DATE)),
                      height_cm    = COALESCE(height_cm, ?),
                      country      = COALESCE(country, ?),
                      debut_year   = COALESCE(debut_year, ?),
                      key_wikidata = COALESCE(key_wikidata, ?)
                   WHERE name = ? OR unique_name = ?""",
                [r["dob"], r["height_cm"], r["country"], r["debut_year"],
                 r["qid"], r["name"], r["name"]],
            ).rowcount or 0
            matched += updated
    con.close()
    return matched
