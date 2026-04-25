"""GDELT 2.0 DOC API — global news event database.

GDELT monitors the world's broadcast, print, and web news in 100+ languages.
The DOC API is free and key-less. We query for cricket-related articles and
land them into the same `news` table with sentiment + entity tags.
"""

from __future__ import annotations

import json
from datetime import datetime

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .. import config
from ..db import connect
from .news import _entities, _vocab

_analyzer = SentimentIntensityAnalyzer()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _get(params: dict) -> dict:
    r = requests.get(
        config.GDELT_DOC_API,
        params={"format": "json", **params},
        headers={"User-Agent": "cricket_pipeline/0.1"},
        timeout=30,
    )
    r.raise_for_status()
    try:
        return r.json()
    except json.JSONDecodeError:
        return {}


def fetch(query: str = "cricket", maxrecords: int = 100, hours: int = 72) -> int:
    data = _get({
        "query": query,
        "maxrecords": min(maxrecords, 250),
        "timespan": f"{hours}h",
        "sort": "datedesc",
    })
    articles = data.get("articles") or []
    if not articles:
        return 0

    vocab = _vocab()
    rows = []
    for a in articles:
        title = a.get("title", "")
        text = title  # GDELT doc API doesn't include body
        rows.append({
            "url":          a.get("url"),
            "published_at": _parse_dt(a.get("seendate")),
            "source":       a.get("domain") or "gdelt",
            "title":        title,
            "summary":      None,
            "entities":     json.dumps(_entities(text, vocab)),
            "sentiment":    _analyzer.polarity_scores(text)["compound"],
        })

    rows = [r for r in rows if r["url"]]
    if not rows:
        return 0
    con = connect()
    con.executemany(
        """INSERT OR REPLACE INTO news
           (url, published_at, source, title, summary, entities, sentiment)
           VALUES ($url, $published_at, $source, $title, $summary, $entities, $sentiment)""",
        rows,
    )
    con.close()
    return len(rows)


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
