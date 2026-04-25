"""NewsAPI integration — broader cricket news than RSS feeds.

Free tier (developer): 100 requests/day, 24-hour delay on articles. Set
NEWSAPI_KEY in env. Lands articles into the same `news` table the RSS
ingester writes to (with sentiment + entity tags).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .. import config
from ..db import connect
from .news import _entities, _vocab

_analyzer = SentimentIntensityAnalyzer()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _get(endpoint: str, params: dict) -> dict:
    if not config.NEWSAPI_KEY:
        raise RuntimeError("Set NEWSAPI_KEY in your environment first.")
    p = {"apiKey": config.NEWSAPI_KEY, **params}
    r = requests.get(f"{config.NEWSAPI_BASE}/{endpoint}", params=p, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch(query: str = "cricket", days: int = 7, language: str = "en") -> int:
    since = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    data = _get("everything", {
        "q": query,
        "from": since,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": 100,
    })
    articles = data.get("articles", []) or []
    if not articles:
        return 0

    vocab = _vocab()
    rows = []
    for a in articles:
        title = a.get("title") or ""
        summary = (a.get("description") or "") + " " + (a.get("content") or "")
        text = f"{title}. {summary}"
        rows.append({
            "url":          a.get("url"),
            "published_at": _parse_dt(a.get("publishedAt")),
            "source":       (a.get("source") or {}).get("name") or "newsapi",
            "title":        title,
            "summary":      summary[:2000],
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
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
