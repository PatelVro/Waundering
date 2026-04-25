"""Cricket news RSS ingester with VADER sentiment scoring.

Pulls multiple feeds (Cricinfo, Cricbuzz, ICC, Wisden), normalises them, runs
a lexicon-based sentiment score on the title+summary, and lightly tags entities
by string-matching against the players and venues already in the DB.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from time import mktime

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .. import config
from ..db import connect

_analyzer = SentimentIntensityAnalyzer()


def _published_at(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        v = entry.get(key)
        if v:
            return datetime.fromtimestamp(mktime(v))
    return None


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _entities(text: str, vocab: set[str]) -> list[str]:
    """Cheap keyword tagger — case-sensitive substring match against known names."""
    if not text or not vocab:
        return []
    found = [name for name in vocab if name and name in text]
    return sorted(set(found))


def _vocab() -> set[str]:
    con = connect()
    players = {r[0] for r in con.execute("SELECT DISTINCT name FROM players WHERE name IS NOT NULL").fetchall()}
    venues  = {r[0] for r in con.execute("SELECT DISTINCT venue FROM matches WHERE venue IS NOT NULL").fetchall()}
    teams   = {r[0] for r in con.execute(
        "SELECT DISTINCT team_home FROM matches UNION SELECT DISTINCT team_away FROM matches"
    ).fetchall() if r[0]}
    con.close()
    return players | venues | teams


def fetch_feed(name: str, url: str, vocab: set[str]) -> list[dict]:
    parsed = feedparser.parse(url)
    rows = []
    for e in parsed.entries:
        title = e.get("title", "")
        summary = _strip_html(e.get("summary", "") or e.get("description", ""))
        text = f"{title}. {summary}"
        score = _analyzer.polarity_scores(text)["compound"]
        rows.append({
            "url":          e.get("link"),
            "published_at": _published_at(e),
            "source":       name,
            "title":        title,
            "summary":      summary[:2000],
            "entities":     json.dumps(_entities(text, vocab)),
            "sentiment":    score,
        })
    return rows


def ingest(sources: list[str] | None = None) -> int:
    feeds = config.NEWS_FEEDS
    if sources:
        feeds = {k: v for k, v in feeds.items() if k in sources}
    vocab = _vocab()
    all_rows: list[dict] = []
    for name, url in feeds.items():
        try:
            rows = fetch_feed(name, url, vocab)
            print(f"  {name:<14} {len(rows):>4} items")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  {name:<14} error: {e}")
    if not all_rows:
        return 0
    con = connect()
    con.executemany(
        """INSERT OR REPLACE INTO news
           (url, published_at, source, title, summary, entities, sentiment)
           VALUES ($url, $published_at, $source, $title, $summary, $entities, $sentiment)""",
        [r for r in all_rows if r["url"]],
    )
    con.close()
    return len(all_rows)
