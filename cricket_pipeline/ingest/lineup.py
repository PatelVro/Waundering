"""Real-time playing-XI scraper (Cricbuzz match info pages).

XIs are usually announced ~30 minutes before toss. This module pulls the
two squads from Cricbuzz `/cricket-match-squads/<matchId>/<slug>` pages and
returns clean name lists. Cached briefly because lineups change rarely once
announced.

The HTML structure:
    <div class="cb-col cb-col-100 cb-min-stats">
        <div class="cb-col cb-col-50 ...">  (team A)
            <a class="cb-player-name-left">Player Name</a>
            ...
        </div>
        <div class="cb-col cb-col-50 ...">  (team B)
            ...
        </div>
    </div>

Cricbuzz tweaks markup periodically. This scraper is best-effort with a
fallback to plain text mining if the structured parse fails.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (cricket_pipeline lineup scraper)",
    "Accept": "text/html,application/xhtml+xml",
}

# Cache lineups for 5 minutes — they don't change once announced, but during
# the announcement window we want some freshness.
_CACHE_TTL_S = 5 * 60


def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"lineup_{h}.html"


def _is_fresh(p: Path) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < _CACHE_TTL_S


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch(url: str) -> str:
    cache = _cache_path(url)
    if _is_fresh(cache):
        return cache.read_text(encoding="utf-8")
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    cache.write_text(r.text, encoding="utf-8")
    return r.text


def _clean(name: str) -> str:
    name = re.sub(r"\([^)]*\)", "", name)            # strip "(c)", "(wk)", etc.
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_toss(html: str) -> dict:
    """Pull toss-winner + decision from the page text. Returns
    {'toss_winner': 'Team Name', 'toss_decision': 'bat'|'field', None if absent}."""
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    # Cricbuzz copy: "Team Name won the toss & opt to bowl/bat"
    m = re.search(r"([A-Z][A-Za-z' \.\-]+?)\s+won the toss\s*(?:&|and)\s*opt(?:ed)?\s+to\s+(bat|bowl|field)",
                  text, re.IGNORECASE)
    if not m:
        m = re.search(r"([A-Z][A-Za-z' \.\-]+?)\s+won the toss\s*(?:&|and)\s*chose\s+to\s+(bat|bowl|field)",
                      text, re.IGNORECASE)
    if not m:
        return {"toss_winner": None, "toss_decision": None}
    decision = m.group(2).lower()
    if decision == "bowl":
        decision = "field"
    return {"toss_winner": m.group(1).strip(), "toss_decision": decision}


def parse_squads(html: str) -> dict:
    """Returns {'team_a': str, 'team_b': str, 'xi_a': [...], 'xi_b': [...],
                'bench_a': [...], 'bench_b': [...], 'announced': bool,
                'toss_winner': str | None, 'toss_decision': 'bat'|'field' | None}."""
    soup = BeautifulSoup(html, "lxml")

    teams = []
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""
    m = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+(?:Squad|squad|Team))", title)
    if m:
        teams = [m.group(1).strip(), m.group(2).strip()]

    # Cricbuzz uses two columns, each with player anchors.
    columns = soup.select("div.cb-play11-lft-col, div.cb-play11-rt-col")
    xis: list[list[str]] = [[], []]
    bench: list[list[str]] = [[], []]
    if columns:
        # Each column has "Playing XI" and "Bench" subheaders
        for i, col in enumerate(columns[:2]):
            current = "xi"
            for el in col.find_all(["div", "a"], recursive=True):
                txt = el.get_text(" ", strip=True)
                if not txt:
                    continue
                low = txt.lower()
                if "bench" in low or "subs" in low or "substitute" in low:
                    current = "bench"
                    continue
                if "playing xi" in low or "predicted" in low:
                    current = "xi"
                    continue
                # Player anchors
                if el.name == "a" and "cb-player" in " ".join(el.get("class", [])):
                    target = xis if current == "xi" else bench
                    target[i].append(_clean(txt))

    if not any(xis):
        # Fallback: any list of <a class="cb-player...">
        anchors = soup.select("a.cb-player-name-left, a.cb-player-name-right, a[class*='cb-player']")
        names = [_clean(a.get_text()) for a in anchors if a.get_text(strip=True)]
        # Heuristic split: first 11 = team A, next 11 = team B
        if len(names) >= 22:
            xis = [names[:11], names[11:22]]

    toss = parse_toss(html)
    return {
        "team_a":   teams[0] if len(teams) > 0 else None,
        "team_b":   teams[1] if len(teams) > 1 else None,
        "xi_a":     xis[0],
        "xi_b":     xis[1],
        "bench_a":  bench[0],
        "bench_b":  bench[1],
        "announced": (len(xis[0]) >= 11 and len(xis[1]) >= 11),
        **toss,
    }


def fetch(url: str) -> dict:
    """Public entry point. URL is a Cricbuzz match-squads or scorecard page."""
    if "cricbuzz.com" not in url:
        raise ValueError("Only Cricbuzz match URLs supported (for now)")
    html = _fetch(url)
    return parse_squads(html)


def fetch_by_match_id(match_id: str | int, slug: str = "match") -> dict:
    url = f"https://www.cricbuzz.com/cricket-match-squads/{match_id}/{slug}"
    return fetch(url)
