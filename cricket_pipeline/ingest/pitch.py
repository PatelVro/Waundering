"""Pitch & conditions report scraper.

Cricbuzz publishes pitch info in two places:
  1. The match-news preview pages (article body)
  2. The live match page commentary just before/after toss

We scrape the live match page text + any "preview" article body, then
classify the text into bucket scores via keyword regexes:
   pitch_dry       — dry, dusty, brown, cracks, abrasive
   pitch_green     — green, grass, seam-friendly, juicy
   pitch_pace      — pace, bounce, carry, fast, hard
   pitch_spin      — spin-friendly, turning, slow, dust
   pitch_flat      — flat, road, batting paradise, true
   pitch_low       — low scoring, two-paced, sluggish, tired
   pitch_dew       — dew, wet evening, dewy, second innings advantage

Each score in [0, 1] = matched_terms / max_terms. Stored per match for
downstream feature use. Falls back gracefully when text is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..db import connect


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

_CACHE_DIR = config.CACHE_DIR / "pitch"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_S = 3600      # 1 hour: pitch reports rarely change once published


# ---- keyword buckets ----
PATTERNS = {
    "dry":   r"\b(dry|dusty|brown|cracks?|abrasive|bare|worn|crumbl(ing|y)|dust\b)",
    "green": r"\b(green(ish)?|grass(y)?|seam(-?friendly)?|juicy|live|swing(-?friendly)?|moisture)",
    "pace":  r"\b(pacy|bounc(y|e)|carry|fast|hard|quick(er)?|good\s+pace)",
    "spin":  r"\b(spin(-?friendly)?|turn(s|ing)?|slow(er)?\s+(track|surface)|grip(ping)?|tweak(ers)?)",
    "flat":  r"\b(flat|road|batting\s+paradise|true|belter|placid|even-?paced)",
    "low":   r"\b(low(-|\s)?scoring|two-?paced|slug(g|ish)|tired|sticky|stop(s|ping)?\s+a\s+bit)",
    "dew":   r"\b(dew|dewy|wet\s+(evening|outfield|ball)|chasing\s+side\s+(?:will\s+be\s+|would\s+)?favoured|second\s+innings\s+advantage|moist\s+evening)",
}

PATTERNS_COMPILED = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}

# Each bucket gets a max count for normalization (so we don't over-reward articles
# that just repeat the same word a lot).
MAX_HITS = {k: 5 for k in PATTERNS}


def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return _CACHE_DIR / f"pitch_{h}.html"


def _is_fresh(p: Path) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < _CACHE_TTL_S


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=15))
def _fetch(url: str) -> str:
    cache = _cache_path(url)
    if _is_fresh(cache): return cache.read_text(encoding="utf-8")
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    cache.write_text(r.text, encoding="utf-8")
    return r.text


def extract_text(html: str) -> str:
    """Strip HTML to readable text. Prefer article body and 'pitch' / 'conditions' sections."""
    soup = BeautifulSoup(html, "lxml")
    chunks = []

    # Cricbuzz article body
    for tag_class in ("cb-nws-para", "cb-nws-intr", "cb-com-ln"):
        for el in soup.select(f"[class*={tag_class}]"):
            txt = el.get_text(" ", strip=True)
            if txt: chunks.append(txt)

    # Generic <p> text — useful for previews
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if txt and len(txt) > 30: chunks.append(txt)

    # Anything containing pitch/conditions keywords
    full = " ".join(chunks) if chunks else soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", full)


_ANCHOR = re.compile(
    r"\b(pitch|wicket|surface|track|deck|conditions?|outfield|toss|dew|"
    r"playing\s+conditions|first\s+innings|second\s+innings|batting\s+first|chase)\b",
    re.I,
)


def _pitch_sentences(text: str) -> list[str]:
    """Pull only sentences that contain a pitch-related anchor word.
    Reduces false positives from generic commentary ("fast bowler", "spin
    bowler", "Green plays a..." etc.)."""
    if not text: return []
    sents = re.split(r"(?<=[\.\!\?])\s+", text)
    return [s for s in sents if _ANCHOR.search(s)]


def score_text(text: str) -> dict:
    """Convert pitch text → bucket scores in [0, 1]. Only counts keyword
    matches inside sentences that contain a pitch-anchor word.
    """
    if not text:
        return {f"pitch_{k}": 0.0 for k in PATTERNS}
    sents = _pitch_sentences(text)
    if not sents:
        return {f"pitch_{k}": 0.0 for k in PATTERNS}
    joined = " ".join(sents)
    out = {}
    for k, pat in PATTERNS_COMPILED.items():
        n_hits = len(pat.findall(joined))
        out[f"pitch_{k}"] = round(min(n_hits / MAX_HITS[k], 1.0), 3)
    out["_n_pitch_sentences"] = len(sents)
    return out


def fetch_for_match(match_id: str, slug: str = "match") -> dict | None:
    """Pull live + preview text for a Cricbuzz match and return pitch scores."""
    urls = [
        f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/{slug}",
        # Preview is usually a separate news article; we don't always know its slug.
        # Try a generic search page query as a fallback.
    ]
    text_parts = []
    for u in urls:
        try:
            html = _fetch(u)
            text_parts.append(extract_text(html))
        except Exception as e:
            print(f"pitch fetch {u} failed: {e}")
            continue
    if not text_parts: return None
    text = " ".join(text_parts)
    scores = score_text(text)
    scores["_text_len"] = len(text)
    scores["_match_id"] = match_id
    return scores


# ---- store / read from DB ----

def _ensure_table():
    con = connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS pitch_reports (
            match_id    VARCHAR PRIMARY KEY,
            fetched_at  TIMESTAMP,
            text_len    INTEGER,
            pitch_dry   DOUBLE,
            pitch_green DOUBLE,
            pitch_pace  DOUBLE,
            pitch_spin  DOUBLE,
            pitch_flat  DOUBLE,
            pitch_low   DOUBLE,
            pitch_dew   DOUBLE,
            source      VARCHAR DEFAULT 'cricbuzz'
        )
    """)
    con.close()


def store_for_match(match_id: str, slug: str = "match") -> bool:
    _ensure_table()
    out = fetch_for_match(match_id, slug)
    if not out: return False
    con = connect()
    con.execute("""
        INSERT OR REPLACE INTO pitch_reports
          (match_id, fetched_at, text_len, pitch_dry, pitch_green, pitch_pace,
           pitch_spin, pitch_flat, pitch_low, pitch_dew, source)
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, 'cricbuzz')
    """, [match_id, out.get("_text_len", 0),
          out["pitch_dry"], out["pitch_green"], out["pitch_pace"],
          out["pitch_spin"], out["pitch_flat"], out["pitch_low"],
          out["pitch_dew"]])
    con.close()
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--slug",     default="match")
    args = ap.parse_args()
    if args.match_id:
        out = fetch_for_match(args.match_id, args.slug)
        print(json.dumps(out, indent=2))
        if store_for_match(args.match_id, args.slug):
            print("stored.")
