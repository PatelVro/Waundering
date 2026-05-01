"""Project-wide content filter.

Anything matching `BLOCKED_TEAMS` or `BLOCKED_SLUG_PATTERN` is excluded from:
  - orchestrator discovery / tracking / prediction
  - dashboard data export (recent_matches, top_teams, predictions)
"""
from __future__ import annotations

import re

# International + franchise teams to exclude
BLOCKED_TEAMS = {
    # Pakistan international
    "Pakistan", "Pakistan A", "Pakistan U19",
    # PSL franchises (current + historical names)
    "Lahore Qalandars", "Karachi Kings", "Quetta Gladiators",
    "Islamabad United", "Multan Sultans", "Peshawar Zalmi",
    "Hyderabad Kingsmen", "Rawalpindiz", "Rawalpindi",
}

# Slug fragments that mark a Pakistan-centric tournament/match
BLOCKED_SLUG_PATTERN = re.compile(
    r"(pakistan|psl|\bpak[-/]|\bpak\b|hbl-pakistan|pakistan-super-league|"
    r"national-t20-cup|quaid-e-azam-trophy|caribbean-tour-of-pakistan|"
    r"\bpsz\b|\blhq\b|\bqtg\b|\bkrk\b|\bisu\b|\bms\b(?!\w)|\bhydk\b|\brwp\b)",
    re.I,
)


def is_blocked_team(name: str | None) -> bool:
    if not name: return False
    n = name.strip()
    if n in BLOCKED_TEAMS: return True
    n_low = n.lower()
    return n_low.startswith("pakistan") or n in {b.lower() for b in BLOCKED_TEAMS}


def is_blocked_slug(slug: str | None) -> bool:
    if not slug: return False
    return bool(BLOCKED_SLUG_PATTERN.search(slug))


def is_blocked_match(home: str | None, away: str | None) -> bool:
    return is_blocked_team(home) or is_blocked_team(away)
