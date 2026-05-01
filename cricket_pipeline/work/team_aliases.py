"""Canonical team-name aliases.

Bookmakers and CricSheet sometimes use different spellings for the same team
(Bangalore vs Bengaluru, USA vs United States of America, etc.). We normalize
to the CricSheet form since that's what the rest of our pipeline uses.

Used at:
  - cricket_pipeline.ingest.odds._store_event (incoming odds → canonical)
  - cricket_pipeline.work.odds_features.book_consensus (lookup robustness)
"""
from __future__ import annotations

# Map ANY alias → canonical CricSheet form
_ALIASES = {
    # IPL
    "royal challengers bangalore":   "Royal Challengers Bengaluru",
    "rcb":                            "Royal Challengers Bengaluru",
    "kkr":                            "Kolkata Knight Riders",
    "csk":                            "Chennai Super Kings",
    "mi":                             "Mumbai Indians",
    "rr":                             "Rajasthan Royals",
    "srh":                            "Sunrisers Hyderabad",
    "lsg":                            "Lucknow Super Giants",
    "gt":                             "Gujarat Titans",
    "dc":                             "Delhi Capitals",
    "pbks":                           "Punjab Kings",

    # Internationals — common bookmaker spellings
    "usa":                            "United States of America",
    "united states":                  "United States of America",
    "uae":                            "United Arab Emirates",
    "u.a.e.":                         "United Arab Emirates",
    "p.n.g":                          "Papua New Guinea",
    "png":                            "Papua New Guinea",

    # Women's variants (the BAN/AUS/IND etc. women's teams use 'Women' suffix)
    "india women":                    "India Women",
    "australia women":                "Australia Women",
    "england women":                  "England Women",
    "south africa women":             "South Africa Women",
    "new zealand women":              "New Zealand Women",
    "bangladesh women":               "Bangladesh Women",
    "sri lanka women":                "Sri Lanka Women",
    "west indies women":              "West Indies Women",
    "pakistan women":                 "Pakistan Women",

    # PSL franchises (filtered out per project rule, but keep mappings consistent)
    "lahore qalandars":               "Lahore Qalandars",
    "karachi kings":                  "Karachi Kings",
    "quetta gladiators":              "Quetta Gladiators",
    "islamabad united":               "Islamabad United",
    "multan sultans":                 "Multan Sultans",
    "peshawar zalmi":                 "Peshawar Zalmi",
}


def canonicalize(name: str | None) -> str | None:
    if not name: return name
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    return name.strip()


# ---- format canonicalization ----
# Different sources use different format strings. Cricsheet says "T20",
# Cricbuzz says "Twenty20", ICC rankings says "t20i", The Odds API uses sport
# keys like "cricket_ipl". Normalise to the schema's expected vocabulary
# (Test, ODI, T20, IT20) so cross-source joins and feature filters don't
# silently drop matches.
_FORMAT_ALIASES = {
    "test":       "Test",
    "tests":      "Test",
    "first-class": "Test",

    "odi":        "ODI",
    "odis":       "ODI",
    "one day":    "ODI",
    "one-day":    "ODI",

    "t20":        "T20",
    "twenty20":   "T20",

    "t20i":       "IT20",
    "it20":       "IT20",
    "i20":        "IT20",

    "t10":        "T10",
    "the hundred": "Hundred",
    "100-ball":   "Hundred",
}


def canonicalize_format(fmt: str | None) -> str | None:
    """Canonical format string for cross-source joins.

    Returns one of: 'Test', 'ODI', 'T20', 'IT20', 'T10', 'Hundred', or the
    input verbatim if unrecognised (so callers can detect novel formats).
    """
    if not fmt:
        return fmt
    key = fmt.strip().lower()
    return _FORMAT_ALIASES.get(key, fmt.strip())
