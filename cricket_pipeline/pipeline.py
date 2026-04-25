"""Cricket pipeline CLI.

Usage:
    python -m cricket_pipeline.pipeline cricsheet --dataset t20s_json --limit 50
    python -m cricket_pipeline.pipeline people
    python -m cricket_pipeline.pipeline venues --limit 100
    python -m cricket_pipeline.pipeline rankings
    python -m cricket_pipeline.pipeline statsguru --stat batting --fmt t20i
    python -m cricket_pipeline.pipeline statsguru-split --stat batting --fmt t20i --groupby year
    python -m cricket_pipeline.pipeline weather --limit 20
    python -m cricket_pipeline.pipeline owm --limit 20
    python -m cricket_pipeline.pipeline stats
"""

from __future__ import annotations

import argparse

from .db import connect
from .ingest import (
    cricsheet,
    openweather,
    people,
    rankings,
    statsguru,
    venues,
    weather,
)


def cmd_cricsheet(args):
    print(cricsheet.ingest(dataset=args.dataset, limit=args.limit, force=args.force))


def cmd_people(args):
    n = people.ingest(force=args.force)
    print(f"stored {n} player registry rows")


def cmd_venues(args):
    n = venues.enrich_from_matches(limit=args.limit)
    print(f"geocoded {n} venues")


def cmd_rankings(args):
    print(rankings.ingest_all())


def cmd_statsguru(args):
    rows = statsguru.fetch_player_overall(stat=args.stat, fmt=args.fmt)
    n = statsguru.store_splits(rows, fmt=args.fmt, split_type="overall", split_key="all")
    print(f"stored {n} rows")


def cmd_statsguru_split(args):
    rows = statsguru.fetch_split(
        stat=args.stat,
        fmt=args.fmt,
        groupby=args.groupby,
        opposition=args.opposition,
        year=args.year,
    )
    key = args.opposition or (str(args.year) if args.year else args.groupby or "all")
    n = statsguru.store_splits(
        rows, fmt=args.fmt, split_type=args.groupby or "filtered", split_key=key
    )
    print(f"stored {n} rows under split_type={args.groupby or 'filtered'}, key={key}")


def cmd_weather(args):
    n = weather.backfill_from_matches(limit=args.limit)
    print(f"stored historical weather for {n} (venue, date) pairs")


def cmd_owm(args):
    n = openweather.fetch_all_venues(limit=args.limit, include_forecast=not args.current_only)
    print(f"openweather: refreshed {n} venues")


def cmd_stats(args):
    con = connect()
    queries = [
        ("matches",         "SELECT COUNT(*) FROM matches"),
        ("balls",           "SELECT COUNT(*) FROM balls"),
        ("innings",         "SELECT COUNT(*) FROM innings"),
        ("players",         "SELECT COUNT(*) FROM players"),
        ("player_splits",   "SELECT COUNT(*) FROM player_splits"),
        ("rankings",        "SELECT COUNT(*) FROM rankings"),
        ("venues_geocoded", "SELECT COUNT(*) FROM venues WHERE lat IS NOT NULL"),
        ("weather_days",    "SELECT COUNT(*) FROM weather_daily"),
        ("news",            "SELECT COUNT(*) FROM news"),
        ("distinct_venues", "SELECT COUNT(DISTINCT venue) FROM matches"),
    ]
    for label, q in queries:
        n = con.execute(q).fetchone()[0]
        print(f"{label:<18} {n:>10}")
    con.close()


def main():
    p = argparse.ArgumentParser(prog="cricket_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cricsheet", help="Ingest a CricSheet zip")
    c.add_argument("--dataset", default="t20s_json")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--force", action="store_true")
    c.set_defaults(func=cmd_cricsheet)

    pp = sub.add_parser("people", help="Ingest CricSheet people registry")
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=cmd_people)

    v = sub.add_parser("venues", help="Geocode venues via Nominatim")
    v.add_argument("--limit", type=int, default=None)
    v.set_defaults(func=cmd_venues)

    r = sub.add_parser("rankings", help="Scrape ICC rankings (player + team)")
    r.set_defaults(func=cmd_rankings)

    s = sub.add_parser("statsguru", help="Fetch a Statsguru overall leaderboard")
    s.add_argument("--stat", choices=["batting", "bowling"], default="batting")
    s.add_argument("--fmt",  choices=["test", "odi", "t20i", "ipl"], default="t20i")
    s.set_defaults(func=cmd_statsguru)

    ss = sub.add_parser("statsguru-split", help="Fetch Statsguru with groupby/opposition/year filters")
    ss.add_argument("--stat", choices=["batting", "bowling"], default="batting")
    ss.add_argument("--fmt",  choices=["test", "odi", "t20i", "ipl"], default="t20i")
    ss.add_argument("--groupby", choices=["year", "ground", "opposition", "season"], default=None)
    ss.add_argument("--opposition", default=None, help="e.g. India, Australia")
    ss.add_argument("--year", type=int, default=None)
    ss.set_defaults(func=cmd_statsguru_split)

    w = sub.add_parser("weather", help="Backfill Visual Crossing historical weather")
    w.add_argument("--limit", type=int, default=50)
    w.set_defaults(func=cmd_weather)

    o = sub.add_parser("owm", help="Refresh OpenWeatherMap current + forecast for venues")
    o.add_argument("--limit", type=int, default=50)
    o.add_argument("--current-only", action="store_true")
    o.set_defaults(func=cmd_owm)

    st = sub.add_parser("stats", help="Show row counts")
    st.set_defaults(func=cmd_stats)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
