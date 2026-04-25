"""Cricket pipeline CLI.

Usage:
    python -m cricket_pipeline.pipeline cricsheet --dataset t20s_json --limit 50
    python -m cricket_pipeline.pipeline statsguru --stat batting --fmt t20i
    python -m cricket_pipeline.pipeline weather --limit 20
    python -m cricket_pipeline.pipeline stats
"""

from __future__ import annotations

import argparse

from .db import connect
from .ingest import cricsheet, statsguru, weather


def cmd_cricsheet(args):
    result = cricsheet.ingest(
        dataset=args.dataset, limit=args.limit, force=args.force
    )
    print(result)


def cmd_statsguru(args):
    rows = statsguru.fetch_player_overall(stat=args.stat, fmt=args.fmt)
    n = statsguru.store_splits(rows, fmt=args.fmt, split_type="overall", split_key="all")
    print(f"stored {n} rows")


def cmd_weather(args):
    n = weather.backfill_from_matches(limit=args.limit)
    print(f"stored weather for {n} (venue, date) pairs")


def cmd_stats(args):
    con = connect()
    for label, q in [
        ("matches",         "SELECT COUNT(*) FROM matches"),
        ("balls",           "SELECT COUNT(*) FROM balls"),
        ("innings",         "SELECT COUNT(*) FROM innings"),
        ("players_splits",  "SELECT COUNT(*) FROM player_splits"),
        ("weather_days",    "SELECT COUNT(*) FROM weather_daily"),
        ("distinct_venues", "SELECT COUNT(DISTINCT venue) FROM matches"),
    ]:
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

    s = sub.add_parser("statsguru", help="Fetch a Statsguru overall leaderboard")
    s.add_argument("--stat", choices=["batting", "bowling"], default="batting")
    s.add_argument("--fmt",  choices=["test", "odi", "t20i", "ipl"], default="t20i")
    s.set_defaults(func=cmd_statsguru)

    w = sub.add_parser("weather", help="Backfill Visual Crossing weather for loaded matches")
    w.add_argument("--limit", type=int, default=50)
    w.set_defaults(func=cmd_weather)

    st = sub.add_parser("stats", help="Show row counts")
    st.set_defaults(func=cmd_stats)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
