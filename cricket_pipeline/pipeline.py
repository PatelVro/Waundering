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

from .db import connect, install_views
from .ingest import (
    cricbuzz,
    cricinfo_profiles,
    cricsheet,
    cricsheet_players,
    fixtures,
    lineup,
    gdelt,
    news,
    newsapi,
    openweather,
    partnerships,
    people,
    rankings,
    statsguru,
    umpires,
    venues,
    weather,
    wikidata,
    wikipedia,
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


def cmd_news(args):
    n = news.ingest(sources=args.sources)
    print(f"stored {n} news items")


def cmd_wiki(args):
    n = wikipedia.enrich_all(limit=args.limit)
    print(f"enriched {n} venues from Wikipedia")


def cmd_umpires(args):
    n = umpires.populate()
    print(f"umpires table populated: {n} unique officials")


def cmd_fixtures(args):
    n = fixtures.fetch_scores()
    print(f"stored {n} fixtures")


def cmd_profiles(args):
    n = cricinfo_profiles.enrich_all(limit=args.limit, only_active=not args.all)
    print(f"enriched {n} player profiles from Cricinfo")


def cmd_views(args):
    install_views()
    print("views installed: v_venue_profile, v_phase_metrics, v_batter_profile, "
          "v_bowler_profile, v_matchup, v_umpire_lbw, v_toss_impact")


def cmd_datasets(args):
    from . import config
    for k, v in config.CRICSHEET_ZIPS.items():
        print(f"  {k:<35} {v}")


def cmd_partnerships(args):
    n = partnerships.derive(replace=not args.append)
    print(f"derived {n} partnerships")


def cmd_cricbuzz(args):
    n = cricbuzz.snapshot_many(args.match_ids)
    print(f"captured {n} live snapshots")


def cmd_newsapi(args):
    n = newsapi.fetch(query=args.query, days=args.days)
    print(f"stored {n} NewsAPI articles")


def cmd_gdelt(args):
    n = gdelt.fetch(query=args.query, hours=args.hours, maxrecords=args.max)
    print(f"stored {n} GDELT articles")


def cmd_wikidata(args):
    n = wikidata.merge()
    print(f"matched {n} Wikidata cricketer records onto players")


def cmd_csplayers(args):
    n = cricsheet_players.backfill(datasets=args.datasets or None)
    print(f"backfilled country on {n} player rows from CricSheet match files")


def cmd_match_train(args):
    from .model import match as M
    M.train(format_filter=args.fmt, device=args.device)


def cmd_match_predict(args):
    import json as _json
    if args.ensemble:
        from .model.match import predict_match_ensemble
        out = predict_match_ensemble(
            home=args.home, away=args.away, venue=args.venue,
            format_=args.fmt,
            toss_winner=args.toss_winner,
            toss_decision=args.toss_decision,
            ref_date=args.ref_date,
        )
    else:
        from .model.match import predict_match
        out = predict_match(
            home=args.home, away=args.away, venue=args.venue,
            format_=args.fmt,
            toss_winner=args.toss_winner,
            toss_decision=args.toss_decision,
            ref_date=args.ref_date,
        )
    print(_json.dumps(out, indent=2, default=str))


def cmd_lineup(args):
    import json as _json
    out = lineup.fetch(args.url) if args.url else lineup.fetch_by_match_id(args.match_id)
    print(_json.dumps(out, indent=2, default=str))


def cmd_match_forecast(args):
    """End-to-end forecast — winner, scores, top players, key matchups."""
    from . import forecast as F
    home_xi = [n.strip() for n in args.home_xi.split(",")] if args.home_xi else None
    away_xi = [n.strip() for n in args.away_xi.split(",")] if args.away_xi else None
    fc = F.forecast(
        home=args.home, away=args.away, venue=args.venue,
        home_xi=home_xi, away_xi=away_xi,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
        ref_date=args.ref_date,
        n_sim=args.n_sim,
    )
    if args.json:
        import dataclasses, json as _json
        print(_json.dumps(dataclasses.asdict(fc), indent=2, default=str))
    else:
        print(F.render(fc))


def cmd_prematch(args):
    """Poll a Cricbuzz match page for XIs + toss until both are present, then
    write the result to data/cache/prematch_<match_id>.json so a subsequent
    `match-forecast` can pick them up. Cron-friendly."""
    import json as _json
    import time as _time
    from pathlib import Path
    deadline = _time.time() + args.max_wait_seconds
    last = None
    while _time.time() < deadline:
        try:
            data = lineup.fetch(args.url)
        except Exception as e:
            print(f"  fetch error: {e}")
            data = None
        if data:
            last = data
            ann = data.get("announced")
            toss = data.get("toss_winner")
            print(f"  XIs={'yes' if ann else 'no'}  toss={toss or 'no'}")
            if ann and toss:
                break
        _time.sleep(args.poll_seconds)
    out = Path(args.out) if args.out else None
    if out and last:
        out.write_text(_json.dumps(last, indent=2))
        print(f"  wrote {out}")
    print(_json.dumps(last, indent=2, default=str))


def cmd_live_track(args):
    """Poll Cricbuzz live data and update data.json with in-play win probabilities."""
    from pathlib import Path
    from . import live_tracker as LT

    out = Path(args.out) if args.out else LT.DATA_JSON

    if args.auto:
        LT.auto_run(
            home_hint=args.home_hint,
            away_hint=args.away_hint,
            interval=args.interval,
            n_sim=args.n_sim,
            out_path=out,
        )
    else:
        if not args.match_id:
            print("Provide --match-id or use --auto to discover a live match.")
            return
        LT.run(
            match_id=args.match_id,
            home=args.home or "",
            away=args.away or "",
            venue=args.venue,
            interval=args.interval,
            n_sim=args.n_sim,
            out_path=out,
        )


def cmd_daily_refresh(args):
    """Re-pull data, rebuild views, retrain match model. Cron-friendly."""
    from .ingest import cricsheet
    from .model import match as M
    print("=== Daily refresh ===")
    for ds in args.datasets:
        print(f"\n[1/3] Re-ingesting {ds} …")
        cricsheet.ingest(dataset=ds, force=args.force)
    print("\n[2/3] Reinstalling views …")
    install_views()
    print("\n[3/3] Retraining match model …")
    M.train(format_filter=args.fmt)
    print("\nDone.")


def cmd_model(args):
    import json as _json
    if args.action == "train":
        if args.type == "sequence":
            from .model import sequence as S
            S.train(format_filter=args.fmt, limit=args.limit, epochs=args.epochs,
                    device=None if args.device == "auto" else args.device)
        else:
            from .model import train as M
            M.train(format_filter=args.fmt, limit=args.limit, device=args.device)
    elif args.action == "predict":
        if args.type == "sequence":
            from .model.sequence import predict_sequence
            history = _json.loads(args.state)
            if isinstance(history, dict):
                history = [history]
            out = predict_sequence(history)
        else:
            from .model.predict import predict_ball
            out = predict_ball(_json.loads(args.state))
        print(_json.dumps(out, indent=2))
    elif args.action == "simulate":
        from .model.simulate import simulate_innings
        out = simulate_innings(_json.loads(args.state),
                               n_sim=args.n_sim, seed=args.seed)
        print(_json.dumps(out, indent=2))


def cmd_stats(args):
    con = connect()
    queries = [
        ("matches",            "SELECT COUNT(*) FROM matches"),
        ("balls",              "SELECT COUNT(*) FROM balls"),
        ("innings",            "SELECT COUNT(*) FROM innings"),
        ("players",            "SELECT COUNT(*) FROM players"),
        ("players_enriched",   "SELECT COUNT(*) FROM players WHERE enriched_at IS NOT NULL"),
        ("player_splits",      "SELECT COUNT(*) FROM player_splits"),
        ("rankings",           "SELECT COUNT(*) FROM rankings"),
        ("venues_geocoded",    "SELECT COUNT(*) FROM venues WHERE lat IS NOT NULL"),
        ("venues_enriched",    "SELECT COUNT(*) FROM venues WHERE capacity IS NOT NULL"),
        ("weather_days",       "SELECT COUNT(*) FROM weather_daily"),
        ("news",               "SELECT COUNT(*) FROM news"),
        ("fixtures",           "SELECT COUNT(*) FROM fixtures"),
        ("umpires",            "SELECT COUNT(*) FROM umpires"),
        ("match_officials",    "SELECT COUNT(*) FROM match_officials"),
        ("partnerships",       "SELECT COUNT(*) FROM partnerships"),
        ("live_state",         "SELECT COUNT(*) FROM live_state"),
        ("distinct_venues",    "SELECT COUNT(DISTINCT venue) FROM matches"),
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

    n = sub.add_parser("news", help="Pull cricket news RSS with sentiment + entity tags")
    n.add_argument("--sources", nargs="*", default=None,
                   help="Subset of: espncricinfo cricbuzz icc wisden")
    n.set_defaults(func=cmd_news)

    wk = sub.add_parser("wiki", help="Enrich venues from Wikipedia (capacity, ends, established)")
    wk.add_argument("--limit", type=int, default=None)
    wk.set_defaults(func=cmd_wiki)

    um = sub.add_parser("umpires", help="Populate umpires table from matches we already have")
    um.set_defaults(func=cmd_umpires)

    fx = sub.add_parser("fixtures", help="Fetch upcoming/live fixtures via CricAPI")
    fx.set_defaults(func=cmd_fixtures)

    pr = sub.add_parser("profiles", help="Enrich players from Cricinfo profile pages")
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--all", action="store_true",
                    help="Don't restrict to players we have ball data for")
    pr.set_defaults(func=cmd_profiles)

    vw = sub.add_parser("views", help="Install/refresh analytical views (free, derived)")
    vw.set_defaults(func=cmd_views)

    ds = sub.add_parser("datasets", help="List available CricSheet zip datasets")
    ds.set_defaults(func=cmd_datasets)

    pt = sub.add_parser("partnerships", help="Derive partnerships from balls")
    pt.add_argument("--append", action="store_true",
                    help="Don't wipe partnerships table first")
    pt.set_defaults(func=cmd_partnerships)

    cb = sub.add_parser("cricbuzz", help="Snapshot live match state from Cricbuzz")
    cb.add_argument("match_ids", nargs="+", help="Cricbuzz match IDs")
    cb.set_defaults(func=cmd_cricbuzz)

    na = sub.add_parser("newsapi", help="Fetch broader cricket news via NewsAPI")
    na.add_argument("--query", default="cricket")
    na.add_argument("--days", type=int, default=7)
    na.set_defaults(func=cmd_newsapi)

    gd = sub.add_parser("gdelt", help="Pull global cricket news via GDELT 2.0 (no key)")
    gd.add_argument("--query", default="cricket")
    gd.add_argument("--hours", type=int, default=72)
    gd.add_argument("--max", type=int, default=100)
    gd.set_defaults(func=cmd_gdelt)

    wd = sub.add_parser("wikidata", help="Enrich players from Wikidata SPARQL")
    wd.set_defaults(func=cmd_wikidata)

    cp = sub.add_parser("cs-players",
                        help="Backfill players.country from cached CricSheet match JSON")
    cp.add_argument("--datasets", nargs="*",
                    help="CricSheet datasets to walk (default: all_json)")
    cp.set_defaults(func=cmd_csplayers)

    mt = sub.add_parser("match-train",
                        help="Train the match-outcome model (binary win classifier)")
    mt.add_argument("--fmt", default="T20",
                    help="Format filter (T20, IT20, ODI, Test)")
    mt.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "gpu", "cuda"],
                    help="LightGBM device (auto = use GPU if available)")
    mt.set_defaults(func=cmd_match_train)

    mp = sub.add_parser("match-predict",
                        help="Predict P(home wins) for a single upcoming match")
    mp.add_argument("--home",   required=True)
    mp.add_argument("--away",   required=True)
    mp.add_argument("--venue",  required=True)
    mp.add_argument("--fmt",    default="T20")
    mp.add_argument("--toss-winner",   default=None)
    mp.add_argument("--toss-decision", default=None,
                    choices=[None, "bat", "field"], help="bat or field")
    mp.add_argument("--ref-date", default=None,
                    help="As-of date for form lookups (default = today). YYYY-MM-DD")
    mp.add_argument("--ensemble", action="store_true",
                    help="Blend match model with form + h2h priors (recommended)")
    mp.set_defaults(func=cmd_match_predict)

    ln = sub.add_parser("lineup",
                        help="Fetch announced playing XIs from a Cricbuzz match URL")
    ln.add_argument("--url",      default=None,
                    help="Cricbuzz match-squads URL (preferred)")
    ln.add_argument("--match-id", default=None,
                    help="Cricbuzz match id (alt to --url)")
    ln.set_defaults(func=cmd_lineup)

    mf = sub.add_parser("match-forecast",
                        help="End-to-end forecast: winner, scores, top players, matchups")
    mf.add_argument("--home",   required=True)
    mf.add_argument("--away",   required=True)
    mf.add_argument("--venue",  required=True)
    mf.add_argument("--home-xi", default=None,
                    help="Comma-separated playing XI for home team")
    mf.add_argument("--away-xi", default=None,
                    help="Comma-separated playing XI for away team")
    mf.add_argument("--toss-winner",   default=None)
    mf.add_argument("--toss-decision", default=None, choices=[None, "bat", "field"])
    mf.add_argument("--ref-date", default=None,
                    help="As-of date for form lookups. YYYY-MM-DD")
    mf.add_argument("--n-sim", type=int, default=2000)
    mf.add_argument("--json",  action="store_true",
                    help="Emit JSON instead of formatted text")
    mf.set_defaults(func=cmd_match_forecast)

    pm = sub.add_parser("prematch",
                        help="Poll Cricbuzz for XIs + toss; write to a cache file")
    pm.add_argument("--url", required=True,
                    help="Cricbuzz match-squads URL")
    pm.add_argument("--max-wait-seconds", type=int, default=3600,
                    help="Stop polling after this many seconds (default 1h)")
    pm.add_argument("--poll-seconds", type=int, default=120,
                    help="Sleep between polls")
    pm.add_argument("--out", default=None,
                    help="Write last successful response to this path")
    pm.set_defaults(func=cmd_prematch)

    dr = sub.add_parser("daily-refresh",
                        help="Re-ingest, reinstall views, retrain match model (cron-friendly)")
    dr.add_argument("--datasets", nargs="+", default=["ipl_json"],
                    help="CricSheet datasets to refresh (e.g. ipl_json t20s_json)")
    dr.add_argument("--fmt", default="T20,IT20",
                    help="Format filter for retrain")
    dr.add_argument("--force", action="store_true",
                    help="Force re-download even if cached zip exists")
    dr.set_defaults(func=cmd_daily_refresh)

    md = sub.add_parser("model", help="Train / predict / simulate the ball-outcome model")
    md.add_argument("action", choices=["train", "predict", "simulate"])
    md.add_argument("--type", choices=["lgbm", "sequence"], default="lgbm",
                    help="Model architecture: lgbm (default) or sequence Transformer")
    md.add_argument("--fmt", default="IT20",
                    help="Format filter for training (e.g. IT20, T20, ODI)")
    md.add_argument("--limit", type=int, default=None)
    md.add_argument("--epochs", type=int, default=8,
                    help="Sequence-model training epochs (ignored for lgbm)")
    md.add_argument("--state", default="{}",
                    help="JSON of ball state, or list of state dicts for sequence predict")
    md.add_argument("--n-sim", type=int, default=5000)
    md.add_argument("--seed", type=int, default=None)
    md.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "gpu", "cuda"],
                    help="Compute device — for sequence model uses CUDA, "
                         "for lgbm uses LightGBM's GPU build")
    md.set_defaults(func=cmd_model)

    lt = sub.add_parser(
        "live-track",
        help=(
            "Poll a live Cricbuzz match, re-run Monte Carlo win-probability "
            "each over, and write live_match into data.json for the dashboard."
        ),
    )
    lt.add_argument("--match-id", default=None,
                    help="Cricbuzz numeric match ID (from the live-scores URL)")
    lt.add_argument("--auto", action="store_true",
                    help="Auto-discover a live IPL match instead of supplying --match-id")
    lt.add_argument("--home", default=None,
                    help="Home team name (used for venue lookup + display)")
    lt.add_argument("--away", default=None,
                    help="Away team name")
    lt.add_argument("--home-hint", default=None,
                    help="Fragment of home team name for auto-discovery (e.g. 'Rajasthan')")
    lt.add_argument("--away-hint", default=None,
                    help="Fragment of away team name for auto-discovery (e.g. 'Sunrisers')")
    lt.add_argument("--venue", default=None,
                    help="Venue name (falls back to fixtures table if omitted)")
    lt.add_argument("--interval", type=int, default=60,
                    help="Seconds between polls (default 60 = once per over)")
    lt.add_argument("--n-sim", type=int, default=1000,
                    help="Monte Carlo simulations per update (default 1000)")
    lt.add_argument("--out", default=None,
                    help="Path to data.json (default: auto-detected project root)")
    lt.set_defaults(func=cmd_live_track)

    st = sub.add_parser("stats", help="Show row counts")
    st.set_defaults(func=cmd_stats)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
