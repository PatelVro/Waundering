# Cricket Prediction — Data Pipeline

Starter ingestion pipeline for a cricket prediction system. Lands ball-by-ball
history, player career splits, and venue weather into a single DuckDB file you
can query or feed into a model.

## Layout

```
cricket_pipeline/
├── config.py            # URLs, env vars, cache/data paths
├── pipeline.py          # CLI entry point
├── db/
│   ├── schema.sql       # matches, innings, balls, players, weather
│   └── connection.py    # DuckDB connection factory
├── ingest/
│   ├── cricsheet.py     # download + parse CricSheet zips
│   ├── statsguru.py     # scraper (polite, cached) for Statsguru tables
│   └── weather.py       # Visual Crossing historical weather
├── examples/
│   └── basic_query.py   # sample analytics queries
└── data/                # created on first run: DuckDB file + cache/
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r cricket_pipeline/requirements.txt
```

## One-time env

```bash
export STATSGURU_CONTACT="you@example.com"        # identifies your scraper
export VISUAL_CROSSING_KEY="your_key_here"        # free tier at visualcrossing.com
```

## Quick start

From the repo root (`/home/user/Waundering`):

```bash
# 1. Pull a small T20I sample from CricSheet
python -m cricket_pipeline.pipeline cricsheet --dataset t20s_json --limit 100

# 2. Fetch a Statsguru leaderboard (cached to disk; rate-limited)
python -m cricket_pipeline.pipeline statsguru --stat batting --fmt t20i

# 3. Backfill weather for the first 20 match/venue pairs
python -m cricket_pipeline.pipeline weather --limit 20

# 4. Row counts + sample queries
python -m cricket_pipeline.pipeline stats
python -m cricket_pipeline.examples.basic_query
```

## Available CricSheet datasets

`all_json`, `tests_json`, `odis_json`, `t20s_json`, `ipl_json`, `bbl_json`,
`psl_json`, `the_hundred`. Start small (`--limit 100`) before pulling
`all_json` (thousands of matches).

## What's in the DB after ingestion

| Table           | Grain                            |
|-----------------|----------------------------------|
| `matches`       | one row per match                |
| `innings`       | one row per innings              |
| `balls`         | one row per delivery (source of truth) |
| `player_splits` | Statsguru leaderboard rows       |
| `weather_daily` | one row per (venue, match date)  |
| `venues`        | optional manual enrichment       |

## Extending

- **Ball-tracking (Hawk-Eye):** add a `ball_tracking` table keyed on
  `(match_id, innings_no, over_no, ball_in_over)` — same PK as `balls`.
- **Biometrics:** add a `player_state_daily` table keyed on `(player_id, date)`.
- **News/sentiment:** add a `news` table keyed on `(match_id, source, published_at)`.

## Notes on scraping

Statsguru has no public API. This scraper caches responses and sleeps between
requests (tune via `STATSGURU_SLEEP` env var). Use only for personal / research
work and respect ESPNCricinfo's terms. For commercial use, license data from a
paid provider (Opta, CricViz, Roanuz, SportMonks, Entity Sports).

## Next steps toward a model

1. Verify the DB has data — `python -m cricket_pipeline.pipeline stats`
2. Export a ball-level feature table (joining balls + matches + weather)
3. Train a baseline XGBoost to predict `runs_total` on the next ball
4. Wrap it in a Monte Carlo rollout to simulate full matches
