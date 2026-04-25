# Cricket Prediction — Data Pipeline

Starter ingestion pipeline for a cricket prediction system. Lands ball-by-ball
history, player career splits, and venue weather into a single DuckDB file you
can query or feed into a model.

## Layout

```
cricket_pipeline/
├── config.py              # URLs, env vars, cache/data paths
├── pipeline.py            # CLI entry point
├── db/
│   ├── schema.sql         # matches, innings, balls, players, weather, rankings, news
│   └── connection.py      # DuckDB connection factory
├── ingest/
│   ├── cricsheet.py       # ball-by-ball zips
│   ├── people.py          # CricSheet people.csv → cross-provider IDs
│   ├── statsguru.py       # leaderboards + groupby splits (year / opposition / ground)
│   ├── rankings.py        # ICC men's player + team rankings
│   ├── venues.py          # geocode venues via Nominatim
│   ├── wikipedia.py       # venue infobox enrichment (capacity, ends, established)
│   ├── weather.py         # Visual Crossing historical weather
│   ├── openweather.py     # OpenWeatherMap current + 5-day forecast
│   ├── news.py            # multi-source RSS + VADER sentiment + entity tags
│   ├── umpires.py         # derives umpires table from matches.umpires
│   └── fixtures.py        # CricAPI upcoming/live match list
├── examples/
│   └── basic_query.py     # sample analytics queries
└── data/                  # created on first run: DuckDB file + cache/
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r cricket_pipeline/requirements.txt
```

## One-time env

```bash
export STATSGURU_CONTACT="you@example.com"        # identifies your scrapers (Statsguru, Nominatim, Wikipedia)
export VISUAL_CROSSING_KEY="your_key_here"        # free tier at visualcrossing.com
export OPENWEATHER_KEY="your_key_here"            # free tier at openweathermap.org
export CRICAPI_KEY="your_key_here"                # free tier at cricapi.com (fixtures only)
```

## Quick start

From the repo root (`/home/user/Waundering`):

```bash
# 1. Pull a small T20I sample from CricSheet
python -m cricket_pipeline.pipeline cricsheet --dataset t20s_json --limit 100

# 2. Load CricSheet people registry (Cricinfo / Cricbuzz / BCCI ids)
python -m cricket_pipeline.pipeline people

# 3. Geocode every venue seen in matches (lat/lon via Nominatim)
python -m cricket_pipeline.pipeline venues --limit 200

# 4. ICC rankings (Test/ODI/T20I, batting/bowling/allrounder, plus team)
python -m cricket_pipeline.pipeline rankings

# 5. Statsguru — overall leaderboard
python -m cricket_pipeline.pipeline statsguru --stat batting --fmt t20i

# 6. Statsguru — by year / opposition / ground
python -m cricket_pipeline.pipeline statsguru-split --stat batting --fmt t20i --groupby year
python -m cricket_pipeline.pipeline statsguru-split --stat batting --fmt t20i --opposition India

# 7. Weather — historical (Visual Crossing) for matches
python -m cricket_pipeline.pipeline weather --limit 20

# 8. Weather — current + 5-day forecast (OpenWeatherMap) for geocoded venues
python -m cricket_pipeline.pipeline owm --limit 50

# 9. News + sentiment (Cricinfo / Cricbuzz / ICC / Wisden RSS)
python -m cricket_pipeline.pipeline news

# 10. Wikipedia venue enrichment (capacity, ends, established year)
python -m cricket_pipeline.pipeline wiki --limit 100

# 11. Umpires table (derived purely from already-loaded matches)
python -m cricket_pipeline.pipeline umpires

# 12. Upcoming + live fixtures (needs CRICAPI_KEY)
python -m cricket_pipeline.pipeline fixtures

# 13. Row counts + sample queries
python -m cricket_pipeline.pipeline stats
python -m cricket_pipeline.examples.basic_query
```

## Available CricSheet datasets

`all_json`, `tests_json`, `odis_json`, `t20s_json`, `ipl_json`, `bbl_json`,
`psl_json`, `the_hundred`. Start small (`--limit 100`) before pulling
`all_json` (thousands of matches).

## What's in the DB after ingestion

| Table           | Grain                                                       |
|-----------------|-------------------------------------------------------------|
| `matches`       | one row per match                                           |
| `innings`       | one row per innings                                         |
| `balls`         | one row per delivery (source of truth)                      |
| `players`       | CricSheet identifier → Cricinfo/Cricbuzz/BCCI               |
| `player_splits` | Statsguru leaderboard / split rows                          |
| `rankings`      | ICC men's rankings snapshot                                 |
| `venues`        | venue, lat/lon, capacity, ends, established                 |
| `weather_daily` | one row per (venue, date) — historical + live               |
| `news`          | RSS items with sentiment + entity tags                      |
| `fixtures`      | upcoming + live matches                                     |
| `umpires`       | matches officiated, formats                                 |

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
