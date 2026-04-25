# Cricket Prediction — Data Pipeline

Starter ingestion pipeline for a cricket prediction system. Lands ball-by-ball
history, player career splits, and venue weather into a single DuckDB file you
can query or feed into a model.

## Layout

```
cricket_pipeline/
├── config.py              # URLs, env vars, cache/data paths
├── pipeline.py            # CLI entry point
├── model/
│   ├── features.py        # join v_ball_state with aggregate views
│   ├── train.py           # LightGBM runs + wicket models
│   ├── predict.py         # single-ball / batch inference
│   └── simulate.py        # Monte Carlo innings rollout
├── db/
│   ├── schema.sql         # 12 tables — matches, innings, balls, players, ...
│   ├── views.sql          # derived analytical views (venue profile, phase metrics, ...)
│   └── connection.py      # DuckDB connection factory + view installer
├── ingest/
│   ├── cricsheet.py            # ball-by-ball zips (27 datasets — see `pipeline datasets`)
│   ├── people.py               # CricSheet people.csv → cross-provider IDs
│   ├── cricinfo_profiles.py    # Cricinfo profile pages → dob, role, batting/bowling style
│   ├── statsguru.py            # leaderboards + groupby splits
│   ├── rankings.py             # ICC men's player + team rankings
│   ├── venues.py               # geocode venues via Nominatim
│   ├── wikipedia.py            # venue infobox enrichment
│   ├── weather.py              # Visual Crossing historical weather
│   ├── openweather.py          # OpenWeatherMap current + 5-day forecast
│   ├── news.py                 # multi-source RSS + sentiment + entity tags
│   ├── newsapi.py              # NewsAPI (broader news, 100 req/day free)
│   ├── gdelt.py                # GDELT 2.0 global news event database (no key)
│   ├── wikidata.py             # SPARQL: dob, height, debut year, country
│   ├── umpires.py              # derives umpires from matches.umpires
│   ├── fixtures.py             # CricAPI upcoming/live matches
│   ├── partnerships.py         # derived partnerships from ball-by-ball
│   └── cricbuzz.py             # best-effort live match-state snapshots
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
export NEWSAPI_KEY="your_key_here"                # 100 req/day free at newsapi.org
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

# 13. Cricinfo player profiles (dob, role, batting/bowling style)
python -m cricket_pipeline.pipeline profiles --limit 200

# 14. Install/refresh derived analytical views (free, no fetch)
python -m cricket_pipeline.pipeline views

# 15. Browse all available CricSheet datasets
python -m cricket_pipeline.pipeline datasets

# 16. Derive partnerships from ball-by-ball data
python -m cricket_pipeline.pipeline partnerships

# 17. NewsAPI (100 req/day free; needs NEWSAPI_KEY)
python -m cricket_pipeline.pipeline newsapi --query cricket --days 7

# 18. Live match snapshots from Cricbuzz (unofficial, brittle)
python -m cricket_pipeline.pipeline cricbuzz <CricbuzzMatchId> [<MatchId2> ...]

# 19. GDELT — global cricket news (no key required)
python -m cricket_pipeline.pipeline gdelt --hours 72

# 20. Wikidata player enrichment (height, debut year, etc.)
python -m cricket_pipeline.pipeline wikidata

# 21. Row counts + sample queries
python -m cricket_pipeline.pipeline stats
python -m cricket_pipeline.examples.basic_query
```

## Available CricSheet datasets (free, ball-by-ball)

International: `tests_json`, `odis_json`, `t20s_json`, `all_json`.
Men's franchise: `ipl_json`, `bbl_json`, `psl_json`, `cpl_json`, `lpl_json`,
`bpl_json`, `sa20_json`, `ilt20_json`, `mlc_json`, `the_hundred_men`,
`vitality_blast`, `super_smash`, `syed_mushtaq_ali`.
Men's first-class / list A: `county_championship`, `ranji_trophy`,
`sheffield_shield`, `royal_one_day_cup`.
Women's: `women_t20s`, `women_odis`, `women_tests`, `wpl_json`, `wbbl_json`,
`the_hundred_women`.

Run `python -m cricket_pipeline.pipeline datasets` for the URL list.

## Analytical views (after `pipeline views`)

| View                 | What it gives you                                              |
|----------------------|-----------------------------------------------------------------|
| `v_venue_profile`    | per-venue avg first-innings score, toss-winner-wins-pct        |
| `v_phase_metrics`    | run rate / wicket% / dot% / boundary% by phase per format      |
| `v_batter_profile`   | runs, SR, average, dismissals from ball data                   |
| `v_bowler_profile`   | wickets, economy, average, balls bowled                        |
| `v_matchup`          | bowler-vs-batter raw counts (feed into a Bayesian shrink)      |
| `v_umpire_lbw`       | LBW propensity per umpire (real bias signal)                   |
| `v_toss_impact`      | does winning the toss matter at this venue?                    |
| `v_ball_state`       | one row per delivery with running totals, RR, RRR, phase       |
| `v_batter_form`      | rolling 10-innings runs, SR, average per batter                |
| `v_bowler_workload`  | overs bowled in last 7 / 30 / 90 days per bowler               |
| `v_top_partnerships` | partnership leaderboard sortable by runs                       |
| `v_bowler_spells`    | per-spell stats (start/end over, runs, wickets, economy)       |
| `v_fielding_profile` | catches, run-outs, stumpings, c&b per fielder                  |
| `v_pitch_deterioration` | run rate / wicket% / boundary% by innings number per venue  |
| `v_time_metrics`     | run rate + wicket% by year × month × format                    |

## Available CricSheet datasets

`all_json`, `tests_json`, `odis_json`, `t20s_json`, `ipl_json`, `bbl_json`,
`psl_json`, `the_hundred`. Start small (`--limit 100`) before pulling
`all_json` (thousands of matches).

## What's in the DB after ingestion

| Table              | Grain                                                       |
|--------------------|-------------------------------------------------------------|
| `matches`          | one row per match                                           |
| `innings`          | one row per innings                                         |
| `balls`            | one row per delivery (source of truth)                      |
| `players`          | identifier + cross-provider IDs + dob/role/style/country    |
| `player_splits`    | Statsguru leaderboard / split rows                          |
| `rankings`         | ICC men's rankings snapshot                                 |
| `venues`           | venue, lat/lon, capacity, ends, established                 |
| `weather_daily`    | one row per (venue, date) — historical + live               |
| `news`             | RSS items with sentiment + entity tags                      |
| `fixtures`         | upcoming + live matches                                     |
| `umpires`          | matches officiated, formats                                 |
| `match_officials`  | every umpire / TV umpire / referee per match                |
| `partnerships`     | derived from balls — wicket #, batters, runs, balls         |
| `live_state`       | rolling Cricbuzz live snapshots                             |

## Extending

- **Ball-tracking (Hawk-Eye):** add a `ball_tracking` table keyed on
  `(match_id, innings_no, over_no, ball_in_over)` — same PK as `balls`.
- **Biometrics:** add a `player_state_daily` table keyed on `(player_id, date)`.
- **News/sentiment:** add a `news` table keyed on `(match_id, source, published_at)`.

## Modelling — ball-outcome predictor

After data is loaded and views installed, you can train the prototype model:

```bash
# train on T20I balls (smaller, faster), then predict + simulate
python -m cricket_pipeline.pipeline model train --fmt IT20 --limit 200000

# predict one ball
python -m cricket_pipeline.pipeline model predict \
  --state '{"format":"IT20","venue":"Eden Gardens","phase":"death", \
            "over_no":18,"ball_in_over":1,"runs_so_far":150,"wickets_so_far":4, \
            "deliveries_so_far":108,"legal_balls_left":12, \
            "current_run_rate":8.33,"required_run_rate":12.5}'

# Monte Carlo a chase
python -m cricket_pipeline.pipeline model simulate \
  --n-sim 5000 --seed 0 \
  --state '{"format":"IT20","target":180, ...}'

# end-to-end worked example
python -m cricket_pipeline.examples.model_demo
```

What's inside `model/`:

| File         | Purpose                                                     |
|--------------|-------------------------------------------------------------|
| `features.py`| joins `v_ball_state` + player/venue/form/weather views      |
| `train.py`   | LightGBM runs (multiclass) + wicket (binary) — saves to `data/models/` |
| `predict.py` | scores one ball or a batch                                  |
| `simulate.py`| vectorised Monte Carlo innings rollout                      |

Outputs:
- `runs_probs`: distribution over {0,1,2,3,4,5+,6}
- `wicket_prob`: probability the ball ends in a non-runout dismissal
- `expected_runs`: scalar
- For simulations: mean / p10 / p50 / p90 / histogram / `win_prob` (if target set)

Caveats (read these before using outputs in anger):
1. **Temporal leakage** — career aggregates (`v_batter_profile` etc.) are computed across all data including future balls. The reported test metrics are therefore optimistic. Fix: bucket aggregates by year and join the year-aware aggregate.
2. **Strike rotation** — the simulator approximates this; it doesn't swap the batter's *features* mid-innings.
3. **No batting order queue** — when a wicket falls the simulator keeps the same striker label. Pass an explicit batting order list to extend.
4. **Single ball-tracking layer is missing** — without Hawk-Eye, the model can't learn from pace, swing, seam, RPM. That's the next big jump if you can license it.

## Notes on scraping

Statsguru has no public API. This scraper caches responses and sleeps between
requests (tune via `STATSGURU_SLEEP` env var). Use only for personal / research
work and respect ESPNCricinfo's terms. For commercial use, license data from a
paid provider (Opta, CricViz, Roanuz, SportMonks, Entity Sports).

## Roadmap

The data foundation is broad enough; the next jumps are quality and breadth:

- **Time-aware feature aggregates** to remove leakage in career stats
- **Calibrate** the runs and wicket probabilities (Platt / isotonic on a holdout)
- **Bayesian shrinkage** on bowler-vs-batter matchup priors
- **Bigger architectures**: LSTM / Transformer over sequences of recent balls
- **Hawk-Eye / TrackMan** ball-tracking features (pace, swing, seam, RPM)
- **Player profile updates from CricSheet match files** to fill `country` / `batting_hand` / `bowling_type` directly from match metadata when Cricinfo can't be reached
