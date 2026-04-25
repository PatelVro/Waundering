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

### One-shot installer (detects your GPU automatically)

```bash
./scripts/install.sh
# or, equivalently:
make install
```

What it does:
1. Detects OS + NVIDIA GPU + CUDA driver version (via `nvidia-smi`)
2. Creates a `.venv/` (skip with `--no-venv` if you have your own env)
3. Installs CPU base deps (DuckDB, pandas, sklearn, LightGBM, …)
4. Installs **PyTorch with the matching CUDA wheel** (`cu118` / `cu121` / `cu124`) — sequence-model training will use the GPU automatically
5. Runs `scripts/gpu_check.py` to verify

**Force CPU-only** (no GPU on this machine, or you don't want it):

```bash
./scripts/install.sh --cpu
```

**Optional: LightGBM with GPU build** (requires Boost + OpenCL dev headers; modest speedup):

```bash
./scripts/install.sh --lightgbm-gpu
```

### Manual install (if you prefer)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r cricket_pipeline/requirements.txt
# For NVIDIA GPU, replace torch with the CUDA wheel that matches your driver:
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

### Verify

```bash
make gpu-check
# or
python scripts/gpu_check.py
```

You should see:
```
✓ GPU0: NVIDIA GeForce RTX 4060  driver 555.42.02  vram 8188 MiB
✓ matmul on GPU works.
```

### What actually uses the GPU

| Component | GPU benefit | Notes |
|---|---|---|
| **Sequence Transformer** (`model/sequence.py`) | **~30× faster** | Auto-detects CUDA. Pass `--device cuda` to force. |
| **LightGBM ball model** | ~2-3× faster | Only with `--lightgbm-gpu` build; small datasets see no win |
| **LightGBM match model** | ~1× | Too few rows (~1k) for GPU to matter |
| Data ingestion / DuckDB | none | CPU-bound, already fast (10s for full IPL) |
| Monte Carlo simulator | none | Already vectorised in numpy + LightGBM batched |

Practical message: **the GPU mainly speeds up `model train --type sequence`.** For the ball-outcome and match-outcome models, CPU is fine.

### Pass `--device` explicitly

```bash
# Force CUDA for sequence model
python -m cricket_pipeline.pipeline model train --type sequence --device cuda

# Force CPU on a GPU machine (e.g. for reproducibility comparison)
python -m cricket_pipeline.pipeline match-train --device cpu

# Auto-detect (default)
python -m cricket_pipeline.pipeline model train --device auto
```

### Convenience targets

```bash
make install            # install + GPU detect
make refresh            # daily-refresh
make sequence-train     # GPU-accelerated training
make forecast HOME='Lucknow Super Giants' AWAY='Kolkata Knight Riders' \
              VENUE='Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow'
make clean              # wipe DuckDB + cache + models
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

# 2b. Backfill players.country from already-cached CricSheet zips (free, no scrape)
python -m cricket_pipeline.pipeline cs-players

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

## Live-match operations — predicting tomorrow's game

The pipeline is built so the same machine that ingests data also runs the
forecast. Here's the operational flow for an upcoming IPL match:

### Day before / morning of (cron)

```bash
# Re-pull CricSheet + reinstall views + retrain match model. ~15s on T20+IT20.
python -m cricket_pipeline.pipeline daily-refresh --datasets ipl_json --fmt T20,IT20
```

A reasonable cron: `0 5 * * *  python -m cricket_pipeline.pipeline daily-refresh --datasets ipl_json`.

### ~1 hour before toss (poll for XI + toss)

Find the Cricbuzz match-squads URL for the fixture (linked off the match
preview page) and run:

```bash
python -m cricket_pipeline.pipeline prematch \
  --url "https://www.cricbuzz.com/cricket-match-squads/<id>/<slug>" \
  --max-wait-seconds 3600 \
  --poll-seconds 120 \
  --out cricket_pipeline/data/cache/prematch_<id>.json
```

This polls every 2 minutes for up to an hour. As soon as both XIs and the
toss line appear, it writes the JSON to disk and exits.

### Right before the match (forecast)

```bash
python -m cricket_pipeline.pipeline match-forecast \
  --home "Lucknow Super Giants" \
  --away "Kolkata Knight Riders" \
  --venue "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow" \
  --home-xi "MR Marsh,AT Markram,RR Pant,N Pooran,..." \
  --away-xi "AM Rahane,FH Allen,C Green,..." \
  --toss-winner "Lucknow Super Giants" \
  --toss-decision bat
```

Outputs the full forecast in human-readable form. Add `--json` for
machine-readable output. All XI / toss inputs are optional — the model uses
fallback heuristics where they're missing.

### What each piece of real-time info shifts

| Input | When announced | Effect on prediction |
|---|---|---|
| Daily form refresh | Morning of match | Catches up on overnight matches |
| Playing XIs | ~30 min before toss | Player-level top scorer / wicket picks become accurate |
| Toss winner + decision | ~30 min before toss | Removes the toss-marginalisation; ±5-8 pp swing |
| Pitch report | Hour before / first 5 overs | Manual review — feed into venue prior |
| Dew | Second-innings only | Manual: re-run forecast with `--toss-decision field` weight |

### When the scrapers fail

Cricbuzz aggressively blocks cloud-IP user agents (returns 403). This is
expected from CI / cloud / corporate-VPN environments. From a residential
IP it works fine. Workarounds:

1. Run from a local machine or a residential VPS
2. Pass `--home-xi` and `--away-xi` manually from any other source (IPLT20.com,
   ESPNCricinfo, official team Twitter)
3. Pass `--toss-winner` and `--toss-decision` manually after toss

The pipeline's "last mile" is human-in-the-loop on purpose — chasing
real-time scraping reliability across providers is a losing game.

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

| File          | Purpose                                                     |
|---------------|-------------------------------------------------------------|
| `features.py` | joins `v_ball_state` + player/venue/form/weather views      |
| `train.py`    | LightGBM runs (multiclass) + wicket (binary)                |
| `calibrate.py`| isotonic calibration on a held-out 10% slice                |
| `predict.py`  | scores one ball or a batch (LightGBM)                       |
| `simulate.py` | vectorised Monte Carlo innings rollout                      |
| `sequence.py` | Transformer over the last 12 balls (PyTorch) — captures momentum |

### Two model architectures

The CLI supports both via `--type`:

```bash
# 1. Independent ball model (LightGBM, default — fast, strong baseline)
python -m cricket_pipeline.pipeline model train --type lgbm --fmt IT20

# 2. Sequence model (Transformer — captures momentum from the last 12 balls)
python -m cricket_pipeline.pipeline model train --type sequence --fmt IT20 --epochs 8
```

The sequence model is a small Transformer (~2 layers, 4 heads, d_model=64)
with learned embeddings for batter / bowler / venue. It looks at the last
`SEQ_LEN=12` deliveries within the same innings and predicts the same two
heads (runs multiclass + wicket binary). For early balls in an innings the
sequence is left-padded with a key-padding mask.

Use it when:
- You want better accuracy on momentum-driven moments (death overs, run chases)
- You have a GPU available (CPU training works but is slower — ~15 minutes
  on a typical CPU for 200k T20 balls × 8 epochs)

Stick with `--type lgbm` when:
- You need millisecond inference latency
- You're using the Monte Carlo simulator (which currently routes through the
  LightGBM predictor; integrating the sequence model is a follow-up)

### Predicting with the sequence model

`--state` accepts either a single state dict (auto-wrapped) or a list of
state dicts representing the last few balls (oldest-first). The last entry
is what gets predicted; earlier entries are context.

```bash
python -m cricket_pipeline.pipeline model predict --type sequence \
  --state '[{...prev ball 1...}, {...prev ball 2...}, {...current ball...}]'
```

Outputs:
- `runs_probs`: distribution over {0,1,2,3,4,5+,6}
- `wicket_prob`: probability the ball ends in a non-runout dismissal
- `expected_runs`: scalar
- For simulations: mean / p10 / p50 / p90 / histogram / `win_prob` (if target set)

Caveats (read these before using outputs in anger):
1. ~~Temporal leakage~~ **Fixed** — features now join `v_batter_history` /
   `v_bowler_history` which compute career & rolling stats *strictly before*
   each ball's match.
2. ~~Uncalibrated probabilities~~ **Fixed** — isotonic calibration is fit on a
   held-out 10% slice and applied automatically in `predict_ball`,
   `predict_batch`, and the simulator. Both raw and calibrated log-loss are
   reported.
3. **Strike rotation** — the simulator approximates this; it doesn't swap the
   batter's *features* mid-innings.
4. **No batting order queue** — when a wicket falls the simulator keeps the
   same striker label. Pass an explicit batting order list to extend.
5. **Ball-tracking layer is missing** — without Hawk-Eye, the model can't
   learn from pace, swing, seam, RPM. That's the next big jump if you can
   license it.

## Notes on scraping

Statsguru has no public API. This scraper caches responses and sleeps between
requests (tune via `STATSGURU_SLEEP` env var). Use only for personal / research
work and respect ESPNCricinfo's terms. For commercial use, license data from a
paid provider (Opta, CricViz, Roanuz, SportMonks, Entity Sports).

## Roadmap

The data foundation is broad enough; the next jumps are quality and breadth:

- ~~**Time-aware feature aggregates**~~ ✅ done — features now join the
  `v_batter_history` / `v_bowler_history` views which use windows
  `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` so the model only sees
  data from before the current match.
- ~~**Calibrate** the runs and wicket probabilities~~ ✅ done — isotonic on a
  10% holdout, applied automatically by `predict_ball`, `predict_batch`, and
  the simulator.
- ~~**Bayesian shrinkage** on bowler-vs-batter matchup priors~~ ✅ done — see
  `v_matchup_shrunk` (k=30 prior balls toward the bowler's overall rate).
- ~~**Player profile updates from CricSheet match files**~~ ✅ done — see
  `pipeline cs-players`, which walks cached zips and fills `players.country`
  from the most-recent international team each player appears in.
- ~~**Bigger architectures**: LSTM / Transformer over sequences of recent balls~~
  ✅ done — see `model/sequence.py` (small Transformer, SEQ_LEN=12, learned
  batter/bowler/venue embeddings, isotonic-calibrated heads). Run via
  `pipeline model train --type sequence`. Simulator integration is a follow-up.
- **Hawk-Eye / TrackMan** ball-tracking features (pace, swing, seam, RPM)
- **Strike rotation + batting-order queue** in the simulator (currently
  approximated)
