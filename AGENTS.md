# AGENTS.md

Briefing for AI agents (Claude Code, Cursor, Codex, Copilot agent mode, …) that
have just checked out this repo. Read this in full before doing anything; it
will save you from re-discovering everything by exploring.

---

## 1. What this repo is

There are two unrelated projects in the same directory tree. **Don't confuse them.**

| Project | Where | What it is |
|---|---|---|
| **VoltOps marketing site** | `index.html`, `styles.css`, `script.js`, `serve-phone.sh`, `share-public.sh`, `check.sh`, `README.md` | A static landing page. Ignore unless the user explicitly asks about it. |
| **Cricket prediction pipeline** | `cricket_pipeline/`, `scripts/`, `Makefile` | The substantive project. End-to-end data → model → forecast for cricket matches. **This is what most user requests are about.** |

If a user asks something cricket-related ("predict tomorrow's IPL match", "train the model", "data sources", etc.) → it's the pipeline.

---

## 2. First-run install (one shot)

```bash
./scripts/install.sh
```

What it does (read the script before running on a real laptop — it creates a venv):
1. Detects OS + NVIDIA GPU + CUDA version via `nvidia-smi`
2. Creates `.venv/` (skip with `--no-venv`)
3. Installs base deps from `cricket_pipeline/requirements.txt`
4. Installs PyTorch with the matching CUDA wheel (cu118 / cu121 / cu124) if a GPU is present, else CPU PyTorch
5. Optionally builds LightGBM with GPU support (`--lightgbm-gpu`, best-effort)
6. Runs `scripts/gpu_check.py` to verify

**Then verify**: `python scripts/gpu_check.py` (or `make gpu-check`). It exits 0 if at least PyTorch is usable.

**No GPU machine?** `./scripts/install.sh --cpu` — everything still works, just slower for sequence training.

---

## 3. Common commands (the only ones most users need)

```bash
# 1. Pull / refresh data + retrain the match model — ~15 seconds for IPL
make refresh
# equivalent: python -m cricket_pipeline.pipeline daily-refresh --datasets ipl_json

# 2. Get a complete forecast for a single match
make forecast \
  HOME='Lucknow Super Giants' \
  AWAY='Kolkata Knight Riders' \
  VENUE='Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow'
# equivalent: python -m cricket_pipeline.pipeline match-forecast --home … --away … --venue …

# 3. Pre-match polling for XIs + toss (run ~1h before toss)
python -m cricket_pipeline.pipeline prematch \
  --url "https://www.cricbuzz.com/cricket-match-squads/<id>/<slug>" \
  --max-wait-seconds 3600

# 4. Track a live match in real-time (auto-updates data.json every over)
make live-track                          # auto-discover any live IPL match
make live-track MATCH_ID=151902          # explicit Cricbuzz match ID
# equivalent (with team names for venue lookup):
python -m cricket_pipeline.pipeline live-track --auto \
  --home-hint "Rajasthan" --away-hint "Sunrisers" \
  --interval 60 --n-sim 1000

# 5. Train models manually
make match-train       # match-outcome (binary classifier)
make ball-train        # ball-outcome (LightGBM, runs + wicket heads)
make sequence-train    # Transformer (uses GPU automatically — 30x faster on CUDA)
```

Full subcommand list: `python -m cricket_pipeline.pipeline --help`.

---

## 4. Architecture in 30 seconds

```
17 INGESTERS → 14 TABLES (DuckDB) → 15 VIEWS → 3 MODELS → FORECAST
```

- **Ingesters** (`cricket_pipeline/ingest/`): pull from CricSheet (ball-by-ball), Cricinfo, Wikidata, Statsguru, ICC rankings, Nominatim, Wikipedia, Visual Crossing, OpenWeatherMap, RSS news (9 sources), GDELT, NewsAPI, Cricbuzz live state, CricAPI fixtures, Cricbuzz lineup pages.
- **DB**: a single DuckDB file at `cricket_pipeline/data/cricket.duckdb`. Schema in `cricket_pipeline/db/schema.sql`. Derived analytical views in `cricket_pipeline/db/views.sql`.
- **Models** (`cricket_pipeline/model/`):
  - `train.py` — LightGBM ball-outcome (runs multiclass + wicket binary), isotonic-calibrated
  - `match.py` — LightGBM match-outcome (binary win classifier) + ensemble blender (60/25/15 match-model / form / h2h)
  - `sequence.py` — small Transformer over the last 12 balls, PyTorch + CUDA
  - `simulate.py` — vectorised Monte Carlo innings rollout
  - `calibrate.py` — isotonic regression for both heads
- **Forecast** (`cricket_pipeline/forecast.py`): single function returning a `MatchForecast` dataclass — winner, score distribution, top batters/bowlers, key matchups
- **CLI** (`cricket_pipeline/pipeline.py`): every subcommand listed in section 3

Trained model artefacts live in `cricket_pipeline/data/models/` (gitignored).

---

## 5. The data flow when forecasting a match

1. `daily-refresh` re-downloads CricSheet zips and bulk-inserts into DuckDB (~10s for full IPL via registered DataFrame insert).
2. Views are reinstalled — the time-aware `v_batter_history` and `v_bowler_history` use `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` to avoid temporal leakage.
3. Match model retrains on `v_match_features`. Format filter accepts comma-separated lists like `T20,IT20`. Default is T20+IT20.
4. `match-forecast` runs:
   - `predict_match_ensemble` — blends the LightGBM match model with form prior + h2h prior
   - `simulate_innings` — Monte Carlo rollout (5000 sims) for each team batting first
   - SQL queries against the DB — top batters / bowlers in recent IPL window per side
   - `v_matchup_shrunk` — Bayesian-shrunk bowler-vs-batter risks
   - Outputs everything to a single `MatchForecast` and pretty-prints it

---

## 6. GPU usage — what actually benefits

| Component | GPU benefit | Notes |
|---|---|---|
| **Sequence Transformer** (`model train --type sequence`) | **~30x faster** | Auto-uses CUDA. Pass `--device cuda` to force. |
| LightGBM ball model | ~2-3x | Only with `--lightgbm-gpu` build |
| LightGBM match model | ~1x | Too few rows for GPU to matter |
| DuckDB / data ingestion | none | Already fast on CPU (10s for full IPL) |
| Monte Carlo simulator | none | Vectorised on CPU, batched LightGBM call |

`--device {auto,cpu,gpu,cuda}` is wired on `model train` and `match-train`. Auto-detect probes a tiny LightGBM call to test the GPU build.

---

## 7. Known gotchas — read before debugging

- **CricSheet ingestion is bulk-insert** (registered DataFrame). If you see slow rows-per-sec, check `ingest/cricsheet.py::_flush` is using `con.register("_stage", df)` not `executemany`.
- **DuckDB rejects duplicate window names across views in the same script**. The bowler-history view uses `wb` instead of `w`. Don't rename.
- **Pandas dtypes for LightGBM**: `required_run_rate` must be `float('nan')` not `None`, otherwise pandas keeps it as object dtype and LightGBM rejects it. See `model/simulate.py`.
- **Cricbuzz returns 403 from cloud IPs**. Lineup scraping works only from a residential IP. From CI / sandbox / VPN, fall back to `--home-xi` / `--away-xi` flags or paste-from-Twitter.
- **Reddit RSS needs a custom User-Agent**. `news.py` already sets one (default UA gets blocked).
- **Statsguru / Cricinfo / Nominatim / Wikipedia** want `STATSGURU_CONTACT` env var. They're polite-rate-limited via `tenacity`.

---

## 8. Where to dig deeper

| Question | Read |
|---|---|
| Full pipeline overview, all data sources, all view definitions | `cricket_pipeline/README.md` (the long one) |
| How the model works (features, training, calibration, ensembling) | `cricket_pipeline/README.md` "Modelling" section |
| Live-match operational flow (cron + prematch polling) | `cricket_pipeline/README.md` "Live-match operations" section |
| GPU install + verification | `scripts/install.sh`, `scripts/gpu_check.py` |
| Database schema | `cricket_pipeline/db/schema.sql` |
| Derived analytical views | `cricket_pipeline/db/views.sql` |
| Working code examples | `cricket_pipeline/examples/` |

---

## 9. Behavioral notes for AI agents

- **Don't add documentation files unless the user asks.** This file is the exception because it's the agent briefing.
- **Don't add backwards-compatibility shims** — if a function signature changes, update all call sites.
- **The branch `claude/cricket-gpu-setup` is the active development branch** at the time this file was written. Check `git branch --show-current` and `git log origin/main..HEAD` before pushing.
- **Tests / lint configs are minimal**. There is no test suite. Validate by running `python -m compileall -q cricket_pipeline/` and the relevant `make` target.
- **PRs**: create as draft (`draft: true`) — that matches the convention used so far. The user merges manually.
- **GitHub interactions** go via the `mcp__github__*` tools when available. The repo scope is restricted to `patelvro/waundering` only.
- **When the user asks for a prediction** (e.g. "predict tomorrow's match"), the canonical path is:
  1. `make refresh` (or `daily-refresh` subcommand)
  2. `make forecast HOME=… AWAY=… VENUE=…`
  3. Show the human-readable output. Don't paraphrase — the structured output already explains itself.

---

## 10. The honest ceiling

Without ball-tracking data (Hawk-Eye / TrackMan), the model's wicket AUC plateaus around 0.57 and match-outcome AUC around 0.55. This is the ceiling of public data. See `cricket_pipeline/README.md` "Roadmap" for what it would cost to push past that. Don't promise the user better than honest probabilities — T20 is genuinely high variance.
