# Bootstrap guide for remote agents

You are working on **Waundering**, a cricket prediction pipeline at
`PatelVro/Waundering`. This document tells you what's in the repo, what's
deliberately *not* in the repo, and how to get from "git clone" to a
running model fast.

## What's in the repo

- **Source code.** All ingest / feature / model / orchestrator / dashboard
  code under `cricket_pipeline/` plus the React dashboard at the root.
- **Documentation.**
  - `HOW_TO_USE.md` — operator playbooks (Cycles 0-13)
  - `cricket_pipeline/work/progress_log.md` — running log of every cycle's
    experiments, what shipped, what regressed
  - `cricket_pipeline/work/runs/*.md` — error analyses against the
    production stacked ensemble (T20 + ODI)
  - `cricket_pipeline/work/runs/*.json` — per-cycle metrics (accuracy,
    Brier, log-loss, ECE) so you can see the trajectory
- **Methodology.** All experiment scripts live under `cricket_pipeline/work/`:
  - `recency_experiment.py` (Cycle 7 — recency weighting)
  - `step5_venue_window_experiment.py` (Cycle 9 — venue windowing, regressed)
  - `step6_margin_experiment.py` (Cycle 10 — margin features, regressed)
  - `step8_weather_experiment.py` (Cycle 12 — weather features, T20 only)
- **Calibrators.** `cricket_pipeline/work/runs/calibrators/*.joblib` are
  the per-tier isotonic calibrators applied by `predict_match.py`.
- **Post-match learnings.** `learnings/*.md` are auto-generated post-match
  reviews — predicted vs actual, model rationale, base-learner agreement.
  These are the highest-density methodology artifacts in the repo.
- **Tests.** `cricket_pipeline/work/tests/` — 69 unit tests covering bet
  engine, match-phase machine, walk-forward harness, leakage audits.
- **Config templates.** `.env.example` lists every env var the pipeline
  reads, with safe defaults.

## What's deliberately NOT in the repo

These are gitignored on purpose — don't try to commit them:

- **`cricket_pipeline/data/cricket.duckdb`** (~7 GB). The source-of-truth
  ball-by-ball database. Regenerate from CricSheet zips: see "Bootstrap"
  below.
- **`.env`** — API keys (Odds API, etc.) and any Polymarket / Betfair
  credentials. Never commit. Copy `.env.example` and fill in your own.
- **`data.json`, `data/preds/`, `predictions/`** — live dashboard state
  with bet stakes, bankroll, PnL. Privacy-sensitive. Regenerate from
  the orchestrator if needed.
- **`cricket_pipeline/work/runs/orchestrator_state.json`,
  `live_match*.json`, `match_timeline.jsonl`, `*.pid`** — runtime state
  that changes every tick. Stale by the time you read it.
- **`*.log`** — all log files. Rotated; don't track.

## Bootstrap from a fresh clone (15-30 min)

```bash
# 1. Clone + venv
git clone https://github.com/PatelVro/Waundering.git
cd Waundering
python -m venv .venv
source .venv/Scripts/activate     # Windows: .venv\Scripts\activate
pip install -r cricket_pipeline/requirements.txt

# 2. Secrets
cp .env.example .env
# Edit .env: at minimum set THE_ODDS_API_KEY (free tier at
# https://the-odds-api.com). For pure model work you can leave it empty.

# 3. Initialise DB schema + views
python -m cricket_pipeline.pipeline views

# 4. Seed minimal data (~5 min — enough to train + predict)
python -m cricket_pipeline.pipeline cricsheet --dataset t20s_json --limit 200
python -m cricket_pipeline.pipeline cricsheet --dataset ipl_json   --limit 200

# 5. Verify everything wires up
python -m pytest cricket_pipeline/ -q

# 6. Train a baseline model
python -m cricket_pipeline.work.error_analysis_v2 --tag t20

# 7. Predict an upcoming match
python -m cricket_pipeline.work.predict_match \
  --home "Rajasthan Royals" --away "Sunrisers Hyderabad" \
  --venue "Sawai Mansingh Stadium, Jaipur" \
  --format T20 --date 2026-05-15 --fast
```

## Key entry points by task

| If you want to… | Start with |
|:---|:---|
| Understand the strategic plan | This file → then `cricket_pipeline/work/progress_log.md` |
| Make a model change | `cricket_pipeline/work/ensemble.py`, `predict_match.py` |
| Add a feature | `cricket_pipeline/work/features_v2.py` (run `audit_all` after) |
| Add an SQL view | `cricket_pipeline/db/views.sql` + verify with `audit_all` |
| Run a walk-forward backtest | `cricket_pipeline/work/walk_forward.py` |
| Search hyperparameters | `cricket_pipeline/work/odi_model.py` (uses disjoint meta-test) |
| Tune calibration | `cricket_pipeline/work/tier_calibration.py` |
| Run live tracking | `cricket_pipeline/work/live_match.py`, `live_tracker.py` |
| Place a bet (manual mode) | `cricket_pipeline/work/bet_engine.py --scan` |
| Operator workflow | `HOW_TO_USE.md` Playbooks A-F |

## What to look at before claiming a model improvement

The most important thing: **prior cycle metrics may be inflated by leakage**
(see `cricket_pipeline/work/walk_forward.py` docstring for why). Before
shipping a "+Xpp" claim:

1. Run the walk-forward harness on at least 3 quarterly windows.
2. Confirm `audit_all(df, strict=True)` passes on the feature frame.
3. For Optuna runs, the only honest number is `honest_test_logloss` (NOT
   `meta_test_logloss`) in `runs/odi_best_params.json`.
4. Calibration matters as much as accuracy — check ECE per tier × format.

## Where to find recent work

Most recent commits and their summaries:

```bash
git log --oneline -20
```

The `progress_log.md` and `learnings/SUMMARY.md` are kept current with
methodology decisions and cycle results. If you've been asked to work on
something and it's not obvious from these, ask the user — don't guess.

## What's intentionally left for follow-up

- SQL view rewrite for `v_batter_history` / `v_bowler_history` same-day
  double-header window (Agent 10) — needs schema migration + retrain on
  full ball history.
- CricSheet free-coverage backfill (Ranji, County, WBBL, WPL, MLC) —
  hours of compute, not engineering.
- Player ID resolution + alias graph (Agent 13) — depends on access to
  a curated alias seed set.

## Out-of-bounds for autonomous work

Without explicit authorization, do NOT:
- Place real bets (`BET_MODE=polymarket` or `betfair`). Manual / paper
  mode only.
- Rotate or rotate-then-publish API keys (the operator does this).
- Force-push to `main` or rewrite committed history.
- Touch `cricket.duckdb` directly (it's the source of truth — use the
  ingest pipeline).
