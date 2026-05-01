# Cricket Prediction Terminal — Decision Guide

**Goal of this document:** show you how to read every number on the dashboard
and turn the right ones into money. Sections are ordered **by usefulness for
betting decisions** — start at the top.

> Quick context: you're in Canada → bet engine is in **manual mode**. The
> system generates signals; you place them on BetMGM / FanDuel / theScore /
> Bet99 / etc. Defaults are sized for a $1,000 bankroll (override via `BANKROLL`
> in `.env`).

---

## TL;DR — the only 4 things that matter day-to-day

1. **Open tickets** — your action queue. Every line is a pre-sized recommended bet.
2. **Model-vs-market edge** — where the bet thesis lives. ≥3pp + ≤8pp = act. Outside that = ignore.
3. **Live in-match win-probability** — for in-play bets after the first 5 overs.
4. **Bet ledger PnL / ROI** — the truth that grades the rest.

Everything else is context to help you decide whether to override the engine.

---

## Tier 1 — Decision-driving sections

### 1. Open tickets (`#bets` → "Open tickets")

**What it shows:** every recommended bet the engine has flagged but you
haven't placed yet. Each card carries the eight things you need to act.

| Field | What it means | How to use it |
|---|---|---|
| **Pick** | The team the model rates better-than-priced | Bet this side, not the model's outright favorite |
| **Odds** | Highest decimal price across 28-33 bookmakers | Use this as your *minimum* line — don't take less |
| **Stake** | Capped at half-Kelly *and* 5% of bankroll | Treat as a hard ceiling. Sizing down is fine; up is not |
| **Edge** | Model probability minus book consensus, in percentage points | Your expected long-run advantage per bet |
| **Kelly %** | Fractional Kelly stake as % of bankroll | The *math-optimal* size; capped at 5% so blowups don't ruin you |

**Decision rule:**
- Edge 3-5pp → place at full recommended stake.
- Edge 5-8pp → place, but consider sizing down 30-50% (longer tail of false positives).
- Edge >8pp → ignored automatically (`BET_EDGE_MAX_PP=8`). If you really want to bet, do it manually with a small stake — these are usually model errors, not free money.
- Edge <3pp → engine doesn't ticket it. Fine to fade.

**After placing on the bookie:**
```
.venv/Scripts/python.exe -m cricket_pipeline.work.bet_engine --mark <id8> --status placed
```
After the match settles:
```
.venv/Scripts/python.exe -m cricket_pipeline.work.bet_engine --mark <id8> --status won  # or lost
```

---

### 2. Model-vs-market edge box (`#vsbook` + per-card "VALUE" badge)

**What it shows:** for each match where odds are flowing, our model's
de-vigged probability vs. the consensus of all 28-33 bookmakers, with the
disagreement (`edge_pp`) as the headline number.

**Reading the edge:**

| Edge magnitude | Read |
|---|---|
| **0-3pp** | Model agrees with market. No bet — vig eats your tiny edge. |
| **3-8pp** | **Sweet spot.** Real disagreements are usually small. This is where edges that hold up over many bets live. |
| **8-15pp** | Suspicious. Could be a real injury / lineup signal we're missing. Investigate before betting. |
| **15+pp** | Almost certainly a model blind spot — stale lineup, missed news, wrong pitch read. Auto-blocked. |

**Why books are usually right:** when 30+ books all converge on a price,
they've collectively absorbed every public signal — injuries, dew, pitch,
weather, captain's last-3 toss decisions, you name it. Beating them
consistently by >5pp is extraordinarily hard.

---

### 3. Live in-match win-probability (`#live` banner)

**What it shows:** when a tracked match is in progress, the live state from
Cricbuzz (score, overs, current run-rate, required run-rate, striker /
bowler) plus our model's projected end-of-innings score for the batting
team.

| Field | What it means |
|---|---|
| **score** | live score / wickets |
| **overs** | overs bowled in current innings |
| **current_rr** | runs/over so far |
| **required_rr** | only meaningful in 2nd-innings chases — runs/over needed to win |
| **target** | target the chasing side needs |
| **rem_balls / rem_runs** | balls + runs remaining in chase |
| **live_prediction.p10/p50/p90** | 1st-innings projected final score (Monte-Carlo, 5,000 sims) |

**Decision use cases for live data → see playbook #4 below.**

---

### 4. Bet ledger / PnL (`#bets` summary cards)

**What it shows:** running tally of every settled bet. **This is the only
honest measure of whether the engine is making money.**

| Card | Meaning | Decision use |
|---|---|---|
| **Mode** | paper / manual / polymarket | Paper PnL is hypothetical; manual PnL is real |
| **PnL** | profit/loss in stake currency | The number that matters |
| **ROI** | PnL ÷ total staked | True profitability per dollar risked |
| **W-L-pending** | settled-bet record | Sample size for trusting the system |

**Trust thresholds:**
- < 30 settled bets → **noisy**, don't conclude anything
- 30-100 bets, ROI > 0 → **encouraging**
- 100+ bets, ROI > 3% → **system is real**
- 100+ bets, ROI < 0 → **stop, debug, don't keep feeding it**

---

## Tier 2 — Context sections (read these before betting)

### 5. Headline prediction (the big number on each match card)

**What it shows:** stacked-ensemble probability of each team winning + the
favored side. This is the "what does the model think" answer, before any
market comparison.

**How to use it:**
- **Always cross-reference with the market consensus.** A 60% model pick
  means very different things if the market also has them at 60% (no edge)
  vs. 40% (huge disagreement → check why).
- **Confidence label** is just a UI hint:
  - *toss-up* (edge < 20pp) — both sides realistic
  - *lean* (20-40pp) — model has a clear preference
  - *favorite* (40-60pp) — strong call
  - *strong favorite* (>60pp) — model is highly confident; usually means a big talent/Elo gap

### 6. Per-base-learner breakdown (`Base learners` block on each card)

**What it shows:** the 5 individual model predictions before stacking.

| Learner | What it brings |
|---|---|
| **lgbm_num** | LightGBM on numeric features only (Elo, form, h2h) |
| **lgbm_cat** | LightGBM with categorical IDs (team/venue) |
| **xgb** | XGBoost — different bias, similar features |
| **cat** | CatBoost — only in non-`--fast` mode |
| **lr** | Logistic regression — sanity-check baseline |
| **ensemble** | Final stacked output (LR meta-learner on calibration set) |

**Decision use:**
- **Disagreement among base learners is a yellow flag.** If LGBM says 65%
  and LR says 35%, the ensemble is averaging confusion.
- **Tight cluster** (all within 5pp of each other) = robust signal.
- **Wide spread** = treat the ensemble probability with skepticism, even if
  it shows a "value bet."

### 7. Bookmaker consensus + dot plot

**What it shows:** every bookmaker's de-vigged implied probability on the
home team, plotted as dots, with our model marked separately.

**How to use it:**
- **Tight book cluster** (all dots within 3pp) = sharp market, hard to beat.
- **Wide spread** (8+pp between min and max) = soft market, more chance for
  a real edge, but also less reliable consensus.
- **Vig %** (overround) on the table — lower = friendlier. Pinnacle/Betfair
  ≈ 2-3%, recreational books ≈ 6-10%.

### 8. Totals (over/under) — 1st-innings score curve

**What it shows:** Monte-Carlo distribution of the team-batting-first's
final score, with P10 / P50 / P90 + an over/under ladder at five lines
around the median.

**Reading the curves:**
- **Spread** between P10 and P90 ≈ how much volatility we expect. Tight
  curve (50-run spread) = confident. Wide (80-run spread) = high variance
  match.
- **Two scenarios** plotted (each side batting first) — toss matters here.

**Decision rule:** when bookmaker totals odds are available *and* book line
is >10 runs from our P50 in either direction, that's a value bet.

> Bet engine doesn't yet auto-place totals bets — see Phase 3d. Use the
> manual workflow for now.

### 9. Top-scorer ladders — per team

**What it shows:** for each side's announced XI, the model's probability
that this player is the team's top scorer.

**How to use it:**
- **Random pick from XI** ≈ 9% (1/11). Our model averages **24.5%** on the
  actual top-scorer pre-match. So picks at 14-18% are real signal.
- **Best opportunities:** when a player ranks top-1 in our list at 14-18%
  but the bookie has them at 7+ odds (book implied <14%), there's edge.
- **Caveat:** lineup proxy uses each team's *most recent prior XI*. If a
  star is rested or returning, our list will lag actual today's XI.

---

## Tier 3 — Background context (occasional reference)

### 10. Calibration metrics (`#performance` cards)

| Metric | Meaning | What's good |
|---|---|---|
| **Acc** | Top-1 accuracy on held-out matches | T20 75.8% · ODI 67.8% · all 74% |
| **AUC** | Ranking quality (1.0 = perfect, 0.5 = coin flip) | T20 0.84 — strong |
| **Brier** | Mean squared error on probability | Lower = better. T20 0.165 |
| **ECE** | Expected calibration error (avg \|predicted − observed\|) | T20 3.16% — well calibrated |

**Decision use:** these are the **prior** for trusting model probabilities.
If T20 ECE = 3% then "model says 70%" should be right ~67-73% of the time.
If you ever see ECE > 8% on a format, scale down stakes on that format.

### 11. Elo league tables (`#teams`)

**What it shows:** per-format chronological Elo ratings for top teams.

**Decision use:** the single best one-number proxy for team strength.
- **Difference of 50 Elo** ≈ 7pp probability shift
- **100 Elo** ≈ 14pp shift
- Mostly used as a sanity-check on the model: if the model's pick disagrees
  with raw Elo by more than the per-team form/h2h adjustments would
  explain, dig in before staking.

### 12. Recent matches (`#recent`)

Last 30 matches the engine has ingested. Use to:
- Sanity-check that the corpus is current (newest match should be in the last 24-48h).
- Spot streaks/upsets that future predictions should account for.

### 13. Data inventory (`#data`)

13,364 matches · 5.86M balls · 297K player XIs across 649 venues. Just
a corpus snapshot — useful only when comparing against new data ingests.

---

## Money-making playbooks

### Playbook A — Pre-match h2h value bets (the bread-and-butter)

**When:** every day, ~30 min before the toss of any tracked match.

**Trigger:** an open ticket appears with edge in the **3-8pp** range.

**Action:**
1. Open the ticket card → confirm the **best_odds** field.
2. Open your bookmaker → check current price for that side.
3. **Only place if your bookie's price ≥ best_odds × 0.97** (3% slippage tolerance).
4. Stake: the recommended `stake` field, or scale down to 50-70% if you're conservative.
5. Mark `--status placed` immediately.

**Wait conditions:**
- ⏳ **Wait for the announced XI** if the engine hasn't seen a recent prior match for either team (e.g. season opener) — our lineup proxy will be stale.
- ⏳ **Don't bet >2 hours before toss** for night games in subcontinent venues. Toss decision + dew can swing book lines 5-8pp; better to let the line settle first.

**Expected outcome:** with edges in 3-8pp range, expected win-rate slightly
above implied probability. Over 50+ such bets, ROI of +2 to +5% is
realistic if the model holds up.

---

### Playbook B — Fade the over-confident model picks

**When:** any time you see a card with edge 12-25pp.

**Read:** these get auto-blocked from tickets, but they're interesting
*signal* about market vs. model disagreement.

**Action — DON'T just bet the model.** Instead:
1. Check news / Cricinfo for the announced XI of both teams (typically
   posted 30 min before toss).
2. Check pitch report (if available) and weather.
3. **If you find the missing info** (e.g., a star player rested, dew expected) →
   the book is right; either fade the model side at moderate stake, or pass.
4. **If you genuinely can't find what the market knows** → trust the books
   over us. Pass.

**Wait conditions:**
- ⏳ Always wait for confirmed XI before acting on these.

---

### Playbook C — Totals (over/under) plays

**When:** after a match's odds are flowing, check the totals lines.

**Trigger:** book's main line (typically the median, e.g. "175.5") differs
from our `first_innings_p50` by **>10 runs** in the same direction.

| Our P50 vs. book line | Bet |
|---|---|
| Our P50 = 195, book line 175.5 | **Over** at -110 |
| Our P50 = 155, book line 175.5 | **Under** at -110 |
| Within 10 runs | Pass |

**Edge sanity check:** read the over/under ladder on the prediction card.
At the book's line, look at our `p_over`. If `p_over > 0.55` for an Over
bet at -110 (implied 52.4%), edge ≈ 3pp — bet small. If `p_over > 0.60`,
edge ≈ 8pp — confident bet (but cap stake).

**Wait conditions:**
- ⏳ **Wait for the toss** — the bat-first scenario shifts the median by
  10-20 runs depending on venue. Use the matching scenario, not the average.
- ⏳ **Don't bet totals on Tests** — pitch deterioration over 5 days is
  outside our model's scope; book lines factor that in.

---

### Playbook D — Live in-play (chase only, after over 5)

**When:** any tracked match in 2nd innings, after the first 5 overs.

**Setup:** pull `current_rr` and `required_rr` from the live banner.

| Required vs. current | Read | Suggested action |
|---|---|---|
| RRR < CRR by 1+ runs | Chase comfortable | Take the chasing side at any +odds |
| RRR within ±0.5 of CRR | True coin flip | Pass — book has it priced |
| RRR > CRR by 1.5-3 runs | Pressure mounting | Layered: bet bowling side small as RRR climbs |
| RRR > CRR by >3 runs after 12 overs | Probably over | Bet bowling side moderate |

**Wait conditions:**
- ⏳ **Wait until over 5** — too much variance before powerplay ends.
- ⏳ **Don't live-bet T20s where the chasing side is >7 wickets in hand
  after 15 overs** — they've usually accelerated by then.
- ⏳ **Stop trading after over 18** — line moves too fast, slippage kills any edge.

---

### Playbook E — Top-scorer at long odds

**When:** book has top-batsman markets up (BetMGM, theScore — often 2h before toss).

**Trigger:** the model's #1-ranked player on a team has prob ≥14% but the
bookmaker price is **8.0 or longer** (book implied ≤12.5%).

**Action:**
- Stake small — top-scorer markets are high-variance. Cap at 1-2% of bankroll.
- Best practice: bet the model's #1 *and* #2 on the same team if both
  qualify — diversifies the variance.

**Wait conditions:**
- ⏳ **Wait for confirmed XI.** This market is 100% lineup-dependent. If
  the announced XI differs from our proxy, re-pull the prediction.
- ⏳ **Skip Tests** — too long; injuries / declarations dominate.

---

### Playbook F — Calibration arbitrage (longer-term)

**When:** the model-vs-book Brier comparison harness (`#vsbook`) has 30+
settled samples.

**Read:**
- **Model Brier < Book Brier** by 0.005+ → our model is sharper than the
  market. Run the value-bet engine more aggressively (drop edge threshold
  to 2pp).
- **Model Brier ≈ Book Brier** → comparable. Stick with default 3pp threshold.
- **Model Brier > Book Brier** → market is sharper. Raise threshold to 5pp,
  bet only the strongest disagreements, mostly fade the model.

**Wait conditions:**
- ⏳ **Don't act on this signal until n ≥ 30**. Below that, the comparison is
  pure noise.

---

## Quick wait-condition reference

| Situation | Wait until |
|---|---|
| Season opener / new tournament | At least 5 matches per team under our belt (form features need data) |
| Night T20 in subcontinent | 30 min before toss (let dew/toss factor priced in) |
| Any match with a 16+ pp edge | You've manually verified injury/lineup news |
| Live in-play | Over 5 of 2nd innings has been bowled |
| Trusting calibration claims | 30+ settled bets in the ledger |
| Acting on top-scorer market | Confirmed XI is published |
| Letting paper-mode "validate" the system | 100+ settled bets |

---

## Setup reference

```bash
# Enable odds (REQUIRED — no odds = no value bets)
echo "THE_ODDS_API_KEY=your_key_here" > .env

# Optional bet engine config
echo "BET_MODE=manual"               >> .env
echo "BANKROLL=1000.0"               >> .env
echo "BET_EDGE_THRESHOLD_PP=3.0"     >> .env
echo "BET_EDGE_MAX_PP=8.0"           >> .env
echo "BET_KELLY_CAP=0.5"             >> .env
echo "BET_MAX_STAKE_PCT=0.05"        >> .env

# Start (or restart) the orchestrator
.venv/Scripts/python.exe -m cricket_pipeline.work.start_orchestrator
```

Dashboard: **http://127.0.0.1:4173**.

CLI cheatsheet:
```bash
# show pending tickets
.venv/Scripts/python.exe -m cricket_pipeline.work.bet_engine --tickets

# mark a placed/won/lost
.venv/Scripts/python.exe -m cricket_pipeline.work.bet_engine --mark <id8> --status placed
.venv/Scripts/python.exe -m cricket_pipeline.work.bet_engine --mark <id8> --status won
.venv/Scripts/python.exe -m cricket_pipeline.work.bet_engine --mark <id8> --status lost

# ledger summary
.venv/Scripts/python.exe -m cricket_pipeline.work.bet_engine --summary

# refresh the model-vs-book comparison after matches settle
.venv/Scripts/python.exe -m cricket_pipeline.work.compare_to_books
```

---

## Honest expectations

- **Most days will have no value bets.** That's fine. Discipline is the edge.
- **No single bet ROI matters.** Variance is brutal in cricket. Judge after 100 bets.
- **The model has zero special information about the future.** It only knows
  what's in the data — XIs, form, venue, Elo. If something happens off-data
  (injury announced an hour before toss, captain change), the books know
  before we do.
- **3-5% ROI over a long sample is a great result.** Anyone promising more
  is selling something.

If after 100 bets you're flat or down, the system isn't working — debug it
or stop. If you're up 3%+, keep going and consider raising bankroll
gradually.
