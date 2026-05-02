# Post-Match Review: Bangladesh vs New Zealand

| | |
|---|---|
| **Date** | 2026-04-28 |
| **Venue** | Bir Sreshtho Flight Lieutenant Matiur Rahman Stadium, Chattogram |
| **Format** | T20 |
| **Result** | Bangladesh won by 6 wkts |

## Prediction vs Result

| Metric | Value |
|---|---|
| Predicted winner | New Zealand |
| Actual winner | Bangladesh |
| P(home=Bangladesh wins) | 29.2% |
| Model confidence | 41.5% (edge 41.5 pp) |
| **Correct?** | NO |

## Wrong Prediction Analysis

### Pattern Flags

> ELO upset: New Zealand had a 151-point ELO advantage but lost. The model's ELO signal strongly favoured the wrong side — check if ELO is over-weighted relative to form.

> Model vs market: bookmakers gave the home team 25.4pp more probability than our model — and they were right. Our model may be miscalibrated on this matchup type.

> High-confidence all-model miss: every base learner agreed (0.43, 0.33, 0.39, 0.40, 0.29) but the prediction was wrong — either a genuine upset or an untracked variable (lineup change, injury, pitch, weather).

### Actionable Learnings

- Review ELO weighting for Bir Sreshtho Flight Lieutenant Matiur Rahman Stadium, Chattogram. A 151-pt gap predicted confidently but failed.
- Bookmakers had 25.4pp edge over our model on the correct side. Review feature weights or recalibrate the stacked ensemble on recent T20 data.
- Flag for manual review: high-confidence all-model miss. Check for untracked pre-match signals: key player absent, unusual pitch or weather, or opposition quality in the runup.

### Retraining Flag

**Flag for model weight review:** YES  
**Confidence this is learnable:** MEDIUM

## Model Details

### Ensemble Components

| Base Model | P(home wins) |
|---|---|
| lgbm_num | 0.43 |
| lgbm_cat | 0.33 |
| xgb | 0.39 |
| cat | — |
| lr | 0.40 |
| ensemble | 0.29 |
| **ensemble** | **0.29** |

### Key Features

| Feature | Bangladesh (t1) | New Zealand (t2) |
|---|---|---|
| ELO rating | 2,307 | 2,457 |
| ELO diff (t1-t2) | -150.5 | — |
| Last-5 win% | 100% | 60% |
| Last-10 win% | 100% | 50% |
| H2H win% | 23% | — |
| H2H meetings | 394.0 | — |
| Batting form SR | 84.2 | 67.8 |
| Bowl career econ | 5.26 | 6.26 |
| Venue toss win% | 33% | — |

### Model vs Bookmakers

| | Value |
|---|---|
| Books sampled | 33 |
| Market P(home) | 54.6% |
| Market P(away) | 45.4% |
| Model edge vs market (home) | -25.4pp |
| Model edge vs market (away) | +25.4pp |

### Projected Scores (Monte Carlo)

First innings: p10=115, median=153, p90=189
