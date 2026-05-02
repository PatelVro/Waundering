# Post-Match Review: Bangladesh vs New Zealand

| | |
|---|---|
| **Date** | 2026-04-27 |
| **Venue** | Bir Sreshtho Flight Lieutenant Matiur Rahman Stadium, Chattogram |
| **Format** | T20 |
| **Result** | Bangladesh won by 6 wkts |

## Prediction vs Result

| Metric | Value |
|---|---|
| Predicted winner | Bangladesh |
| Actual winner | Bangladesh |
| P(home=Bangladesh wins) | 51.0% |
| Model confidence | 1.9% (edge 1.9 pp) |
| **Correct?** | YES |

## Outcome: Correct Prediction

The model correctly favoured **Bangladesh** with 1.9pp edge.

**Why the model was right:**

- ELO: New Zealand had a 135-point rating advantage (t1=1782, t2=1918).
- H2H: New Zealand leads the head-to-head (22% home win rate over 82 meetings).
- Consensus: all base models agreed (range 0.24) — ensemble confidence was genuine.

## Model Details

### Ensemble Components

| Base Model | P(home wins) |
|---|---|
| lgbm_num | 0.62 |
| lgbm_cat | 0.45 |
| xgb | 0.66 |
| cat | — |
| lr | 0.42 |
| ensemble | 0.51 |
| **ensemble** | **0.51** |

### Key Features

| Feature | Bangladesh (t1) | New Zealand (t2) |
|---|---|---|
| ELO rating | 1,782 | 1,918 |
| ELO diff (t1-t2) | -135.1 | — |
| Last-5 win% | 100% | 60% |
| Last-10 win% | 90% | 70% |
| H2H win% | 22% | — |
| H2H meetings | 82.0 | — |
| Batting form SR | 95.8 | 97.7 |
| Bowl career econ | 5.72 | 6.98 |
| Venue toss win% | 33% | — |

### Model vs Bookmakers

| | Value |
|---|---|
| Books sampled | 33 |
| Market P(home) | 54.6% |
| Market P(away) | 45.4% |
| Model edge vs market (home) | -3.7pp |
| Model edge vs market (away) | +3.7pp |

### Projected Scores (Monte Carlo)

First innings: p10=128, median=161, p90=195
