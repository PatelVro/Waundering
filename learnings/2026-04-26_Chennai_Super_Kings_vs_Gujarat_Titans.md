# Post-Match Review: Chennai Super Kings vs Gujarat Titans

| | |
|---|---|
| **Date** | 2026-04-26 |
| **Venue** | MA Chidambaram Stadium, Chennai |
| **Format** | T20 |
| **Result** | Gujarat Titans won by 8 wkts |

## Prediction vs Result

| Metric | Value |
|---|---|
| Predicted winner | Gujarat Titans |
| Actual winner | Gujarat Titans |
| P(home=Chennai Super Kings wins) | 47.2% |
| Model confidence | 5.6% (edge 5.6 pp) |
| **Correct?** | YES |

## Outcome: Correct Prediction

The model correctly favoured **Gujarat Titans** with 5.6pp edge.

**Why the model was right:**

- Form: Chennai Super Kings showed better recent batting SR (146.0 vs 134.3).
- Consensus: all base models agreed (range 0.16) — ensemble confidence was genuine.

## Model Details

### Ensemble Components

| Base Model | P(home wins) |
|---|---|
| lgbm_num | 0.49 |
| lgbm_cat | 0.40 |
| xgb | 0.50 |
| cat | 0.50 |
| lr | 0.56 |
| ensemble | 0.47 |
| **ensemble** | **0.47** |

### Key Features

| Feature | Chennai Super Kings (t1) | Gujarat Titans (t2) |
|---|---|---|
| ELO rating | 1,523 | 1,503 |
| ELO diff (t1-t2) | +19.5 | — |
| Last-5 win% | 60% | 60% |
| Last-10 win% | 60% | 60% |
| H2H win% | 50% | — |
| H2H meetings | 16.0 | — |
| Batting form SR | 146.0 | 134.3 |
| Bowl career econ | 7.69 | 6.11 |
| Venue toss win% | — | — |

### Projected Scores (Monte Carlo)

First innings: p10=147, median=179, p90=218
