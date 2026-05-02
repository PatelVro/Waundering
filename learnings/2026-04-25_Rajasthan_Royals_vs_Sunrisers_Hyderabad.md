# Post-Match Review: Rajasthan Royals vs Sunrisers Hyderabad

| | |
|---|---|
| **Date** | 2026-04-25 |
| **Venue** | Sawai Mansingh Stadium, Jaipur |
| **Format** | T20 |
| **Result** | Sunrisers Hyderabad won by 5 wkts |

## Prediction vs Result

| Metric | Value |
|---|---|
| Predicted winner | Sunrisers Hyderabad |
| Actual winner | Sunrisers Hyderabad |
| P(home=Rajasthan Royals wins) | 33.3% |
| Model confidence | 33.4% (edge 33.4 pp) |
| **Correct?** | YES |

## Outcome: Correct Prediction

The model correctly favoured **Sunrisers Hyderabad** with 33.4pp edge.

**Why the model was right:**

- Form: Sunrisers Hyderabad showed better recent batting SR (136.7 vs 110.1).
- Consensus: all base models agreed (range 0.24) — ensemble confidence was genuine.

## Model Details

### Ensemble Components

| Base Model | P(home wins) |
|---|---|
| lgbm_num | 0.50 |
| lgbm_cat | 0.26 |
| xgb | 0.49 |
| cat | 0.42 |
| lr | 0.47 |
| ensemble | 0.33 |
| **ensemble** | **0.33** |

### Key Features

| Feature | Rajasthan Royals (t1) | Sunrisers Hyderabad (t2) |
|---|---|---|
| ELO rating | 1,540 | 1,567 |
| ELO diff (t1-t2) | -26.4 | — |
| Last-5 win% | 60% | 60% |
| Last-10 win% | 60% | 70% |
| H2H win% | 41% | — |
| H2H meetings | 22.0 | — |
| Batting form SR | 110.1 | 136.7 |
| Bowl career econ | 6.42 | 7.07 |
| Venue toss win% | 67% | — |

### Projected Scores (Monte Carlo)

First innings: p10=158, median=197, p90=224
