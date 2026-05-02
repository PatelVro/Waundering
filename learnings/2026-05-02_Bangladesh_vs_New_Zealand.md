# Post-Match Review: Bangladesh vs New Zealand

| | |
|---|---|
| **Date** | 2026-05-02 |
| **Venue** | Shere Bangla National Stadium, Dhaka |
| **Format** | T20 |
| **Result** | Bangladesh won by 6 wkts |

## Prediction vs Result

| Metric | Value |
|---|---|
| Predicted winner | New Zealand |
| Actual winner | Bangladesh |
| P(home=Bangladesh wins) | 24.6% |
| Model confidence | 50.8% (edge 50.8 pp) |
| **Correct?** | NO |

## Wrong Prediction Analysis

### Pattern Flags

> Model vs market: bookmakers gave the home team 30.0pp more probability than our model — and they were right. Our model may be miscalibrated on this matchup type.

> High-confidence all-model miss: every base learner agreed (0.33, 0.31, 0.42, 0.29, 0.25, 0.25) but the prediction was wrong — either a genuine upset or an untracked variable (lineup change, injury, pitch, weather).

### Actionable Learnings

- Bookmakers had 30.0pp edge over our model on the correct side. Review feature weights or recalibrate the stacked ensemble on recent T20 data.
- Flag for manual review: high-confidence all-model miss. Check for untracked pre-match signals: key player absent, unusual pitch or weather, or opposition quality in the runup.

### Retraining Flag

**Flag for model weight review:** YES  
**Confidence this is learnable:** HIGH

## Model Details

### Ensemble Components

| Base Model | P(home wins) |
|---|---|
| lgbm_num | 0.35 |
| lgbm_cat | 0.35 |
| xgb | 0.44 |
| cat | — |
| lr | 0.29 |
| ensemble_raw | 0.33 |
| ensemble | 0.33 |
| **ensemble** | **0.25** |

### Key Features

| Feature | Bangladesh (t1) | New Zealand (t2) |
|---|---|---|
| ELO rating | 2,767 | 2,846 |
| ELO diff (t1-t2) | -79.4 | — |
| Last-5 win% | 100% | 80% |
| Last-10 win% | 100% | 60% |
| H2H win% | 23% | — |
| H2H meetings | 1103.0 | — |
| Batting form SR | 69.4 | 62.1 |
| Bowl career econ | 5.17 | 5.98 |
| Venue toss win% | — | — |

### Model vs Bookmakers

| | Value |
|---|---|
| Books sampled | 33 |
| Market P(home) | 54.6% |
| Market P(away) | 45.4% |
| Model edge vs market (home) | -21.9pp |
| Model edge vs market (away) | +21.9pp |

### Projected Scores (Monte Carlo)

First innings: p10=115, median=158, p90=186
