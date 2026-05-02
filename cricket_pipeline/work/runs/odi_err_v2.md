# Error analysis · `odi` (formats: ODI)

_Generated against the **production stacked ensemble** (LGBM-num + LGBM-cat + XGB + CatBoost + LR + LR meta)._


## Headline

- **Test set:** 444 matches (time-based hold-out, latest slice)
- **Accuracy:** 68.69%
- **Brier:** 0.200   |   **Logloss:** 0.582   |   **ECE:** 5.35%
- **Upsets:** 34.2% of test (lower-Elo team won), model caught **36.2%** of them


## Per-base-learner

| learner | acc | brier | logloss |
|---|---|---|---|
| lgbm_num | 64.2% | 0.209 | 0.60 |
| lgbm_cat | 64.9% | 0.209 | 0.60 |
| xgb | 65.3% | 0.205 | 0.59 |
| cat | 66.4% | 0.203 | 0.59 |
| lr | 66.4% | 0.212 | 0.61 |
| ensemble | 68.7% | 0.200 | 0.58 |


## Per-tier — where the accuracy actually lives

Tier-1 = top-flight leagues (IPL, Big Bash, CPL, MLC, Hundred, SA20, ILT20, BPL, LPL, ICC men's WC + Champions Trophy) and bilateral series between full Test nations.
Tier-2 = women's competitions, qualifier tournaments, ICC CWC League 2, associate-nation tri-series, etc. — these often dominate the test set but are not the matches you bet on day-to-day.

| tier | n | acc | brier | ece |
|---|---|---|---|---|
| tier1 | 163.00 | 66.9% | 0.213 | 0.091 |
| tier2_assoc | 169.00 | 76.3% | 0.166 | 0.064 |
| tier2_main | 25.00 | 84.0% | 0.130 | 0.186 |
| tier2_other | 87.00 | 52.9% | 0.260 | 0.137 |


## Reliability — Tier 1 only

Confidence × accuracy. Well-calibrated would have `acc ≈ avg_p` for each row.

| conf_bin | n | acc | avg_p |
|---|---|---|---|
| 0-20% | 46 | 54.3% | 0.55 |
| 20-40% | 40 | 65.0% | 0.65 |
| 40-60% | 49 | 69.4% | 0.75 |
| 60-80% | 27 | 85.2% | 0.85 |
| 80-100% | 1 | 100.0% | 0.91 |


## Reliability — All tiers

| conf_bin | n | acc | avg_p |
|---|---|---|---|
| 0-20% | 122 | 54.9% | 0.55 |
| 20-40% | 119 | 60.5% | 0.65 |
| 40-60% | 114 | 75.4% | 0.75 |
| 60-80% | 82 | 89.0% | 0.84 |
| 80-100% | 7 | 100.0% | 0.91 |


## Tier-1 accuracy by competition (n ≥ 15)

| competition | n | acc | brier | avg_conf |
|---|---|---|---|---|
| ICC Cricket World Cup | 39 | 76.9% | 0.149 | 44.0% |


## Tier-1 accuracy by year

| year | n | acc | brier | avg_conf |
|---|---|---|---|---|
| 2023.00 | 50.00 | 74.0% | 0.173 | 44.8% |
| 2024.00 | 38.00 | 65.8% | 0.238 | 30.3% |
| 2025.00 | 63.00 | 63.5% | 0.218 | 36.3% |


## Tier-1 accuracy by Elo gap

Bigger Elo gap → easier prediction. If accuracy is flat across gaps, the model isn't learning team strength well.

| elo_gap | n | acc | avg_conf |
|---|---|---|---|
| close (≤30) | 19 | 57.9% | 29.8% |
| moderate (30-80) | 24 | 79.2% | 26.1% |
| wide (80-150) | 35 | 71.4% | 30.4% |
| very wide (150+) | 85 | 63.5% | 44.4% |


## Top 20 high-confidence misses (Tier 1 if available)

| start_date | competition | team_home | team_away | venue | winner | pred_p_t1 | elo_diff_pre | h2h_t1_winpct |
|---|---|---|---|---|---|---|---|---|
| 2023-10-17 | ICC Cricket World Cup | Netherlands | South Africa | Himachal Pradesh Cricket Association Stadium, Dharamsala | Netherlands | 0.12 | -536.00 | 0.00 |
| 2024-08-04 | India tour of Sri Lanka | Sri Lanka | India | R Premadasa Stadium, Colombo | Sri Lanka | 0.13 | -259.00 | 0.27 |
| 2024-08-07 | India tour of Sri Lanka | Sri Lanka | India | R Premadasa Stadium, Colombo | Sri Lanka | 0.15 | -202.00 | 0.28 |
| 2023-12-09 | England tour of West Indies | England | West Indies | Kensington Oval, Bridgetown, Barbados | West Indies | 0.80 | 218.00 | 0.71 |
| 2023-12-03 | England tour of West Indies | England | West Indies | Sir Vivian Richards Stadium, North Sound, Antigua | West Indies | 0.80 | 250.00 | 0.72 |
| 2025-12-03 | South Africa tour of India | India | South Africa | Shaheed Veer Narayan Singh International Stadium, Raipur | South Africa | 0.77 | 191.00 | 0.56 |
| 2023-12-23 | Bangladesh tour of New Zealand | New Zealand | Bangladesh | McLean Park, Napier | Bangladesh | 0.77 | 192.00 | 0.76 |
| 2024-10-07 | Ireland vs South Africa | Ireland | South Africa | Zayed Cricket Stadium, Abu Dhabi | Ireland | 0.24 | -278.00 | 0.12 |
| 2025-01-11 | Sri Lanka tour of New Zealand | Sri Lanka | New Zealand | Eden Park, Auckland | Sri Lanka | 0.25 | -9.00 | 0.42 |
| 2026-03-11 | Pakistan tour of Bangladesh | Pakistan | Bangladesh | Shere Bangla National Stadium, Mirpur | Bangladesh | 0.75 | 67.00 | 0.76 |
| 2026-04-20 | New Zealand tour of Bangladesh | New Zealand | Bangladesh | Shere Bangla National Stadium, Mirpur | Bangladesh | 0.74 | 267.00 | 0.76 |
| 2025-08-10 | Pakistan tour of West Indies | Pakistan | West Indies | Brian Lara Stadium, Tarouba, Trinidad | West Indies | 0.74 | 153.00 | 0.60 |
| 2025-10-29 | England tour of New Zealand | England | New Zealand | Seddon Park, Hamilton | New Zealand | 0.73 | 16.00 | 0.59 |
| 2025-02-18 | Ireland tour of Zimbabwe | Ireland | Zimbabwe | Harare Sports Club | Zimbabwe | 0.73 | 84.00 | 0.65 |
| 2023-11-06 | ICC Cricket World Cup | Sri Lanka | Bangladesh | Arun Jaitley Stadium, Delhi | Bangladesh | 0.72 | 149.00 | 0.78 |
| 2025-10-18 | West Indies tour of Bangladesh | Bangladesh | West Indies | Shere Bangla National Stadium, Mirpur | Bangladesh | 0.28 | -12.00 | 0.49 |
| 2025-02-14 | Australia tour of Sri Lanka | Sri Lanka | Australia | R Premadasa Stadium, Colombo | Sri Lanka | 0.28 | -204.00 | 0.34 |
| 2024-11-24 | Pakistan tour of Zimbabwe | Zimbabwe | Pakistan | Queens Sports Club, Bulawayo | Zimbabwe | 0.29 | -238.00 | 0.06 |
| 2026-04-23 | New Zealand tour of Bangladesh | Bangladesh | New Zealand | Bir Sreshtho Flight Lieutenant Matiur Rahman Stadium, Chattogram | Bangladesh | 0.30 | -217.00 | 0.26 |
| 2025-08-12 | Pakistan tour of West Indies | West Indies | Pakistan | Brian Lara Stadium, Tarouba, Trinidad | West Indies | 0.31 | -111.00 | 0.42 |


## Files
- Per-row CSV: `odi_err_v2.csv`
- This report:  `odi_err_v2.md`
