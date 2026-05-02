# Error analysis · `t20` (formats: T20,IT20)

_Generated against the **production stacked ensemble** (LGBM-num + LGBM-cat + XGB + CatBoost + LR + LR meta)._


## Headline

- **Test set:** 1,562 matches (time-based hold-out, latest slice)
- **Accuracy:** 76.18%
- **Brier:** 0.164   |   **Logloss:** 0.496   |   **ECE:** 2.61%
- **Upsets:** 34.1% of test (lower-Elo team won), model caught **55.6%** of them


## Per-base-learner

| learner | acc | brier | logloss |
|---|---|---|---|
| lgbm_num | 74.3% | 0.166 | 0.50 |
| lgbm_cat | 74.2% | 0.176 | 0.53 |
| xgb | 75.1% | 0.168 | 0.50 |
| cat | 75.2% | 0.168 | 0.51 |
| lr | 73.9% | 0.178 | 0.53 |
| ensemble | 76.2% | 0.164 | 0.50 |


## Per-tier — where the accuracy actually lives

Tier-1 = top-flight leagues (IPL, Big Bash, CPL, MLC, Hundred, SA20, ILT20, BPL, LPL, ICC men's WC + Champions Trophy) and bilateral series between full Test nations.
Tier-2 = women's competitions, qualifier tournaments, ICC CWC League 2, associate-nation tri-series, etc. — these often dominate the test set but are not the matches you bet on day-to-day.

| tier | n | acc | brier | ece |
|---|---|---|---|---|
| tier1 | 468.00 | 77.6% | 0.157 | 0.060 |
| tier2_assoc | 708.00 | 76.7% | 0.159 | 0.038 |
| tier2_main | 22.00 | 63.6% | 0.267 | 0.283 |
| tier2_other | 364.00 | 74.2% | 0.176 | 0.056 |


## Reliability — Tier 1 only

Confidence × accuracy. Well-calibrated would have `acc ≈ avg_p` for each row.

| conf_bin | n | acc | avg_p |
|---|---|---|---|
| 0-20% | 100 | 60.0% | 0.55 |
| 20-40% | 95 | 69.5% | 0.65 |
| 40-60% | 97 | 72.2% | 0.75 |
| 60-80% | 135 | 94.1% | 0.85 |
| 80-100% | 41 | 97.6% | 0.92 |


## Reliability — All tiers

| conf_bin | n | acc | avg_p |
|---|---|---|---|
| 0-20% | 285 | 58.2% | 0.55 |
| 20-40% | 285 | 63.9% | 0.65 |
| 40-60% | 341 | 74.8% | 0.75 |
| 60-80% | 410 | 86.8% | 0.85 |
| 80-100% | 241 | 95.9% | 0.92 |


## Tier-1 accuracy by competition (n ≥ 15)

| competition | n | acc | brier | avg_conf |
|---|---|---|---|---|
| ICC Men's T20 World Cup | 49 | 83.7% | 0.124 | 50.1% |
| Indian Premier League | 186 | 79.6% | 0.127 | 49.2% |
| Bangladesh Premier League | 33 | 75.8% | 0.199 | 45.1% |
| Big Bash League | 43 | 69.8% | 0.203 | 37.5% |
| Major League Cricket | 33 | 69.7% | 0.229 | 37.3% |
| Caribbean Premier League | 32 | 62.5% | 0.240 | 47.1% |


## Tier-1 accuracy by year

| year | n | acc | brier | avg_conf |
|---|---|---|---|---|
| 2026.00 | 196.00 | 81.1% | 0.140 | 47.8% |
| 2025.00 | 272.00 | 75.0% | 0.169 | 44.4% |


## Tier-1 accuracy by Elo gap

Bigger Elo gap → easier prediction. If accuracy is flat across gaps, the model isn't learning team strength well.

| elo_gap | n | acc | avg_conf |
|---|---|---|---|
| close (≤30) | 102 | 73.5% | 41.4% |
| moderate (30-80) | 106 | 87.7% | 48.3% |
| wide (80-150) | 141 | 73.8% | 47.8% |
| very wide (150+) | 119 | 76.5% | 45.1% |


## Top 20 high-confidence misses (Tier 1 if available)

| start_date | competition | team_home | team_away | venue | winner | pred_p_t1 | elo_diff_pre | h2h_t1_winpct |
|---|---|---|---|---|---|---|---|---|
| 2026-01-09 | Bangladesh Premier League | Noakhali Express | Rangpur Riders | Sylhet International Cricket Stadium | Noakhali Express | 0.09 | -238.00 | nan |
| 2025-12-19 | Big Bash League | Perth Scorchers | Brisbane Heat | Brisbane Cricket Ground, Woolloongabba, Brisbane | Brisbane Heat | 0.86 | 151.00 | 0.58 |
| 2026-01-18 | Bangladesh Premier League | Dhaka Capitals | Chattogram Royals | Shere Bangla National Stadium, Mirpur | Dhaka Capitals | 0.14 | -140.00 | 0.00 |
| 2025-04-05 | Indian Premier League | Delhi Capitals | Chennai Super Kings | MA Chidambaram Stadium, Chepauk, Chennai | Delhi Capitals | 0.15 | 60.00 | 0.42 |
| 2025-07-13 | Major League Cricket | MI New York | Washington Freedom | Grand Prairie Stadium, Dallas | MI New York | 0.15 | -132.00 | 0.33 |
| 2026-01-11 | Bangladesh Premier League | Noakhali Express | Dhaka Capitals | Sylhet International Cricket Stadium | Noakhali Express | 0.16 | -18.00 | 0.00 |
| 2025-09-10 | Caribbean Premier League | Guyana Amazon Warriors | Antigua and Barbuda Falcons | Providence Stadium, Guyana | Antigua and Barbuda Falcons | 0.82 | 148.00 | 1.00 |
| 2026-02-22 | ICC Men's T20 World Cup | South Africa | India | Narendra Modi Stadium, Ahmedabad | South Africa | 0.19 | -211.00 | 0.35 |
| 2025-06-22 | Major League Cricket | Seattle Orcas | Los Angeles Knight Riders | Grand Prairie Stadium, Dallas | Los Angeles Knight Riders | 0.81 | 27.00 | 0.33 |
| 2026-01-02 | Big Bash League | Melbourne Stars | Brisbane Heat | Brisbane Cricket Ground, Woolloongabba, Brisbane | Brisbane Heat | 0.80 | 146.00 | 0.38 |
| 2025-04-08 | Indian Premier League | Punjab Kings | Chennai Super Kings | Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur | Punjab Kings | 0.20 | -13.00 | 0.71 |
| 2025-07-04 | Major League Cricket | San Francisco Unicorns | Texas Super Kings | Central Broward Regional Park Stadium Turf Ground, Lauderhill | San Francisco Unicorns | 0.20 | -4.00 | 0.50 |
| 2025-08-26 | Caribbean Premier League | Guyana Amazon Warriors | St Lucia Kings | Daren Sammy National Cricket Stadium, Gros Islet, St Lucia | St Lucia Kings | 0.79 | 51.00 | 0.56 |
| 2026-03-15 | ICC Men's T20 World Cup Sub Regional Americas Qualifier B | Mexico | Suriname | Jimmy Powell Oval, Cayman Islands | Suriname | 0.78 | 29.00 | 1.00 |
| 2025-07-02 | Major League Cricket | Texas Super Kings | Washington Freedom | Central Broward Regional Park Stadium Turf Ground, Lauderhill | Texas Super Kings | 0.22 | -75.00 | 0.00 |
| 2026-01-20 | Bangladesh Premier League | Rangpur Riders | Sylhet Titans | Shere Bangla National Stadium, Mirpur | Sylhet Titans | 0.77 | 98.00 | 0.50 |
| 2025-08-20 | Caribbean Premier League | Antigua and Barbuda Falcons | Trinbago Knight Riders | Sir Vivian Richards Stadium, North Sound, Antigua | Antigua and Barbuda Falcons | 0.23 | -151.00 | 1.00 |
| 2025-07-22 | Pakistan tour of Bangladesh | Bangladesh | Pakistan | Shere Bangla National Stadium, Mirpur | Bangladesh | 0.23 | -13.00 | 0.24 |
| 2025-11-05 | West Indies tour of New Zealand | West Indies | New Zealand | Eden Park, Auckland | West Indies | 0.24 | -81.00 | 0.28 |
| 2025-09-12 | Caribbean Premier League | Trinbago Knight Riders | Barbados Royals | Kensington Oval, Bridgetown, Barbados | Barbados Royals | 0.76 | 157.00 | 0.70 |


## Files
- Per-row CSV: `t20_err_v2.csv`
- This report:  `t20_err_v2.md`
