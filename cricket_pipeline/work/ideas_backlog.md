# Experiment Backlog

Ranked roughly by expected leverage. Pop top, run cycle, append at bottom.

## High leverage (data + features)

1. Per-format Elo rating system (compute chronologically across all matches).
2. Recent form features (last 3 / 5 / 10 matches win rate, weighted by recency).
3. Last-N at venue, last-N vs opponent.
4. Time-based train/val/test split (this is mandatory for honest measurement).
5. Venue features: avg 1st-innings score, chase win rate, toss-winner-wins-pct.
6. Rest days since each team's last match.
7. Format-specific models (separate T20 / ODI / Test).
8. Bigger Cricsheet ingest — all_json instead of t20s sample.

## Medium leverage

9. Player-level aggregates from balls (when XI known): top-N batter form, top-N
   bowler form on team.
10. Dew flag for night matches at dew-prone venues.
11. Tournament stage (group vs knockout) one-hot.
12. Home/away/neutral indicator.
13. Toss-winner × format interaction.
14. Last-meeting weighted recency H2H.
15. Calibration with isotonic on a held-out calibration slice.
16. Stacking ensemble: LightGBM + XGBoost + LogReg.
17. Recency-weighted training (weight recent matches more).

## Lower leverage / try after the above

18. CatBoost with categorical features native handling.
19. Optuna hyperparameter search with time-series CV.
20. Sequence model over recent team-form deltas.
21. Add weather features (Visual Crossing — needs API key).
22. News sentiment (lower-quality signal but free).
