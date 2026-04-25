# Cricket Domain Notes

Living document. Notes on what actually drives cricket match outcomes,
translated into candidate features.

## What drives a cricket match outcome (general)

- **Toss + decision** in places where dew or pitch wear matters (e.g. day-night
  T20s in subcontinent: chasing wins ~55-60%; in early-tournament Sheffield
  Shield bowling first wins because of green wicket).
- **Recent form** of XIs, specifically with weighted recency. Career averages
  matter much less than rolling form over last 5-10 innings in the same format.
- **Venue effects**: average first-innings score, batting-first-vs-chasing win
  rate, spin vs pace effectiveness, day vs night results, ground dimensions.
- **Player matchups**: batter X vs left-arm spin, bowler Y at this venue, etc.
- **Strength differential**: per-format Elo / Glicko of the two teams.
- **Pitch + dew**: dry/dusty (spin-friendly), green/seaming, dew (helps chasers
  in night T20Is/ODIs). Dew prediction often shifts model 5-8pp.
- **Captain + balance**: 5 bowling options vs 4, batting depth, all-rounder
  count.
- **Context**: knockout vs group, home vs away vs neutral, rest days, travel.
- **Injury / availability** of star players.

## Format-specific dynamics

### T20 (and IT20)
- Toss matters most when dew is expected (night games subcontinent).
- Powerplay (6 overs) and death (16-20) are the highest-variance phases.
- Top order strike rate often matters more than average.
- 1st innings score is highly predictable from recent venue average +
  team batting strength; 2nd innings is conditional on dew + chase pressure.

### ODI
- Middle overs (11-40) and death-over partnerships matter.
- Spin-pace mix at venue is a strong signal.
- Toss matters less than T20 (more time absorbs randomness).

### Test
- Conditions on day 1 vs day 4 differ massively — pitch deterioration is the
  biggest variable.
- Series context (1-0 down) shifts approach.
- Wickets to bowler-style mix.

## Statistical pitfalls to avoid

- Career batting average is a backward-looking, slow-moving stat — use rolling
  10-innings runs/SR/avg.
- "Form" computed including the match being predicted = leakage.
- Random train/test split on time-series sports data = leakage. Always use
  time-based split.

## Candidate features (translating notes into model inputs)

- Per-format Elo for both teams (custom, computed chronologically).
- Last-N-matches win rate for each team (N ∈ {3, 5, 10}).
- Last-N-matches at this venue for each team.
- Last-N matches against this opponent.
- Toss winner and decision (one-hot).
- Venue: avg first-innings score, chase win rate, mean RPO.
- Days since each team's last match (rest factor).
- Tournament stage (group/knockout), neutral-venue flag.
- Rolling per-player runs / SR / wickets / economy aggregated to team level
  (when XIs known).
- Dew flag for night matches at known dew-prone venues.
