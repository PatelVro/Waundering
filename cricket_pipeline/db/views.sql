-- Derived analytical views. Run after data is loaded.
-- These are pure SQL — no fetch — and become input features for a model.

-- Per-venue, per-format scoring profile
CREATE OR REPLACE VIEW v_venue_profile AS
SELECT
    m.venue,
    m.format,
    COUNT(DISTINCT m.match_id)                                AS matches,
    AVG(i1.total_runs)                                        AS avg_first_innings,
    AVG(i2.total_runs)                                        AS avg_second_innings,
    AVG(i1.total_runs - i2.total_runs)                        AS first_innings_advantage,
    AVG(CASE WHEN m.toss_decision = 'bat'   THEN 1 ELSE 0 END)  AS bat_first_pct,
    AVG(CASE WHEN m.toss_winner   = m.winner THEN 1 ELSE 0 END) AS toss_winner_won_pct
FROM matches m
LEFT JOIN innings i1 ON i1.match_id = m.match_id AND i1.innings_no = 1
LEFT JOIN innings i2 ON i2.match_id = m.match_id AND i2.innings_no = 2
GROUP BY m.venue, m.format;

-- Phase metrics (powerplay / middle / death) for limited overs only
CREATE OR REPLACE VIEW v_phase_metrics AS
SELECT
    m.format,
    CASE
      WHEN b.over_no <  6 THEN '1_powerplay'
      WHEN b.over_no < 15 THEN '2_middle'
      ELSE                        '3_death'
    END                                            AS phase,
    COUNT(*)                                        AS balls,
    ROUND(6.0 * SUM(b.runs_total) / COUNT(*), 2)    AS run_rate,
    ROUND(100.0 * SUM(CASE WHEN b.is_wicket THEN 1 ELSE 0 END) / COUNT(*), 3) AS wicket_pct,
    ROUND(100.0 * SUM(CASE WHEN b.runs_batter = 0 AND b.runs_extras = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS dot_pct,
    ROUND(100.0 * SUM(CASE WHEN b.runs_batter IN (4, 6) THEN 1 ELSE 0 END) / COUNT(*), 2)                AS boundary_pct
FROM balls b
JOIN matches m USING (match_id)
WHERE m.format IN ('T20', 'IT20', 'ODI')
GROUP BY m.format, phase;

-- Per-batter career profile from ball data
CREATE OR REPLACE VIEW v_batter_profile AS
SELECT
    batter,
    COUNT(*)                                        AS balls_faced,
    SUM(runs_batter)                                AS runs,
    ROUND(100.0 * SUM(runs_batter) / COUNT(*), 2)   AS strike_rate,
    SUM(CASE WHEN runs_batter IN (4, 6) THEN 1 ELSE 0 END)                                AS boundaries,
    SUM(CASE WHEN is_wicket AND wicket_kind NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END) AS dismissals,
    ROUND(1.0 * SUM(runs_batter) /
          NULLIF(SUM(CASE WHEN is_wicket AND wicket_kind NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END), 0), 2) AS average
FROM balls
WHERE batter IS NOT NULL
GROUP BY batter;

-- Per-bowler career profile from ball data
CREATE OR REPLACE VIEW v_bowler_profile AS
SELECT
    bowler,
    COUNT(*)                                        AS balls_bowled,
    SUM(runs_total)                                 AS runs_conceded,
    SUM(CASE WHEN is_wicket AND wicket_kind NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END) AS wickets,
    ROUND(6.0 * SUM(runs_total) / COUNT(*), 2)      AS economy,
    ROUND(1.0 * SUM(runs_total) /
          NULLIF(SUM(CASE WHEN is_wicket AND wicket_kind NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END), 0), 2) AS average
FROM balls
WHERE bowler IS NOT NULL
GROUP BY bowler;

-- Bowler-vs-batter matchup (Bayesian-friendly: keep low-sample matchups)
CREATE OR REPLACE VIEW v_matchup AS
SELECT
    bowler,
    batter,
    COUNT(*)                                        AS balls,
    SUM(runs_batter)                                AS runs,
    SUM(CASE WHEN is_wicket AND wicket_kind NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END) AS dismissals,
    ROUND(100.0 * SUM(runs_batter) / COUNT(*), 2)   AS strike_rate
FROM balls
WHERE bowler IS NOT NULL AND batter IS NOT NULL
GROUP BY bowler, batter;

-- Umpire LBW propensity (a real bias signal worth modelling)
CREATE OR REPLACE VIEW v_umpire_lbw AS
SELECT
    o.name                                                  AS umpire,
    COUNT(DISTINCT o.match_id)                              AS matches,
    SUM(CASE WHEN b.wicket_kind = 'lbw' THEN 1 ELSE 0 END)  AS lbws_given,
    ROUND(1.0 * SUM(CASE WHEN b.wicket_kind = 'lbw' THEN 1 ELSE 0 END)
              / NULLIF(COUNT(DISTINCT o.match_id), 0), 2)   AS lbws_per_match
FROM match_officials o
JOIN balls b USING (match_id)
WHERE o.role = 'umpire'
GROUP BY o.name;

-- Toss-impact summary by venue (does winning the toss matter here?)
CREATE OR REPLACE VIEW v_toss_impact AS
SELECT
    venue,
    format,
    COUNT(*)                                                                        AS matches,
    AVG(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END)                           AS toss_winner_win_pct,
    AVG(CASE WHEN toss_decision = 'bat'   AND toss_winner = winner THEN 1 ELSE 0 END) AS bat_first_win_pct,
    AVG(CASE WHEN toss_decision = 'field' AND toss_winner = winner THEN 1 ELSE 0 END) AS field_first_win_pct
FROM matches
WHERE toss_winner IS NOT NULL AND winner IS NOT NULL
GROUP BY venue, format;

-- Ball-state features — one row per delivery with running totals.
-- This is the canonical model-input table for ball-outcome prediction.
CREATE OR REPLACE VIEW v_ball_state AS
WITH b AS (
    SELECT
        b.*,
        m.format, m.venue,
        i.target,
        i.total_runs AS innings_total_runs,
        SUM(b.runs_total) OVER (
            PARTITION BY b.match_id, b.innings_no
            ORDER BY b.over_no, b.ball_in_over
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS runs_so_far,
        SUM(CASE WHEN b.is_wicket THEN 1 ELSE 0 END) OVER (
            PARTITION BY b.match_id, b.innings_no
            ORDER BY b.over_no, b.ball_in_over
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS wickets_so_far,
        ROW_NUMBER() OVER (
            PARTITION BY b.match_id, b.innings_no
            ORDER BY b.over_no, b.ball_in_over
        ) - 1 AS deliveries_so_far
    FROM balls b
    JOIN matches m USING (match_id)
    LEFT JOIN innings i ON i.match_id = b.match_id AND i.innings_no = b.innings_no
)
SELECT
    match_id, innings_no, over_no, ball_in_over,
    format, venue,
    batter, non_striker, bowler,
    runs_total, is_wicket, wicket_kind,
    COALESCE(runs_so_far, 0)     AS runs_so_far,
    COALESCE(wickets_so_far, 0)  AS wickets_so_far,
    deliveries_so_far,
    CASE
      WHEN format IN ('T20', 'IT20') THEN 120 - deliveries_so_far
      WHEN format = 'ODI'            THEN 300 - deliveries_so_far
      ELSE NULL
    END                          AS legal_balls_left,
    CASE WHEN deliveries_so_far > 0
         THEN ROUND(6.0 * COALESCE(runs_so_far, 0) / deliveries_so_far, 2)
    END                          AS current_run_rate,
    CASE WHEN target IS NOT NULL
              AND format IN ('T20', 'IT20', 'ODI')
              AND (CASE WHEN format IN ('T20','IT20') THEN 120 ELSE 300 END - deliveries_so_far) > 0
         THEN ROUND(
                6.0 * (target - COALESCE(runs_so_far, 0))
                    / (CASE WHEN format IN ('T20','IT20') THEN 120 ELSE 300 END - deliveries_so_far),
              2)
    END                          AS required_run_rate,
    CASE
      WHEN over_no <  6 THEN 'powerplay'
      WHEN over_no < 15 THEN 'middle'
      ELSE                   'death'
    END                          AS phase
FROM b;

-- Rolling 10-innings batter form (recent runs and SR)
CREATE OR REPLACE VIEW v_batter_form AS
WITH innings_agg AS (
    SELECT
        b.batter,
        b.match_id, b.innings_no,
        m.start_date,
        SUM(b.runs_batter) AS runs,
        COUNT(*)           AS balls,
        BOOL_OR(b.is_wicket AND b.player_out = b.batter) AS dismissed
    FROM balls b
    JOIN matches m USING (match_id)
    WHERE b.batter IS NOT NULL
    GROUP BY b.batter, b.match_id, b.innings_no, m.start_date
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY batter ORDER BY start_date DESC) AS rn
    FROM innings_agg
)
SELECT
    batter,
    SUM(runs)                                 AS last10_runs,
    SUM(balls)                                AS last10_balls,
    ROUND(100.0 * SUM(runs) / NULLIF(SUM(balls), 0), 2)              AS last10_sr,
    ROUND(1.0 * SUM(runs) / NULLIF(SUM(CASE WHEN dismissed THEN 1 ELSE 0 END), 0), 2)
                                              AS last10_avg,
    SUM(CASE WHEN runs >= 50 THEN 1 ELSE 0 END) AS last10_fifties_plus
FROM ranked
WHERE rn <= 10
GROUP BY batter;

-- Bowler workload — overs bowled in a recent window (proxy for fatigue)
CREATE OR REPLACE VIEW v_bowler_workload AS
WITH per_match AS (
    SELECT
        b.bowler,
        m.start_date,
        COUNT(*) / 6.0 AS overs_bowled
    FROM balls b
    JOIN matches m USING (match_id)
    WHERE b.bowler IS NOT NULL
    GROUP BY b.bowler, m.start_date
)
SELECT
    bowler,
    SUM(CASE WHEN start_date >= CURRENT_DATE -  7 THEN overs_bowled ELSE 0 END) AS overs_last_7d,
    SUM(CASE WHEN start_date >= CURRENT_DATE - 30 THEN overs_bowled ELSE 0 END) AS overs_last_30d,
    SUM(CASE WHEN start_date >= CURRENT_DATE - 90 THEN overs_bowled ELSE 0 END) AS overs_last_90d,
    MAX(start_date)                                                              AS last_match
FROM per_match
GROUP BY bowler;

-- Partnership leaderboard view (after partnerships table is populated)
CREATE OR REPLACE VIEW v_top_partnerships AS
SELECT
    p.match_id, m.format, m.start_date, m.venue,
    p.innings_no, p.wicket_no,
    p.batter1, p.batter2,
    p.runs, p.balls,
    ROUND(100.0 * p.runs / NULLIF(p.balls, 0), 2) AS sr
FROM partnerships p
JOIN matches m USING (match_id)
ORDER BY p.runs DESC;
