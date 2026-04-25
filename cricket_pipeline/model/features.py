"""Build the model-ready feature table by joining v_ball_state with
time-aware player history, venue and weather aggregates.

The joins use `v_batter_history` and `v_bowler_history` which compute
career and rolling stats *strictly before* each ball's match — so features
only see the past and there's no temporal leakage.
"""

from __future__ import annotations

import pandas as pd

from ..db import connect, install_views

CATEGORICAL = [
    "format", "venue",
    "batter_hand", "bowler_type",
    "phase",
]

NUMERIC = [
    "innings_no", "over_no", "ball_in_over",
    "runs_so_far", "wickets_so_far", "deliveries_so_far",
    "legal_balls_left", "current_run_rate", "required_run_rate",
    "batter_sr", "batter_avg", "batter_balls",
    "bowler_econ", "bowler_avg", "bowler_balls",
    "batter_form_sr", "batter_form_runs", "batter_form_balls",
    "bowler_workload_7d", "bowler_workload_30d",
    "venue_avg_first_innings", "venue_toss_winner_won_pct",
    "temp_c", "humidity", "wind_kmh",
]

TARGETS = ["y_runs", "y_runs_bucket", "y_wicket"]

FEATURE_SQL = """
SELECT
    bs.match_id, bs.innings_no, bs.over_no, bs.ball_in_over,
    bs.format, bs.venue,
    bs.batter, bs.bowler,
    bs.runs_so_far, bs.wickets_so_far, bs.deliveries_so_far,
    bs.legal_balls_left, bs.current_run_rate, bs.required_run_rate,
    bs.phase,

    bs.runs_total                                                AS y_runs,
    CASE
      WHEN bs.is_wicket
       AND bs.wicket_kind NOT IN ('run out', 'retired hurt') THEN 1
      ELSE 0
    END                                                          AS y_wicket,

    bh.career_sr                                                 AS batter_sr,
    bh.career_avg                                                AS batter_avg,
    bh.career_balls                                              AS batter_balls,

    wh.career_econ                                               AS bowler_econ,
    wh.career_avg                                                AS bowler_avg,
    wh.career_balls                                              AS bowler_balls,

    bh.form_sr                                                   AS batter_form_sr,
    bh.form_runs                                                 AS batter_form_runs,
    bh.form_balls                                                AS batter_form_balls,

    wh.workload_7d                                               AS bowler_workload_7d,
    wh.workload_30d                                              AS bowler_workload_30d,

    pb.batting_hand                                              AS batter_hand,
    pn.bowling_type                                              AS bowler_type,

    vp.avg_first_innings                                         AS venue_avg_first_innings,
    vp.toss_winner_won_pct                                       AS venue_toss_winner_won_pct,

    wd.temp_c, wd.humidity, wd.wind_kmh,

    m.start_date
FROM v_ball_state bs
LEFT JOIN v_batter_history  bh ON bh.batter = bs.batter AND bh.match_id = bs.match_id
LEFT JOIN v_bowler_history  wh ON wh.bowler = bs.bowler AND wh.match_id = bs.match_id
LEFT JOIN players           pb ON pb.name   = bs.batter
LEFT JOIN players           pn ON pn.name   = bs.bowler
LEFT JOIN matches           m  ON m.match_id = bs.match_id
LEFT JOIN v_venue_profile   vp ON vp.venue = bs.venue AND vp.format = bs.format
LEFT JOIN weather_daily     wd ON wd.venue = bs.venue AND wd.date  = m.start_date
WHERE bs.format IN ('T20', 'IT20', 'ODI')
  AND bs.batter IS NOT NULL
  AND bs.bowler IS NOT NULL
"""


def _bucket_runs(r: int) -> int:
    """Multi-class buckets for ball-outcome: 0,1,2,3,4,5(other),6."""
    if r in (0, 1, 2, 3, 4, 6):
        return r
    return 5


def build(format_filter: str | None = None, limit: int | None = None) -> pd.DataFrame:
    install_views()
    con = connect()
    sql = FEATURE_SQL
    if format_filter:
        sql = sql.replace(
            "WHERE bs.format IN ('T20', 'IT20', 'ODI')",
            f"WHERE bs.format = '{format_filter}'",
        )
    if limit:
        sql += f" LIMIT {limit}"
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return df

    df["y_runs_bucket"] = df["y_runs"].fillna(0).astype(int).map(_bucket_runs)
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def split_by_date(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-aware split: most-recent `test_frac` becomes the test set."""
    df = df.sort_values("start_date").reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
