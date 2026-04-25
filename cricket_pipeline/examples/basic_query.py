"""Example queries you can run once data is loaded.

    python -m cricket_pipeline.examples.basic_query
"""

from cricket_pipeline.db import connect


def main():
    con = connect()

    print("\n=== Top 10 run-scorers in loaded data ===")
    rows = con.execute("""
        SELECT batter, SUM(runs_batter) AS runs, COUNT(*) AS balls_faced,
               ROUND(100.0 * SUM(runs_batter) / NULLIF(COUNT(*), 0), 2) AS sr
        FROM balls
        WHERE batter IS NOT NULL
        GROUP BY batter
        ORDER BY runs DESC
        LIMIT 10
    """).fetchall()
    for r in rows:
        print(r)

    print("\n=== Bowler-vs-batter matchup (min 10 balls) ===")
    rows = con.execute("""
        SELECT bowler, batter,
               COUNT(*) AS balls,
               SUM(runs_batter) AS runs,
               SUM(CASE WHEN is_wicket AND wicket_kind NOT IN ('run out', 'retired hurt')
                        THEN 1 ELSE 0 END) AS dismissals
        FROM balls
        WHERE bowler IS NOT NULL AND batter IS NOT NULL
        GROUP BY bowler, batter
        HAVING balls >= 10
        ORDER BY dismissals DESC, runs ASC
        LIMIT 15
    """).fetchall()
    for r in rows:
        print(r)

    print("\n=== Phase-wise run rate (T20 only) ===")
    rows = con.execute("""
        SELECT
          CASE
            WHEN over_no < 6  THEN '1_powerplay'
            WHEN over_no < 15 THEN '2_middle'
            ELSE '3_death'
          END AS phase,
          ROUND(6.0 * SUM(runs_total) / COUNT(*), 2) AS run_rate,
          COUNT(*) AS balls
        FROM balls b
        JOIN matches m USING (match_id)
        WHERE m.format IN ('T20', 'IT20')
        GROUP BY phase
        ORDER BY phase
    """).fetchall()
    for r in rows:
        print(r)

    con.close()


if __name__ == "__main__":
    main()
