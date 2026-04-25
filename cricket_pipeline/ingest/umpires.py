"""Build the umpires table from data we already have in `matches`.

No external fetch — pure SQL transform. Splits the comma-separated
`matches.umpires` column and aggregates appearances + formats officiated.
"""

from __future__ import annotations

from ..db import connect


def populate() -> int:
    con = connect()
    con.execute(
        """CREATE OR REPLACE TEMP VIEW _umps AS
           SELECT TRIM(unnest(string_split(umpires, ','))) AS name,
                  format
           FROM matches
           WHERE umpires IS NOT NULL AND umpires != ''"""
    )
    con.execute("DELETE FROM umpires")
    con.execute(
        """INSERT INTO umpires (name, matches, formats)
           SELECT name,
                  COUNT(*) AS matches,
                  string_agg(DISTINCT format, ',') AS formats
           FROM _umps
           WHERE name != ''
           GROUP BY name"""
    )
    n = con.execute("SELECT COUNT(*) FROM umpires").fetchone()[0]
    con.close()
    return n
