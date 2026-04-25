"""Derive partnerships from ball-by-ball data.

A partnership starts when a new batter comes in and ends when one of the two
batters is dismissed (or the innings ends). Tracking this cleanly in pure SQL
is painful because of strike rotation, so we walk balls in Python.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..db import connect


def _iter_innings(con) -> Iterable[tuple]:
    return con.execute(
        """SELECT match_id, innings_no, over_no, ball_in_over,
                  batter, non_striker, runs_batter, runs_extras, runs_total,
                  is_wicket, player_out, extras_type
           FROM balls
           ORDER BY match_id, innings_no, over_no, ball_in_over"""
    ).fetchall()


def _is_legal(extras_type: str | None) -> bool:
    if not extras_type:
        return True
    et = extras_type.split(",")
    return not any(t in ("wides", "noballs") for t in et)


def derive(replace: bool = True) -> int:
    con = connect()
    if replace:
        con.execute("DELETE FROM partnerships")

    rows: list[dict] = []
    cur_key: tuple | None = None
    wicket_no = 0
    p_runs = p_balls = 0
    p_bat1 = p_bat2 = None
    p_start = None
    p_end = None
    last_over = None

    def flush(unbeaten: bool):
        if cur_key is None or p_bat1 is None:
            return
        rows.append({
            "match_id":   cur_key[0],
            "innings_no": cur_key[1],
            "wicket_no":  wicket_no,
            "batter1":    p_bat1,
            "batter2":    p_bat2,
            "runs":       p_runs,
            "balls":      p_balls,
            "start_over": p_start,
            "end_over":   p_end if p_end is not None else last_over,
            "unbeaten":   unbeaten,
        })

    for r in _iter_innings(con):
        (match_id, innings_no, over_no, ball_in_over, batter, non_striker,
         _rb, _re, runs_total, is_wicket, player_out, extras_type) = r
        key = (match_id, innings_no)
        over_decimal = over_no + (ball_in_over - 1) / 6.0

        if key != cur_key:
            if cur_key is not None:
                flush(unbeaten=True)
            cur_key = key
            wicket_no = 1
            p_bat1, p_bat2 = batter, non_striker
            p_runs = p_balls = 0
            p_start = over_decimal
            p_end = None

        if p_bat1 not in (batter, non_striker) or p_bat2 not in (batter, non_striker):
            p_bat1, p_bat2 = batter, non_striker

        p_runs += int(runs_total or 0)
        if _is_legal(extras_type):
            p_balls += 1
        last_over = over_decimal

        if is_wicket and player_out and player_out != "retired hurt":
            p_end = over_decimal
            flush(unbeaten=False)
            wicket_no += 1
            survivor = non_striker if player_out == batter else batter
            p_bat1 = survivor
            p_bat2 = None
            p_runs = p_balls = 0
            p_start = over_decimal
            p_end = None

    flush(unbeaten=True)

    if rows:
        con.executemany(
            """INSERT OR REPLACE INTO partnerships
               (match_id, innings_no, wicket_no, batter1, batter2,
                runs, balls, start_over, end_over, unbeaten)
               VALUES ($match_id, $innings_no, $wicket_no, $batter1, $batter2,
                       $runs, $balls, $start_over, $end_over, $unbeaten)""",
            rows,
        )
    con.close()
    return len(rows)
