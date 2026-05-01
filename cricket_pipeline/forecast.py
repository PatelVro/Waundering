"""End-to-end match forecast — every signal the pipeline has, in one call.

Combines:
  * `model.match.predict_match_ensemble` (LightGBM + form + h2h)
  * `model.simulate.simulate_innings`    (ball-level Monte Carlo)
  * Recent player form from the DB       (top batters / bowlers per side)
  * Bowler-vs-batter matchup risks       (Bayesian-shrunk)

Designed to be runnable both from CLI (`pipeline match-forecast`) and from
Python. Accepts optional overrides for toss and explicit XIs once they're
announced (~30 min pre-toss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .db import connect


@dataclass
class TeamForecast:
    name: str
    set_score_p10: int
    set_score_p50: int
    set_score_p90: int
    top_batters: list[tuple]
    top_bowlers: list[tuple]


@dataclass
class MatchForecast:
    home: str
    away: str
    venue: str
    p_home_wins: float
    p_away_wins: float
    favored: str
    edge_pct: float
    components: dict
    home_team: TeamForecast
    away_team: TeamForecast
    matchups_home_bowls: list[tuple]
    matchups_away_bowls: list[tuple]
    inputs_used: dict


def _state(con, batter, bowler, venue, target=None) -> dict:
    bsr  = (con.execute("SELECT strike_rate FROM v_batter_profile WHERE batter=?",
                        [batter]).fetchone() or (130.0,))[0] or 130
    becn = (con.execute("SELECT economy FROM v_bowler_profile WHERE bowler=?",
                        [bowler]).fetchone() or (8.5,))[0] or 8.5
    venue_row = con.execute(
        """SELECT avg_first_innings, toss_winner_won_pct
           FROM v_venue_profile WHERE venue=? AND format='T20'""",
        [venue],
    ).fetchone()
    venue_avg = (venue_row or (175,))[0] or 175
    venue_toss = (venue_row or (0, 0.55))[1] or 0.55
    return {
        "format": "T20", "venue": venue, "batter": batter, "bowler": bowler,
        "batter_hand": "Right hand Bat", "bowler_type": "Right arm Fast",
        "phase": "powerplay", "innings_no": 2 if target else 1,
        "over_no": 0, "ball_in_over": 1,
        "runs_so_far": 0, "wickets_so_far": 0, "deliveries_so_far": 0,
        "legal_balls_left": 120, "current_run_rate": 0.0,
        "required_run_rate": float("nan") if target is None else 6.0 * target / 120,
        "batter_sr": bsr, "batter_avg": 25, "batter_balls": 1000,
        "bowler_econ": becn, "bowler_avg": 28, "bowler_balls": 1000,
        "batter_form_sr": bsr, "batter_form_runs": 100, "batter_form_balls": 80,
        "bowler_workload_30d": 18, "bowler_workload_7d": 6,
        "venue_avg_first_innings": venue_avg, "venue_toss_winner_won_pct": venue_toss,
        "temp_c": 30.0, "humidity": 55.0, "wind_kmh": 7.0,
        "target": target,
    }


def _team_recent_xi(con, team: str, since: str, limit: int = 15) -> list[str]:
    """Return the players most frequently in the team's recent XIs."""
    rows = con.execute("""
        SELECT mx.player
        FROM match_xi mx
        JOIN matches m USING (match_id)
        WHERE mx.team = ? AND m.start_date >= ? AND m.format = 'T20'
        GROUP BY mx.player
        ORDER BY COUNT(*) DESC
        LIMIT ?
    """, [team, since, limit]).fetchall()
    return [r[0] for r in rows]


def _top_batters(con, squad: list[str], since: str, limit: int) -> list[tuple]:
    if not squad:
        return []
    placeholders = ",".join("?" * len(squad))
    return con.execute(f"""
        SELECT b.batter,
               SUM(b.runs_batter)                       AS runs,
               COUNT(*)                                 AS balls,
               ROUND(100.0 * SUM(b.runs_batter)
                           / NULLIF(COUNT(*), 0), 1)    AS sr,
               COUNT(DISTINCT b.match_id)               AS innings
        FROM balls b
        JOIN matches m USING (match_id)
        WHERE b.batter IN ({placeholders})
          AND m.start_date >= ? AND m.format = 'T20'
        GROUP BY b.batter
        HAVING balls >= 20
        ORDER BY sr DESC, runs DESC
        LIMIT ?
    """, list(squad) + [since, limit]).fetchall()


def _top_bowlers(con, squad: list[str], since: str, limit: int) -> list[tuple]:
    if not squad:
        return []
    placeholders = ",".join("?" * len(squad))
    return con.execute(f"""
        SELECT b.bowler,
               COUNT(*) / 6.0                                              AS overs,
               SUM(b.runs_total)                                           AS runs,
               SUM(CASE WHEN b.is_wicket
                         AND b.wicket_kind NOT IN ('run out','retired hurt')
                        THEN 1 ELSE 0 END)                                 AS wickets,
               ROUND(6.0 * SUM(b.runs_total) / COUNT(*), 2)                AS econ
        FROM balls b
        JOIN matches m USING (match_id)
        WHERE b.bowler IN ({placeholders})
          AND m.start_date >= ? AND m.format = 'T20'
        GROUP BY b.bowler
        HAVING overs >= 6
        ORDER BY econ ASC, wickets DESC
        LIMIT ?
    """, list(squad) + [since, limit]).fetchall()


def _matchups(con, bowlers: list[str], batters: list[str], k: int) -> list[tuple]:
    if not bowlers or not batters:
        return []
    bp = ",".join("?" * len(bowlers))
    bt = ",".join("?" * len(batters))
    return con.execute(f"""
        SELECT bowler, batter, balls, runs, wkts,
               ROUND(shrunk_wkt_per_ball * 100, 2)  AS wkt_pct,
               ROUND(shrunk_runs_per_ball, 2)       AS rpb
        FROM v_matchup_shrunk
        WHERE bowler IN ({bp}) AND batter IN ({bt})
          AND balls >= 12
        ORDER BY wkt_pct DESC, rpb ASC
        LIMIT ?
    """, list(bowlers) + list(batters) + [k]).fetchall()


def forecast(
    home: str, away: str, venue: str,
    home_xi: list[str] | None = None,
    away_xi: list[str] | None = None,
    home_opener: str | None = None,
    away_opener: str | None = None,
    home_bowler: str | None = None,
    away_bowler: str | None = None,
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    ref_date: str | None = None,
    since: str | None = None,
    n_sim: int = 2000,
) -> MatchForecast:
    """Produce a full forecast. All XI / opener arguments are optional —
    if missing, the pipeline falls back to historical openers/bowlers per
    side or to whatever the public preview suggested.
    """
    from .model.match     import predict_match_ensemble
    from .model.simulate  import simulate_innings

    # Default the "since" cutoff to the recent IPL 2026 window if unset.
    if since is None:
        con = connect()
        latest = (con.execute("SELECT MAX(start_date) FROM matches").fetchone() or (None,))[0]
        con.close()
        if latest:
            from datetime import timedelta
            since = (latest - timedelta(days=45)).isoformat()
        else:
            since = "2024-01-01"

    ens = predict_match_ensemble(
        home=home, away=away, venue=venue,
        format_="T20", toss_winner=toss_winner,
        toss_decision=toss_decision, ref_date=ref_date,
    )

    con = connect()

    # Resolve squads from DB when not explicitly provided
    if not home_xi:
        home_xi = _team_recent_xi(con, home, since)
    if not away_xi:
        away_xi = _team_recent_xi(con, away, since)

    # Pick openers / bowlers for the rollout: explicit > top batter/bowler in DB.
    if home_opener is None and home_xi:
        home_opener = home_xi[0]
    if away_opener is None and away_xi:
        away_opener = away_xi[0]
    if home_bowler is None and home_xi:
        bowlers = _top_bowlers(con, home_xi, since, 1)
        home_bowler = bowlers[0][0] if bowlers else home_xi[-1]
    if away_bowler is None and away_xi:
        bowlers = _top_bowlers(con, away_xi, since, 1)
        away_bowler = bowlers[0][0] if bowlers else away_xi[-1]

    # Hard fallback only if DB has no data at all for these teams
    home_opener = home_opener or home
    home_bowler = home_bowler or home
    away_opener = away_opener or away
    away_bowler = away_bowler or away

    # Score distributions (each side batting first)
    home_set = simulate_innings(_state(con, home_opener, away_bowler, venue),
                                n_sim=n_sim, seed=1)
    away_set = simulate_innings(_state(con, away_opener, home_bowler, venue),
                                n_sim=n_sim, seed=2)

    home_top_b = _top_batters(con, home_xi or [home_opener], since, 5)
    away_top_b = _top_batters(con, away_xi or [away_opener], since, 5)
    home_top_w = _top_bowlers(con, home_xi or [home_bowler], since, 5)
    away_top_w = _top_bowlers(con, away_xi or [away_bowler], since, 5)

    # Matchups
    matchups_home = _matchups(con, home_xi or [home_bowler],
                                  away_xi or [away_opener], 5)
    matchups_away = _matchups(con, away_xi or [away_bowler],
                                  home_xi or [home_opener], 5)
    con.close()

    home_t = TeamForecast(
        name=home,
        set_score_p10=home_set["p10"], set_score_p50=home_set["p50"],
        set_score_p90=home_set["p90"],
        top_batters=home_top_b, top_bowlers=home_top_w,
    )
    away_t = TeamForecast(
        name=away,
        set_score_p10=away_set["p10"], set_score_p50=away_set["p50"],
        set_score_p90=away_set["p90"],
        top_batters=away_top_b, top_bowlers=away_top_w,
    )

    return MatchForecast(
        home=home, away=away, venue=venue,
        p_home_wins=ens["p_home_wins"],
        p_away_wins=ens["p_away_wins"],
        favored=ens["favored"],
        edge_pct=ens["edge_pct"],
        components=ens["components"],
        home_team=home_t, away_team=away_t,
        matchups_home_bowls=matchups_home,
        matchups_away_bowls=matchups_away,
        inputs_used={
            "home_xi":       home_xi,
            "away_xi":       away_xi,
            "home_opener":   home_opener, "away_opener":   away_opener,
            "home_bowler":   home_bowler, "away_bowler":   away_bowler,
            "toss_winner":   toss_winner, "toss_decision": toss_decision,
            "ref_date":      ref_date, "since":           since,
        },
    )


def render(f: MatchForecast) -> str:
    """Pretty-print a MatchForecast to a string."""
    L = []
    L.append("=" * 72)
    L.append(f"  {f.home} vs {f.away}")
    L.append(f"  {f.venue}")
    L.append("=" * 72)
    L.append("")
    L.append("📊 WIN PROBABILITY (ensemble, calibrated)")
    L.append(f"  {f.home:<32} {f.p_home_wins:>6.1%}")
    L.append(f"  {f.away:<32} {f.p_away_wins:>6.1%}")
    L.append(f"  Edge: {f.edge_pct:.1f} pp toward {f.favored}")
    c = f.components
    L.append(f"  Components → match {c.get('match_model', 0):.1%}, "
             f"form {c.get('form_prior', 0):.1%}, h2h {c.get('h2h_prior', 0):.1%}")
    L.append("")
    L.append("📈 PROJECTED SCORES (Monte Carlo)")
    L.append(f"  {f.home} batting first → median {f.home_team.set_score_p50:>3} "
             f"(p10–p90: {f.home_team.set_score_p10}–{f.home_team.set_score_p90})")
    L.append(f"  {f.away} batting first → median {f.away_team.set_score_p50:>3} "
             f"(p10–p90: {f.away_team.set_score_p10}–{f.away_team.set_score_p90})")
    L.append("")
    L.append("🏏 TOP BATTERS (recent window, min 20 balls)")
    for label, rows in (("  " + f.home, f.home_team.top_batters),
                        ("  " + f.away, f.away_team.top_batters)):
        L.append(f"{label}:")
        for r in rows[:3]:
            L.append(f"    {r[0]:<25} {r[1]:>3} runs in {r[2]:>3} balls  "
                     f"SR {r[3]:>5.1f}  ({r[4]} inns)")
    L.append("")
    L.append("🎯 TOP BOWLERS (recent window, min 6 overs)")
    for label, rows in (("  " + f.home, f.home_team.top_bowlers),
                        ("  " + f.away, f.away_team.top_bowlers)):
        L.append(f"{label}:")
        for r in rows[:3]:
            L.append(f"    {r[0]:<25} {r[1]:>4.1f} ov  {r[3]:>2} wkts  econ {r[4]:>4.2f}")
    L.append("")
    L.append("⚠️  KEY MATCHUPS (Bayesian-shrunk)")
    L.append(f"  {f.away} bowlers vs {f.home} batters:")
    for r in f.matchups_away_bowls:
        L.append(f"    {r[0]:<22} vs {r[1]:<18}  {r[2]:>3}b  {r[3]:>3}r  "
                 f"{r[4]} wkts  P(wkt)={r[5]}%  RPB={r[6]}")
    L.append(f"  {f.home} bowlers vs {f.away} batters:")
    for r in f.matchups_home_bowls:
        L.append(f"    {r[0]:<22} vs {r[1]:<18}  {r[2]:>3}b  {r[3]:>3}r  "
                 f"{r[4]} wkts  P(wkt)={r[5]}%  RPB={r[6]}")
    L.append("")
    L.append("=" * 72)
    return "\n".join(L)
