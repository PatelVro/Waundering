"""Leakage-safe feature builder for the match-winner model.

Replaces v_match_features. Two key fixes vs the legacy view:

1. **Venue stats are AS-OF the match's start_date**, not all-time. The legacy
   v_venue_profile bakes in stats from the match being predicted itself.
2. **Adds Elo ratings** computed chronologically, per format. This is the
   single highest-leverage feature for cricket prediction.

Output: pandas DataFrame, one row per match, with the label `y_team1_wins`.
We use `team1` / `team2` (not "home/away") because Cricsheet's team ordering
is not geographic — relabelling kills any false intuition about home advantage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..db import connect, install_views


# ---------- Elo ----------

@dataclass
class EloConfig:
    base: float = 1500.0
    k: float = 24.0
    home_adv: float = 0.0
    fmt_isolate: bool = True
    margin_weight: bool = False   # scale K by win-margin (FiveThirtyEight-style)


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _is_pos_num(x) -> bool:
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    try:
        return float(x) > 0
    except Exception:
        return False


def _margin_multiplier(runs, wickets) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier."""
    m = 1.0
    if _is_pos_num(runs):
        r = float(runs)
        m = math.log(max(r, 1.0) + 1.0) / math.log(11.0)
    elif _is_pos_num(wickets):
        w = float(wickets)
        m = 1.0 + (w / 10.0) * 0.5
    return max(0.5, min(m, 2.5))


def compute_elo_features(matches: pd.DataFrame, cfg: EloConfig | None = None) -> pd.DataFrame:
    """Walk matches in chronological order, compute pre-match Elo for each side.

    Returns: match_id, t1_elo_pre, t2_elo_pre, elo_diff_pre
    """
    cfg = cfg or EloConfig()
    df = matches.sort_values(["start_date", "match_id"]).reset_index(drop=True).copy()

    ratings: dict[tuple[str, str], float] = {}

    def _get(team: str, fmt: str) -> float:
        key = (fmt if cfg.fmt_isolate else "_", team)
        return ratings.get(key, cfg.base)

    def _set(team: str, fmt: str, val: float) -> None:
        key = (fmt if cfg.fmt_isolate else "_", team)
        ratings[key] = val

    t1_pre = np.empty(len(df)); t2_pre = np.empty(len(df))
    has_runs   = "win_margin_runs"     in df.columns
    has_wkts   = "win_margin_wickets"  in df.columns

    for i, row in enumerate(df.itertuples(index=False)):
        t1, t2, fmt = row.team_home, row.team_away, row.format
        r1 = _get(t1, fmt); r2 = _get(t2, fmt)
        t1_pre[i] = r1; t2_pre[i] = r2

        winner = getattr(row, "winner", None)
        if winner is None or (isinstance(winner, float) and math.isnan(winner)):
            continue
        s1 = 1.0 if winner == t1 else (0.0 if winner == t2 else 0.5)
        e1 = _expected(r1, r2)
        k_eff = cfg.k
        if cfg.margin_weight:
            runs = getattr(row, "win_margin_runs", None)   if has_runs else None
            wkts = getattr(row, "win_margin_wickets", None) if has_wkts else None
            k_eff = cfg.k * _margin_multiplier(runs, wkts)
        delta = k_eff * (s1 - e1)
        _set(t1, fmt, r1 + delta)
        _set(t2, fmt, r2 - delta)

    out = pd.DataFrame({
        "match_id":     df["match_id"].values,
        "t1_elo_pre":   t1_pre,
        "t2_elo_pre":   t2_pre,
        "elo_diff_pre": t1_pre - t2_pre,
    })
    return out


# ---------- venue stats AS-OF date ----------

def compute_venue_stats_asof(matches: pd.DataFrame, innings: pd.DataFrame) -> pd.DataFrame:
    """For each match, compute venue stats from matches strictly before its date.

    Columns: match_id, venue_n_prior, venue_avg_first, venue_chase_winrate,
             venue_toss_winner_winpct, venue_bat_first_pct
    """
    m = matches.sort_values(["venue", "format", "start_date", "match_id"]).copy()
    inn1 = innings[innings["innings_no"] == 1][["match_id", "total_runs"]].rename(
        columns={"total_runs": "first_innings"}
    )
    m = m.merge(inn1, on="match_id", how="left")

    # winner of 1st innings batting team?
    m["bat1_team"] = np.where(m["toss_decision"] == "bat", m["toss_winner"],
                              np.where(m["toss_winner"] == m["team_home"], m["team_away"], m["team_home"]))
    # If no toss data, bat1_team will be wrong; protect:
    m.loc[m["toss_winner"].isna() | m["toss_decision"].isna(), "bat1_team"] = np.nan
    m["bat1_won"]    = (m["bat1_team"] == m["winner"]).astype(float)
    m.loc[m["bat1_team"].isna() | m["winner"].isna(), "bat1_won"] = np.nan

    m["toss_won_match"] = (m["toss_winner"] == m["winner"]).astype(float)
    m.loc[m["toss_winner"].isna() | m["winner"].isna(), "toss_won_match"] = np.nan
    m["bat_first"]      = (m["toss_decision"] == "bat").astype(float)
    m.loc[m["toss_decision"].isna(), "bat_first"] = np.nan

    g = m.groupby(["venue", "format"], sort=False)
    # cumulative SUM up to (but not including) the current row → use shift
    def _cum_excl(s):
        return s.shift(1).expanding(min_periods=1).sum()

    def _cnt_excl(s):
        return s.shift(1).expanding(min_periods=1).count()

    m["v_n"]            = g.cumcount()
    m["v_first_sum"]    = g["first_innings"].transform(_cum_excl)
    m["v_first_cnt"]    = g["first_innings"].transform(_cnt_excl)
    m["v_bat1won_sum"]  = g["bat1_won"].transform(_cum_excl)
    m["v_bat1won_cnt"]  = g["bat1_won"].transform(_cnt_excl)
    m["v_tosswon_sum"]  = g["toss_won_match"].transform(_cum_excl)
    m["v_tosswon_cnt"]  = g["toss_won_match"].transform(_cnt_excl)
    m["v_batfirst_sum"] = g["bat_first"].transform(_cum_excl)
    m["v_batfirst_cnt"] = g["bat_first"].transform(_cnt_excl)

    out = pd.DataFrame({
        "match_id":                  m["match_id"].values,
        "venue_n_prior":             m["v_n"].values,
        "venue_avg_first":           (m["v_first_sum"] / m["v_first_cnt"]).values,
        "venue_bat1_winrate":        (m["v_bat1won_sum"] / m["v_bat1won_cnt"]).values,
        "venue_toss_winner_winpct":  (m["v_tosswon_sum"] / m["v_tosswon_cnt"]).values,
        "venue_bat_first_pct":       (m["v_batfirst_sum"] / m["v_batfirst_cnt"]).values,
    })
    return out


# ---------- per-team rolling form ----------

def compute_team_form(matches: pd.DataFrame) -> pd.DataFrame:
    """For each (team, match_id) compute pre-match form windows."""
    # symmetric expansion — one row per (team, match)
    a = matches[["match_id", "start_date", "format", "team_home", "team_away", "winner"]].copy()
    rows = []
    rows.append(a.rename(columns={"team_home": "team", "team_away": "opp"}).assign(
        won=lambda x: (x["winner"] == x["team"]).astype(float).where(x["winner"].notna())
    ))
    rows.append(a.rename(columns={"team_away": "team", "team_home": "opp"}).assign(
        won=lambda x: (x["winner"] == x["team"]).astype(float).where(x["winner"].notna())
    ))
    sym = pd.concat(rows, ignore_index=True).sort_values(["team", "format", "start_date", "match_id"])

    g = sym.groupby(["team", "format"], sort=False)["won"]
    def _roll_mean(s, n):
        return s.shift(1).rolling(n, min_periods=1).mean()

    sym["last3_winpct"]  = g.transform(lambda s: _roll_mean(s, 3))
    sym["last5_winpct"]  = g.transform(lambda s: _roll_mean(s, 5))
    sym["last10_winpct"] = g.transform(lambda s: _roll_mean(s, 10))
    sym["last20_winpct"] = g.transform(lambda s: _roll_mean(s, 20))
    sym["career_n"]      = sym.groupby(["team", "format"]).cumcount()
    sym["days_rest"]     = sym.groupby(["team", "format"])["start_date"].diff().dt.days

    return sym[["match_id", "team",
                "last3_winpct", "last5_winpct", "last10_winpct", "last20_winpct",
                "career_n", "days_rest"]]


# ---------- head-to-head ----------

def compute_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """Pre-match H2H (team_home perspective) using strictly prior meetings."""
    m = matches.sort_values(["start_date", "match_id"]).reset_index(drop=True).copy()

    # canonical pair key
    pair_key = m.apply(lambda r: tuple(sorted([r["team_home"], r["team_away"]])), axis=1)
    m["pair_key"] = pair_key
    m["t1_won"] = np.where(m["winner"] == m["team_home"], 1.0,
                    np.where(m["winner"] == m["team_away"], 0.0, np.nan))

    out_rows = []
    pair_state: dict = {}    # pair -> list of (t1_label, t1_won)
    for r in m.itertuples(index=False):
        key = r.pair_key
        t1, t2 = r.team_home, r.team_away
        hist = pair_state.get(key, [])
        if hist:
            # convert each historical record to "from t1's POV"
            scores = []
            for (h_t1, h_won) in hist:
                if pd.isna(h_won):
                    continue
                scores.append(h_won if h_t1 == t1 else (1 - h_won))
            n = len(scores)
            wp = (sum(scores) / n) if n else np.nan
        else:
            n = 0
            wp = np.nan
        out_rows.append((r.match_id, n, wp))
        # update with this match
        if not pd.isna(r.t1_won):
            pair_state.setdefault(key, []).append((t1, r.t1_won))

    return pd.DataFrame(out_rows, columns=["match_id", "h2h_n_prior", "h2h_t1_winpct"])


# ---------- top-level builder ----------

CATEGORICAL = ["format", "team_home", "team_away", "venue"]
NUMERIC = [
    "toss_winner_is_t1", "toss_decision_is_bat",
    # form
    "t1_last3", "t1_last5", "t1_last10", "t1_last20",
    "t2_last3", "t2_last5", "t2_last10", "t2_last20",
    "form_diff_5", "form_diff_10",
    "t1_career_n", "t2_career_n",
    "t1_days_rest", "t2_days_rest",
    # elo
    "t1_elo_pre", "t2_elo_pre", "elo_diff_pre",
    # h2h
    "h2h_n_prior", "h2h_t1_winpct",
    # venue (as-of)
    "venue_n_prior", "venue_avg_first",
    "venue_bat1_winrate", "venue_toss_winner_winpct", "venue_bat_first_pct",
]


def compute_team_venue_form(matches: pd.DataFrame) -> pd.DataFrame:
    """For each (team, match) compute past win rate at the venue."""
    a = matches[["match_id", "start_date", "venue", "team_home", "team_away", "winner"]].copy()
    rows = []
    rows.append(a.rename(columns={"team_home": "team", "team_away": "opp"}).assign(
        won=lambda x: (x["winner"] == x["team"]).astype(float).where(x["winner"].notna())
    ))
    rows.append(a.rename(columns={"team_away": "team", "team_home": "opp"}).assign(
        won=lambda x: (x["winner"] == x["team"]).astype(float).where(x["winner"].notna())
    ))
    sym = pd.concat(rows, ignore_index=True).sort_values(["team", "venue", "start_date", "match_id"])
    g = sym.groupby(["team", "venue"], sort=False)["won"]
    sym["v_n"]    = sym.groupby(["team", "venue"]).cumcount()
    sym["v_wp"]   = g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    return sym[["match_id", "team", "v_n", "v_wp"]]


def derive_team_country(matches: pd.DataFrame) -> dict[str, str]:
    """For each team, the modal venue.country across its past matches.
    Used as 'home country' proxy for international and franchise teams.
    """
    a = matches[["venue_country", "team_home", "team_away"]].copy()
    a = a[a["venue_country"].notna()]
    long = pd.concat([
        a.rename(columns={"team_home": "team", "venue_country": "country"})[["team", "country"]],
        a.rename(columns={"team_away": "team", "venue_country": "country"})[["team", "country"]],
    ], ignore_index=True)
    counts = long.groupby(["team", "country"]).size().reset_index(name="n")
    idx = counts.groupby("team")["n"].idxmax()
    modal = counts.loc[idx, ["team", "country"]].set_index("team")["country"]
    return modal.to_dict()


def build_features(format_filter: list[str] | None = None,
                   keep_unfinished: bool = False) -> pd.DataFrame:
    """Build leakage-safe feature frame for the match-winner model."""
    install_views()
    con = connect()
    matches = con.execute("""
        SELECT match_id, format, competition, season, start_date, team_home, team_away,
               venue, country AS venue_country,
               toss_winner, toss_decision, winner,
               win_margin_runs, win_margin_wickets
        FROM matches
        WHERE start_date IS NOT NULL
    """).df()
    innings = con.execute("SELECT match_id, innings_no, total_runs FROM innings").df()
    con.close()

    if format_filter:
        matches = matches[matches["format"].isin(format_filter)].copy()
    matches["start_date"] = pd.to_datetime(matches["start_date"])

    elo     = compute_elo_features(matches, EloConfig(margin_weight=True))
    venue   = compute_venue_stats_asof(matches, innings)
    form    = compute_team_form(matches)
    h2h     = compute_h2h(matches)
    tv_form = compute_team_venue_form(matches)
    team_country = derive_team_country(matches)

    # widen team_form into t1 / t2 form frames
    f1 = form.rename(columns={
        "team":            "team_home",
        "last3_winpct":    "t1_last3",
        "last5_winpct":    "t1_last5",
        "last10_winpct":   "t1_last10",
        "last20_winpct":   "t1_last20",
        "career_n":        "t1_career_n",
        "days_rest":       "t1_days_rest",
    })
    f2 = form.rename(columns={
        "team":            "team_away",
        "last3_winpct":    "t2_last3",
        "last5_winpct":    "t2_last5",
        "last10_winpct":   "t2_last10",
        "last20_winpct":   "t2_last20",
        "career_n":        "t2_career_n",
        "days_rest":       "t2_days_rest",
    })

    df = matches.copy()
    df = df.merge(elo,   on="match_id", how="left")
    df = df.merge(venue, on="match_id", how="left")
    df = df.merge(f1,    on=["match_id", "team_home"], how="left")
    df = df.merge(f2,    on=["match_id", "team_away"], how="left")
    df = df.merge(h2h,   on="match_id", how="left")

    tv1 = tv_form.rename(columns={"team": "team_home", "v_n": "t1_venue_n", "v_wp": "t1_venue_winrate"})
    tv2 = tv_form.rename(columns={"team": "team_away", "v_n": "t2_venue_n", "v_wp": "t2_venue_winrate"})
    df = df.merge(tv1, on=["match_id", "team_home"], how="left")
    df = df.merge(tv2, on=["match_id", "team_away"], how="left")

    df["toss_winner_is_t1"]   = (df["toss_winner"] == df["team_home"]).astype(int)
    df["toss_decision_is_bat"] = (df["toss_decision"] == "bat").astype(int)
    df["form_diff_5"]  = df["t1_last5"]  - df["t2_last5"]
    df["form_diff_10"] = df["t1_last10"] - df["t2_last10"]

    df["year"] = df["start_date"].dt.year.astype("Int64")
    df["t1_country_proxy"] = df["team_home"].map(team_country)
    df["t2_country_proxy"] = df["team_away"].map(team_country)
    df["t1_is_home"] = (df["t1_country_proxy"] == df["venue_country"]).astype(int)
    df["t2_is_home"] = (df["t2_country_proxy"] == df["venue_country"]).astype(int)
    df["is_neutral_venue"] = ((df["t1_is_home"] == 0) & (df["t2_is_home"] == 0)).astype(int)

    # label
    df["y_t1_wins"] = np.where(df["winner"] == df["team_home"], 1,
                       np.where(df["winner"] == df["team_away"], 0, np.nan))

    before = len(df)
    if keep_unfinished:
        # keep the row but set y as NaN; downstream code filters with notna() before training
        df.attrs["dropped_unfinished"] = 0
    else:
        df = df.dropna(subset=["y_t1_wins"]).copy()
        df["y_t1_wins"] = df["y_t1_wins"].astype(int)
        df.attrs["dropped_unfinished"] = before - len(df)

    return df


def build_features_with_players(format_filter: list[str] | None = None,
                                 keep_unfinished: bool = False) -> pd.DataFrame:
    """build_features + per-team player aggregates (career SR/avg/econ)."""
    from .player_features import attach_player_features
    df = build_features(format_filter=format_filter, keep_unfinished=keep_unfinished)
    df = attach_player_features(df)
    return df


PLAYER_NUMERIC = [
    "t1_bat_career_sr", "t1_bat_form_sr", "t1_bat_career_avg",
    "t1_bowl_career_econ", "t1_bowl_career_avg",
    "t2_bat_career_sr", "t2_bat_form_sr", "t2_bat_career_avg",
    "t2_bowl_career_econ", "t2_bowl_career_avg",
    "diff_bat_career_sr", "diff_bat_form_sr", "diff_bowl_career_econ",
]


__all__ = [
    "build_features",
    "build_features_with_players",
    "compute_elo_features",
    "compute_venue_stats_asof",
    "compute_team_form",
    "compute_h2h",
    "CATEGORICAL",
    "NUMERIC",
    "PLAYER_NUMERIC",
    "EloConfig",
]
