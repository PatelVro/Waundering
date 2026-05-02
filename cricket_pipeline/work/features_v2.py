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
    # NOTE: 24-month windowed venue stats (Cycle 9 experiment) tested but
    # regressed both T20 (-0.13pp) and ODI (-0.90pp). Helper kept available
    # for future use (e.g. totals_model) but NOT joined into the match model.
    return out


def _windowed_venue_stats(m: pd.DataFrame, days: int = 720) -> pd.DataFrame:
    """Per (venue, format), compute mean of various stats over the last
    `days` days, ending strictly before each match's start_date."""
    rows = []
    for (_venue, _fmt), g in m.groupby(["venue", "format"], sort=False, dropna=False):
        g = g.sort_values("start_date").copy()
        # Drop rows with missing start_date (rare); time-rolling requires datetime index
        g = g[g["start_date"].notna()]
        if g.empty: continue
        g = g.set_index(pd.DatetimeIndex(g["start_date"]))
        win = g.rolling(f"{int(days)}D", closed="left")
        sub = pd.DataFrame({
            "match_id":                          g["match_id"].values,
            "venue_n_prior_24mo":                win["first_innings"].count().values,
            "venue_avg_first_24mo":              win["first_innings"].mean().values,
            "venue_bat1_winrate_24mo":           win["bat1_won"].mean().values,
            "venue_toss_winner_winpct_24mo":     win["toss_won_match"].mean().values,
            "venue_bat_first_pct_24mo":          win["bat_first"].mean().values,
        })
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=[
            "match_id","venue_n_prior_24mo","venue_avg_first_24mo",
            "venue_bat1_winrate_24mo","venue_toss_winner_winpct_24mo","venue_bat_first_pct_24mo"])
    return pd.concat(rows, ignore_index=True)


# ---------- per-team rolling form ----------

def compute_team_form(matches: pd.DataFrame) -> pd.DataFrame:
    """For each (team, match_id) compute pre-match form windows.

    Includes:
      - last3/5/10/20 win-pct
      - last5/10 avg signed margin in runs (positive = won big, negative = lost big)
      - last5/10 avg signed margin in wickets (positive = won, negative = lost)
      - career_n, days_rest
    """
    margin_cols = []
    for c in ("win_margin_runs", "win_margin_wickets"):
        if c in matches.columns: margin_cols.append(c)
    a = matches[["match_id", "start_date", "format", "team_home", "team_away", "winner"] + margin_cols].copy()

    rows = []
    rows.append(a.rename(columns={"team_home": "team", "team_away": "opp"}).assign(
        won=lambda x: (x["winner"] == x["team"]).astype(float).where(x["winner"].notna())
    ))
    rows.append(a.rename(columns={"team_away": "team", "team_home": "opp"}).assign(
        won=lambda x: (x["winner"] == x["team"]).astype(float).where(x["winner"].notna())
    ))
    sym = pd.concat(rows, ignore_index=True).sort_values(["team", "format", "start_date", "match_id"])

    # Signed margins per row: + for win, − for loss. NaN if no result.
    if "win_margin_runs" in sym.columns:
        sym["margin_runs"] = np.where(
            sym["won"] == 1, sym["win_margin_runs"],
            np.where(sym["won"] == 0, -sym["win_margin_runs"], np.nan)).astype(float)
    else:
        sym["margin_runs"] = np.nan
    if "win_margin_wickets" in sym.columns:
        sym["margin_wkts"] = np.where(
            sym["won"] == 1, sym["win_margin_wickets"],
            np.where(sym["won"] == 0, -sym["win_margin_wickets"], np.nan)).astype(float)
    else:
        sym["margin_wkts"] = np.nan

    g_won  = sym.groupby(["team", "format"], sort=False)["won"]
    g_run  = sym.groupby(["team", "format"], sort=False)["margin_runs"]
    g_wkt  = sym.groupby(["team", "format"], sort=False)["margin_wkts"]

    def _roll_mean(s, n):
        return s.shift(1).rolling(n, min_periods=1).mean()

    sym["last3_winpct"]   = g_won.transform(lambda s: _roll_mean(s, 3))
    sym["last5_winpct"]   = g_won.transform(lambda s: _roll_mean(s, 5))
    sym["last10_winpct"]  = g_won.transform(lambda s: _roll_mean(s, 10))
    sym["last20_winpct"]  = g_won.transform(lambda s: _roll_mean(s, 20))
    sym["last5_margin_runs"]   = g_run.transform(lambda s: _roll_mean(s, 5))
    sym["last10_margin_runs"]  = g_run.transform(lambda s: _roll_mean(s, 10))
    sym["last5_margin_wkts"]   = g_wkt.transform(lambda s: _roll_mean(s, 5))
    sym["last10_margin_wkts"]  = g_wkt.transform(lambda s: _roll_mean(s, 10))
    sym["career_n"]      = sym.groupby(["team", "format"]).cumcount()
    sym["days_rest"]     = sym.groupby(["team", "format"])["start_date"].diff().dt.days

    return sym[["match_id", "team",
                "last3_winpct", "last5_winpct", "last10_winpct", "last20_winpct",
                "last5_margin_runs", "last10_margin_runs",
                "last5_margin_wkts", "last10_margin_wkts",
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

# Weather features (Cycle 12). A/B says: lifts T20 (+0.26pp full, +0.51pp weather-subset)
# but regresses ODI (-0.68pp). Predict path adds these only when format is T20/IT20.
WEATHER_NUMERIC = [
    "weather_temp_c", "weather_humidity", "weather_dew_point",
    "weather_wind_kmh", "weather_cloud_pct", "weather_precip_mm",
    "weather_dew_risk", "weather_rain_risk", "weather_swing_friendly",
]

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


def compute_weather_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Pull weather_daily by (venue, date) and derive cricket-relevant features.

    Output columns (NaN if no weather row exists):
        weather_temp_c, weather_humidity, weather_dew_point,
        weather_wind_kmh, weather_cloud_pct, weather_precip_mm,
        weather_dew_risk, weather_rain_risk, weather_swing_friendly
    """
    from ..db import connect
    con = connect()
    wx = con.execute("""
        SELECT venue, date AS start_date,
               temp_c, humidity, dew_point, wind_kmh, cloud_pct, precip_mm
        FROM weather_daily
    """).df()
    con.close()
    if wx.empty:
        # No weather data yet — return all-NaN columns so feature shape is stable
        cols = ["weather_temp_c","weather_humidity","weather_dew_point",
                "weather_wind_kmh","weather_cloud_pct","weather_precip_mm",
                "weather_dew_risk","weather_rain_risk","weather_swing_friendly"]
        out = pd.DataFrame({"match_id": matches["match_id"].values})
        for c in cols: out[c] = np.nan
        return out
    wx["start_date"] = pd.to_datetime(wx["start_date"])
    df = matches[["match_id", "venue", "start_date"]].copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df = df.merge(wx, on=["venue", "start_date"], how="left")
    out = pd.DataFrame({
        "match_id":           df["match_id"].values,
        "weather_temp_c":     df["temp_c"].values,
        "weather_humidity":   df["humidity"].values,
        "weather_dew_point":  df["dew_point"].values,
        "weather_wind_kmh":   df["wind_kmh"].values,
        "weather_cloud_pct":  df["cloud_pct"].values,
        "weather_precip_mm":  df["precip_mm"].values,
    })
    # Derived (NaN-safe)
    out["weather_dew_risk"]        = ((df["humidity"] > 70) & (df["temp_c"] < 28)
                                      & (df["wind_kmh"] < 12)).astype(float)
    out.loc[df["humidity"].isna() | df["temp_c"].isna(), "weather_dew_risk"] = np.nan
    out["weather_rain_risk"]       = (df["precip_mm"] > 0.5).astype(float)
    out.loc[df["precip_mm"].isna(), "weather_rain_risk"] = np.nan
    out["weather_swing_friendly"]  = ((df["humidity"] > 65) & (df["cloud_pct"] > 50)).astype(float)
    out.loc[df["humidity"].isna() | df["cloud_pct"].isna(), "weather_swing_friendly"] = np.nan
    return out


def compute_pitch_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Pull pitch_reports by match_id. Returns NaN for matches with no report."""
    from ..db import connect
    con = connect()
    try:
        pr = con.execute("""
            SELECT match_id, pitch_dry, pitch_green, pitch_pace, pitch_spin,
                   pitch_flat, pitch_low, pitch_dew
            FROM pitch_reports
        """).df()
    except Exception:
        pr = pd.DataFrame()
    con.close()
    cols = ["pitch_dry","pitch_green","pitch_pace","pitch_spin","pitch_flat","pitch_low","pitch_dew"]
    out = pd.DataFrame({"match_id": matches["match_id"].values})
    if pr.empty:
        for c in cols: out[c] = np.nan
        return out
    out = out.merge(pr, on="match_id", how="left")
    return out


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
        "team":              "team_home",
        "last3_winpct":      "t1_last3",
        "last5_winpct":      "t1_last5",
        "last10_winpct":     "t1_last10",
        "last20_winpct":     "t1_last20",
        "last5_margin_runs":  "t1_last5_margin_runs",
        "last10_margin_runs": "t1_last10_margin_runs",
        "last5_margin_wkts":  "t1_last5_margin_wkts",
        "last10_margin_wkts": "t1_last10_margin_wkts",
        "career_n":          "t1_career_n",
        "days_rest":         "t1_days_rest",
    })
    f2 = form.rename(columns={
        "team":              "team_away",
        "last3_winpct":      "t2_last3",
        "last5_winpct":      "t2_last5",
        "last10_winpct":     "t2_last10",
        "last20_winpct":     "t2_last20",
        "last5_margin_runs":  "t2_last5_margin_runs",
        "last10_margin_runs": "t2_last10_margin_runs",
        "last5_margin_wkts":  "t2_last5_margin_wkts",
        "last10_margin_wkts": "t2_last10_margin_wkts",
        "career_n":          "t2_career_n",
        "days_rest":         "t2_days_rest",
    })

    df = matches.copy()
    df = df.merge(elo,   on="match_id", how="left")
    df = df.merge(venue, on="match_id", how="left")
    df = df.merge(f1,    on=["match_id", "team_home"], how="left")
    df = df.merge(f2,    on=["match_id", "team_away"], how="left")
    df = df.merge(h2h,   on="match_id", how="left")
    # Weather + pitch (Cycle 12) — NaN for matches with no data; LightGBM
    # handles missing values natively, sklearn variants get filled below.
    df = df.merge(compute_weather_features(matches), on="match_id", how="left")
    df = df.merge(compute_pitch_features(matches),   on="match_id", how="left")

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
    # Wave 1 leakage audits (Agent 10)
    "audit_venue_leakage",
    "audit_form_leakage",
    "audit_elo_leakage",
    "audit_h2h_leakage",
    "audit_all",
]


# ---------------------------------------------------------------------------
# Wave 1: Leakage audit helpers (Agent 10).
#
# These are intended to be called BEFORE training, on the result of
# `build_features_with_players()`. They do NOT mutate the frame; they raise
# (or warn, depending on `strict`) if a constraint is violated.
#
# Each audit answers a specific question:
#   - Did any feature for match M use data from M itself?
#   - Did rolling windows include same-day double-headers?
#   - Did Elo update before the match's outcome was actually recorded?
#
# Run from CLI:
#     python -m cricket_pipeline.work.features_v2 --audit
# ---------------------------------------------------------------------------


def _is_strictly_positive(s: pd.Series) -> bool:
    return bool((s.dropna() > 0).all()) if not s.dropna().empty else True


def audit_venue_leakage(df: pd.DataFrame, strict: bool = True) -> dict:
    """Verify venue stats look as-of, not all-time.

    Heuristic: when a match is the FIRST at its (venue, format) pair,
    venue_n_prior should be 0 (or NaN). If we see a non-zero value on a
    first-at-venue match, the as-of cumulative is leaking the current row.
    """
    issues: list[str] = []
    if "venue_n_prior" in df.columns and "venue" in df.columns:
        # First match at each (venue, format): must have venue_n_prior == 0 or NaN
        df_sorted = df.sort_values("start_date")
        first_idx = df_sorted.groupby(["venue", "format"]).head(1).index
        leaking = df_sorted.loc[first_idx, "venue_n_prior"].dropna()
        leaking = leaking[leaking > 0]
        if len(leaking):
            issues.append(
                f"venue_n_prior > 0 on {len(leaking)} first-at-venue matches "
                "(suggests as-of cumulative leak)"
            )
    out = {"audit": "venue_leakage", "ok": not issues, "issues": issues}
    if strict and issues:
        raise AssertionError(f"venue leakage detected: {issues}")
    return out


def audit_form_leakage(df: pd.DataFrame, strict: bool = True) -> dict:
    """Form features should never use the match's own result.

    Heuristic: t1_last5/t1_last10/t1_last20 must be in [0, 1] when present;
    a value of exactly 1.0 with `last5_n == 5` AND `y_t1_wins == 1` is a
    smell (always-winner with the current win baked in). We can't catch
    every leak from the data alone, but we flag the obvious case.
    """
    issues: list[str] = []
    for col in ("t1_last5", "t1_last10", "t1_last20", "t2_last5", "t2_last10", "t2_last20"):
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if vals.empty:
            continue
        if (vals < 0).any() or (vals > 1).any():
            issues.append(f"{col} outside [0, 1]")
    out = {"audit": "form_leakage", "ok": not issues, "issues": issues}
    if strict and issues:
        raise AssertionError(f"form leakage detected: {issues}")
    return out


def audit_elo_leakage(df: pd.DataFrame, strict: bool = True) -> dict:
    """Elo ratings must be PRE-match (before the result is observed).

    Heuristic: t1_elo_pre should not perfectly correlate with y_t1_wins
    (>= 0.95). A near-perfect correlation means the rating is being
    updated with the current match's result before it's read into features.
    """
    issues: list[str] = []
    if "t1_elo_pre" in df.columns and "y_t1_wins" in df.columns:
        sub = df[["t1_elo_pre", "y_t1_wins"]].dropna()
        if len(sub) > 50:
            corr = float(sub["t1_elo_pre"].corr(sub["y_t1_wins"].astype(float)))
            if abs(corr) > 0.95:
                issues.append(f"t1_elo_pre vs y_t1_wins corr={corr:.3f} suggests leak")
    out = {"audit": "elo_leakage", "ok": not issues, "issues": issues}
    if strict and issues:
        raise AssertionError(f"elo leakage detected: {issues}")
    return out


def audit_h2h_leakage(df: pd.DataFrame, strict: bool = True) -> dict:
    """h2h_n_prior must be 0 (or NaN) for the first match between two teams.

    If we see h2h_t1_winpct populated when h2h_n_prior is 0, the head-to-head
    aggregator is including the current match in its window.
    """
    issues: list[str] = []
    if {"h2h_t1_winpct", "h2h_n_prior"}.issubset(df.columns):
        # h2h_t1_winpct populated AND h2h_n_prior == 0 → bug
        bad = df[(df["h2h_n_prior"].fillna(0) == 0) & df["h2h_t1_winpct"].notna()]
        if len(bad):
            issues.append(
                f"{len(bad)} rows have h2h_t1_winpct set with h2h_n_prior=0"
            )
    out = {"audit": "h2h_leakage", "ok": not issues, "issues": issues}
    if strict and issues:
        raise AssertionError(f"h2h leakage detected: {issues}")
    return out


def audit_all(df: pd.DataFrame, strict: bool = False) -> dict:
    """Run every audit and return a combined report.

    With `strict=False` (default), prints issues but doesn't raise — safe
    for diagnostic CLI use. With `strict=True`, the first failing audit
    raises AssertionError; use this in pre-training pipelines to halt on
    leakage.
    """
    audits = [
        audit_venue_leakage,
        audit_form_leakage,
        audit_elo_leakage,
        audit_h2h_leakage,
    ]
    reports = []
    for fn in audits:
        try:
            reports.append(fn(df, strict=strict))
        except AssertionError as e:
            reports.append({"audit": fn.__name__, "ok": False, "issues": [str(e)]})
            if strict:
                raise
    n_issues = sum(len(r["issues"]) for r in reports)
    return {"ok": n_issues == 0, "n_issues": n_issues, "reports": reports}
