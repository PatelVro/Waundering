"""Fetch live (or completed) match state from Cricbuzz and emit a JSON blob
the dashboard can render.

Output schema matches what script.js's hydrateLive() expects:
{
  "match_id":      "151902",
  "url":           "...",
  "status":        "Sunrisers Hyderabad won by 5 wkts",
  "is_complete":   true,
  "fetched_at":    "2026-04-25T22:47:01Z",
  "home":          "Rajasthan Royals",
  "away":          "Sunrisers Hyderabad",
  "batting_team":  "Sunrisers Hyderabad",
  "bowling_team":  "Rajasthan Royals",
  "score":         "229/5",
  "overs":         "18.3",
  "current_rr":    12.43,
  "required_rr":   null,
  "target":        229,
  "rem_runs":      0,
  "striker":       {"name":"...", "runs":8, "balls":3, "fours":1, "sixes":0},
  "non_striker":   {"name":"...", "runs":..., "balls":...},
  "bowler":        {"name":"...", "overs":"3.3", "runs":34, "wickets":1},
  "last_overs":    "...",
  "innings": [
     {"team":"RR","score":"228/6","overs":"19.6"},
     {"team":"SRH","score":"229/5","overs":"18.3"}
  ],
  "live_prediction": {
     "mode": "chase",
     "win_prob": 0.92,
     "balls_remaining": 9,
     "p10": 215, "p50": 230, "p90": 245, "mean": 230,
     "n_sim": 5000,
     "venue_avg": 186
  }
}

Usage:
  python -m cricket_pipeline.work.live_match \\
     --cricbuzz-id 151902 \\
     --out cricket_pipeline/work/runs/live_match.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0)",
    "Accept":     "text/html,application/xhtml+xml,application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

ROOT      = Path(__file__).resolve().parents[2]
RUNS_DIR  = ROOT / "cricket_pipeline" / "work" / "runs"


def _live_match_url(match_id: str) -> str:
    return f"https://www.cricbuzz.com/live-cricket-scores/{match_id}"


_SHORT_CODE_TO_NAME = {
    # IPL
    "RR":   "Rajasthan Royals",
    "SRH":  "Sunrisers Hyderabad",
    "CSK":  "Chennai Super Kings",
    "GT":   "Gujarat Titans",
    "LSG":  "Lucknow Super Giants",
    "KKR":  "Kolkata Knight Riders",
    "MI":   "Mumbai Indians",
    "RCB":  "Royal Challengers Bengaluru",
    "DC":   "Delhi Capitals",
    "PBKS": "Punjab Kings",
    # PSL
    "LHQ":  "Lahore Qalandars",
    "PSZ":  "Peshawar Zalmi",
    "QTG":  "Quetta Gladiators",
    "KRK":  "Karachi Kings",
    "ISU":  "Islamabad United",
    "MS":   "Multan Sultans",
    "HYDK": "Hyderabad Kingsmen",
    "RWP":  "Rawalpindiz",
    # Internationals
    "IND":  "India",   "INDW": "India Women",
    "AUS":  "Australia", "AUSW": "Australia Women",
    "ENG":  "England", "ENGW": "England Women",
    "PAK":  "Pakistan",
    "RSA":  "South Africa", "RSAW": "South Africa Women",
    "NZ":   "New Zealand",
    "SL":   "Sri Lanka",
    "BAN":  "Bangladesh",
    "WI":   "West Indies",
    "AFG":  "Afghanistan",
    "IRE":  "Ireland",
    "SCO":  "Scotland",
    "NEP":  "Nepal",
    "UAE":  "United Arab Emirates",
    "USA":  "United States of America",
    "NAM":  "Namibia",
    "OMN":  "Oman",
    "ZIM":  "Zimbabwe",
}


def _team_name_from_short(short: str) -> str | None:
    if not short: return None
    return _SHORT_CODE_TO_NAME.get(short.upper())


# Hardcoded home-venue map, keyed by short team code. Used as a last-ditch
# fallback when Cricbuzz's matchHeader.matchVenue is None and no venueInfo
# block can be matched to the requested match_id. Only covers leagues where
# every franchise has a stable home ground — IPL (extend cautiously to BBL/
# CPL/etc., where home grounds rotate, only after verification).
_HOME_VENUES_BY_LEAGUE = {
    "indian-premier-league": {
        "csk":  "MA Chidambaram Stadium, Chennai",
        "mi":   "Wankhede Stadium, Mumbai",
        "rcb":  "M Chinnaswamy Stadium, Bengaluru",
        "kkr":  "Eden Gardens, Kolkata",
        "dc":   "Arun Jaitley Stadium, Delhi",
        "rr":   "Sawai Mansingh Stadium, Jaipur",
        "pbks": "Maharaja Yadavindra Singh International Cricket Stadium, "
                "Mullanpur",
        "srh":  "Rajiv Gandhi International Stadium, Hyderabad",
        "gt":   "Narendra Modi Stadium, Ahmedabad",
        "lsg":  "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket "
                "Stadium, Lucknow",
    },
}


# Aliases per league: any of these substrings in the slug count as a hit.
# IPL slugs sometimes use the long form, sometimes "ipl-2026", and the slug
# ordering of the team tokens itself isn't always reliable (Cricbuzz can
# flip home/away between scheduled and live state on the same match_id).
_LEAGUE_ALIASES = {
    "indian-premier-league": ("indian-premier-league", "ipl-"),
}


def _venue_from_slug(slug: str | None) -> str | None:
    """Recover a likely venue from a Cricbuzz match slug for known leagues.

    Slugs look like 'gt-vs-rcb-44th-match-indian-premier-league-2026' or
    'csk-vs-mi-37th-match-ipl-2026'. The first team token is treated as
    the home side; the league name is encoded near the end. Final
    fallback after page-scraping has failed — reliable for IPL home
    games, but Cricbuzz occasionally swaps the team order on the same
    match_id, so a small fraction of fixtures will return the wrong
    home team's ground (still better than 'Unknown Venue' for the
    feature builder)."""
    if not slug:
        return None
    s = slug.lower()
    for league, venues in _HOME_VENUES_BY_LEAGUE.items():
        if any(alias in s for alias in _LEAGUE_ALIASES.get(league, (league,))):
            m = re.match(r"^([a-z0-9]+)-vs-[a-z0-9]+", s)
            if m:
                return venues.get(m.group(1))
    return None


def find_live_match_by_teams(home_keywords: list[str], away_keywords: list[str]) -> dict | None:
    """Scan Cricbuzz live-scores page for a match whose slug mentions both teams."""
    r = requests.get("https://www.cricbuzz.com/cricket-match/live-scores", headers=HEADERS, timeout=15)
    r.raise_for_status()
    matches = re.findall(r"/live-cricket-scores/(\d+)/([a-z0-9-]+)", r.text)
    seen = set()
    for mid, slug in matches:
        if mid in seen: continue
        seen.add(mid)
        slow = slug.lower()
        if any(k in slow for k in home_keywords) and any(k in slow for k in away_keywords):
            return {"match_id": mid, "slug": slug, "url": f"https://www.cricbuzz.com/live-cricket-scores/{mid}/{slug}"}
    return None


def fetch_match_state(match_id: str, slug: str | None = None) -> dict:
    """Fetch the Next.js flight data for a match and parse out the live state."""
    suffix = "/" + slug if slug else ""
    url = f"https://www.cricbuzz.com/live-cricket-scores/{match_id}{suffix}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    # Cricbuzz uses Next.js — flight data is in self.__next_f.push chunks
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.+?)"\]\)', r.text, re.S)
    joined = "".join(chunks).encode("utf-8").decode("unicode_escape", errors="replace")

    # Pull the matchHeader and miniscore via brace matching
    out = {
        "match_id":   match_id,
        "url":        url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    # The page may contain MULTIPLE matchHeader blocks (live banner + upcoming
    # cards + the actual match page). Find the one whose matchId matches the
    # URL we requested.
    mh = _find_matchheader_for_id(joined, match_id) or _extract_object(joined, '"matchHeader":')
    if mh:
        out["status"]     = mh.get("status")
        out["short_status"] = mh.get("shortStatus")
        out["state"]      = mh.get("state")
        out["complete"]   = bool(mh.get("complete"))
        out["match_format"] = mh.get("matchFormat")
        out["match_description"] = mh.get("matchDescription")
        out["match_type"] = mh.get("matchType")
        # matchStartTimestamp is in MILLISECONDS UTC. Convert to seconds for
        # the phase machine (`start_ts`). Some upcoming-card matchHeaders
        # also expose `tossStartTimestamp`; we don't use it (toss start is
        # implied by the regular toss-detection path).
        mst = mh.get("matchStartTimestamp")
        if isinstance(mst, (int, float)) and mst > 0:
            out["match_start_ts"] = int(mst) // 1000
        t1 = mh.get("team1") or {}; t2 = mh.get("team2") or {}
        out["team1_name"] = t1.get("teamName"); out["team1_short"] = t1.get("teamSName")
        out["team2_name"] = t2.get("teamName"); out["team2_short"] = t2.get("teamSName")
        toss = mh.get("tossResults") or {}
        if toss:
            out["toss_winner"]   = toss.get("tossWinnerName")
            out["toss_decision"] = toss.get("decision")
        venue = mh.get("matchVenue") or {}
        out["venue"] = venue.get("name") or venue.get("ground")

    # Fallback: regex-pluck team names. Scope to the *matchHeader* slice only —
    # Cricbuzz's page contains a "live now" banner (currently RR vs SRH) that
    # would otherwise dominate a global regex.
    mh_idx = joined.find('"matchHeader"')
    mh_slice = ""
    if mh_idx >= 0:
        # take a generous window after matchHeader; matchHeader objects are <10KB
        mh_slice = joined[mh_idx: mh_idx + 50_000]

    def _scoped(pat, scope=mh_slice or joined):
        return re.search(pat, scope)

    if not out.get("team1_name") and mh_slice:
        m = _scoped(r'"team1"\s*:\s*\{[^}]*?"teamName"\s*:\s*"([^"]+)"[^}]*?"teamSName"\s*:\s*"([^"]+)"')
        if m:
            out["team1_name"], out["team1_short"] = m.group(1), m.group(2)
    if not out.get("team2_name") and mh_slice:
        m = _scoped(r'"team2"\s*:\s*\{[^}]*?"teamName"\s*:\s*"([^"]+)"[^}]*?"teamSName"\s*:\s*"([^"]+)"')
        if m:
            out["team2_name"], out["team2_short"] = m.group(1), m.group(2)

    # Last resort: derive short codes from the slug ("csk-vs-gt-...") and map via DB
    if (not out.get("team1_name") or not out.get("team2_name")) and slug:
        m = re.match(r'([a-z0-9]+)-vs-([a-z0-9]+)', slug.lower())
        if m:
            t1c, t2c = m.group(1).upper(), m.group(2).upper()
            n1 = _team_name_from_short(t1c)
            n2 = _team_name_from_short(t2c)
            if n1 and not out.get("team1_name"):
                out["team1_name"] = n1; out["team1_short"] = t1c
            if n2 and not out.get("team2_name"):
                out["team2_name"] = n2; out["team2_short"] = t2c

    if not out.get("venue"):
        # First try a matchInfo block matching this match_id (it has venueInfo)
        idx = joined.find(f'"matchId":{int(match_id)}')
        if idx > 0:
            window = joined[idx: idx + 4000]
            mv = re.search(r'"venueInfo"\s*:\s*\{[^}]*?"ground"\s*:\s*"([^"]+)"', window)
            if mv:
                out["venue"] = mv.group(1)
                # also try city
                mc = re.search(r'"venueInfo"\s*:\s*\{[^}]*?"city"\s*:\s*"([^"]+)"', window)
                if mc:
                    out["venue_city"] = mc.group(1)
                    if mc.group(1) and mc.group(1) not in mv.group(1):
                        out["venue"] = f"{mv.group(1)}, {mc.group(1)}"
        if not out.get("venue") and mh_slice:
            m = re.search(r'"matchVenue"\s*:\s*\{[^}]*?"name"\s*:\s*"([^"]+)"', mh_slice)
            if m: out["venue"] = m.group(1)
        if not out.get("venue"):
            # Last resort: hardcoded home-venue lookup keyed off the slug.
            # Reliable for IPL (each franchise has a stable home ground);
            # may be wrong for neutral venues or leagues with rotating
            # grounds, but better than the alternative — we tried a "first
            # venueInfo in page" fallback and it returned the live-NOW
            # match's venue for every other match's page, since Cricbuzz
            # shares miniscore widgets at the top of every match URL.
            v = _venue_from_slug(slug)
            if v:
                out["venue"] = v
    if not out.get("toss_winner") and mh_slice:
        mw = _scoped(r'"tossWinnerName"\s*:\s*"([^"]+)"')
        md = _scoped(r'"decision"\s*:\s*"([^"]+)"')
        if mw: out["toss_winner"] = mw.group(1)
        if md: out["toss_decision"] = md.group(1)

    # Only use miniscore if the match is in progress / complete; for upcoming
    # matches the page may contain a stale miniscore from the live banner —
    # detect that by checking if the bat_team_id matches one of our team IDs.
    mh_t1_id = (mh or {}).get("team1", {}).get("teamId")
    mh_t2_id = (mh or {}).get("team2", {}).get("teamId")
    ms = _extract_object(joined, '"miniscore":')
    if ms and mh_t1_id and mh_t2_id:
        bat_id = (ms.get("batTeam") or {}).get("teamId")
        if bat_id and bat_id not in (mh_t1_id, mh_t2_id):
            ms = None    # belongs to a different match (the page banner)
    if ms:
        bat = ms.get("batTeam") or {}
        out["bat_team_id"]    = bat.get("teamId")
        out["bat_team_score"] = bat.get("teamScore")
        out["bat_team_wkts"]  = bat.get("teamWkts")
        out["overs"]          = ms.get("overs")
        out["current_rr"]     = ms.get("currentRunRate")
        out["required_rr"]    = ms.get("requiredRunRate")
        out["target"]         = ms.get("target")
        out["rem_runs"]       = ms.get("remRuns")
        out["rem_balls"]      = ms.get("remBalls")
        out["last_overs"]     = ms.get("recentOvsStats")
        bs = ms.get("batsmanStriker")    or {}
        bn = ms.get("batsmanNonStriker") or {}
        bw = ms.get("bowlerStriker")     or {}
        out["striker"]     = _player(bs, "bat")
        out["non_striker"] = _player(bn, "bat")
        out["bowler"]      = _player(bw, "bowl")

    # Multi-innings totals
    score_obj = _extract_object(joined, '"matchScore":')
    if score_obj:
        out["match_score"] = score_obj
        # build innings list
        inn = []
        for tk in ("team1Score", "team2Score"):
            ts = score_obj.get(tk) or {}
            for ik in ("inngs1", "inngs2"):
                ii = ts.get(ik)
                if ii:
                    inn.append({
                        "team_key":  tk,
                        "innings":   ik,
                        "runs":      ii.get("runs"),
                        "wickets":   ii.get("wickets"),
                        "overs":     ii.get("overs"),
                        "isDeclared": ii.get("isDeclared"),
                    })
        if inn:
            out["innings_summary"] = inn

    return out


def _player(p: dict, kind: str) -> dict:
    if not p: return {}
    if kind == "bat":
        return {
            "name":   p.get("name") or p.get("batName"),
            "runs":   p.get("runs") or p.get("batRuns"),
            "balls":  p.get("balls") or p.get("batBalls"),
            "fours":  p.get("fours") or p.get("batFours"),
            "sixes":  p.get("sixes") or p.get("batSixes"),
            "strike_rate": p.get("strikeRate"),
        }
    return {
        "name":    p.get("name") or p.get("bowlName"),
        "overs":   p.get("overs") or p.get("bowlOvs"),
        "runs":    p.get("runs")  or p.get("bowlRuns"),
        "wickets": p.get("wickets") or p.get("bowlWkts"),
        "economy": p.get("economy") or p.get("bowlEcon"),
    }


def _find_matchheader_for_id(text: str, match_id: str) -> dict | None:
    """Walk every '"matchHeader":' occurrence and return the one whose
    matchId matches `match_id`. Falls back to None if not found."""
    pos = 0
    target = f'"matchId":{int(match_id)}'  # int to avoid quotes
    while True:
        idx = text.find('"matchHeader":', pos)
        if idx < 0: return None
        obj = _extract_object(text[idx:], '"matchHeader":')
        # Quick check: does the raw blob contain the matchId int?
        end = idx + 50_000
        if target in text[idx:end] and obj:
            return obj
        pos = idx + 14


def _extract_object(text: str, marker: str) -> dict | None:
    """Find `marker` and brace-match the JSON object that follows. Returns dict
    or None if not found / not parseable."""
    idx = text.find(marker)
    if idx < 0: return None
    start = text.find("{", idx)
    if start < 0: return None
    depth = 0; in_str = False; esc = False
    for i in range(start, min(len(text), start + 200_000)):
        c = text[i]
        if esc: esc = False; continue
        if c == "\\" and in_str: esc = True; continue
        if c == '"': in_str = not in_str; continue
        if in_str: continue
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i+1]
                try:
                    return json.loads(blob)
                except Exception:
                    return None
    return None


def normalise_for_dashboard(state: dict) -> dict:
    """Translate Cricbuzz-flavoured state into the script.js schema."""
    if not state.get("match_id"):
        return {}
    home = state.get("team1_name"); away = state.get("team2_name")
    score_obj = state.get("match_score") or {}
    t1_score = (score_obj.get("team1Score") or {})
    t2_score = (score_obj.get("team2Score") or {})
    # Determine batting team by matching bat_team_score against each innings runs
    bts = state.get("bat_team_score")
    btw = state.get("bat_team_wkts")
    batting_team = home; bowling_team = away
    def _matches(inn, runs, wkts):
        return inn and inn.get("runs") == runs and (wkts is None or inn.get("wickets") == wkts)
    if t2_score.get("inngs2") and _matches(t2_score["inngs2"], bts, btw):
        batting_team = away; bowling_team = home
    elif t1_score.get("inngs2") and _matches(t1_score["inngs2"], bts, btw):
        batting_team = home; bowling_team = away
    elif _matches(t2_score.get("inngs1"), bts, btw):
        batting_team = away; bowling_team = home
    elif _matches(t1_score.get("inngs1"), bts, btw):
        batting_team = home; bowling_team = away
    elif t2_score.get("inngs2"):
        batting_team = away; bowling_team = home
    elif t1_score.get("inngs2"):
        batting_team = home; bowling_team = away

    score = "—"
    if state.get("bat_team_score") is not None:
        score = f"{state['bat_team_score']}/{state.get('bat_team_wkts', 0)}"

    out = {
        "match_id":      state["match_id"],
        "url":           state.get("url"),
        "status":        state.get("status") or state.get("short_status"),
        "is_complete":   bool(state.get("complete")) or (state.get("state") == "Complete"),
        "fetched_at":    state.get("fetched_at"),
        "home":          home,
        "away":          away,
        "batting_team":  batting_team,
        "bowling_team":  bowling_team,
        "venue":         state.get("venue"),
        "match_format":  state.get("match_format"),
        "match_start_ts": state.get("match_start_ts"),    # epoch seconds UTC; phase machine input
        "toss_winner":   state.get("toss_winner"),
        "toss_decision": state.get("toss_decision"),
        "score":         score,
        "overs":         state.get("overs"),
        "current_rr":    state.get("current_rr"),
        "required_rr":   state.get("required_rr"),
        "target":        state.get("target"),
        "rem_runs":      state.get("rem_runs"),
        "rem_balls":     state.get("rem_balls"),
        "striker":       state.get("striker") or {},
        "non_striker":   state.get("non_striker") or {},
        "bowler":        state.get("bowler") or {},
        "last_overs":    state.get("last_overs"),
        "innings":       _innings_for_dashboard(state, home, away),
    }

    # In-play prediction
    out["live_prediction"] = compute_live_prediction(state, out, batting_team, bowling_team)
    return out


def _innings_for_dashboard(state: dict, home: str, away: str) -> list:
    score_obj = state.get("match_score") or {}
    out = []
    for team, key in [(home, "team1Score"), (away, "team2Score")]:
        ts = score_obj.get(key) or {}
        for ik in ("inngs1", "inngs2"):
            ii = ts.get(ik)
            if not ii: continue
            out.append({
                "team":    team,
                "innings": ik,
                "score":   f"{ii.get('runs')}/{ii.get('wickets')}",
                "overs":   ii.get("overs"),
            })
    return out


def compute_live_prediction(state: dict, dash: dict, batting_team: str, bowling_team: str) -> dict:
    """If chasing (target known), use venue-avg + current state to project win prob.
    Otherwise, project final 1st-innings score."""
    target  = state.get("target")
    rem_balls = state.get("rem_balls")
    cur_runs = state.get("bat_team_score")
    overs    = state.get("overs")
    if state.get("complete"):
        # Match is over; encode the result as a deterministic "prediction"
        winner = state.get("status", "")
        win_for_bat = 1.0 if (winner and ("won" in winner.lower()) and (batting_team and batting_team.split()[0].lower() in winner.lower())) else 0.0
        return {
            "mode":        "chase" if target else "set_score",
            "win_prob":    win_for_bat if target else None,
            "p50":         cur_runs,
            "p10":         cur_runs,
            "p90":         cur_runs,
            "mean":        cur_runs,
            "n_sim":       0,
            "balls_remaining": rem_balls or 0,
            "completed":   True,
        }

    # Live in-play projection
    import numpy as np
    rng = np.random.default_rng(0)
    n_sim = 5000
    if target and rem_balls and cur_runs is not None:
        # Chase model: per-ball runs ~ poisson with mean = needed/balls but bounded.
        # Per-ball wicket prob ~ 0.04 in T20 death overs; combine simply.
        needed = max(target - cur_runs, 0)
        balls_left = max(rem_balls, 1)
        wickets_lost = state.get("bat_team_wkts") or 0
        wkts_left = max(10 - wickets_lost, 0)
        per_ball_run = 1.4   # T20 typical
        per_ball_wkt = 0.05
        # adjust: required RR drives aggression
        rrr = (needed * 6.0) / balls_left
        per_ball_run = max(0.8, min(per_ball_run + 0.15 * (rrr - 8.0), 2.6))
        sims = []
        wins = 0
        for _ in range(n_sim):
            runs = 0; wkts = 0
            for b in range(balls_left):
                if wkts >= wkts_left:
                    break
                # bernoulli wicket
                if rng.random() < per_ball_wkt:
                    wkts += 1
                    continue
                # gaussian-ish run (1.4 mean, sd 1.1)
                runs += max(0, int(rng.normal(per_ball_run, 1.1)))
                if runs >= needed:
                    break
            sims.append(runs); wins += 1 if runs >= needed else 0
        p10, p50, p90 = np.quantile(sims, [0.1, 0.5, 0.9])
        return {
            "mode":            "chase",
            "win_prob":        wins / n_sim,
            "balls_remaining": balls_left,
            "p10":             int(cur_runs + p10),
            "p50":             int(cur_runs + p50),
            "p90":             int(cur_runs + p90),
            "mean":            int(cur_runs + np.mean(sims)),
            "n_sim":           n_sim,
        }

    if cur_runs is not None and overs is not None:
        # Setting model: extrapolate from current rate and venue average
        try:
            ovs = float(overs)
        except Exception:
            ovs = 0.0
        max_overs = 20 if (state.get("match_format") or "").upper() in ("T20", "T20I", "IT20") else 50
        balls_done = int(ovs) * 6 + int(round((ovs - int(ovs)) * 10))
        balls_left = max_overs * 6 - balls_done
        wickets_lost = state.get("bat_team_wkts") or 0
        wkts_left = max(10 - wickets_lost, 0)
        per_ball_run = 1.5
        per_ball_wkt = 0.04
        sims = []
        for _ in range(n_sim):
            runs = 0; wkts = 0
            for b in range(balls_left):
                if wkts >= wkts_left:
                    break
                if rng.random() < per_ball_wkt:
                    wkts += 1; continue
                runs += max(0, int(rng.normal(per_ball_run, 1.1)))
            sims.append(runs)
        p10, p50, p90 = np.quantile(sims, [0.1, 0.5, 0.9])
        return {
            "mode":            "set_score",
            "balls_remaining": balls_left,
            "p10":             int(cur_runs + p10),
            "p50":             int(cur_runs + p50),
            "p90":             int(cur_runs + p90),
            "mean":            int(cur_runs + np.mean(sims)),
            "n_sim":           n_sim,
        }

    return {"mode": "unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cricbuzz-id")
    ap.add_argument("--slug")
    ap.add_argument("--find-home", help="comma-separated keywords for home team slug match")
    ap.add_argument("--find-away", help="comma-separated keywords for away team slug match")
    ap.add_argument("--out", default=str(RUNS_DIR / "live_match.json"))
    args = ap.parse_args()

    mid = args.cricbuzz_id; slug = args.slug
    if not mid:
        if not (args.find_home and args.find_away):
            print("Provide --cricbuzz-id or --find-home + --find-away", file=sys.stderr); sys.exit(2)
        h = [k.strip().lower() for k in args.find_home.split(",")]
        a = [k.strip().lower() for k in args.find_away.split(",")]
        m = find_live_match_by_teams(h, a)
        if not m:
            print("No live match found.", file=sys.stderr); sys.exit(1)
        mid = m["match_id"]; slug = m["slug"]
        print(f"Found Cricbuzz match {mid}: {slug}")

    raw = fetch_match_state(mid, slug)
    dash = normalise_for_dashboard(raw)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dash, indent=2, default=str))

    print(f"\nMatch: {dash.get('home')} vs {dash.get('away')}")
    print(f"Status: {dash.get('status')}")
    print(f"Score:  {dash.get('score')} ({dash.get('overs')} ov)")
    if dash.get("target"):
        print(f"Target: {dash['target']}, need {dash.get('rem_runs')} from {dash.get('rem_balls')} balls")
    if dash.get("striker", {}).get("name"):
        s = dash["striker"]
        print(f"Striker: {s['name']} {s['runs']}({s['balls']})")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
