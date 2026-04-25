"""Live match tracker — polls Cricbuzz for score/state, re-runs the Monte Carlo
simulator each over, and writes a ``live_match`` block into data.json so the
dashboard auto-refreshes with in-play win probabilities.

CLI (via pipeline.py):
    # Explicit match ID:
    python -m cricket_pipeline.pipeline live-track --match-id 123456 \\
        --home "Rajasthan Royals" --away "Sunrisers Hyderabad" \\
        --venue "Sawai Mansingh Stadium, Jaipur"

    # Auto-discover any live IPL match:
    python -m cricket_pipeline.pipeline live-track --auto

    # Auto with team hints:
    python -m cricket_pipeline.pipeline live-track --auto \\
        --home-hint "Rajasthan" --away-hint "Sunrisers"
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config
from .db import connect

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.cricbuzz.com/",
}

ROOT = Path(__file__).resolve().parent.parent   # Waundering/
DATA_JSON = ROOT / "data.json"

# IPL team slug fragments used for auto-discovery
_IPL_SLUGS = frozenset([
    "rr", "srh", "mi", "csk", "rcb", "kkr", "lsg", "dc", "gt", "pbks",
    "rajasthan", "sunrisers", "mumbai", "chennai", "bangalore",
    "kolkata", "lucknow", "delhi", "gujarat", "punjab",
])


# ── Live match discovery ──────────────────────────────────────────────────────

def discover_live_matches() -> list[dict]:
    """Scrape Cricbuzz live-scores page and return list of {match_id, slug, title}."""
    url = "https://www.cricbuzz.com/cricket-match/live-scores"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        print(f"[live_tracker] Could not fetch live list: {e}")
        return []

    # Match URLs like /live-cricket-scores/12345/rr-vs-srh-22nd-match-ipl-2026
    found = re.findall(r'/live-cricket-scores/(\d{4,})/([a-z0-9-]+)', html)
    seen: set[str] = set()
    result = []
    for match_id, slug in found:
        if match_id not in seen:
            seen.add(match_id)
            result.append({
                "match_id": match_id,
                "slug": slug,
                "title": slug.replace("-", " ").title(),
            })
    return result


def find_best_match(
    matches: list[dict],
    home_hint: str | None = None,
    away_hint: str | None = None,
) -> dict | None:
    """Select the most relevant match given optional team hints."""
    if not matches:
        return None

    if home_hint or away_hint:
        h = (home_hint or "").lower()
        a = (away_hint or "").lower()
        for m in matches:
            slug = m["slug"]
            if (h and any(part in slug for part in h.split())) or \
               (a and any(part in slug for part in a.split())):
                return m

    for m in matches:
        slug_parts = set(m["slug"].split("-"))
        if slug_parts & _IPL_SLUGS or "ipl" in m["slug"]:
            return m

    return matches[0]


# ── HTML live-score scraper ───────────────────────────────────────────────────

def fetch_live_state(match_id: str, slug: str | None = None) -> dict | None:
    """Scrape the Cricbuzz live-cricket-scores page and return a match state dict.

    Uses the rendered HTML page (works even when the old JSON API returns HTML).
    Returns None on 404; raises on other network errors.
    """
    url = (
        f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/{slug}"
        if slug
        else f"https://www.cricbuzz.com/live-cricket-scores/{match_id}"
    )
    r = requests.get(url, headers=_HEADERS, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return _parse_live_html(r.text, match_id)


def _parse_live_html(html: str, match_id: str) -> dict:
    """Parse Cricbuzz live-scores HTML into a normalised match state dict.

    Data sources used (in priority order):
      1. Page <title>  — current score, overs, batsmen (refreshed each ball)
      2. overSeparator JSON blob — bowler figures, last-over summary
      3. HTML text search — target, "needs N runs", match result
    """
    # ── Title ─────────────────────────────────────────────────────────────────
    title_m = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""

    # Match complete?  Title will contain "beat", "won", "tied", etc.
    is_complete = bool(
        re.search(r"\b(?:beat|beats|won|tied|drawn|no result|abandoned)\b", title, re.I)
    )

    # ── Score + overs (from title) ─────────────────────────────────────────
    #  Pattern: "IPL | RR 225/5 (19.2) …" or "T20I | IND 187/4 (20) …"
    score_m = re.search(
        r"(?:IPL|T20I?|ODI|Test)\s*\|\s*([A-Z]{2,6})\s+(\d+)[/\-](\d+)\s+\(([0-9.]+)\)",
        title,
    )
    bat_abbrev = ""
    runs = wickets = 0
    overs = "0"
    if score_m:
        bat_abbrev = score_m.group(1)
        runs = int(score_m.group(2))
        wickets = int(score_m.group(3))
        overs = score_m.group(4)

    # ── Batsmen (from title) ───────────────────────────────────────────────
    striker: dict = {}
    non_striker: dict = {}
    #  "(Shimron Hetmyer 11(9) Ravindra Jadeja 4(2))"
    bat_block_m = re.search(r"\(([A-Z][^()]+\(\d+\)[^()]*\(\d+\))\)", title)
    if bat_block_m:
        pairs = re.findall(r"([A-Z][a-zA-Z ]+?)\s+(\d+)\((\d+)\)", bat_block_m.group(1))
        if pairs:
            striker = {"name": pairs[0][0].strip(), "runs": int(pairs[0][1]), "balls": int(pairs[0][2])}
        if len(pairs) >= 2:
            non_striker = {"name": pairs[1][0].strip(), "runs": int(pairs[1][1]), "balls": int(pairs[1][2])}

    # ── Team full names (from title) ───────────────────────────────────────
    teams_m = re.search(r"\|\s*([A-Z][^|]+?)\s+vs\s+([A-Z][^|,]+?),", title)
    team1 = teams_m.group(1).strip() if teams_m else ""
    team2 = teams_m.group(2).strip() if teams_m else ""

    # Resolve full batting team name from 2-6 char abbreviation
    batting_team = _abbrev_to_team(bat_abbrev, team1, team2)
    bowling_team = team2 if batting_team == team1 else team1

    # ── Bowler (from overSeparator JSON blob) ─────────────────────────────
    bowler: dict = {}
    last_overs: str | None = None
    bow_m = re.search(
        r'"bowlerObj":\{"playerId":\d+,"playerName":"([^"]+)","playerScore":"([^"]+)"\}',
        html,
    )
    if bow_m:
        figures = bow_m.group(2)          # "4-0-38-2"
        parts = figures.split("-")
        bowler = {
            "name":    bow_m.group(1),
            "overs":   parts[0] if parts else None,
            "runs":    int(parts[2]) if len(parts) >= 3 else None,
            "wickets": int(parts[3]) if len(parts) >= 4 else None,
        }
    ov_sum_m = re.search(r'"overSummary":"([^"]+)"', html)
    if ov_sum_m:
        last_overs = ov_sum_m.group(1)

    # ── Target / chase ─────────────────────────────────────────────────────
    target: int | None = None
    rem_runs: int | None = None

    #  "needs 176 runs" / "need 176 from 72 balls"
    need_m = re.search(r"needs?\s+(\d+)\s+(?:more\s+)?runs?", html, re.I)
    if need_m:
        rem_runs = int(need_m.group(1))
        target = runs + rem_runs
    else:
        tgt_m = re.search(r"[Tt]arget[:\s]+(\d+)", html)
        if tgt_m:
            target = int(tgt_m.group(1))

    # ── Innings number ─────────────────────────────────────────────────────
    innings = 2 if target is not None else 1
    if innings == 1 and re.search(r"2nd\s+Innings|second\s+innings", html, re.I):
        innings = 2

    # ── Status text ────────────────────────────────────────────────────────
    if is_complete:
        # Grab the result part: "IPL | SRH beat RR by 5 wickets | …"
        res_m = re.search(r"(?:IPL|T20I?|ODI|Test)\s*\|\s*([^|]+)\|", title)
        status = res_m.group(1).strip() if res_m else "Match over"
    else:
        status = "Live"

    # ── Derived metrics ────────────────────────────────────────────────────
    balls_done = _overs_to_balls(overs)
    balls_remaining = max(0, 120 - balls_done)
    crr = round(runs * 6 / max(balls_done, 1), 2)
    rrr: float | None = None
    if target is not None and balls_remaining > 0:
        needed = rem_runs if rem_runs is not None else max(0, target - runs)
        rrr = round(needed * 6 / balls_remaining, 2)

    return {
        "match_id":        match_id,
        "status":          status,
        "is_complete":     is_complete,
        "innings":         innings,
        "team1":           team1,
        "team2":           team2,
        "batting_team":    batting_team,
        "bowling_team":    bowling_team,
        "score":           f"{runs}/{wickets}",
        "runs":            runs,
        "wickets":         wickets,
        "overs":           overs,
        "balls_done":      balls_done,
        "balls_remaining": balls_remaining,
        "current_rr":      crr,
        "required_rr":     rrr,
        "target":          target,
        "rem_runs":        rem_runs,
        "striker":         striker,
        "non_striker":     non_striker,
        "bowler":          bowler,
        "last_overs":      last_overs,
        "last_wicket":     None,
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
    }


def _overs_to_balls(overs_str: str) -> int:
    """Convert "15.3" → 93 (15 full overs + 3 balls)."""
    try:
        s = str(overs_str)
        if "." in s:
            ov, part = s.split(".", 1)
            return int(ov) * 6 + int(part[:1])
        return int(s) * 6
    except (ValueError, TypeError):
        return 0


def _abbrev_to_team(abbrev: str, team1: str, team2: str) -> str:
    """Best-effort mapping from a 2-6 char abbreviation to a full team name."""
    ab = abbrev.lower()
    t1 = team1.lower()
    t2 = team2.lower()
    # Check initials: "SRH" → "Sunrisers Hyderabad" (S+H match first letters of each word)
    t1_words = t1.split()
    t2_words = t2.split()
    t1_initials = "".join(w[0] for w in t1_words if w)
    t2_initials = "".join(w[0] for w in t2_words if w)
    if ab == t1_initials:
        return team1
    if ab == t2_initials:
        return team2
    # Substring of first word
    if ab[:2] in t1 and ab[:2] not in t2:
        return team1
    if ab[:2] in t2 and ab[:2] not in t1:
        return team2
    # First word prefix
    if t1_words and t1_words[0].startswith(ab[:3]):
        return team1
    if t2_words and t2_words[0].startswith(ab[:3]):
        return team2
    return team1  # default


# ── Live win-probability via Monte Carlo ──────────────────────────────────────

def _db_player_stats(
    con,
    striker: str | None,
    bowler: str | None,
    venue: str,
) -> dict:
    """Fetch batter SR, bowler economy, and venue averages from DuckDB."""
    batter_sr, batter_avg = 130.0, 25.0
    bowler_econ, bowler_avg = 8.5, 28.0
    venue_avg, venue_toss = 175.0, 0.55

    if striker:
        row = con.execute(
            "SELECT strike_rate FROM v_batter_profile WHERE batter = ?",
            [striker],
        ).fetchone()
        if row and row[0]:
            batter_sr = float(row[0])

    if bowler:
        row = con.execute(
            "SELECT economy FROM v_bowler_profile WHERE bowler = ?",
            [bowler],
        ).fetchone()
        if row and row[0]:
            bowler_econ = float(row[0])

    if venue:
        row = con.execute(
            """SELECT avg_first_innings, toss_winner_won_pct
               FROM v_venue_profile
               WHERE venue = ? AND format = 'T20'""",
            [venue],
        ).fetchone()
        if row:
            if row[0]:
                venue_avg = float(row[0])
            if row[1]:
                venue_toss = float(row[1])

    return {
        "batter_sr":   batter_sr,
        "batter_avg":  batter_avg,
        "bowler_econ": bowler_econ,
        "bowler_avg":  bowler_avg,
        "venue_avg":   venue_avg,
        "venue_toss":  venue_toss,
    }


def compute_live_prediction(
    state: dict,
    venue: str,
    n_sim: int = 1000,
) -> dict | None:
    """Run Monte Carlo simulation for the remaining innings.

    Returns a dict with win_prob (if chasing), p10/p50/p90 score projection,
    and meta fields. Returns None if models aren't loaded or simulation fails.
    """
    from .model.simulate import simulate_innings

    striker_name = (state.get("striker") or {}).get("name")
    bowler_name  = (state.get("bowler")  or {}).get("name")

    try:
        con = connect()
        stats = _db_player_stats(con, striker_name, bowler_name, venue)
        con.close()
    except Exception as e:
        print(f"[live_tracker] DB lookup skipped: {e}")
        stats = {
            "batter_sr": 130.0, "batter_avg": 25.0,
            "bowler_econ": 8.5, "bowler_avg": 28.0,
            "venue_avg": 175.0, "venue_toss": 0.55,
        }

    overs_float = _safe_float(state.get("overs", "0"), 0.0)
    target      = state.get("target")
    innings     = int(state.get("innings", 1))
    balls_done  = int(state.get("balls_done", 0))
    balls_left  = int(state.get("balls_remaining", 120))
    runs        = int(state.get("runs", 0))
    wickets     = int(state.get("wickets", 0))
    crr         = float(state.get("current_rr") or 0.0)
    rrr         = float(state.get("required_rr") or 0.0) if target else float("nan")

    sim_state = {
        "format":                    "T20",
        "venue":                     venue or "unknown",
        "batter":                    striker_name or "unknown",
        "bowler":                    bowler_name  or "unknown",
        "batter_hand":               "Right hand Bat",
        "bowler_type":               "Right arm Fast",
        "phase":                     _phase(overs_float),
        "innings_no":                innings,
        "over_no":                   int(overs_float),
        "ball_in_over":              int((overs_float % 1) * 10),
        "runs_so_far":               runs,
        "wickets_so_far":            wickets,
        "deliveries_so_far":         balls_done,
        "legal_balls_left":          balls_left,
        "current_run_rate":          crr,
        "required_run_rate":         rrr,
        "batter_sr":                 stats["batter_sr"],
        "batter_avg":                stats["batter_avg"],
        "batter_balls":              1000,
        "batter_form_sr":            stats["batter_sr"],
        "batter_form_runs":          100,
        "batter_form_balls":         80,
        "bowler_econ":               stats["bowler_econ"],
        "bowler_avg":                stats["bowler_avg"],
        "bowler_balls":              1000,
        "bowler_workload_30d":       18,
        "bowler_workload_7d":        6,
        "venue_avg_first_innings":   stats["venue_avg"],
        "venue_toss_winner_won_pct": stats["venue_toss"],
        "temp_c":    30.0,
        "humidity":  55.0,
        "wind_kmh":  7.0,
        "target":    target,
    }

    try:
        result = simulate_innings(sim_state, n_sim=n_sim, seed=None)
    except Exception as e:
        print(f"[live_tracker] simulate_innings failed: {e}")
        return None

    mode = "chase" if target else "set_score"
    win_prob = result.get("win_prob") if target else None

    # First-innings proxy: P(projected final >= venue average)
    if win_prob is None and mode == "set_score":
        import numpy as _np
        finals = _np.array([result["p50"]])  # rough proxy only
        win_prob = float((result["p50"] >= stats["venue_avg"]))

    return {
        "mode":            mode,
        "win_prob":        round(win_prob, 4) if win_prob is not None else None,
        "p10":             result["p10"],
        "p50":             result["p50"],
        "p90":             result["p90"],
        "mean":            round(result["mean"], 1),
        "balls_remaining": balls_left,
        "target":          target,
        "venue_avg":       round(stats["venue_avg"], 1),
        "n_sim":           n_sim,
    }


def _phase(overs: float) -> str:
    if overs < 6:
        return "powerplay"
    if overs < 15:
        return "middle"
    return "death"


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── data.json update ──────────────────────────────────────────────────────────

def update_data_json(live_data: dict, out_path: Path = DATA_JSON) -> None:
    """Merge the live_match block into data.json with an atomic rename."""
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    except Exception:
        existing = {}

    existing["live_match"] = live_data
    existing["live_updated_at"] = datetime.now(timezone.utc).isoformat()

    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    tmp.replace(out_path)


# ── Main polling loop ─────────────────────────────────────────────────────────

def run(
    match_id: str,
    home: str,
    away: str,
    venue: str | None = None,
    slug: str | None = None,
    interval: int = 60,
    n_sim: int = 1000,
    out_path: Path = DATA_JSON,
) -> None:
    """Poll Cricbuzz every `interval` seconds and write live data to data.json."""
    print(f"[live_tracker] Tracking match {match_id}  ({home} vs {away})")
    print(f"[live_tracker] Poll interval: {interval}s  Output: {out_path}")

    if not venue:
        venue = _resolve_venue(home, away)
    print(f"[live_tracker] Venue: {venue or '(unknown)'}")

    last_score = None
    consecutive_errors = 0

    while True:
        try:
            state = fetch_live_state(match_id, slug)
        except Exception as e:
            consecutive_errors += 1
            print(f"[live_tracker] Fetch error ({consecutive_errors}/5): {e}")
            if consecutive_errors >= 5:
                print("[live_tracker] Too many consecutive fetch errors — aborting.")
                break
            time.sleep(interval)
            continue

        if state is None:
            consecutive_errors += 1
            print(f"[live_tracker] Match {match_id}: 404 ({consecutive_errors}/10)")
            if consecutive_errors >= 10:
                print("[live_tracker] Match not found after 10 attempts — aborting.")
                break
            time.sleep(interval)
            continue

        consecutive_errors = 0
        current_score = state["score"]

        if current_score != last_score:
            _log_state(state)
            last_score = current_score

        # Compute in-play win probability
        live_pred = compute_live_prediction(state, venue or "unknown", n_sim=n_sim)

        live_data = {
            **state,
            "home":  home,
            "away":  away,
            "venue": venue or "unknown",
            "live_prediction": live_pred,
        }
        update_data_json(live_data, out_path)

        if state["is_complete"]:
            print(f"[live_tracker] Match complete: {state['status']}")
            break

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[live_tracker] Stopped by user.")
            break

    print("[live_tracker] Tracker finished.")


def auto_run(
    home_hint: str | None = None,
    away_hint: str | None = None,
    interval: int = 60,
    n_sim: int = 1000,
    out_path: Path = DATA_JSON,
) -> None:
    """Auto-discover a live IPL match and start tracking it."""
    print("[live_tracker] Auto-discovering live matches on Cricbuzz...")
    matches = discover_live_matches()

    if not matches:
        print("[live_tracker] No live matches found. Is a match currently in progress?")
        return

    print(f"[live_tracker] Found {len(matches)} live match(es):")
    for m in matches[:8]:
        print(f"  {m['match_id']}: {m['title']}")

    chosen = find_best_match(matches, home_hint, away_hint)
    if not chosen:
        print("[live_tracker] Could not identify a suitable match.")
        return

    print(f"\n[live_tracker] Selected: {chosen['match_id']} — {chosen['title']}")

    home, away, venue = _team_names_from_data_json(out_path, home_hint, away_hint)

    run(
        match_id=chosen["match_id"],
        home=home, away=away, venue=venue,
        slug=chosen.get("slug"),
        interval=interval, n_sim=n_sim, out_path=out_path,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_venue(home: str, away: str) -> str | None:
    """Look up today's fixture venue from the fixtures table."""
    try:
        con = connect()
        today = datetime.now().date().isoformat()
        # Try first word of team name (e.g. "Rajasthan" from "Rajasthan Royals")
        h_word = (home or "").split()[0] if home else ""
        a_word = (away or "").split()[0] if away else ""
        row = con.execute(
            """SELECT venue FROM fixtures
               WHERE (team_home LIKE ? OR team_away LIKE ?)
                 AND start_date = ?
               LIMIT 1""",
            [f"%{h_word}%", f"%{a_word}%", today],
        ).fetchone()
        con.close()
        return row[0] if row else None
    except Exception:
        return None


def _team_names_from_data_json(
    out_path: Path,
    home_hint: str | None,
    away_hint: str | None,
) -> tuple[str, str, str | None]:
    """Pull team names and venue from data.json latest_prediction."""
    home, away, venue = home_hint or "", away_hint or "", None
    try:
        d = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
        m = (d.get("latest_prediction") or {}).get("match") or {}
        home  = home  or m.get("home",  "")
        away  = away  or m.get("away",  "")
        venue = m.get("venue")
    except Exception:
        pass
    return home, away, venue


def _log_state(state: dict) -> None:
    strike     = state.get("striker")  or {}
    bowl       = state.get("bowler")   or {}
    s_name     = strike.get("name")    or "—"
    b_name     = bowl.get("name")      or "—"
    target_str = (
        f"  target {state['target']}  RRR {state['required_rr']}"
        if state.get("target") else ""
    )
    print(
        f"  [{state['overs']} ov] {state['batting_team']} "
        f"{state['score']}  CRR {state['current_rr']}"
        f"{target_str}  |  {s_name} bat / {b_name} bowl  |  {state['status']}"
    )
