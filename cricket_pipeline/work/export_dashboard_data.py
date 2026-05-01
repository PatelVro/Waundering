"""Export everything the cricket dashboard needs into a single data.json
served from the project root.

Includes:
  - latest_prediction: the saved RR vs SRH JSON (or whichever the user last ran)
  - all_predictions: every JSON in predictions/
  - model_metrics: final_summary.json
  - top_teams_t20 / top_teams_odi: per-format Elo leaderboard
  - recent_matches: last 30 with model retro-prediction (where available)
  - team_form_snapshots: last 5 win-pct for top teams
  - data_stats: row counts by table
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..db import connect, install_views
from .features_v2 import build_features
from . import filters as F


ROOT = Path(__file__).resolve().parents[2]   # Waundering/
PREDICTIONS_DIR = ROOT / "predictions"
RUNS_DIR        = ROOT / "cricket_pipeline" / "work" / "runs"
LIVE_PATH       = RUNS_DIR / "live_match.json"
LIVE_MULTI_PATH = RUNS_DIR / "live_matches.json"
OUT_PATH        = ROOT / "data.json"

# Aliases for the design's terminal page (it fetches data/data.json + data/preds/*.json)
DESIGN_DATA_DIR = ROOT / "data"
DESIGN_PREDS_DIR = DESIGN_DATA_DIR / "preds"

# Map full team names → short bookmaker-style codes for the design's PRED_FILES
_TEAM_CODES = {
    "Lucknow Super Giants": "lsg", "Kolkata Knight Riders": "kkr",
    "Delhi Capitals": "dc", "Royal Challengers Bengaluru": "rcb",
    "Chennai Super Kings": "csk", "Gujarat Titans": "gt",
    "Rajasthan Royals": "rr", "Sunrisers Hyderabad": "srh",
    "Mumbai Indians": "mi", "Punjab Kings": "pbks",
    "New Zealand": "nz", "Bangladesh": "ban",
    "India": "ind", "Australia": "aus", "England": "eng",
    "South Africa": "sa", "Sri Lanka": "sl", "West Indies": "wi",
    "Afghanistan": "afg", "Ireland": "ire", "Netherlands": "ned",
    "Scotland": "sco", "Nepal": "nep", "Uganda": "uga",
    "United Arab Emirates": "uae", "United States of America": "usa",
}


def _short_id(team: str | None) -> str | None:
    if not team: return None
    return _TEAM_CODES.get(team) or "".join(w[0] for w in team.split() if w)[:4].lower()


def _write_design_aliases(data_dict: dict, preds_list: list[dict]) -> None:
    """Mirror the export into the layout the design's terminal expects:
       data/data.json
       data/preds/<short_a>_vs_<short_b>.json    (sorted alphabetically)
    Newest match per fixture wins on collision.

    Canonical sorted slug: same fixture always maps to the same alias
    file regardless of which side Cricbuzz currently labels as 'home'.
    Without this, a Cricbuzz home/away flip mid-tournament leaves two
    competing alias files (e.g. gt_vs_rcb.json from before the flip,
    rcb_vs_gt.json from after) — the frontend's static fixture list
    happens to ask for one of them and silently shows stale data.

    Stale non-canonical aliases from earlier writes are pruned to
    prevent the dashboard from reading them.
    """
    DESIGN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    (DESIGN_DATA_DIR / "data.json").write_text(json.dumps(data_dict, indent=2, default=str))

    # newest first by date so first-seen wins
    sorted_preds = sorted(preds_list, key=lambda p: (p.get("match", {}).get("date") or ""),
                           reverse=True)
    canonical = set()
    for p in sorted_preds:
        m = p.get("match") or {}
        h, a = _short_id(m.get("home")), _short_id(m.get("away"))
        if not (h and a): continue
        # Canonical: alphabetical so home/away swaps don't fork the alias
        lo, hi = sorted([h, a])
        slug = f"{lo}_vs_{hi}"
        if slug in canonical: continue
        canonical.add(slug)
        (DESIGN_PREDS_DIR / f"{slug}.json").write_text(json.dumps(p, indent=2, default=str))

    # Prune stale non-canonical sibling aliases — for every alias file in
    # the dir, if its canonical-sorted form differs from its actual filename
    # AND the canonical form was just written, remove the stale one.
    for fp in DESIGN_PREDS_DIR.glob("*.json"):
        stem = fp.stem        # e.g. "rcb_vs_gt"
        if "_vs_" not in stem: continue
        a_, b_ = stem.split("_vs_", 1)
        lo, hi = sorted([a_, b_])
        canonical_stem = f"{lo}_vs_{hi}"
        if stem != canonical_stem and canonical_stem in canonical:
            try:
                fp.unlink()
            except OSError:
                pass


def _load_predictions() -> dict:
    """Load predictions, deduping multi-day forecasts of the same fixture.

    The predictor saves one file per (fixture, date) pair. When a match
    rolls over multiple days without playing, we end up with stale files
    pointing at the same actual upcoming game (e.g. three BAN-vs-NZ files
    for 26/27/28 APR when only one game is played). All three get tagged
    with the same live result by `_attach_pred_result` because that
    function matches by team-pair only — producing duplicate rows in the
    track record. Keep only the prediction with the latest match.date per
    fixture pair.
    """
    out = {"all": [], "latest": None}
    if not PREDICTIONS_DIR.exists():
        return out
    files = sorted(PREDICTIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    by_pair: dict[frozenset, dict] = {}
    for fp in files:
        try:
            data = json.loads(fp.read_text())
            m = data.get("match", {})
            home, away = m.get("home"), m.get("away")
            if F.is_blocked_match(home, away):
                continue
            data["_file"] = fp.name
            key = frozenset([_norm(home), _norm(away)])
            existing = by_pair.get(key)
            if existing is None or (m.get("date") or "") > (existing.get("match", {}).get("date") or ""):
                by_pair[key] = data
        except Exception:
            continue
    out["all"] = sorted(by_pair.values(),
                        key=lambda p: (p.get("match", {}).get("date") or ""),
                        reverse=True)
    if out["all"]:
        out["latest"] = out["all"][0]
    return out


def _model_metrics() -> dict:
    p = RUNS_DIR / "final_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _top_teams_per_format(format_codes: list[str], top_n: int = 12) -> list[dict]:
    """Compute current Elo per team for the given formats. Excludes blocked teams."""
    df = build_features(format_filter=format_codes)
    if df.empty:
        return []
    rows1 = df[["start_date", "team_home", "t1_elo_pre"]].rename(
        columns={"team_home": "team", "t1_elo_pre": "elo"})
    rows2 = df[["start_date", "team_away", "t2_elo_pre"]].rename(
        columns={"team_away": "team", "t2_elo_pre": "elo"})
    long = pd.concat([rows1, rows2], ignore_index=True).dropna()
    long = long[~long["team"].apply(F.is_blocked_team)]
    long = long.sort_values("start_date").groupby("team", as_index=False).tail(1)
    matches = long.sort_values("elo", ascending=False).head(top_n)
    return [{"team": r.team, "elo": round(float(r.elo), 1),
             "as_of": str(r.start_date.date())} for r in matches.itertuples(index=False)]


def _recent_matches(n: int = 30) -> list[dict]:
    con = connect()
    install_views()
    # dedup by (date, teams, venue) — CricSheet zip overlap creates duplicates
    rows = con.execute(f"""
        WITH dedup AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY start_date, team_home, team_away, venue
                       ORDER BY match_id
                   ) AS rk
            FROM matches
            WHERE start_date IS NOT NULL AND winner IS NOT NULL
        )
        SELECT match_id, format, competition, start_date, venue,
               team_home, team_away, winner, win_margin_runs, win_margin_wickets
        FROM dedup
        WHERE rk = 1
        ORDER BY start_date DESC, match_id DESC
        LIMIT {n}
    """).df()
    con.close()
    out = []
    for r in rows.itertuples(index=False):
        if F.is_blocked_match(r.team_home, r.team_away):
            continue
        margin = ""
        runs_v = r.win_margin_runs
        wkts_v = r.win_margin_wickets
        if runs_v is not None and not pd.isna(runs_v) and float(runs_v) > 0:
            margin = f"by {int(runs_v)} runs"
        elif wkts_v is not None and not pd.isna(wkts_v) and float(wkts_v) > 0:
            margin = f"by {int(wkts_v)} wickets"
        out.append({
            "match_id":    r.match_id,
            "format":      r.format,
            "competition": r.competition,
            "date":        str(r.start_date),
            "venue":       r.venue,
            "home":        r.team_home,
            "away":        r.team_away,
            "winner":      r.winner,
            "margin":      margin,
        })
    return out


def _bet_summary() -> dict:
    """Recent bets + PnL summary."""
    try:
        from . import bet_engine as BET
    except Exception:
        return {"available": False}
    summary = BET.pnl_summary()
    con = connect()
    rows = con.execute("""
        SELECT bet_id, placed_at, mode, market, selection, decimal_odds,
               stake, model_p, book_p, edge_pct, status, settled_at, pnl, notes
        FROM bets
        ORDER BY placed_at DESC
        LIMIT 30
    """).fetchall()
    con.close()
    bets = []
    for r in rows:
        try:
            n = json.loads(r[13] or "{}")
        except Exception:
            n = {}
        m = (n.get("match") or {})
        bets.append({
            "bet_id":      (r[0] or "")[:8],
            "placed_at":   r[1],
            "mode":        r[2],
            "market":      r[3],
            "selection":   r[4],
            "decimal_odds": r[5],
            "stake":       r[6],
            "model_p":     r[7],
            "book_p":      r[8],
            "edge_pct":    r[9],
            "status":      r[10],
            "settled_at":  r[11],
            "pnl":         r[12],
            "match":       m,
        })
    summary["available"] = True
    summary["recent_bets"] = bets
    try:
        summary["open_tickets"] = BET.open_tickets()
    except Exception:
        summary["open_tickets"] = []
    return summary


def _data_stats() -> dict:
    con = connect()
    out = {}
    # 'matches' shows the DISTINCT count (CricSheet zip overlap creates duplicates
    # in the raw table). Other tables are joined via match_id and effectively dedup.
    out["matches"] = int(con.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT start_date, team_home, team_away, venue FROM matches
        )
    """).fetchone()[0])
    for tbl in ("innings", "balls", "match_xi"):
        try:
            out[tbl] = int(con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
        except Exception:
            out[tbl] = 0
    out["distinct_venues"] = int(con.execute("SELECT COUNT(DISTINCT venue) FROM matches").fetchone()[0])
    out["distinct_teams"]  = int(con.execute("""
        SELECT COUNT(DISTINCT team) FROM (
            SELECT team_home AS team FROM matches
            UNION SELECT team_away FROM matches
        )
    """).fetchone()[0])
    out["distinct_competitions"] = int(con.execute("""
        SELECT COUNT(DISTINCT competition) FROM matches WHERE competition IS NOT NULL
    """).fetchone()[0])
    con.close()
    return out


def _load_live_match() -> dict | None:
    if not LIVE_PATH.exists():
        return None
    try:
        return json.loads(LIVE_PATH.read_text())
    except Exception:
        return None


def _load_live_matches_multi() -> list[dict]:
    """Optional: a list of multiple matches (upcoming + recent). Used for
    cross-referencing predictions to results when the user has many."""
    out = []
    if LIVE_PATH.exists():
        try:
            single = json.loads(LIVE_PATH.read_text())
            if single:
                out.append(single)
        except Exception:
            pass
    if LIVE_MULTI_PATH.exists():
        try:
            multi = json.loads(LIVE_MULTI_PATH.read_text())
            if isinstance(multi, list):
                out.extend(multi)
        except Exception:
            pass
    # de-dup by match_id, drop blocked teams
    seen = set(); deduped = []
    for d in out:
        if F.is_blocked_match(d.get("home"), d.get("away")):
            continue
        mid = d.get("match_id")
        if mid and mid in seen: continue
        seen.add(mid); deduped.append(d)
    return deduped


def _norm(s):
    return (s or "").strip().lower()


def _matches_pair(pred_match: dict, live: dict) -> bool:
    ph = _norm(pred_match.get("home")); pa = _norm(pred_match.get("away"))
    lh = _norm(live.get("home"));        la = _norm(live.get("away"))
    return (ph == lh and pa == la) or (ph == la and pa == lh)


def _pick_best_live(pred_match: dict, candidates: list) -> dict | None:
    """When multiple live entries match the same team-pair (e.g. a series
    where only one game has played), prefer the entry that gives us a
    real, settled result — so an upcoming/abandoned game doesn't clobber
    a previously-graded prediction. Priority:
      1. is_complete=True with a parseable winner
      2. is_complete=True (even if winner not parseable yet)
      3. anything else (in-progress, upcoming, abandoned)
    """
    matched = [l for l in candidates if _matches_pair(pred_match, l)]
    if not matched:
        return None
    def rank(l):
        if not l.get("is_complete"): return 2
        w = _parse_winner_from_status(l.get("status"), l.get("home"), l.get("away"))
        return 0 if w else 1
    matched.sort(key=rank)
    return matched[0]


def _attach_pred_result(pred: dict, live_pool) -> dict:
    """live_pool is either a single live dict or a list of live dicts. Find
    the best matching one (preferring completed-with-winner) and decorate.

    Sticky grading: once a prediction has been graded `complete` with a
    winner, that result is persisted on the prediction object — a fresh
    live tick that's a different match between the same teams (e.g. an
    abandoned next-day fixture) cannot clobber it."""
    if not pred:
        return pred
    # Sticky: if the prediction already has a complete grade with a winner,
    # don't re-derive it from current live state. The played match is over;
    # later live ticks for the same team-pair are different fixtures.
    existing = pred.get("result") or {}
    if existing.get("status") == "complete" and existing.get("winner"):
        return pred
    candidates = live_pool if isinstance(live_pool, list) else ([live_pool] if live_pool else [])
    if not candidates:
        return pred
    live = _pick_best_live(pred["match"], candidates)
    if not live:
        return pred
    if not live.get("is_complete"):
        return {**pred, "result": {
            "status": "in_progress",
            "live_status": live.get("status"),
            "score": live.get("score"),
            "overs": live.get("overs"),
        }}
    # parse winner from status string
    winner = _parse_winner_from_status(live.get("status"),
                                        live.get("home"), live.get("away"))
    # Fallback A: infer from chase state (target vs final score). Catches the
    # awkward Cricbuzz window where is_complete=True but status hasn't flipped
    # from "need N runs" to "won by N wickets" yet.
    if not winner:
        winner = _winner_from_chase_state(live)
    # Fallback B: look up the matches table for the actual settled result
    if not winner:
        winner = _winner_from_matches_table(pred["match"]["home"],
                                              pred["match"]["away"],
                                              pred["match"]["date"])
    if not winner:
        # Cricbuzz says complete but we can't determine the winner yet — common
        # for a brief window between "need 1 run" and "won by N wickets".
        # Don't mark as complete; let the next live tick or matches table catch up.
        return {**pred, "result": {
            "status":      "awaiting_result",
            "live_status": live.get("status"),
            "score":       live.get("score"),
            "overs":       live.get("overs"),
            "note":        "Cricbuzz flagged complete but winner not yet parsed",
        }}
    actual_t1_wins = 1 if winner == pred["match"]["home"] else (0 if winner == pred["match"]["away"] else None)
    pred_t1_wins  = 1 if pred["prediction"]["p_home_wins"] >= 0.5 else 0
    correct = (actual_t1_wins == pred_t1_wins) if actual_t1_wins is not None else None
    return {**pred, "result": {
        "status":         "complete",
        "winner":         winner,
        "live_status":    live.get("status"),
        "predicted_winner": pred["prediction"]["favored"],
        "correct":        correct,
        "favored_pct":    pred["prediction"]["favored_pct"],
    }}


def _parse_winner_from_status(status: str | None, home: str | None, away: str | None) -> str | None:
    """Parse winner from Cricbuzz status text. Handles:
       'X won by N runs/wickets'        — direct
       'Match tied (X won the Super Over)' — super-over winner
       Returns None for no-result / abandoned / unparseable / pure-tie."""
    if not status: return None
    s = status.lower()
    if "no result" in s or "abandoned" in s:
        return None
    has_winner_kw = ("won" in s) or ("wins" in s)
    if not has_winner_kw:
        return None
    # Pure tie with no super-over context → no winner
    if "tied" in s and "super over" not in s:
        return None
    # Super-over case: prefer the substring inside parentheses if present
    paren = None
    if "super over" in s:
        # try to grab text after the last '(' for cleanest team match
        if "(" in s:
            paren = s[s.rfind("(") + 1:]
    target = paren or s
    # Match the longer team name first to avoid 'India' matching 'India Women'
    for t in sorted([home, away], key=lambda x: -(len(x or ""))):
        if t and t.lower() in target:
            return t
    # Match by short team CODE — Cricbuzz uses codes inside super-over parens
    # ("Match tied (KKR won the Super Over)") rather than full names.
    import re as _re
    for t in (home, away):
        code = _short_id(t)
        if code and _re.search(rf"\b{_re.escape(code)}\b", target):
            return t
    # Last resort: first word match
    for t in (home, away):
        if t and t.lower().split()[0] in target:
            return t
    return None


def _winner_from_chase_state(live: dict) -> str | None:
    """If the match is complete and we can read target + final chase score,
    derive the winner. Cricket convention: target = bowling_team_total + 1, so
    the chasing team wins iff their score >= target. Otherwise the bowling
    team won (covers won-by-runs and ties — ties are rare and a re-fetch will
    correct it once Cricbuzz publishes the official 'won by' string)."""
    if not live.get("is_complete"): return None
    bat = live.get("batting_team")
    bowl = live.get("bowling_team")
    target = live.get("target")
    score = (live.get("score") or "")
    if not (bat and bowl and target and score): return None
    try:
        runs = int(str(score).split("/")[0])
        target = int(target)
    except (ValueError, TypeError):
        return None
    if runs >= target:
        return bat
    if runs < target - 1:
        return bowl
    # exactly target-1 means tied — wait for the next live tick rather than
    # guess a super-over outcome.
    return None


def _margin_from_status(status: str | None) -> str:
    """Extract a 'by N runs' / 'by N wkts' phrase from a Cricbuzz status."""
    if not status:
        return ""
    import re as _re
    m = _re.search(r"won\s+(by\s+\d+\s+(?:runs?|wkts?|wickets?))", status, _re.I)
    if m:
        return m.group(1).lower().replace("wkts", "wickets")
    if "super over" in status.lower():
        return "tied · super over"
    return ""


def _merge_settled_into_recent(recent: list[dict], all_preds: list[dict],
                                 max_total: int = 40) -> list[dict]:
    """Inject settled predictions into recent_matches when CricSheet hasn't
    yet ingested the match. Dedup by (date, sorted-team-pair). Keep newest
    first. Truncates to max_total to match the original list shape."""
    seen = set()
    for r in recent:
        date = str(r.get("date") or "")[:10]
        pair = frozenset([_norm(r.get("home")), _norm(r.get("away"))])
        seen.add((date, pair))
    out = list(recent)
    for p in all_preds:
        res = p.get("result") or {}
        if res.get("status") != "complete" or not res.get("winner"):
            continue
        m = p.get("match") or {}
        date = str(m.get("date") or "")[:10]
        pair = frozenset([_norm(m.get("home")), _norm(m.get("away"))])
        key = (date, pair)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "match_id":    p.get("_file") or f"pred_{date}_{m.get('home')}_{m.get('away')}",
            "format":      m.get("format"),
            "competition": m.get("competition") or "",
            "date":        date,
            "venue":       m.get("venue"),
            "home":        m.get("home"),
            "away":        m.get("away"),
            "winner":      res.get("winner"),
            "margin":      _margin_from_status(res.get("live_status")),
        })
    out.sort(key=lambda r: str(r.get("date") or ""), reverse=True)
    return out[:max_total]


def _persist_result_if_complete(pred: dict) -> None:
    """Write a newly-graded `result` block back to the source prediction
    file on disk so subsequent runs treat the grade as sticky.

    No-op unless `result.status == "complete"` AND the on-disk file is
    missing the grade (or has a different one). Silent on failure — a
    persistence error must not block the dashboard build."""
    if not pred:
        return
    res = pred.get("result") or {}
    if res.get("status") != "complete" or not res.get("winner"):
        return
    fname = pred.get("_file")
    if not fname:
        return
    fp = PREDICTIONS_DIR / fname
    if not fp.exists():
        return
    try:
        on_disk = json.loads(fp.read_text())
        existing = (on_disk.get("result") or {}) if isinstance(on_disk, dict) else {}
        if existing.get("status") == "complete" and existing.get("winner") == res.get("winner"):
            return  # already persisted
        on_disk["result"] = res
        fp.write_text(json.dumps(on_disk, indent=2, default=str))
    except Exception:
        pass


def _load_timeline_events(limit: int = 80) -> list[dict]:
    """Load the last `limit` events from match_timeline.jsonl. Append-only file
    written by orchestrator's phase_loop; cheap to tail-read each export."""
    fp = RUNS_DIR / "match_timeline.jsonl"
    if not fp.exists():
        return []
    try:
        # Read tail efficiently — events are usually <300B each, so just read
        # the whole file (typically <1MB) and slice. If it ever grows large
        # we can swap in a reverse-line iterator.
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        events = []
        for ln in lines[-limit:]:
            ln = ln.strip()
            if not ln: continue
            try:
                events.append(json.loads(ln))
            except Exception:
                continue
        return events
    except Exception:
        return []


def _enrich_timeline_with_match_meta(events: list[dict],
                                       state_by_mid: dict) -> list[dict]:
    """Decorate events with home/away team names so the dashboard doesn't
    have to cross-reference. `state_by_mid` is the map of match_id -> its
    last_state (taken from orchestrator_state.json)."""
    out = []
    for e in events:
        mid = e.get("mid")
        meta = state_by_mid.get(mid) or {}
        out.append({
            **e,
            "home": meta.get("home"),
            "away": meta.get("away"),
        })
    return out


def _load_recent_learnings(limit: int = 10) -> list[dict]:
    """Tail-read the post-match learning ledger for the most recent
    structured attributions. Each entry is a per-version error breakdown
    written by post_match_review.review_one()."""
    fp = ROOT / "learnings" / "post_match_log.jsonl"
    if not fp.exists():
        return []
    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        out = []
        for ln in lines[-limit:]:
            ln = ln.strip()
            if not ln: continue
            try:
                e = json.loads(ln)
                # Compact version: trim full version analyses to just the
                # fields the dashboard needs (the full record stays on disk).
                vers = [{
                    "tag":              v.get("tag"),
                    "predicted_winner": v.get("predicted_winner"),
                    "p_home":           v.get("p_home_wins"),
                    "edge_pct":         v.get("edge_pct"),
                    "correct":          v.get("correct"),
                } for v in (e.get("versions") or [])]
                out.append({
                    "match":       e.get("match"),
                    "actual":      e.get("actual"),
                    "versions":    vers,
                    "deltas":      e.get("deltas"),
                    "attribution": e.get("attribution"),
                    "reviewed_at": e.get("reviewed_at"),
                })
            except Exception:
                continue
        return list(reversed(out))   # newest first
    except Exception:
        return []


def _load_state_meta() -> dict[str, dict]:
    """Read orchestrator_state.json and produce {match_id: {home, away}} so
    timeline events can be enriched without a Cricbuzz round-trip."""
    fp = RUNS_DIR / "orchestrator_state.json"
    if not fp.exists():
        return {}
    try:
        data = json.loads(fp.read_text())
        out = {}
        for mid, e in (data.get("tracked") or {}).items():
            ls = e.get("last_state") or {}
            out[mid] = {"home": ls.get("home"), "away": ls.get("away")}
        return out
    except Exception:
        return {}


def _ensure_versions_shape(pred: dict) -> dict:
    """Backwards-compat: predictions written before the phase machine fired
    don't have versions[] / current. Synthesize a single 'legacy' version
    so the frontend has a consistent shape to render against."""
    if not pred:
        return pred
    if pred.get("versions") and pred.get("current"):
        return pred
    legacy = {
        "tag": "legacy",
        "at": None,
        "prediction":    pred.get("prediction"),
        "base_learners": pred.get("base_learners"),
        "features":      pred.get("features"),
        "model_vs_book": pred.get("model_vs_book"),
        "xi":            pred.get("xi"),
    }
    return {**pred, "versions": [legacy], "current": "legacy"}


def _winner_from_matches_table(home: str, away: str, date_iso: str) -> str | None:
    """Fall back to CricSheet-derived matches table for definitive results."""
    if not (home and away and date_iso):
        return None
    try:
        d = (date_iso or "")[:10]
        con = connect()
        row = con.execute("""
            SELECT winner FROM matches
            WHERE start_date = CAST(? AS DATE)
              AND ((team_home = ? AND team_away = ?) OR (team_home = ? AND team_away = ?))
              AND winner IS NOT NULL
            ORDER BY match_id DESC LIMIT 1
        """, [d, home, away, away, home]).fetchone()
        con.close()
        return row[0] if row else None
    except Exception:
        return None


def main():
    print("Building dashboard data ...")

    preds      = _load_predictions()
    metrics    = _model_metrics()
    live       = _load_live_match()
    live_multi = _load_live_matches_multi()
    print("  computing top teams ...")
    top_t20 = _top_teams_per_format(["T20", "IT20"], top_n=15)
    top_odi = _top_teams_per_format(["ODI"], top_n=12)
    recent  = _recent_matches(40)
    stats   = _data_stats()
    bets    = _bet_summary()
    try:
        from . import compare_to_books
        comparison = compare_to_books.build()
    except Exception as e:
        comparison = {"n": 0, "msg": f"comparison failed: {e}"}

    # Attach result info to predictions when we have live data for same fixture.
    # Persist newly-completed grades back to the source prediction file so a
    # later live tick (different match, same team-pair) can't clobber them.
    pool = live_multi or live
    if preds["latest"]:
        preds["latest"] = _attach_pred_result(preds["latest"], pool)
        _persist_result_if_complete(preds["latest"])
    new_all = []
    for p in preds["all"]:
        graded = _attach_pred_result(p, pool)
        _persist_result_if_complete(graded)
        new_all.append(graded)
    preds["all"] = new_all

    # Merge settled predictions into recent_matches so freshly-played matches
    # appear in the recent-results panel even before CricSheet ingests them.
    recent = _merge_settled_into_recent(recent, preds["all"], max_total=40)

    # Backwards-compat: ensure every prediction has a versions[] / current shape
    # so the frontend can render uniformly across phase-machine-versioned and
    # legacy predictions.
    preds["all"] = [_ensure_versions_shape(p) for p in preds["all"]]
    if preds["latest"]:
        preds["latest"] = _ensure_versions_shape(preds["latest"])

    # Phase-machine timeline (last 80 events, enriched with match meta)
    state_meta = _load_state_meta()
    timeline = _enrich_timeline_with_match_meta(
        _load_timeline_events(limit=80), state_meta
    )
    learnings = _load_recent_learnings(limit=10)

    out = {
        "generated_at":     pd.Timestamp.now().isoformat(),
        "latest_prediction": preds["latest"],
        "all_predictions":   preds["all"],
        "model_metrics":     metrics,
        "top_teams_t20":     top_t20,
        "top_teams_odi":     top_odi,
        "recent_matches":    recent,
        "data_stats":        stats,
        "live_match":        live,
        "bets":              bets,
        "model_vs_book":     comparison,
        "match_timeline":    timeline,
        "learnings":         learnings,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"  wrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")
    print(f"  predictions: {len(preds['all'])}")
    print(f"  recent_matches: {len(recent)}")
    print(f"  top_teams_t20: {len(top_t20)}")

    # Mirror to data/ aliases so the design terminal can fetch
    try:
        _write_design_aliases(out, preds["all"])
        print(f"  design aliases: data/ + data/preds/")
    except Exception as e:
        print(f"  WARN: design alias write failed: {e}")


if __name__ == "__main__":
    main()
