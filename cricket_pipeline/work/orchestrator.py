"""Cricket dashboard orchestrator — runs forever, no manual commands needed.

Three concurrent loops in one process:

  1. discover_loop   (every  5 min): scan Cricbuzz live-scores page for new
                                     match IDs, register them
  2. refresh_loop    (every 30 sec): fetch live state for every tracked match,
                                     write live_matches.json and data.json
  3. predict_loop    (every  5 min): for any tracked match without a saved
                                     prediction, train + predict (one at a
                                     time so we don't bog down the box)

A fourth thread runs the HTTP server on 127.0.0.1:4173.

All errors are caught + logged; loops never die.

Run once, leave running:
  cd Waundering
  .venv/Scripts/python.exe -m cricket_pipeline.work.orchestrator

To run as a fully detached background service on Windows-bash:
  nohup .venv/Scripts/python.exe -m cricket_pipeline.work.orchestrator \\
        > cricket_pipeline/work/runs/orchestrator.log 2>&1 &
"""
from __future__ import annotations

import json
import logging
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import requests

# Load .env from project root (no python-dotenv dependency)
def _load_dotenv():
    import os as _os
    from pathlib import Path as _Path
    fp = _Path(__file__).resolve().parents[2] / ".env"
    if not fp.exists(): return
    for ln in fp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, _, v = ln.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in _os.environ:
            _os.environ[k] = v
_load_dotenv()

from . import live_match as lm
from . import export_dashboard_data as edd
from . import filters as F
from . import bet_engine as BET
from . import match_phase as mp
from ..ingest import odds as ODDS
from ..ingest import lineup as LINEUP
from ..ingest import pitch as PITCH
from ..ingest import open_meteo as METEO


# Suppress the brief CMD/console window flash when subprocess.run() spawns
# predict_match / cricsheet ingest on Windows. Harmless 0 on POSIX so the
# same flag works cross-platform.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------- paths + constants ----------

ROOT             = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR  = ROOT / "predictions"
RUNS_DIR         = ROOT / "cricket_pipeline" / "work" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH         = RUNS_DIR / "orchestrator.log"
STATE_PATH       = RUNS_DIR / "orchestrator_state.json"
LIVE_MULTI_PATH  = RUNS_DIR / "live_matches.json"

DISCOVER_INTERVAL_SEC  = 300
LIVE_INTERVAL_SEC      = 30        # refresh live states + live_matches.json
EXPORT_INTERVAL_SEC    = 180       # rebuild full data.json (slower; recomputes Elo+top_teams)
PREDICT_INTERVAL_SEC   = 300
ODDS_INTERVAL_SEC      = 1800      # 30 min — keep within The Odds API free quota
LINEUP_INTERVAL_SEC    = 120       # poll cricbuzz match-squads pages
INGEST_INTERVAL_SEC    = 86400     # 24h — re-pull CricSheet to capture newly-finished matches
PHASE_INTERVAL_SEC     = 30        # match-phase machine: drive transitions + fire timed actions
HTTP_PORT              = 4173

# CricSheet datasets to refresh daily (small + relevant). Skip the full
# associate-tour stuff — it bloats the corpus without lifting accuracy.
INGEST_DATASETS = (
    "ipl_json", "t20s_json", "odis_json", "tests_json",
    "bbl_json", "cpl_json", "sa20_json", "ilt20_json",
    "the_hundred_men", "vitality_blast",
)

CRICBUZZ_LIVE_URL = "https://www.cricbuzz.com/cricket-match/live-scores"

# Filter: only auto-predict matches in these tournaments (regex on slug).
# Add more as you like; keep it tight to avoid burning compute on women's
# regional T20I qualifier matches that the user doesn't care about.
SLUG_ALLOWLIST = re.compile(
    r"(indian-premier-league|\bipl-\d{4}|"
    r"\bt20-world-cup\b|\bodi-world-cup\b|"
    r"big-bash-league|caribbean-premier-league|sa20|ilt20|"
    r"major-league-cricket|the-hundred|bangladesh-premier-league|"
    r"lanka-premier-league|"
    # Internationals — bilateral series in main formats
    r"\d+(st|nd|rd|th)-(t20i|odi)\b|"
    r"-(test|test-match)-\d{4}\b)",
    re.I,
)
# Slugs to explicitly skip even if matched above
SLUG_BLOCKLIST = re.compile(
    r"(women|youth|under-?19|unofficial|tour-match|warm-up|practice)",
    re.I,
)


# ---------- logging ----------

LOG = logging.getLogger("orchestrator")
LOG.setLevel(logging.INFO)
_handler_file = logging.FileHandler(LOG_PATH, encoding="utf-8")
_handler_stream = logging.StreamHandler(sys.stdout)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_handler_file.setFormatter(_fmt); _handler_stream.setFormatter(_fmt)
LOG.addHandler(_handler_file); LOG.addHandler(_handler_stream)


# ---------- shared state ----------

class State:
    def __init__(self):
        self.lock = threading.Lock()
        # tracked: mid -> {slug, last_state_dict, prediction_done(bool)}
        self.tracked: dict[str, dict] = {}
        self.shutdown = threading.Event()
        self._load()

    def _load(self):
        if not STATE_PATH.exists(): return
        try:
            data = json.loads(STATE_PATH.read_text())
            self.tracked = data.get("tracked", {})
            LOG.info(f"Loaded state: tracking {len(self.tracked)} matches")
        except Exception as e:
            LOG.warning(f"State load failed: {e}")

    def _save(self):
        try:
            STATE_PATH.write_text(json.dumps({"tracked": self.tracked}, indent=2, default=str))
        except Exception as e:
            LOG.warning(f"State save failed: {e}")

    def register(self, mid: str, slug: str):
        with self.lock:
            existing = self.tracked.get(mid)
            if existing:
                existing["slug"] = slug
            else:
                self.tracked[mid] = {"slug": slug, "prediction_done": False, "last_state": None}
                LOG.info(f"+ tracking new match {mid}: {slug}")
            self._save()

    def update_state(self, mid: str, state: dict):
        """Replace last_state with a fresh fetch — but defensively preserve
        a previously-known venue if the new fetch returned None/empty.
        Cricbuzz strips matchHeader.matchVenue from past-match pages, so
        a re-fetch of a completed fixture would otherwise wipe a venue
        we'd correctly extracted while the match was live."""
        with self.lock:
            entry = self.tracked.get(mid)
            if not entry: return
            old = entry.get("last_state") or {}
            old_venue = old.get("venue")
            if old_venue and not state.get("venue"):
                state = {**state, "venue": old_venue}
            entry["last_state"] = state
            self._save()

    def mark_predicted(self, mid: str):
        with self.lock:
            entry = self.tracked.get(mid)
            if entry: entry["prediction_done"] = True
            self._save()

    def matches_to_predict(self) -> list[tuple[str, dict]]:
        """Tracked matches the legacy predict_loop should handle.

        The phase machine (phase_loop) now owns prediction firing for any
        match with a known start_ts — it emits versioned predictions
        (pre_match_v0 / pre_start_v1 / toss_aware_v2) at the right
        moments. predict_loop falls back to handling matches where the
        phase machine hasn't taken over yet (no start_ts, or start_ts
        is in the past with no completion signal).

        Returns matches that:
          - have home/away
          - are not complete
          - don't have today's prediction file
          - either have NO start_ts (phase machine can't act yet) OR
            no phase-machine prediction has fired yet
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Action keys phase_loop owns
        phase_owned = {"SCHEDULED.pre_match_v0",
                        "PRE_START.pre_start_v1",
                        "PRE_START.toss_aware_v2"}
        with self.lock:
            out = []
            for mid, e in self.tracked.items():
                state = e.get("last_state") or {}
                home, away = state.get("home"), state.get("away")
                if not home or not away: continue
                if state.get("is_complete"): continue
                # Phase machine has a clock for this match AND has fired (or is
                # about to fire) — skip in predict_loop to avoid double work.
                if e.get("start_ts") and any(k in (e.get("actions_fired") or {})
                                              for k in phase_owned):
                    continue
                fname = _canonical_fname(home, away, today)
                todays_file = PREDICTIONS_DIR / fname
                if todays_file.exists():
                    continue
                out.append((mid, {**state, "slug": e["slug"]}))
            return out

    def all_states(self) -> list[dict]:
        with self.lock:
            out = []
            for mid, e in self.tracked.items():
                state = e.get("last_state")
                if state: out.append(state)
            return out

    def list_tracked(self) -> list[tuple[str, str]]:
        with self.lock:
            return [(mid, e.get("slug","")) for mid, e in self.tracked.items()]

    # ---- Live XI hooks (Step 2) ----
    def set_xi(self, mid: str, xi_a: list[str], xi_b: list[str], team_a: str, team_b: str):
        """Record an announced XI for a tracked match. If the XI is new (or
        differs from what we last saw), clear `prediction_done` so the
        predict_loop re-runs with the announced lineup."""
        with self.lock:
            entry = self.tracked.get(mid)
            if not entry: return False
            existing = entry.get("announced_xi") or {}
            new = {"team_a": team_a, "team_b": team_b,
                   "xi_a": list(xi_a), "xi_b": list(xi_b),
                   "fetched_at": datetime.now(timezone.utc).isoformat()}
            same = (existing.get("team_a") == team_a and existing.get("team_b") == team_b
                    and existing.get("xi_a") == new["xi_a"] and existing.get("xi_b") == new["xi_b"])
            entry["announced_xi"] = new
            if not same:
                entry["prediction_done"] = False    # force re-prediction
                entry["xi_changed_at"]   = new["fetched_at"]
                self._save()
                LOG.info(f"+ XI announced for match {mid}: {team_a}={len(xi_a)}, {team_b}={len(xi_b)} → re-predict queued")
                return True
            self._save()
            return False

    def announced_xi(self, mid: str) -> dict | None:
        with self.lock:
            entry = self.tracked.get(mid)
            return (entry or {}).get("announced_xi")


STATE = State()


# ---------- prediction helper ----------

def _safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s).strip("_")


def _canonical_fname(home: str, away: str, date: str) -> str:
    """Canonical prediction filename for a fixture. Sorts team names so the
    same match always maps to the same file regardless of which side
    Cricbuzz currently labels as home — Cricbuzz silently swaps home/away
    on some pages mid-match, which would otherwise split the version
    trajectory across two files (one per ordering)."""
    a = _safe_filename(home or "")
    b = _safe_filename(away or "")
    lo, hi = sorted([a, b])
    return f"{lo}_vs_{hi}_{date}.json"


def _normalise_toss_decision(raw: str | None) -> str | None:
    """Cricbuzz uses 'Bowling'/'Batting' (long form) or 'bowl'/'bat' (short).
    predict_match.py CLI accepts only 'bat' or 'field'. Map appropriately —
    in cricket, opting to bowl == fielding first."""
    if not raw: return None
    r = raw.strip().lower()
    if r in ("bat", "batting"):                     return "bat"
    if r in ("bowl", "bowling", "field", "fielding"): return "field"
    return None


def predict_match(state: dict, force: bool = False,
                   xi_home: list[str] | None = None,
                   xi_away: list[str] | None = None,
                   toss_winner: str | None = None,
                   toss_decision: str | None = None) -> bool:
    """Run predict_match.py as a subprocess. Returns True on success.

    Subprocess isolation matters: predict_match imports + trains a 5-model
    ensemble per call, leaks DuckDB connections, etc. Running it as a
    subprocess sidesteps those concerns (and lets us cap memory if needed).

    `force=True` overwrites an existing saved prediction. `xi_home/xi_away`
    pass announced playing XIs to override the proxy lookup. `toss_winner/
    toss_decision` are forwarded as `--toss-winner/--toss-decision` so the
    feature builder can use the actual toss state when known. None means
    "model predicts without toss info" — used for pre-toss prediction
    versions so retroactive firings don't accidentally bake in info the
    model wouldn't have had at that point.
    """
    home  = state["home"]; away = state["away"]
    venue = state.get("venue") or "Unknown Venue"
    fmt   = state.get("match_format") or "T20"
    date  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    toss_decision = _normalise_toss_decision(toss_decision)

    fname = _canonical_fname(home, away, date)
    out_path = PREDICTIONS_DIR / fname
    if out_path.exists() and not force:
        LOG.info(f"Prediction already exists for {home} vs {away} on {date}, skipping")
        return True

    tag = "RE-PREDICT" if (out_path.exists() and force) else "PREDICT"
    extras = []
    if xi_home or xi_away:
        extras.append(f"xi: home={len(xi_home or [])}/away={len(xi_away or [])}")
    if toss_winner:
        extras.append(f"toss: {toss_winner} -> {toss_decision or '?'}")
    LOG.info(f"{tag}  {home} vs {away}  ({fmt} @ {venue}, {date})"
             + (f"  [{', '.join(extras)}]" if extras else ""))
    py = sys.executable
    cmd = [py, "-m", "cricket_pipeline.work.predict_match",
           "--home", home, "--away", away, "--venue", venue,
           "--format", fmt, "--date", date,
           "--save", str(out_path.relative_to(ROOT)),
           "--fast"]                                   # autonomous loop uses fast ensemble
    if force:
        cmd.append("--force")
    if xi_home: cmd += ["--xi-home", ",".join(xi_home)]
    if xi_away: cmd += ["--xi-away", ",".join(xi_away)]
    if toss_winner and toss_decision:
        cmd += ["--toss-winner", toss_winner, "--toss-decision", toss_decision]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=_NO_WINDOW,
        )
        dt = time.time() - t0
        if proc.returncode != 0:
            LOG.error(f"predict_match exited {proc.returncode} after {dt:.0f}s — stderr tail:\n{proc.stderr[-500:]}")
            return False
        # parse a couple of lines for log
        last = [ln for ln in proc.stdout.splitlines() if "Favored:" in ln or "P(" in ln]
        for ln in last[-3:]:
            LOG.info(f"   {ln.strip()}")
        LOG.info(f"   saved -> {out_path.name}  ({dt:.0f}s)")
        return True
    except subprocess.TimeoutExpired:
        LOG.error(f"predict_match TIMED OUT after 900s for {home} vs {away}")
        return False
    except Exception as e:
        LOG.error(f"predict_match failed: {e}\n{traceback.format_exc()}")
        return False


# ---------- loops ----------

def discover_loop():
    LOG.info(f"discover_loop started (every {DISCOVER_INTERVAL_SEC}s)")
    while not STATE.shutdown.is_set():
        try:
            r = requests.get(CRICBUZZ_LIVE_URL,
                             headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"},
                             timeout=20)
            if r.status_code == 200:
                ids = re.findall(r"/live-cricket-scores/(\d+)/([a-z0-9-]+)", r.text)
                seen = set(); n_new = 0
                for mid, slug in ids:
                    if mid in seen: continue
                    seen.add(mid)
                    if not SLUG_ALLOWLIST.search(slug):
                        continue
                    if SLUG_BLOCKLIST.search(slug):
                        continue
                    if F.is_blocked_slug(slug):
                        continue
                    if mid not in {m for m, _ in STATE.list_tracked()}:
                        n_new += 1
                    STATE.register(mid, slug)
                if n_new:
                    LOG.info(f"discovered {n_new} new tracked matches "
                             f"(total tracked: {len(STATE.list_tracked())})")
            else:
                LOG.warning(f"discover: HTTP {r.status_code}")
        except Exception as e:
            LOG.warning(f"discover failed: {e}")
        STATE.shutdown.wait(DISCOVER_INTERVAL_SEC)


def _fetch_one(mid: str, slug: str) -> tuple[str, dict | None]:
    try:
        raw = lm.fetch_match_state(mid, slug)
        d = lm.normalise_for_dashboard(raw)
        if d and F.is_blocked_match(d.get("home"), d.get("away")):
            return mid, None
        return mid, d
    except Exception as e:
        LOG.warning(f"fetch {mid} failed: {e}")
        return mid, None


def live_loop():
    """Fast — fetch live states for tracked matches, write live_*.json files.
    Does NOT rebuild data.json (that's the slow Elo recompute)."""
    LOG.info(f"live_loop started (every {LIVE_INTERVAL_SEC}s)")
    while not STATE.shutdown.is_set():
        t0 = time.time()
        try:
            tracked = STATE.list_tracked()
            if tracked:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda t: _fetch_one(*t), tracked))
                n_ok = 0
                for mid, st in results:
                    if st:
                        STATE.update_state(mid, st)
                        n_ok += 1
                states = STATE.all_states()
                LIVE_MULTI_PATH.write_text(json.dumps(states, indent=2, default=str))
                # Auto-settle any pending bets whose match just completed
                try:
                    res = BET.settle_bets_against_results(states)
                    if res.get("settled"):
                        LOG.info(f"bets: settled {res['settled']} pending bets from completed matches")
                except Exception as e:
                    LOG.warning(f"bet settlement failed: {e}")
                # featured = actually-playing match preferred (toss done,
                # score updating). Falls through to non-complete (covers
                # SCHEDULED) and finally to anything. Avoids the case where
                # an abandoned-no-toss fixture from yesterday outranks a
                # truly live match in the dashboard's LiveStrip.
                featured = (
                    next((s for s in states if mp.is_in_play(s.get("status"))), None)
                    or next((s for s in states if not s.get("is_complete")
                             and not mp.is_abandoned(s.get("status"))), None)
                    or next((s for s in states if not s.get("is_complete")), None)
                    or (states[0] if states else None)
                )
                if featured:
                    (RUNS_DIR / "live_match.json").write_text(json.dumps(featured, indent=2, default=str))
                LOG.info(f"live: {n_ok}/{len(tracked)} states OK ({time.time()-t0:.1f}s)")
        except Exception as e:
            LOG.warning(f"live refresh failed: {e}\n{traceback.format_exc()}")
        STATE.shutdown.wait(LIVE_INTERVAL_SEC)


def export_loop():
    """Slow — rebuild full data.json (Elo + top_teams + recent_matches).
    Triggered every EXPORT_INTERVAL_SEC. Retries on transient DuckDB locks
    (a parallel predict_match subprocess holds the write-lock briefly)."""
    LOG.info(f"export_loop started (every {EXPORT_INTERVAL_SEC}s)")
    STATE.shutdown.wait(5)
    while not STATE.shutdown.is_set():
        t0 = time.time()
        attempts = 0; max_attempts = 6
        while attempts < max_attempts and not STATE.shutdown.is_set():
            attempts += 1
            try:
                edd.main()
                LOG.info(f"export: data.json refreshed ({time.time()-t0:.1f}s, try {attempts})")
                break
            except Exception as e:
                msg = str(e)
                if "being used by another process" in msg or "Could not set lock" in msg:
                    LOG.info(f"export: DuckDB busy (try {attempts}/{max_attempts}), waiting 10s")
                    STATE.shutdown.wait(10)
                    continue
                LOG.warning(f"export failed: {e}\n{traceback.format_exc()}")
                break
        STATE.shutdown.wait(EXPORT_INTERVAL_SEC)


def odds_loop():
    """Pull bookmaker odds — quota-aware so the free tier (500/mo) lasts.

    Strategy:
      1. Only poll sports where we currently track an upcoming match.
      2. Poll cadence depends on time-until-toss for the soonest tracked match:
            > 24h:   every 4h
            6-24h:   every  2h
            < 6h:    every 30min
            in-play: every 5min
    """
    LOG.info("odds_loop started (quota-aware)")
    has_key = bool(__import__("os").environ.get("THE_ODDS_API_KEY"))
    if not has_key:
        LOG.info("odds_loop: THE_ODDS_API_KEY not set — odds disabled "
                 "(set the env var and restart to enable bookmaker features)")
    STATE.shutdown.wait(15)
    while not STATE.shutdown.is_set():
        if not has_key:
            STATE.shutdown.wait(3600); continue

        # Decide which sports + how soon to poll again
        states = STATE.all_states()
        sport_keys, next_wait = _odds_sports_and_cadence(states)
        if not sport_keys:
            LOG.info("odds: no upcoming matches in any tracked sport — sleeping 1h")
            STATE.shutdown.wait(3600); continue

        t0 = time.time()
        try:
            summary = ODDS.fetch_and_store(sport_keys=sport_keys)
            LOG.info(f"odds: {summary['rows']} rows / {summary['events']} events "
                     f"across {len(sport_keys)} sport(s) ({time.time()-t0:.1f}s); "
                     f"next in {next_wait//60} min")
        except Exception as e:
            LOG.warning(f"odds fetch failed: {e}")
        STATE.shutdown.wait(next_wait)


def _odds_sports_and_cadence(states: list[dict]) -> tuple[list[str], int]:
    """Map our tracked-match competitions/slugs → odds-API sport keys.
    Return (sport_keys, sleep_seconds)."""
    if not states: return ([], 3600)
    sport_set = set()
    soonest_min_until_toss = None
    now = time.time()
    in_play = False
    for s in states:
        slug = (s.get("slug") or "").lower()
        # map slug → sport_key
        if "indian-premier-league" in slug or "ipl" in slug:
            sport_set.add("cricket_ipl")
        elif "big-bash" in slug:
            sport_set.add("cricket_big_bash")
        elif "caribbean-premier" in slug:
            sport_set.add("cricket_caribbean_premier_league")
        elif "test" in slug:
            sport_set.add("cricket_test_match")
        elif "odi" in slug:
            sport_set.add("cricket_odi")
        elif "t20i" in slug:
            sport_set.add("cricket_t20i")
        # status / timing
        status = (s.get("status") or "").lower()
        if "won" not in status and any(k in status for k in ("over", "ball", "innings", "trail")):
            in_play = True
        # try to read commence time from status "Match starts at Apr 27, 14:00 GMT"
        import re
        m = re.search(r"starts at ([A-Za-z]+ \d+, \d+:\d+) GMT", s.get("status") or "")
        if m:
            try:
                from datetime import datetime, timezone
                # Year may be missing; assume current
                yr = datetime.now(timezone.utc).year
                t = datetime.strptime(f"{m.group(1)} {yr}", "%b %d, %H:%M %Y").replace(tzinfo=timezone.utc)
                mins = (t.timestamp() - now) / 60
                if mins > 0 and (soonest_min_until_toss is None or mins < soonest_min_until_toss):
                    soonest_min_until_toss = mins
            except Exception: pass

    if in_play:
        return (sorted(sport_set), 5 * 60)
    if soonest_min_until_toss is None:
        return (sorted(sport_set), 4 * 3600)
    if soonest_min_until_toss < 6 * 60:
        return (sorted(sport_set), 30 * 60)
    if soonest_min_until_toss < 24 * 60:
        return (sorted(sport_set), 2 * 3600)
    return (sorted(sport_set), 4 * 3600)


def ingest_loop():
    """Once a day, re-pull CricSheet datasets so newly-finished matches land
    in the `matches` / `innings` / `balls` tables. This:
      1. Grades any predictions whose live source couldn't parse the winner
         (the `_winner_from_matches_table` fallback in export_dashboard_data
         picks up the official CricSheet result once published).
      2. Feeds future training cycles with fresh data automatically.

    First run sleeps 5 min after start so we don't compete with the initial
    discover/live/export bursts."""
    LOG.info(f"ingest_loop started (every {INGEST_INTERVAL_SEC // 3600}h, "
             f"{len(INGEST_DATASETS)} datasets)")
    STATE.shutdown.wait(300)
    while not STATE.shutdown.is_set():
        py = sys.executable
        for ds in INGEST_DATASETS:
            if STATE.shutdown.is_set(): break
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [py, "-m", "cricket_pipeline.pipeline", "cricsheet",
                     "--dataset", ds],
                    cwd=str(ROOT), capture_output=True, text=True, timeout=900,
                    env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
                    creationflags=_NO_WINDOW,
                )
                last = (proc.stdout or "").splitlines()[-1] if proc.stdout else ""
                LOG.info(f"ingest: {ds:<22} {last[:80]}  ({time.time()-t0:.0f}s)")
            except subprocess.TimeoutExpired:
                LOG.warning(f"ingest: {ds} timed out (15min)")
            except Exception as e:
                LOG.warning(f"ingest: {ds} failed: {e}")
        STATE.shutdown.wait(INGEST_INTERVAL_SEC)


def lineup_loop():
    """Poll Cricbuzz match-squads pages for tracked, not-yet-complete matches.
    When a full XI is announced (≥11 names per side), record it via STATE.set_xi
    which clears `prediction_done` so predict_loop re-runs with the announced
    lineup. ~2 min cadence — XIs are stable once published, so polling lightly
    is fine."""
    LOG.info(f"lineup_loop started (every {LINEUP_INTERVAL_SEC}s)")
    STATE.shutdown.wait(25)
    while not STATE.shutdown.is_set():
        try:
            polled = 0; new_announces = 0
            for mid, e in list(STATE.tracked.items()):
                state = e.get("last_state") or {}
                if state.get("is_complete"): continue
                if (e.get("announced_xi") or {}).get("xi_a") and len((e.get("announced_xi") or {}).get("xi_a", [])) >= 11:
                    # already have a full XI — skip until a new poll cycle
                    # we still re-poll occasionally (every ~6 cycles) to catch late changes
                    if (e.get("xi_poll_n") or 0) % 6 != 0:
                        e["xi_poll_n"] = (e.get("xi_poll_n") or 0) + 1
                        continue
                slug = e.get("slug","")
                try:
                    info = LINEUP.fetch_by_match_id(mid, slug)
                except Exception as fetch_err:
                    LOG.debug(f"lineup fetch {mid} failed: {fetch_err}")
                    continue
                polled += 1
                if not info.get("announced"):
                    e["xi_poll_n"] = (e.get("xi_poll_n") or 0) + 1
                    continue
                # We have a full XI — record it
                changed = STATE.set_xi(mid, info["xi_a"], info["xi_b"],
                                         info["team_a"], info["team_b"])
                if changed: new_announces += 1
                e["xi_poll_n"] = (e.get("xi_poll_n") or 0) + 1
                # Also pull a pitch report when the XI lands (pitch is usually
                # published in the same news cycle as the XI announcement).
                try:
                    PITCH.store_for_match(mid, slug)
                except Exception as pe:
                    LOG.debug(f"pitch store {mid} failed: {pe}")
                # And fetch weather forecast for the venue+date
                try:
                    state = e.get("last_state") or {}
                    venue = state.get("venue")
                    # Match date — derive from the live status text or default to today
                    from datetime import date as _date
                    METEO.fetch_forecast(venue or "Unknown", _date.today())
                except Exception as we:
                    LOG.debug(f"weather forecast {mid} failed: {we}")
            if polled or new_announces:
                LOG.info(f"lineup: polled {polled} matches, {new_announces} new XI announcement(s)")
        except Exception as ex:
            LOG.warning(f"lineup_loop iteration failed: {ex}")
        STATE.shutdown.wait(LINEUP_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Match-phase machine
# ---------------------------------------------------------------------------

PHASE_LOG = RUNS_DIR / "match_timeline.jsonl"


def _phase_log(event: dict) -> None:
    """Append-only event log of phase transitions and action firings."""
    try:
        event = {**event, "at": datetime.now(timezone.utc).isoformat()}
        with PHASE_LOG.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass    # observability is best-effort


def _resolve_start_ts(state: dict, entry: dict) -> int | None:
    """Find start_ts for a match, in order of authority:
       1. live_match's matchStartTimestamp (most reliable)
       2. parse Cricbuzz status text "Match starts at Apr 30, 14:00 GMT"
       3. existing entry.start_ts (don't downgrade once set)"""
    raw_ts = state.get("match_start_ts")
    if isinstance(raw_ts, (int, float)) and raw_ts > 0:
        return int(raw_ts)
    parsed = mp.parse_start_ts_from_status(state.get("status"))
    if parsed:
        return parsed
    return entry.get("start_ts")


_VERSION_ORDER = {
    "pre_match_v0":  0,
    "pre_start_v1":  1,
    "toss_aware_v2": 2,
    "legacy":       -1,    # before phase machine
}


def _wrap_prediction_with_version(out_path: Path, version_tag: str,
                                    prior_versions: list[dict] | None = None) -> None:
    """Wrap a `predictions/*.json` file's prediction body in a `versions[]`
    array. `predict_match.py` always overwrites the entire file when it
    runs, so the caller must capture the prior `versions` array BEFORE
    invoking predict_match and pass it here — otherwise each new version
    silently destroys the trajectory.

    Idempotent across same-tag re-fires (replaces in place). Sorts by
    natural phase order so the dashboard always renders chronologically."""
    if not out_path.exists():
        return
    try:
        d = json.loads(out_path.read_text())
        snapshot = {
            "tag":            version_tag,
            "at":             datetime.now(timezone.utc).isoformat(),
            "prediction":     d.get("prediction"),
            "base_learners":  d.get("base_learners"),
            "features":       d.get("features"),
            "model_vs_book":  d.get("model_vs_book"),
            "xi":             d.get("xi"),
        }
        # Build the merged versions list: prior history minus any same-tag
        # entry (re-fires replace), plus the new snapshot.
        merged = [v for v in (prior_versions or [])
                   if v.get("tag") != version_tag]
        merged.append(snapshot)
        merged.sort(key=lambda v: _VERSION_ORDER.get(v.get("tag"), 99))
        d["versions"] = merged
        d["current"]  = version_tag
        out_path.write_text(json.dumps(d, indent=2, default=str))
    except Exception as e:
        LOG.warning(f"version-wrap failed for {out_path.name}: {e}")


def _read_prior_versions(out_path: Path) -> list[dict]:
    """Snapshot the existing `versions[]` array on disk, BEFORE predict_match
    overwrites the file. Pass this into _wrap_prediction_with_version after
    the predict call to preserve trajectory history."""
    if not out_path.exists():
        return []
    try:
        d = json.loads(out_path.read_text())
        return list(d.get("versions") or [])
    except Exception:
        return []


def _xi_for_state(mid: str, home: str) -> tuple[list[str] | None, list[str] | None]:
    """Look up the announced XI from STATE and align team_a/team_b to home/away."""
    xi = STATE.announced_xi(mid)
    if not (xi and xi.get("xi_a") and xi.get("xi_b")):
        return None, None
    ta, tb = xi["team_a"], xi["team_b"]
    if (ta or "").lower() == (home or "").lower():
        return xi["xi_a"], xi["xi_b"]
    if (tb or "").lower() == (home or "").lower():
        return xi["xi_b"], xi["xi_a"]
    return None, None       # team-name mismatch — fall back to proxy


def _fire_phase_action(mid: str, entry: dict, action: str) -> bool:
    """Execute a phase action. Returns True iff caller should record firing.

    Pre-match / pre-start / toss-aware actions all run predict_match with
    different feature inputs, then post-process the output file into a
    versions[] array tagged with the version. Settle and review are
    boundary markers — actual settling happens in live_loop's bet
    settlement; review happens via the existing post-match-review module.
    """
    state = entry.get("last_state") or {}
    home, away = state.get("home"), state.get("away")
    # Some tracked matches sit at the discovery edge with home/away
    # unresolved (Cricbuzz hasn't loaded the teams panel, or it's a
    # placeholder entry from the live-scores scan). predict_match would
    # crash on a None argv element; bail early so the phase-tick loop
    # retries on the next tick once team names land. Pitch/weather and
    # bookkeeping actions can still proceed where possible.
    if not home or not away:
        if action in (mp.A_PRE_MATCH_PRED, mp.A_PRE_START_PRED, mp.A_TOSS_AWARE_PRED):
            return False    # don't record as fired; retry next tick
    venue = state.get("venue") or "Unknown Venue"
    fmt   = state.get("match_format") or "T20"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fname = _canonical_fname(home or "", away or "", today)
    out_path = PREDICTIONS_DIR / fname

    # Pitch + weather snapshot — best-effort, lineup_loop also does this
    # opportunistically; firing here makes it idempotent and timestamped.
    if action == mp.A_PITCH_WEATHER:
        try:
            slug = entry.get("slug") or ""
            if slug:
                PITCH.store_for_match(mid, slug)
        except Exception as e:
            LOG.debug(f"pitch action {mid}: {e}")
        try:
            from datetime import date as _date
            METEO.fetch_forecast(venue or "Unknown", _date.today())
        except Exception as e:
            LOG.debug(f"weather action {mid}: {e}")
        return True

    # Versioned predictions — pre_match (no XI, no toss), pre_start (XI ok,
    # no toss), toss_aware (XI + toss). Each version represents the model's
    # call given the info available at that phase, so we explicitly WITHHOLD
    # downstream info from upstream versions even when firing late — keeps
    # the trajectory honest if the orchestrator catches up after a window.
    if action in (mp.A_PRE_MATCH_PRED, mp.A_PRE_START_PRED, mp.A_TOSS_AWARE_PRED):
        version_tag = {
            mp.A_PRE_MATCH_PRED:  "pre_match_v0",
            mp.A_PRE_START_PRED:  "pre_start_v1",
            mp.A_TOSS_AWARE_PRED: "toss_aware_v2",
        }[action]
        xi_home = xi_away = None
        if action != mp.A_PRE_MATCH_PRED:
            xi_home, xi_away = _xi_for_state(mid, home)
        toss_w = toss_d = None
        if action == mp.A_TOSS_AWARE_PRED:
            toss_w = entry.get("toss_winner") or state.get("toss_winner")
            toss_d = entry.get("toss_decision") or state.get("toss_decision")
        synthetic_state = {
            **state,
            "home": home, "away": away,
            "venue": venue, "match_format": fmt,
        }
        # Strip toss from the state dict for non-toss-aware versions so the
        # snapshot represents "model without toss info"
        if action != mp.A_TOSS_AWARE_PRED:
            synthetic_state.pop("toss_winner", None)
            synthetic_state.pop("toss_decision", None)
        # Capture trajectory BEFORE predict_match overwrites the file
        prior_versions = _read_prior_versions(out_path)
        ok = predict_match(synthetic_state, force=True,
                            xi_home=xi_home, xi_away=xi_away,
                            toss_winner=toss_w, toss_decision=toss_d)
        if ok:
            _wrap_prediction_with_version(out_path, version_tag,
                                           prior_versions=prior_versions)
            STATE.mark_predicted(mid)
        return ok

    # Boundary markers — return True so they're recorded as fired.
    # Actual settling is done by live_loop's BET.settle_bets_against_results.
    # Review is a future enhancement; for now we just mark the phase.
    if action == mp.A_SETTLE:
        return True
    if action == mp.A_REVIEW:
        # Phase-aware per-match review writes a structured ledger line
        # to learnings/post_match_log.jsonl with per-version attribution.
        # Use the match's start date (not today's UTC) so we look up the
        # correct prediction file when reviewing fixtures completed earlier.
        try:
            from .. import post_match_review as pmr
            match_date = today
            if entry.get("start_ts"):
                match_date = datetime.fromtimestamp(
                    int(entry["start_ts"]), tz=timezone.utc
                ).strftime("%Y-%m-%d")
            if hasattr(pmr, "review_one"):
                res = pmr.review_one(home=home, away=away, date=match_date)
                if res:
                    LOG.info(f"review: ledger entry for {home} vs {away} "
                             f"({match_date}) — final_correct="
                             f"{res.get('attribution',{}).get('final_correct')}")
            elif hasattr(pmr, "main"):
                pmr.main()
        except Exception as e:
            LOG.debug(f"post-match review {mid}: {e}")
        return True

    return False


def _phase_tick(mid: str, entry: dict) -> None:
    """One phase-machine step for one match. All STATE writes go through
    STATE.lock; the read snapshot above runs lock-free for speed."""
    state = entry.get("last_state") or {}

    # 1) Resolve / refresh start_ts
    fresh_ts = _resolve_start_ts(state, entry)
    old_ts = entry.get("start_ts")
    if fresh_ts and fresh_ts != old_ts:
        with STATE.lock:
            e = STATE.tracked.get(mid)
            if e:
                if mp.is_meaningful_reschedule(old_ts, fresh_ts):
                    # Match was meaningfully rescheduled — rewind any phase
                    # actions for the current phase and after (so they re-fire
                    # against the new clock).
                    cur_phase = e.get("phase") or mp.Phase.SCHEDULED.value
                    mp.reset_actions_for_phase_and_after(e, cur_phase)
                    _phase_log({"event": "rescheduled", "mid": mid,
                                  "old_ts": old_ts, "new_ts": fresh_ts})
                e["start_ts"] = fresh_ts
                STATE._save()
        entry = STATE.tracked.get(mid, entry)

    # 2) Detect toss (from live status text) once we're in PRE_START / LIVE
    if not entry.get("toss_seen_at"):
        winner, decision = mp.detect_toss(state)
        if winner:
            with STATE.lock:
                e = STATE.tracked.get(mid)
                if e:
                    e["toss_seen_at"] = mp.now_utc()
                    e["toss_winner"] = winner
                    e["toss_decision"] = decision
                    STATE._save()
            _phase_log({"event": "toss", "mid": mid,
                          "toss_winner": winner, "toss_decision": decision})
            entry = STATE.tracked.get(mid, entry)

    # 3) Compute phase + transition if changed
    new_phase = mp.compute_next_phase(entry)
    if new_phase != entry.get("phase"):
        old_phase = entry.get("phase")
        with STATE.lock:
            e = STATE.tracked.get(mid)
            if e:
                e["phase"] = new_phase
                hist = e.setdefault("phase_history", [])
                hist.append({"phase": new_phase,
                              "at": datetime.now(timezone.utc).isoformat(),
                              "from": old_phase})
                if new_phase == mp.Phase.COMPLETE.value and not e.get("completed_at"):
                    e["completed_at"] = mp.now_utc()
                STATE._save()
        LOG.info(f"phase: {mid} {old_phase or 'NEW'} -> {new_phase} "
                 f"({state.get('home')} vs {state.get('away')})")
        _phase_log({"event": "transition", "mid": mid,
                      "from": old_phase, "to": new_phase})
        entry = STATE.tracked.get(mid, entry)

    # 4) Fire any due actions (idempotently)
    due = mp.due_actions(entry)
    for action in due:
        if STATE.shutdown.is_set():
            break
        ok = False
        try:
            ok = _fire_phase_action(mid, entry, action)
        except Exception as e:
            LOG.warning(f"phase action {action} failed for {mid}: {e}")
        _phase_log({"event": "action", "mid": mid,
                      "action": action, "ok": ok})
        if ok:
            with STATE.lock:
                ent = STATE.tracked.get(mid)
                if ent:
                    af = ent.setdefault("actions_fired", {})
                    af[action] = datetime.now(timezone.utc).isoformat()
                    STATE._save()


def phase_loop():
    """Drives the match-phase state machine. Lightweight — reads STATE.tracked
    snapshot, computes phase transitions, fires due actions. Coexists with
    predict_loop / lineup_loop: predict_loop continues to handle matches
    that aren't yet phase-tracked (legacy path), while phase_loop owns the
    versioned predictions for matches that have a `start_ts`."""
    LOG.info(f"phase_loop started (every {PHASE_INTERVAL_SEC}s)")
    STATE.shutdown.wait(15)
    while not STATE.shutdown.is_set():
        try:
            with STATE.lock:
                items = list(STATE.tracked.items())
            for mid, entry in items:
                if STATE.shutdown.is_set():
                    break
                try:
                    _phase_tick(mid, entry)
                except Exception as e:
                    LOG.warning(f"phase tick {mid} failed: {e}\n{traceback.format_exc()}")
        except Exception as e:
            LOG.warning(f"phase_loop iteration failed: {e}")
        STATE.shutdown.wait(PHASE_INTERVAL_SEC)


def predict_loop():
    LOG.info(f"predict_loop started (every {PREDICT_INTERVAL_SEC}s)")
    STATE.shutdown.wait(20)
    while not STATE.shutdown.is_set():
        try:
            todo = STATE.matches_to_predict()
            if todo:
                LOG.info(f"predict: {len(todo)} match(es) need a prediction")
                for mid, state in todo:
                    if STATE.shutdown.is_set(): break
                    # If we have an announced XI for this match, pass it through
                    # AND force-overwrite the prior proxy-based prediction.
                    xi = STATE.announced_xi(mid)
                    if xi and xi.get("xi_a") and xi.get("xi_b"):
                        # Map team_a/team_b → home/away by matching against state names
                        ta, tb = xi["team_a"], xi["team_b"]
                        h, a   = state["home"], state["away"]
                        if (ta or "").lower() == (h or "").lower():
                            xi_home, xi_away = xi["xi_a"], xi["xi_b"]
                        elif (tb or "").lower() == (h or "").lower():
                            xi_home, xi_away = xi["xi_b"], xi["xi_a"]
                        else:
                            # name mismatch (Cricbuzz vs CricSheet aliases) — skip XI override
                            LOG.warning(f"  XI team-name mismatch for {mid} ({ta}/{tb} vs {h}/{a}); using proxy")
                            xi_home = xi_away = None
                        ok = predict_match(state, force=True, xi_home=xi_home, xi_away=xi_away)
                    else:
                        ok = predict_match(state)
                    if ok:
                        STATE.mark_predicted(mid)
                # After predictions land, scan for value bets
                try:
                    bet_summary = BET.scan_predictions_dir(PREDICTIONS_DIR)
                    if bet_summary["placed"]:
                        LOG.info(f"bets: placed {len(bet_summary['placed'])} new bet(s) "
                                 f"(mode={BET.BET_MODE})")
                        for b in bet_summary["placed"]:
                            LOG.info(f"   {b['selection']:<28} stake={b['stake']:.2f}  "
                                     f"odds={b['odds']:.2f}  edge={b['edge_pp']:+.1f}pp  "
                                     f"id={b['bet_id'][:8]}")
                except Exception as e:
                    LOG.warning(f"bet scan failed: {e}\n{traceback.format_exc()}")
        except Exception as e:
            LOG.error(f"predict loop iteration failed: {e}\n{traceback.format_exc()}")
        STATE.shutdown.wait(PREDICT_INTERVAL_SEC)


# ---------- HTTP server ----------

class _SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a, **kw):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def http_loop():
    server = HTTPServer(("127.0.0.1", HTTP_PORT), _SilentHandler)
    LOG.info(f"http server listening on http://127.0.0.1:{HTTP_PORT}")
    while not STATE.shutdown.is_set():
        server.handle_request()
    server.server_close()


# ---------- main ----------

def main():
    # Ensure HTTP server cwd is the project root
    import os
    os.chdir(str(ROOT))

    LOG.info("==== orchestrator start ====")
    LOG.info(f"root = {ROOT}")
    LOG.info(f"predictions dir = {PREDICTIONS_DIR}")

    def _sigterm(*_):
        LOG.info("shutdown signal received")
        STATE.shutdown.set()
    try:
        signal.signal(signal.SIGINT, _sigterm)
        signal.signal(signal.SIGTERM, _sigterm)
    except Exception: pass

    threads = []
    for fn in (discover_loop, live_loop, export_loop, predict_loop, odds_loop,
                lineup_loop, ingest_loop, phase_loop, http_loop):
        t = threading.Thread(target=fn, name=fn.__name__, daemon=True)
        t.start()
        threads.append(t)

    try:
        # block forever; let signals stop us
        while not STATE.shutdown.is_set():
            time.sleep(1.0)
    finally:
        STATE.shutdown.set()
        LOG.info("orchestrator stopping")


if __name__ == "__main__":
    main()
