"""Match-phase state machine for the orchestrator.

Each tracked match flows through:

    DISCOVERED → SCHEDULED → PRE_START → LIVE → COMPLETE → REVIEWED
                                         ↘ ABANDONED ↗

Toss is a *boundary event* recorded inside PRE_START via `toss_seen_at`,
not a separate phase — re-prediction is just a versioned action keyed
off that field.

This module is I/O-free: pure functions only. All Cricbuzz fetching is
done elsewhere (`live_match.py`, `ingest/lineup.py`); we just classify
a per-match dict into a phase and report which time-windowed actions
are due. That makes the phase machine fully testable with a mocked
clock + recorded `live_matches.json` snapshots.

Per-match dict shape (extends `STATE.tracked[mid]`):
    {
        "slug":           str,            # Cricbuzz slug, set by discover_loop
        "last_state":     dict,            # output of live_match.normalise_for_dashboard
        "phase":          Phase,           # current phase (this module's responsibility)
        "phase_history":  list[dict],      # [{phase, at, from}]
        "start_ts":       int | None,      # UTC epoch seconds; None until derived
        "toss_seen_at":   int | None,      # UTC epoch when toss was first observed
        "toss_winner":    str | None,
        "toss_decision":  "bat"|"bowl"|"field"|None,
        "completed_at":   int | None,      # UTC epoch when match flipped to COMPLETE
        "actions_fired":  dict,            # {"<PHASE>.<ACTION>": "<iso_ts>"} idempotency log
        # legacy fields kept for backwards compat:
        "prediction_done": bool,
        "announced_xi":    dict,
        "xi_changed_at":   str,
        "xi_poll_n":       int,
    }
"""
from __future__ import annotations

import enum
import re
import time
from datetime import datetime, timezone


# ===========================================================================
# Phase enum + action keys
# ===========================================================================

class Phase(str, enum.Enum):
    DISCOVERED = "DISCOVERED"
    SCHEDULED  = "SCHEDULED"
    PRE_START  = "PRE_START"
    LIVE       = "LIVE"
    COMPLETE   = "COMPLETE"
    REVIEWED   = "REVIEWED"
    ABANDONED  = "ABANDONED"


# Action keys (used as idempotency keys in entry["actions_fired"])
A_PITCH_WEATHER    = "SCHEDULED.pitch_weather"
A_PRE_MATCH_PRED   = "SCHEDULED.pre_match_v0"
A_PRE_START_PRED   = "PRE_START.pre_start_v1"
A_TOSS_AWARE_PRED  = "PRE_START.toss_aware_v2"
A_SETTLE           = "COMPLETE.settle"
A_REVIEW           = "COMPLETE.review"

# Time windows, in seconds, relative to start_ts (or completed_at)
W_PRE_MATCH = 30 * 60          # T - 30 min: pitch / weather / pre_match_v0
W_PRE_START =  5 * 60          # T -  5 min: lineup poll + pre_start_v1
W_REVIEW    = 60 * 60          # T + complete + 1 h: post-match review


# ===========================================================================
# Time helpers
# ===========================================================================

def now_utc() -> int:
    """Single source of truth for "now". Replaceable in tests via monkeypatch."""
    return int(time.time())


# Cricbuzz status patterns:
#   "Match starts at Apr 30, 14:00 GMT"
#   "Match starts in 12 mins"           (skipped — relative; no absolute time)
_STARTS_AT_RE = re.compile(
    r"match\s+starts?\s+(?:at\s+)?"
    r"([A-Z][a-z]{2}\s+\d{1,2}),?\s*(\d{1,2}:\d{2})\s*(GMT|UTC|IST)?",
    re.IGNORECASE,
)


def parse_start_ts_from_status(status: str | None,
                                year_hint: int | None = None,
                                now_func=now_utc) -> int | None:
    """Parse a Cricbuzz-style 'Match starts at Apr 30, 14:00 GMT' string into
    a UTC epoch. Returns None if unparseable.

    Year defaults to current UTC year (or `year_hint`). Handles year-rollover:
    if the parsed date would be more than 180 days in the past, advance one year.
    Treats IST as UTC+5:30 explicitly; absent / GMT / UTC tz suffixes are UTC.
    """
    if not status:
        return None
    m = _STARTS_AT_RE.search(status)
    if not m:
        return None
    date_part, time_part, tz_part = m.group(1), m.group(2), (m.group(3) or "").upper()
    yr = year_hint if year_hint is not None else datetime.now(timezone.utc).year
    try:
        dt = datetime.strptime(f"{date_part} {yr} {time_part}", "%b %d %Y %H:%M")
    except ValueError:
        return None
    dt = dt.replace(tzinfo=timezone.utc)
    if tz_part == "IST":
        # IST is UTC+5:30; subtract 5h30m to get UTC
        dt = datetime.fromtimestamp(int(dt.timestamp()) - (5 * 3600 + 30 * 60), tz=timezone.utc)
    # Year rollover heuristic
    now = datetime.fromtimestamp(now_func(), tz=timezone.utc)
    if (now - dt).days > 180:
        dt = dt.replace(year=yr + 1)
    return int(dt.timestamp())


# ===========================================================================
# Status classifiers
# ===========================================================================

def is_abandoned(status: str | None) -> bool:
    if not status:
        return False
    s = status.lower()
    return ("abandon" in s) or ("no result" in s)


def has_winner_text(status: str | None) -> bool:
    """True when the status says 'X won by N runs/wkts' or 'Match tied (super over)'."""
    if not status:
        return False
    s = status.lower()
    if is_abandoned(s):
        return False
    return ("won by" in s) or ("won the super over" in s) or ("won the match" in s)


def is_in_play(status: str | None) -> bool:
    """Heuristic: a match is in-play when the status text references active
    play markers (toss done OR active scoring) and isn't abandoned/complete."""
    if not status:
        return False
    s = status.lower()
    if is_abandoned(s) or has_winner_text(s):
        return False
    play_markers = (
        "opt to bat", "opt to bowl", "opt to field",
        "elected to bat", "elected to bowl", "elected to field",
        "won the toss",
        "innings break", "drinks break",
        "trail", "lead", "need", "require", "chasing",
        " ov ", " ov,", "overs)",  # score-line cadence
        "in over",
    )
    return any(m in s for m in play_markers)


def detect_toss(state: dict) -> tuple[str | None, str | None]:
    """Parse 'X opt to bowl' / 'Y elected to bat' / 'X won the toss and chose to field'
    from the live status. Returns (toss_winner, toss_decision) or (None, None).
    `toss_decision` ∈ {"bat", "bowl", "field"}."""
    s = (state.get("status") or "").strip()
    if not s:
        return None, None
    # Form 1: "Bangladesh opt to bowl"
    m = re.match(r"^([A-Za-z][\w\s']+?)\s+opt\s+to\s+(bat|bowl|field)\b", s, re.I)
    if m:
        return m.group(1).strip(), m.group(2).lower()
    # Form 2: "Bangladesh elected to bat"
    m = re.match(r"^([A-Za-z][\w\s']+?)\s+elected\s+to\s+(bat|bowl|field)\b", s, re.I)
    if m:
        return m.group(1).strip(), m.group(2).lower()
    # Form 3: "Bangladesh won the toss and chose to field"
    m = re.match(
        r"^([A-Za-z][\w\s']+?)\s+won\s+the\s+toss\s+and\s+(?:chose|elected|opted)\s+to\s+(bat|bowl|field)\b",
        s, re.I,
    )
    if m:
        return m.group(1).strip(), m.group(2).lower()
    return None, None


# ===========================================================================
# Phase classifier — pure
# ===========================================================================

def compute_next_phase(entry: dict, now: int | None = None) -> str:
    """Pure phase classifier. Given a per-match entry, return the phase the
    match SHOULD be in right now. Caller is responsible for detecting whether
    that's a transition (entry["phase"] != return value).
    """
    now = now if now is not None else now_utc()
    state = entry.get("last_state") or {}
    status = state.get("status") or ""
    is_complete = bool(state.get("is_complete"))

    # Terminal classifications first — abandoned trumps complete (Cricbuzz
    # sometimes leaves abandoned matches with is_complete=False indefinitely)
    if is_abandoned(status):
        return Phase.ABANDONED.value
    if is_complete or has_winner_text(status):
        return Phase.COMPLETE.value

    # In-play override — Cricbuzz's match-clock can be off by a few minutes;
    # trust the status text when it's actively reporting play.
    if is_in_play(status):
        return Phase.LIVE.value

    start_ts = entry.get("start_ts")
    if start_ts is None:
        # Can't time-window; promote DISCOVERED → SCHEDULED only when we
        # have basic match metadata to act on.
        if state.get("home") and state.get("away"):
            current = entry.get("phase")
            return Phase.SCHEDULED.value if current == Phase.SCHEDULED.value else Phase.DISCOVERED.value
        return Phase.DISCOVERED.value

    if now < start_ts - W_PRE_MATCH:
        return Phase.SCHEDULED.value
    if now < start_ts:
        return Phase.PRE_START.value
    return Phase.LIVE.value


# ===========================================================================
# Action scheduler — pure
# ===========================================================================

def due_actions(entry: dict, now: int | None = None) -> list[str]:
    """Return action keys that are due to fire RIGHT NOW for this match.

    Time-driven (not phase-driven): once a time window opens, the action
    fires on the next tick whenever the match is in a non-terminal phase,
    regardless of which phase the machine has advanced to. This prevents
    silent skips when phase transitions race past action windows — e.g.
    SCHEDULED → PRE_START at exactly T-30m would otherwise leave
    pre_match_v0 stranded; PRE_START → LIVE on early toss detection
    would strand pre_start_v1.

    Idempotent — actions already in `entry["actions_fired"]` are skipped.
    Returned in firing order (lightest first so a failure in the heavier
    one doesn't block the lighter ones).
    """
    now = now if now is not None else now_utc()
    fired = entry.get("actions_fired") or {}
    phase = entry.get("phase") or Phase.DISCOVERED.value
    start_ts = entry.get("start_ts")
    out: list[str] = []

    # Terminal phases never fire pre-game actions.
    if phase in (Phase.ABANDONED.value, Phase.REVIEWED.value):
        # ABANDONED gets nothing; REVIEWED already wrote ledger
        return out

    pre_game_eligible = phase not in (
        Phase.DISCOVERED.value, Phase.COMPLETE.value, Phase.REVIEWED.value, Phase.ABANDONED.value,
    )

    # Pre-match snapshot opens at T-30m and fires once. If we miss the
    # window because the phase raced past, fire on the next tick — late
    # is better than never (the action handler strips XI/toss state so
    # the snapshot still represents a "no extra info" prediction).
    if pre_game_eligible and start_ts is not None and now >= start_ts - W_PRE_MATCH:
        for k in (A_PITCH_WEATHER, A_PRE_MATCH_PRED):
            if k not in fired:
                out.append(k)

    # Pre-start prediction opens at T-5m. Same rule.
    if pre_game_eligible and start_ts is not None and now >= start_ts - W_PRE_START:
        if A_PRE_START_PRED not in fired:
            out.append(A_PRE_START_PRED)

    # Toss-aware fires once per match the moment toss is observed.
    if pre_game_eligible and entry.get("toss_seen_at") and A_TOSS_AWARE_PRED not in fired:
        out.append(A_TOSS_AWARE_PRED)

    # Post-match
    if phase == Phase.COMPLETE.value:
        if A_SETTLE not in fired:
            out.append(A_SETTLE)
        completed_at = entry.get("completed_at") or now
        if now >= completed_at + W_REVIEW and A_REVIEW not in fired:
            out.append(A_REVIEW)

    return out


# ===========================================================================
# Reschedule handling
# ===========================================================================

def is_meaningful_reschedule(old_ts: int | None, new_ts: int | None,
                              tolerance_sec: int = 600) -> bool:
    """Return True if the new start time is meaningfully different from the
    old one and we should rewind action keys. Default tolerance: 10 min."""
    if old_ts is None or new_ts is None:
        return False
    return abs(new_ts - old_ts) > tolerance_sec


def reset_actions_for_phase_and_after(entry: dict, phase: str) -> None:
    """In-place: clear `actions_fired` keys for the given phase and all
    subsequent phases. Used when a match is rescheduled and earlier
    phase actions need to re-fire against the new clock.
    REVIEWED is durable — never reset (the learning ledger is on disk)."""
    fired = entry.get("actions_fired") or {}
    drop_phases = {
        Phase.SCHEDULED.value: {Phase.SCHEDULED.value, Phase.PRE_START.value},
        Phase.PRE_START.value: {Phase.PRE_START.value},
    }.get(phase, set())
    if not drop_phases:
        return
    for k in list(fired.keys()):
        prefix = k.split(".", 1)[0]
        if prefix in drop_phases:
            fired.pop(k, None)
    entry["actions_fired"] = fired
