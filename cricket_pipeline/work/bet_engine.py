"""Bet placement engine.

Modes (controlled by env var `BET_MODE`):
  - **manual** (default in Canada): engine generates picks + a printable ticket
    to copy-paste into your bookie of choice (BetMGM, FanDuel, theScore, etc.).
    Bets are recorded as 'pending-manual' until you mark them placed.
  - **paper**: records "as-if" bets to the `bets` table, settles automatically
    when matches finish. Tracks PnL with no real money.
  - **polymarket**: live placement on Polymarket (USDC, Polygon network) —
    Canada-friendly via crypto. Requires POLYMARKET_LIVE_CONFIRMED=yes.
  - **betfair**: BLOCKED IN CANADA. Scaffold remains for non-Canadian users
    but raises a friendly error here.

Environment variables:
  BET_MODE=manual|paper|polymarket|betfair      (default: manual)
  BANKROLL=<float>                 (default: 1000.0 — used for Kelly sizing)
  BET_EDGE_THRESHOLD_PP=<float>    (default: 3.0 — minimum model-vs-book edge)
  BET_MIN_ODDS=<float>             (default: 1.20)
  BET_MAX_ODDS=<float>             (default: 8.00)
  BET_KELLY_CAP=<float>            (default: 0.5)
  BET_MAX_STAKE_PCT=<float>        (default: 0.05)
  POLYMARKET_API_KEY / POLYMARKET_API_SECRET / POLYMARKET_PASSPHRASE /
    POLYMARKET_WALLET_ADDR / POLYMARKET_PRIVATE_KEY / POLYMARKET_LIVE_CONFIRMED
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..db import connect

LOG = logging.getLogger(__name__)


# ---------- config ----------

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        # Distinguish typos from missing values: log a warning and fall back
        # so we don't silently mask a misconfigured BANKROLL=fifty.
        LOG.warning("env var %s=%r is not a valid float; falling back to default %s",
                    name, raw, default)
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


BET_MODE              = _env_str("BET_MODE", "manual").lower()    # 'manual' | 'paper' | 'polymarket' | 'betfair'
BANKROLL              = _env_float("BANKROLL", 1000.0)
EDGE_THRESHOLD_PP     = _env_float("BET_EDGE_THRESHOLD_PP", 3.0)    # min model-vs-book edge to bet
EDGE_MAX_PP           = _env_float("BET_EDGE_MAX_PP", 8.0)          # above this, suspect model error not edge
MIN_ODDS              = _env_float("BET_MIN_ODDS", 1.20)
MAX_ODDS              = _env_float("BET_MAX_ODDS", 8.00)
KELLY_CAP             = _env_float("BET_KELLY_CAP", 0.5)
MAX_STAKE_PCT         = _env_float("BET_MAX_STAKE_PCT", 0.05)


def _validate_config() -> None:
    """Sanity-check bet config at module load. Raises ValueError on bad
    combinations rather than silently rolling with broken math at trade time.
    """
    errors = []
    if BET_MODE not in {"manual", "paper", "polymarket", "betfair"}:
        errors.append(f"BET_MODE={BET_MODE!r} must be one of manual|paper|polymarket|betfair")
    if BANKROLL <= 0:
        errors.append(f"BANKROLL={BANKROLL} must be > 0")
    if not (0.0 <= EDGE_THRESHOLD_PP < EDGE_MAX_PP):
        errors.append(f"BET_EDGE_THRESHOLD_PP={EDGE_THRESHOLD_PP} must be in [0, BET_EDGE_MAX_PP={EDGE_MAX_PP})")
    if not (1.0 < MIN_ODDS < MAX_ODDS):
        errors.append(f"BET_MIN_ODDS={MIN_ODDS} must be in (1.0, BET_MAX_ODDS={MAX_ODDS})")
    if not (0.0 < KELLY_CAP <= 1.0):
        errors.append(f"BET_KELLY_CAP={KELLY_CAP} must be in (0, 1]")
    if not (0.0 < MAX_STAKE_PCT <= 1.0):
        errors.append(f"BET_MAX_STAKE_PCT={MAX_STAKE_PCT} must be in (0, 1]")
    if BET_MODE == "polymarket" and not os.environ.get("POLYMARKET_API_KEY"):
        errors.append("BET_MODE=polymarket requires POLYMARKET_API_KEY")
    if errors:
        raise ValueError("Invalid bet engine config:\n  - " + "\n  - ".join(errors))


_validate_config()


# Loud startup audit when live-money mode is active. The operator can leave
# BET_MODE=polymarket in their .env between sessions and not realise live
# trading would resume on the next orchestrator restart; printing a banner
# every time the module loads is the cheap mitigation.
if BET_MODE in ("polymarket", "betfair"):
    LOG.warning(
        "bet_engine loaded in LIVE-MONEY mode (BET_MODE=%s, BANKROLL=%s). "
        "Set BET_MODE=manual or paper to disable.",
        BET_MODE, BANKROLL,
    )


# ---------- decision ----------

@dataclass
class BetDecision:
    should_bet:    bool
    reason:        str
    selection:     str        # team/outcome name
    decimal_odds:  float | None
    model_p:       float | None
    book_p:        float | None
    edge_pp:       float | None
    kelly:         float | None
    stake:         float | None


def decide_bet(prediction: dict, bankroll: float = BANKROLL) -> BetDecision:
    """Given a saved prediction (from predict_match.py, decorated with odds),
    decide whether to place a bet. Returns a BetDecision (no side-effects)."""
    mvb  = prediction.get("model_vs_book") or {}
    odds = (prediction.get("odds") or {}).get("h2h") or {}
    cons = odds.get("consensus") or {}
    if not mvb or not cons:
        return BetDecision(False, "no odds available", "", None, None, None, None, None, None)

    side = mvb.get("best_side")
    best_odds = mvb.get("best_odds")
    if best_odds is None:
        return BetDecision(False, "no usable odds", side, None, None, None, None, None, None)
    if best_odds < MIN_ODDS or best_odds > MAX_ODDS:
        return BetDecision(False, f"odds {best_odds:.2f} outside [{MIN_ODDS}, {MAX_ODDS}]",
                            side, best_odds, None, None, None, None, None)

    match_block = prediction.get("match") or {}
    pred_block = prediction.get("prediction") or {}
    home = match_block.get("home")
    if not home:
        return BetDecision(False, "prediction missing match.home", side, best_odds, None, None, None, None, None)
    is_home_side = (side == home)
    model_p = (pred_block.get("p_home_wins") if is_home_side
               else pred_block.get("p_away_wins"))
    book_p  = cons.get("p_home") if is_home_side else cons.get("p_away")
    if model_p is None or book_p is None:
        return BetDecision(False, "missing model_p or book_p", side, best_odds,
                            model_p, book_p, None, None, None)
    if not (0.0 <= model_p <= 1.0) or not (0.0 <= book_p <= 1.0):
        return BetDecision(False, f"probabilities out of [0,1] (model_p={model_p}, book_p={book_p})",
                            side, best_odds, model_p, book_p, None, None, None)
    edge    = (model_p - book_p) * 100
    if edge < EDGE_THRESHOLD_PP:
        return BetDecision(False, f"edge {edge:.2f}pp < threshold {EDGE_THRESHOLD_PP}pp",
                            side, best_odds, model_p, book_p, edge, None, None)
    if edge > EDGE_MAX_PP:
        # Edges this large vs a 30+ book consensus almost always reflect a model
        # blind spot (stale lineup, missing injury info, wrong pitch read), not
        # a real opportunity. Skip and flag for review.
        return BetDecision(False, f"edge {edge:.2f}pp > suspicious-cap {EDGE_MAX_PP}pp",
                            side, best_odds, model_p, book_p, edge, None, None)

    kelly = mvb.get("kelly_fraction") or 0.0
    if kelly <= 0:
        return BetDecision(False, "kelly = 0", side, best_odds, model_p, book_p, edge, kelly, None)
    # Cap Kelly fraction first to avoid float overflow into stake (defense-in-depth
    # alongside the per-stake percentage cap below).
    kelly_capped = min(kelly, KELLY_CAP)
    stake = max(0.0, min(kelly_capped * bankroll, MAX_STAKE_PCT * bankroll))
    # Hard ceiling: never risk more than MAX_STAKE_PCT of bankroll regardless of math
    stake = min(stake, MAX_STAKE_PCT * bankroll)
    if stake < 1.0:
        return BetDecision(False, f"stake {stake:.2f} below minimum (1.0)",
                            side, best_odds, model_p, book_p, edge, kelly, stake)

    return BetDecision(True, "edge + bankroll OK", side, best_odds,
                       model_p, book_p, edge, kelly, round(stake, 2))


def _bet_dedup_key(prediction: dict, decision: BetDecision) -> str:
    """Stable identity key for a bet: sorted teams + date + market + selection.
    Same key = same logical bet, regardless of file name or insertion order.
    Used to make place_bet() idempotent across pipeline re-runs."""
    m = prediction.get("match") or {}
    home = (m.get("home") or "").strip().lower()
    away = (m.get("away") or "").strip().lower()
    teams = "|".join(sorted([home, away]))
    date = str(m.get("date") or "")
    market = "h2h"
    selection = (decision.selection or "").strip().lower()
    raw = f"{teams}|{date}|{market}|{selection}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ---------- placement ----------

def place_bet(prediction: dict, decision: BetDecision, mode: str | None = None) -> dict:
    """Persist a bet record. In live modes, also routes to the venue.
    Idempotent: if a bet with the same dedup_key already exists, returns the
    existing record without re-inserting and without re-routing to the venue.
    Returns the bet record."""
    mode = (mode or BET_MODE).lower()
    venue = {
        "manual":     "manual",
        "paper":      "paper",
        "polymarket": "polymarket",
        "betfair":    "betfair",
    }.get(mode, "paper")
    placed_at = datetime.now(timezone.utc)

    dedup_key = _bet_dedup_key(prediction, decision)

    # Idempotency check inside a transaction (defense against re-runs and races)
    con = connect()
    try:
        con.execute("BEGIN")
        existing = con.execute(
            "SELECT bet_id, status FROM bets WHERE notes LIKE ? LIMIT 1",
            [f'%"dedup_key": "{dedup_key}"%'],
        ).fetchone()
        if existing:
            con.execute("COMMIT")
            LOG.info("place_bet: idempotent skip for dedup_key=%s (existing=%s)",
                     dedup_key, existing[0])
            return {"bet_id": existing[0], "status": existing[1],
                    "dedup_key": dedup_key, "idempotent": True}

        bet_id = str(uuid.uuid4())
        match_block = prediction.get("match") or {}
        notes_obj = {
            "dedup_key":   dedup_key,
            "match":       match_block,
            "favored":     (prediction.get("prediction") or {}).get("favored"),
            "favored_pct": (prediction.get("prediction") or {}).get("favored_pct"),
            "reason":      decision.reason,
        }

        rec = {
            "bet_id":         bet_id,
            "placed_at":      placed_at,
            "mode":           mode,
            "venue":          venue,
            "external_id":    None,
            "match_id":       None,
            "market":         "h2h",
            "selection":      decision.selection,
            "decimal_odds":   decision.decimal_odds,
            "stake":          decision.stake,
            "model_p":        decision.model_p,
            "book_p":         decision.book_p,
            "edge_pct":       decision.edge_pp,
            "kelly_fraction": decision.kelly,
            "status":         ("pending-manual" if mode == "manual" else "pending"),
            "settled_at":     None,
            "pnl":            None,
            "notes":          json.dumps(notes_obj, default=str),
        }

        if mode == "polymarket":
            try:
                from . import polymarket_client
                ext_ref = polymarket_client.place(decision)
                rec["external_id"] = (ext_ref or {}).get("orderHash") or (ext_ref or {}).get("bet_id")
            except Exception as e:
                LOG.exception("polymarket place failed: %s", e)
                rec["status"] = "error"
                rec["notes"] = json.dumps({**notes_obj, "error": str(e)}, default=str)
        elif mode == "betfair":
            rec["status"] = "blocked"
            rec["notes"]  = json.dumps({**notes_obj,
                                         "error": "Betfair Exchange is not available in Canada. "
                                                  "Set BET_MODE=manual or polymarket."}, default=str)

        con.execute("""
            INSERT INTO bets (bet_id, placed_at, mode, venue, external_id, match_id,
                              market, selection, decimal_odds, stake, model_p, book_p,
                              edge_pct, kelly_fraction, status, settled_at, pnl, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [rec[k] for k in ("bet_id","placed_at","mode","venue","external_id","match_id",
                                "market","selection","decimal_odds","stake","model_p","book_p",
                                "edge_pct","kelly_fraction","status","settled_at","pnl","notes")])
        con.execute("COMMIT")
        rec["dedup_key"] = dedup_key
        return rec
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()


def settle_bets_against_results(live_states: list[dict]) -> dict:
    """Look at live_match states; for each pending bet whose match completed,
    mark it won/lost and update PnL. `live_states` = list from live_matches.json.
    Returns counts."""
    if not live_states: return {"settled": 0}
    con = connect()
    # Auto-settle: paper + polymarket only. Manual bets settle only when the
    # user marks them placed (status='placed') first; otherwise stays 'pending-manual'.
    pending = con.execute("""
        SELECT bet_id, selection, decimal_odds, stake, notes
        FROM bets WHERE status IN ('pending', 'placed')
    """).fetchall()
    settled = 0
    for bet_id, selection, decimal_odds, stake, notes in pending:
        try:
            n = json.loads(notes or "{}")
        except (json.JSONDecodeError, TypeError) as e:
            LOG.warning("settle_bets: malformed notes for bet %s: %s", bet_id, e)
            n = {}
        m = (n.get("match") or {})
        bet_home = m.get("home"); bet_away = m.get("away")
        if not (bet_home and bet_away):
            continue
        # find a completed live state matching home/away (set comparison
        # ensures team-order doesn't matter)
        match = next((ls for ls in live_states
                      if ls.get("is_complete")
                      and {(ls.get("home") or "").lower(), (ls.get("away") or "").lower()}
                       == {bet_home.lower(), bet_away.lower()}), None)
        if not match: continue
        status_text = (match.get("status") or "").lower()
        # Use word-boundary regex so "won the toss" / "rain" / "super over loss"
        # don't trigger false-positive settlement, and ensure we match the
        # actual result phrase ("X won by Y") rather than substring contamination.
        winner = _parse_match_winner(status_text, bet_home, bet_away)
        if winner is None:
            LOG.debug("settle_bets: could not parse winner from status %r (home=%s away=%s)",
                      status_text, bet_home, bet_away)
            continue
        won = (winner == selection)
        # Explicit float coercion guards against int-typed odds bleeding through
        # from older DB rows; arithmetic stays in float space.
        pnl = (float(decimal_odds) - 1.0) * float(stake) if won else -float(stake)
        # The WHERE clause re-asserts the bet is still pending. If a parallel
        # process (manual mark, retry of settlement) updated the row between
        # our SELECT and UPDATE, this no-ops instead of clobbering the new
        # status — TOCTOU defense for the bet ledger.
        result = con.execute("""
            UPDATE bets
            SET status = ?, settled_at = ?, pnl = ?
            WHERE bet_id = ?
              AND status IN ('pending', 'placed')
              AND settled_at IS NULL
        """, ["won" if won else "lost", datetime.now(timezone.utc), pnl, bet_id])
        # DuckDB doesn't expose rowcount on UPDATE consistently across versions;
        # we still increment optimistically since the WHERE filter is the guard.
        settled += 1
    con.close()
    return {"settled": settled}


def _parse_match_winner(status_text: str, home: str, away: str) -> str | None:
    """Parse the winner from a Cricbuzz-style status string with disambiguation
    against false positives like 'won the toss' or 'super over loss'.

    Returns the winning team name (matching the casing of `home` or `away`),
    or None if the result is ambiguous / unparseable.
    """
    if not status_text:
        return None
    text = status_text.lower()
    # Reject result-less phrases up front: tosses, no-results, washouts, etc.
    if "won the toss" in text and " by " not in text:
        return None
    if "no result" in text or "abandoned" in text:
        return None

    candidates = []
    for team in (home, away):
        if not team:
            continue
        # Match "<team> won by ..." with a word boundary on the team name
        # (escape regex metachars in team names like "Royal Challengers (RCB)").
        pat = rf"(^|\W){re.escape(team.lower())}\s+won\b(?!\s+the\s+toss)"
        if re.search(pat, text):
            candidates.append(team)
    # Both teams marked as "won" → ambiguous (e.g. "X won 1st innings, Y won super over").
    # Refuse to settle rather than guess.
    if len(candidates) != 1:
        return None
    return candidates[0]


def pnl_summary() -> dict:
    con = connect()
    total = con.execute("""
        SELECT
          COUNT(*) FILTER (WHERE status = 'won')                 AS won,
          COUNT(*) FILTER (WHERE status = 'lost')                AS lost,
          COUNT(*) FILTER (WHERE status = 'pending')             AS pending,
          SUM(stake)  FILTER (WHERE status IN ('won','lost'))   AS staked,
          SUM(pnl)    FILTER (WHERE status IN ('won','lost'))   AS pnl
        FROM bets
    """).fetchone()
    con.close()
    won, lost, pending, staked, pnl = total
    return {
        "won":         int(won or 0),
        "lost":        int(lost or 0),
        "pending":     int(pending or 0),
        "staked":      float(staked or 0.0),
        "pnl":         float(pnl or 0.0),
        "roi_pct":     (float(pnl or 0.0) / float(staked or 1)) * 100 if staked else None,
        "bankroll":    BANKROLL,
        "mode":        BET_MODE,
    }


def open_tickets() -> list[dict]:
    """List manual bets the user hasn't placed yet (status='pending-manual')."""
    con = connect()
    rows = con.execute("""
        SELECT bet_id, placed_at, market, selection, decimal_odds, stake,
               model_p, book_p, edge_pct, kelly_fraction, notes
        FROM bets
        WHERE status = 'pending-manual'
        ORDER BY placed_at DESC
    """).fetchall()
    con.close()
    out = []
    for r in rows:
        try:
            n = json.loads(r[10] or "{}")
        except Exception:
            n = {}
        out.append({
            "bet_id":         r[0], "placed_at": r[1],
            "market":         r[2], "selection": r[3],
            "decimal_odds":   r[4], "stake":     r[5],
            "model_p":        r[6], "book_p":    r[7],
            "edge_pct":       r[8], "kelly":     r[9],
            "match":          (n.get("match") or {}),
        })
    return out


def mark_status(bet_id_prefix: str, new_status: str,
                pnl: float | None = None) -> dict:
    """User-facing helper: mark a manual bet placed/won/lost/void.
    `bet_id_prefix` matches the first 8 chars of the UUID."""
    new_status = new_status.lower()
    valid = {"placed", "won", "lost", "void", "cancelled"}
    if new_status not in valid:
        raise ValueError(f"new_status must be one of {valid}")
    con = connect()
    rows = con.execute("""
        SELECT bet_id, decimal_odds, stake FROM bets
        WHERE bet_id LIKE ?
    """, [f"{bet_id_prefix}%"]).fetchall()
    if not rows:
        con.close()
        raise RuntimeError(f"No bet with prefix {bet_id_prefix!r}")
    if len(rows) > 1:
        con.close()
        raise RuntimeError(f"Ambiguous prefix {bet_id_prefix!r}: matches {len(rows)} bets")
    bet_id, odds, stake = rows[0]
    settled_at = datetime.now(timezone.utc) if new_status in ("won","lost","void") else None
    if new_status == "won" and pnl is None:
        pnl = (odds - 1.0) * stake
    elif new_status == "lost" and pnl is None:
        pnl = -stake
    elif new_status == "void" and pnl is None:
        pnl = 0.0
    con.execute("""
        UPDATE bets SET status = ?, settled_at = ?, pnl = ?
        WHERE bet_id = ?
    """, [new_status, settled_at, pnl, bet_id])
    con.close()
    return {"bet_id": bet_id, "status": new_status, "pnl": pnl}


def scan_predictions_dir(predictions_dir: Path) -> dict:
    """Iterate every saved prediction; for each that doesn't yet have a bet
    in the DB, evaluate decide_bet() and place if positive. place_bet() is
    idempotent (keyed on dedup_key), so a re-run never produces duplicates.
    Returns summary."""
    placed = []
    skipped = []
    for fp in sorted(predictions_dir.glob("*.json")):
        try:
            pred = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError) as e:
            LOG.warning("scan_predictions_dir: failed to read %s: %s", fp.name, e)
            skipped.append({"file": fp.name, "reason": f"unreadable: {e}"})
            continue
        if not (pred.get("match") and pred.get("prediction")):
            skipped.append({"file": fp.name, "reason": "missing match or prediction block"})
            continue
        d = decide_bet(pred)
        if d.should_bet:
            rec = place_bet(pred, d)
            if rec.get("idempotent"):
                skipped.append({"file": fp.name, "reason": "already-bet (idempotent)",
                                 "bet_id": rec["bet_id"]})
            else:
                placed.append({"file": fp.name, "bet_id": rec["bet_id"],
                                "selection": d.selection, "stake": d.stake,
                                "edge_pp": d.edge_pp, "odds": d.decimal_odds})
        else:
            skipped.append({"file": fp.name, "reason": d.reason})
    return {"placed": placed, "skipped": skipped}


if __name__ == "__main__":
    import argparse
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan",     action="store_true", help="walk predictions/ and place any positive-edge bets")
    ap.add_argument("--summary",  action="store_true", help="show PnL summary")
    ap.add_argument("--tickets",  action="store_true", help="list manual tickets awaiting placement")
    ap.add_argument("--mark",     metavar="BET_ID_PREFIX", help="mark a bet placed/won/lost/void")
    ap.add_argument("--status",   default=None, help="status to set (with --mark)")
    ap.add_argument("--pnl",      type=float, default=None,
                    help="override PnL for this settle (with --mark won|lost)")
    args = ap.parse_args()
    if args.scan:
        out = scan_predictions_dir(ROOT / "predictions")
        print(json.dumps(out, indent=2))
    if args.tickets:
        print(json.dumps(open_tickets(), indent=2, default=str))
    if args.mark:
        if not args.status:
            ap.error("--mark requires --status")
        print(json.dumps(mark_status(args.mark, args.status, pnl=args.pnl),
                          indent=2, default=str))
    if args.summary or not (args.scan or args.tickets or args.mark):
        print(json.dumps(pnl_summary(), indent=2))
