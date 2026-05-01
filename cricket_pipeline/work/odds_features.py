"""Convert raw bookmaker odds into model-ready features.

Two key transforms:
  1. **Implied probability** = 1 / decimal_odds
  2. **De-vig**: bookmakers price in margin (overround); to recover the true
     implied probability, divide by the sum of implied probs across all outcomes
     of the same market. This gives `book_p_home + book_p_away ≈ 1.0`.

Aggregation: for each (match, market), compute the consensus (mean across
bookmakers, after de-vigging) plus min/max for spread.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

from ..db import connect
from ..ingest.odds import latest_for_match


def implied_prob(decimal_odds: float | None) -> float | None:
    """Implied probability from decimal odds, with bounds enforcement.

    Decimal odds <= 1.0 represent zero/negative profit and are invalid;
    odds > 1000 are absurd outliers (10000:1 longshots, malformed feeds)
    and we reject rather than feed them downstream where they'd skew
    calibration. Result is clipped to [1e-6, 1-1e-6] to keep log-loss
    consumers numerically stable.
    """
    if decimal_odds is None:
        return None
    try:
        odds = float(decimal_odds)
    except (TypeError, ValueError):
        return None
    if not (1.0 < odds <= 1000.0):
        return None
    p = 1.0 / odds
    # Clamp to the same bounds the loss functions expect
    return max(1e-6, min(1.0 - 1e-6, p))


def devig_two_way(p_a: float | None, p_b: float | None) -> tuple[float | None, float | None]:
    """Standard normalization de-vig: divide both by their sum."""
    if p_a is None or p_b is None: return (None, None)
    s = p_a + p_b
    if s <= 0: return (None, None)
    return (p_a / s, p_b / s)


def book_consensus(home: str, away: str, market: str = "h2h") -> dict:
    """Return per-bookmaker + consensus de-vigged probabilities for the
    most-recent snapshot of an h2h market.

    Returns:
       {
         "n_books": int,
         "snapshot_at": iso-string,
         "by_book": {bookmaker: {"p_home": x, "p_away": y, "raw_home": ..., "raw_away": ...}, ...},
         "consensus": {"p_home": mean, "p_away": mean,
                       "p_home_min": ..., "p_home_max": ...,
                       "spread_pp": (max-min)*100},
       }
    """
    rows = latest_for_match(home, away, market=market)
    if not rows:
        return {"n_books": 0, "snapshot_at": None, "by_book": {}, "consensus": None}

    by_book_outcomes: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_book_outcomes[r["bookmaker"]][r["selection"]] = r["decimal_odds"]

    by_book: dict[str, dict] = {}
    for bk, outcomes in by_book_outcomes.items():
        # Map outcomes to home/away (Odds API names them as full team names)
        odds_home = outcomes.get(home)
        odds_away = outcomes.get(away)
        # Some bookmakers may publish a Draw market for limited overs; ignore here.
        raw_h = implied_prob(odds_home)
        raw_a = implied_prob(odds_away)
        ph, pa = devig_two_way(raw_h, raw_a)
        if ph is None: continue
        by_book[bk] = {
            "odds_home": odds_home, "odds_away": odds_away,
            "raw_p_home": raw_h, "raw_p_away": raw_a,
            "p_home": ph, "p_away": pa,
            "vig_pct": (raw_h + raw_a - 1) * 100 if (raw_h and raw_a) else None,
        }
    if not by_book:
        return {"n_books": 0, "snapshot_at": rows[0]["snapshot_at"], "by_book": {}, "consensus": None}

    p_homes = np.array([d["p_home"] for d in by_book.values()])
    p_aways = np.array([d["p_away"] for d in by_book.values()])
    consensus = {
        "p_home":     float(p_homes.mean()),
        "p_away":     float(p_aways.mean()),
        "p_home_min": float(p_homes.min()),
        "p_home_max": float(p_homes.max()),
        "spread_pp":  float((p_homes.max() - p_homes.min()) * 100),
        "n_books":    int(len(by_book)),
    }
    return {
        "n_books":     len(by_book),
        "snapshot_at": rows[0]["snapshot_at"],
        "by_book":     by_book,
        "consensus":   consensus,
    }


def line_movement(home: str, away: str, market: str = "h2h",
                   hours: int = 24) -> dict | None:
    """How much did consensus implied prob move in the last `hours` hours?
    Positive = market is moving toward the home team."""
    con = connect()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = con.execute("""
        SELECT snapshot_at, bookmaker, selection, decimal_odds
        FROM odds_snapshot
        WHERE market = ?
          AND snapshot_at >= ?
          AND ((home_team = ? AND away_team = ?)
            OR (home_team = ? AND away_team = ?))
        ORDER BY snapshot_at ASC
    """, [market, cutoff, home, away, away, home]).fetchall()
    con.close()
    if len(rows) < 2: return None
    by_snap: dict[datetime, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for snap, bk, sel, price in rows:
        by_snap[snap][bk][sel] = price
    snaps = sorted(by_snap.keys())
    def _consensus(snap):
        ps = []
        for bk, outs in by_snap[snap].items():
            oh = outs.get(home); oa = outs.get(away)
            ph, pa = devig_two_way(implied_prob(oh), implied_prob(oa))
            if ph is not None: ps.append(ph)
        return float(np.mean(ps)) if ps else None
    p_first = _consensus(snaps[0])
    p_last  = _consensus(snaps[-1])
    if p_first is None or p_last is None: return None
    return {
        "delta_pp_home":   (p_last - p_first) * 100,
        "p_home_first":    p_first,
        "p_home_last":     p_last,
        "first_snap":      snaps[0].isoformat() if hasattr(snaps[0], "isoformat") else str(snaps[0]),
        "last_snap":       snaps[-1].isoformat() if hasattr(snaps[-1], "isoformat") else str(snaps[-1]),
        "n_snaps":         len(snaps),
    }


def edge_pct(model_p: float, book_p: float) -> float:
    """Model edge in percentage points. Positive = model thinks the price is too long."""
    return (model_p - book_p) * 100


def kelly_fraction(model_p: float, decimal_odds: float, kelly_cap: float = 0.5) -> float:
    """Fractional Kelly stake (as fraction of bankroll). Returns 0 if no edge.
    Default cap = half-Kelly to reduce variance."""
    if decimal_odds <= 1 or model_p <= 0 or model_p >= 1:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - model_p
    f = (b * model_p - q) / b
    if f <= 0: return 0.0
    return min(f * kelly_cap, 0.05)   # also cap at 5% of bankroll on any single bet


def attach_odds_to_prediction(pred: dict) -> dict:
    """Take a prediction dict (the predict_match.py JSON shape) and decorate it
    with `odds_consensus` + `model_vs_book_edge` (in pp) and a `value_bet` tag."""
    home = pred["match"]["home"]; away = pred["match"]["away"]
    cons = book_consensus(home, away, market="h2h")
    move = line_movement(home, away, market="h2h", hours=24)
    out = dict(pred)
    out["odds"] = {"h2h": cons, "line_movement_24h": move}
    if cons.get("consensus"):
        c = cons["consensus"]
        model_p_home = pred["prediction"]["p_home_wins"]
        edge_home = edge_pct(model_p_home, c["p_home"])
        edge_away = edge_pct(1 - model_p_home, c["p_away"])
        # The side worth betting is the one where the MODEL beats the BOOK
        # (positive edge), not necessarily the one the model favors outright.
        # Example: model 53% home, book 69% home — model favors home, but the
        # book has it as a much bigger favorite, so the value bet is AWAY.
        if edge_home >= edge_away:
            side, side_team, side_p_model = "home", home, model_p_home
        else:
            side, side_team, side_p_model = "away", away, 1 - model_p_home
        # best (highest) decimal odds across bookmakers for that side
        best_odds = max(
            (b.get(f"odds_{side}") or 0) for b in cons["by_book"].values()
        ) or None
        kelly = kelly_fraction(side_p_model, best_odds) if best_odds else 0.0
        edge_chosen = edge_home if side == "home" else edge_away
        out["model_vs_book"] = {
            "edge_home_pp":    round(edge_home, 2),
            "edge_away_pp":    round(edge_away, 2),
            "best_side":       side_team,
            "best_side_edge_pp": round(edge_chosen, 2),
            "best_odds":       best_odds,
            "kelly_fraction":  round(kelly, 4),
            "value_bet":       (edge_chosen >= 3.0),
        }
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--market", default="h2h")
    args = ap.parse_args()
    print(json.dumps(book_consensus(args.home, args.away, args.market),
                     indent=2, default=str))
