"""Continuous accuracy comparison: our model vs bookmaker consensus.

For every saved prediction that has both:
   - `prediction.p_home_wins`             (our model)
   - `odds.h2h.consensus.p_home`          (book consensus, de-vigged)
   - a settled outcome from matches/live  (actual winner)

we record a row with:
   model_p, book_p, actual (1=home won, 0=away won),
   brier_model, brier_book, hit_model (1 if model_p>0.5 == actual),
   hit_book (same for book)

Aggregated metrics (saved to runs/comparison.json):
   n, brier_model, brier_book, acc_model, acc_book, brier_delta_pp
   plus a "blended" baseline: 0.6 model + 0.4 book

The harness updates whenever the orchestrator runs export_loop, so the
sample size grows automatically as more matches finish.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..db import connect


ROOT       = Path(__file__).resolve().parents[2]
PRED_DIR   = ROOT / "predictions"
RUNS_DIR   = ROOT / "cricket_pipeline" / "work" / "runs"
OUT_PATH   = RUNS_DIR / "comparison.json"


def _norm(s):
    return (s or "").strip().lower()


def _actual_winner_for(home: str, away: str, date: str) -> str | None:
    """Look up the winner of the match in the matches table, or in live_matches.json."""
    con = connect()
    row = con.execute("""
        SELECT winner FROM matches
        WHERE start_date = CAST(? AS DATE)
          AND ((team_home = ? AND team_away = ?) OR (team_home = ? AND team_away = ?))
          AND winner IS NOT NULL
        LIMIT 1
    """, [date, home, away, away, home]).fetchone()
    con.close()
    if row and row[0]: return row[0]
    # fall back to live_matches.json (recent / completed cricbuzz state)
    fp = RUNS_DIR / "live_matches.json"
    if not fp.exists(): return None
    try:
        for s in json.loads(fp.read_text()):
            if not s.get("is_complete"): continue
            sh, sa = _norm(s.get("home")), _norm(s.get("away"))
            if {sh, sa} != {_norm(home), _norm(away)}: continue
            txt = (s.get("status") or "").lower()
            for t in (s.get("home"), s.get("away")):
                if t and t.lower() in txt and "won" in txt:
                    return t
    except Exception:
        pass
    return None


def build() -> dict:
    rows = []
    for fp in sorted(PRED_DIR.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception: continue
        m = d.get("match") or {}
        cons = (((d.get("odds") or {}).get("h2h") or {}).get("consensus") or {})
        if not cons.get("p_home"):
            continue
        winner = _actual_winner_for(m.get("home"), m.get("away"), m.get("date"))
        if not winner:
            continue
        actual = 1 if _norm(winner) == _norm(m.get("home")) else (
                  0 if _norm(winner) == _norm(m.get("away")) else None)
        if actual is None: continue
        model_p = d["prediction"]["p_home_wins"]
        book_p  = cons["p_home"]
        blend_p = 0.6 * model_p + 0.4 * book_p
        rows.append({
            "match":        f'{m["home"]} vs {m["away"]} ({m["date"]})',
            "actual_home":  actual,
            "model_p":      round(model_p, 4),
            "book_p":       round(book_p, 4),
            "blend_p":      round(blend_p, 4),
            "hit_model":    int((model_p >= 0.5) == bool(actual)),
            "hit_book":     int((book_p  >= 0.5) == bool(actual)),
            "hit_blend":    int((blend_p >= 0.5) == bool(actual)),
            "brier_model":  round((model_p - actual) ** 2, 4),
            "brier_book":   round((book_p  - actual) ** 2, 4),
            "brier_blend":  round((blend_p - actual) ** 2, 4),
        })

    n = len(rows)
    if n == 0:
        summary = {
            "n": 0, "msg": "no settled matches with both model + book probabilities yet",
            "rows": [],
        }
    else:
        def _avg(k): return round(sum(r[k] for r in rows) / n, 4)
        summary = {
            "n":             n,
            "brier_model":   _avg("brier_model"),
            "brier_book":    _avg("brier_book"),
            "brier_blend":   _avg("brier_blend"),
            "acc_model":     _avg("hit_model"),
            "acc_book":      _avg("hit_book"),
            "acc_blend":     _avg("hit_blend"),
            "rows":          rows,
        }
        summary["brier_delta_pp"] = round(
            (summary["brier_book"] - summary["brier_model"]) * 100, 2)
        summary["acc_delta_pp"]   = round(
            (summary["acc_model"] - summary["acc_book"]) * 100, 2)

    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    out = build()
    n = out.get("n", 0)
    if n == 0:
        print(out.get("msg")); print(out.get("generated_at"))
    else:
        print(f"Comparison sample: {n} settled matches with both probs available\n")
        print(f"  Accuracy:  model {out['acc_model']*100:5.1f}%   book {out['acc_book']*100:5.1f}%   "
              f"blend {out['acc_blend']*100:5.1f}%")
        print(f"  Brier:     model {out['brier_model']:.4f}   book {out['brier_book']:.4f}   "
              f"blend {out['brier_blend']:.4f}")
        print(f"  Δ (model−book) accuracy: {out['acc_delta_pp']:+.2f}pp")
        print(f"  Δ (book−model) Brier:    {out['brier_delta_pp']:+.2f}pp  (positive = model better)")
        print()
        print("Per-match:")
        for r in out["rows"]:
            print(f"  {r['match']:<55} actual={'H' if r['actual_home'] else 'A'}  "
                  f"model={r['model_p']*100:5.1f}%  book={r['book_p']*100:5.1f}%  "
                  f"hit_model={r['hit_model']} hit_book={r['hit_book']}")
