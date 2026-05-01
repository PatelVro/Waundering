"""Post-match learning module.

Reads saved prediction files from `predictions/` (orchestrator-generated
stacked-LR-ensemble format) and compares them against actual match results.
Results come from the `result` field in `data/preds/` copies (updated
post-match by the live tracker) or from the DB as a fallback.

For each completed match:
  - Correct predictions  -> document what the model got right and why.
  - Wrong predictions    -> separate learnable systematic errors from luck
                           (narrow margins), flag for model weight review.

Usage (via pipeline CLI):
    python -m cricket_pipeline.pipeline post-match-review
    python -m cricket_pipeline.pipeline post-match-review --days-back 7
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT             = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR  = ROOT / "predictions"
PREDS_ALT_DIR    = ROOT / "data" / "preds"   # copies with result field
LEARNINGS_DIR    = ROOT / "learnings"

# A margin this narrow suggests luck, not systematic failure
_LUCK_RUNS     = 5
_LUCK_WICKETS  = 2

# Only analyse wrong predictions above this edge (model was confident)
MIN_EDGE_PP = 15.0

# Thresholds for pattern detection
_ELO_GAP_UPSET   = 80.0   # >80 ELO pts difference is a notable upset
_BASE_DISAGREE   = 0.25   # >0.25 range across base learners = internal disagreement
_TOSS_VENUE_PCT  = 0.62   # venue where toss winner wins >62% of the time
_FORM_SR_GAP     = 15.0   # >15 SR pts difference counts as meaningful form gap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")


def _doc_path(match_date: str, home: str, away: str) -> Path:
    return LEARNINGS_DIR / f"{match_date}_{_slug(home)}_vs_{_slug(away)}.md"


def _already_reviewed(match_date: str, home: str, away: str) -> bool:
    return _doc_path(match_date, home, away).exists()


def _parse_margin(live_status: str) -> tuple[Optional[int], Optional[int]]:
    """Return (margin_runs, margin_wickets) from a live_status string."""
    if not live_status:
        return None, None
    m = re.search(r"won by (\d+)\s*wkt", live_status, re.I)
    if m:
        return None, int(m.group(1))
    m = re.search(r"won by (\d+)\s*run", live_status, re.I)
    if m:
        return int(m.group(1)), None
    return None, None


def _base_learner_range(base_learners: dict) -> float:
    vals = [v for v in base_learners.values() if v is not None]
    return max(vals) - min(vals) if len(vals) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Prediction file loading
# ---------------------------------------------------------------------------

def _load_pred_files() -> list[dict]:
    """Load all prediction files, merging result data from data/preds/ copies."""
    if not PREDICTIONS_DIR.exists():
        return []

    # Index the data/preds/ files by their _file reference or by normalised name
    alt_by_file: dict[str, dict] = {}
    if PREDS_ALT_DIR.exists():
        for p in PREDS_ALT_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                ref = d.get("_file") or p.name
                alt_by_file[ref] = d
                # Also index by normalised match key for fuzzy lookup
                m = d.get("match", {})
                key = f"{m.get('date')}_{_slug(m.get('home',''))}_{_slug(m.get('away',''))}"
                alt_by_file[key] = d
            except Exception:
                pass

    preds = []
    for p in sorted(PREDICTIONS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            d["_source_file"] = str(p)

            # Try to merge result from alt dir
            m = d.get("match", {})
            key = f"{m.get('date')}_{_slug(m.get('home',''))}_{_slug(m.get('away',''))}"
            alt = alt_by_file.get(p.name) or alt_by_file.get(key)
            if alt and alt.get("result"):
                d["result"] = alt["result"]

            preds.append(d)
        except Exception:
            pass
    return preds


def _db_result(home: str, away: str, match_date: str) -> Optional[dict]:
    """Try the DB (read-only) for actual match result."""
    try:
        from .db import connect
        import duckdb
        db_path = Path(__file__).resolve().parent / "data" / "cricket.duckdb"
        con = duckdb.connect(str(db_path), read_only=True)
        row = con.execute("""
            SELECT winner, win_margin_runs, win_margin_wickets
            FROM matches
            WHERE (team_home LIKE ? OR team_away LIKE ?)
              AND (team_home LIKE ? OR team_away LIKE ?)
              AND start_date = CAST(? AS DATE)
            LIMIT 1
        """, [f"%{home.split()[0]}%", f"%{home.split()[0]}%",
              f"%{away.split()[0]}%", f"%{away.split()[0]}%",
              match_date]).fetchone()
        con.close()
        if row and row[0]:
            return {"winner": row[0], "margin_runs": row[1], "margin_wickets": row[2]}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _analyse(pred: dict) -> dict:
    """Return a structured analysis dict for one prediction + result."""
    match  = pred.get("match", {})
    result = pred.get("result") or {}
    feats  = pred.get("features", {})
    base   = pred.get("base_learners", {})
    mvb    = pred.get("model_vs_book", {})
    p_dict = pred.get("prediction", {})

    home = match.get("home", "")
    away = match.get("away", "")

    # ---- Result determination ----
    status = result.get("status")
    actual_winner = result.get("winner")

    if not actual_winner:
        return {"status": "pending", "home": home, "away": away,
                "date": match.get("date")}

    predicted_winner = p_dict.get("favored", "")
    p_home  = float(p_dict.get("p_home_wins", 0.5))
    edge_pp = float(p_dict.get("edge_pct", abs(p_home - 0.5) * 200))
    confidence = abs(p_home - 0.5) * 2
    correct = (predicted_winner == actual_winner)

    base_out = {
        "status":           "correct" if correct else "wrong",
        "home":             home,
        "away":             away,
        "date":             match.get("date"),
        "venue":            match.get("venue", ""),
        "predicted_winner": predicted_winner,
        "actual_winner":    actual_winner,
        "p_home_wins":      p_home,
        "edge_pct":         edge_pp,
        "confidence":       confidence,
        "correct":          correct,
    }

    if correct:
        return base_out

    # ---- Wrong prediction analysis ----
    live_status = result.get("live_status", "")
    margin_runs, margin_wkts = _parse_margin(live_status)

    luck_signals: list[str] = []
    if margin_runs is not None and 0 < margin_runs <= _LUCK_RUNS:
        luck_signals.append(f"won by only {margin_runs} runs")
    if margin_wkts is not None and 0 < margin_wkts <= _LUCK_WICKETS:
        luck_signals.append(f"won by last {margin_wkts} wicket(s)")

    is_luck = bool(luck_signals)
    interesting = edge_pp >= MIN_EDGE_PP and not is_luck

    pattern_flags:       list[str] = []
    actionable_learnings: list[str] = []

    if interesting:
        elo_diff = feats.get("elo_diff_pre", 0) or 0   # positive = home stronger
        actual_home_won = (actual_winner == home)

        # 1. ELO-based upset
        if abs(elo_diff) >= _ELO_GAP_UPSET:
            favoured_by_elo = home if elo_diff > 0 else away
            underdog_won = (actual_winner != favoured_by_elo)
            if underdog_won:
                pattern_flags.append(
                    f"ELO upset: {favoured_by_elo} had a {abs(elo_diff):.0f}-point "
                    f"ELO advantage but lost. The model's ELO signal strongly favoured "
                    f"the wrong side — check if ELO is over-weighted relative to form."
                )
                actionable_learnings.append(
                    f"Review ELO weighting for {match.get('venue','this venue')}. "
                    f"A {abs(elo_diff):.0f}-pt gap predicted confidently but failed."
                )
            else:
                pattern_flags.append(
                    f"ELO-favoured team lost despite {abs(elo_diff):.0f}-pt advantage — "
                    f"ELO alone was insufficient; other signals overrode it."
                )

        # 2. Internal base-model disagreement
        bl_range = _base_learner_range(base)
        if bl_range >= _BASE_DISAGREE:
            bl_str = ", ".join(
                f"{k}={v:.2f}" for k, v in base.items() if v is not None
            )
            pattern_flags.append(
                f"Base-learner disagreement: range={bl_range:.2f} across models "
                f"({bl_str}). The ensemble appeared confident (edge {edge_pp:.1f}pp) "
                f"but the individual models were split — this edge was brittle."
            )
            actionable_learnings.append(
                "When base models disagree by >0.25, treat ensemble confidence as "
                "inflated. Consider adding a calibration penalty for high-disagreement cases."
            )

        # 3. Bookmaker divergence
        book_edge = mvb.get("edge_away_pp") if not actual_home_won else mvb.get("edge_home_pp")
        if book_edge and abs(book_edge) >= 15:
            better_side = "away" if not actual_home_won else "home"
            pattern_flags.append(
                f"Model vs market: bookmakers gave the {better_side} team "
                f"{abs(book_edge):.1f}pp more probability than our model — "
                f"and they were right. Our model may be miscalibrated on this matchup type."
            )
            actionable_learnings.append(
                f"Bookmakers had {abs(book_edge):.1f}pp edge over our model on the "
                f"correct side. Review feature weights or recalibrate the stacked ensemble "
                f"on recent {match.get('match',{}).get('format','T20')} data."
            )

        # 4. Batting form signal conflict
        t1_sr = feats.get("t1_bat_form_sr") or 0
        t2_sr = feats.get("t2_bat_form_sr") or 0
        form_favours_home = t1_sr > t2_sr
        if abs(t1_sr - t2_sr) >= _FORM_SR_GAP and form_favours_home != actual_home_won:
            stronger_sr = home if form_favours_home else away
            pattern_flags.append(
                f"Batting form mismatch: {stronger_sr} had higher form SR "
                f"({max(t1_sr, t2_sr):.1f} vs {min(t1_sr, t2_sr):.1f}) but lost. "
                f"Batting form SR may have been given too much weight over bowling quality."
            )
            t1_econ = feats.get("t1_bowl_career_econ") or 0
            t2_econ = feats.get("t2_bowl_career_econ") or 0
            if abs(t1_econ - t2_econ) > 0.5:
                bowl_edge = away if t1_econ > t2_econ else home
                actionable_learnings.append(
                    f"Bowling economy favoured {bowl_edge} "
                    f"(econ {min(t1_econ, t2_econ):.2f} vs {max(t1_econ, t2_econ):.2f}) "
                    f"and they won — consider increasing bowling econ weight vs batting SR."
                )

        # 5. Toss advantage at venue
        venue_toss_pct = feats.get("venue_toss_winner_winpct") or 0.5
        if venue_toss_pct >= _TOSS_VENUE_PCT:
            # Did the toss winner win?
            # (we don't have toss_winner in features clearly, but we know
            # the actual winner — if it matches who benefits from toss at this
            # venue, flag it)
            pattern_flags.append(
                f"Toss-sensitive venue: toss winner wins {venue_toss_pct:.0%} of matches "
                f"here. If the toss winner also won the match, this may be underweighted."
            )

        # 6. All-model, high-confidence miss
        all_vals = [v for v in base.values() if v is not None]
        if all_vals:
            all_agreed_home = all(v > 0.55 for v in all_vals)
            all_agreed_away = all(v < 0.45 for v in all_vals)
            if (all_agreed_home or all_agreed_away) and confidence >= 0.35:
                pattern_flags.append(
                    f"High-confidence all-model miss: every base learner agreed "
                    f"({', '.join(f'{v:.2f}' for v in all_vals)}) but the prediction "
                    f"was wrong — either a genuine upset or an untracked variable "
                    f"(lineup change, injury, pitch, weather)."
                )
                actionable_learnings.append(
                    "Flag for manual review: high-confidence all-model miss. "
                    "Check for untracked pre-match signals: key player absent, "
                    "unusual pitch or weather, or opposition quality in the runup."
                )

    learnable = interesting and len(actionable_learnings) > 0

    return {
        **base_out,
        "is_luck":              is_luck,
        "luck_signals":         luck_signals,
        "interesting":          interesting,
        "pattern_flags":        pattern_flags,
        "actionable_learnings": actionable_learnings,
        "learnable":            learnable,
        "margin_runs":          margin_runs,
        "margin_wkts":          margin_wkts,
    }


# ---------------------------------------------------------------------------
# Document writers
# ---------------------------------------------------------------------------

def _fmt_float(v, fmt=".2f") -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "—"


def _write_match_doc(pred: dict, analysis: dict) -> Path:
    match  = pred.get("match", {})
    feats  = pred.get("features", {})
    base   = pred.get("base_learners", {})
    totals = pred.get("totals", {})
    mvb    = pred.get("model_vs_book", {})
    p_dict = pred.get("prediction", {})
    result = pred.get("result") or {}
    odds   = pred.get("odds", {}).get("h2h", {})

    home  = match.get("home", "?")
    away  = match.get("away", "?")
    venue = match.get("venue", "Unknown")
    date_s = str(match.get("date", ""))
    fmt   = match.get("format", "T20")

    status        = analysis.get("status", "unknown")
    correct       = analysis.get("correct", False)
    pred_winner   = analysis.get("predicted_winner", "—")
    actual_winner = analysis.get("actual_winner", "—")
    p_home        = analysis.get("p_home_wins", 0.5)
    edge_pp       = analysis.get("edge_pct", 0.0)
    confidence    = analysis.get("confidence", 0.0)
    no_model      = status == "no_model"
    correct_str   = ("N/A" if no_model else ("YES" if correct else "NO"))

    path = _doc_path(date_s, home, away)
    LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)

    margin_str = ""
    live_s = result.get("live_status", "")
    if live_s:
        m = re.search(r"won by .+", live_s, re.I)
        if m:
            margin_str = m.group(0)

    lines: list[str] = [
        f"# Post-Match Review: {home} vs {away}",
        "",
        f"| | |",
        f"|---|---|",
        f"| **Date** | {date_s} |",
        f"| **Venue** | {venue} |",
        f"| **Format** | {fmt} |",
        f"| **Result** | {actual_winner} {margin_str} |",
        "",
        "## Prediction vs Result",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Predicted winner | {pred_winner} |",
        f"| Actual winner | {actual_winner} |",
        f"| P(home={home} wins) | {p_home:.1%} |",
        f"| Model confidence | {confidence:.1%} (edge {edge_pp:.1f} pp) |",
        f"| **Correct?** | {correct_str} |",
        "",
    ]

    if status == "pending":
        lines += ["*Result not yet available — re-run after the match completes.*", ""]

    elif status == "no_model":
        lines += ["*Model not trained at review time — no prediction to compare.*", ""]

    elif status == "correct":
        # Brief correct-prediction section with reasoning
        elo_diff = feats.get("elo_diff_pre", 0) or 0
        bl_range = _base_learner_range(base)
        lines += [
            "## Outcome: Correct Prediction",
            "",
            f"The model correctly favoured **{actual_winner}** with {edge_pp:.1f}pp edge.",
            "",
            "**Why the model was right:**",
            "",
        ]
        if abs(elo_diff) >= 30:
            elo_fav = home if elo_diff > 0 else away
            lines.append(
                f"- ELO: {elo_fav} had a {abs(elo_diff):.0f}-point rating advantage "
                f"(t1={feats.get('t1_elo_pre',0):.0f}, t2={feats.get('t2_elo_pre',0):.0f})."
            )
        h2h = feats.get("h2h_t1_winpct")
        if h2h is not None and abs(h2h - 0.5) > 0.1:
            h2h_fav = home if h2h > 0.5 else away
            lines.append(
                f"- H2H: {h2h_fav} leads the head-to-head "
                f"({h2h:.0%} home win rate over {feats.get('h2h_n_prior',0):.0f} meetings)."
            )
        t1_sr = feats.get("t1_bat_form_sr") or 0
        t2_sr = feats.get("t2_bat_form_sr") or 0
        if abs(t1_sr - t2_sr) >= 10:
            better_form = home if t1_sr > t2_sr else away
            lines.append(
                f"- Form: {better_form} showed better recent batting SR "
                f"({max(t1_sr,t2_sr):.1f} vs {min(t1_sr,t2_sr):.1f})."
            )
        if bl_range < _BASE_DISAGREE:
            lines.append(
                f"- Consensus: all base models agreed (range {bl_range:.2f}) — "
                f"ensemble confidence was genuine."
            )
        lines.append("")

    else:
        # Wrong prediction analysis
        lines += ["## Wrong Prediction Analysis", ""]

        if analysis.get("is_luck"):
            lines += [
                "### Verdict: Luck / Narrow-Margin Loss",
                "",
                "The margin indicates this was largely luck, not a systematic model failure:",
                "",
            ]
            for sig in analysis.get("luck_signals", []):
                lines.append(f"- {sig}")
            lines += [
                "",
                "> **Learning action:** None. Narrow margins are within the model's expected "
                "variance. Do not adjust weights based on this result.",
                "",
            ]

        elif not analysis.get("interesting"):
            lines += [
                "### Verdict: Low-Confidence Miss (noise band)",
                "",
                f"Model edge was only {edge_pp:.1f}pp — within the coin-flip zone. "
                "This wrong prediction is not informative enough to act on.",
                "",
            ]

        else:
            flags = analysis.get("pattern_flags", [])
            learnings = analysis.get("actionable_learnings", [])

            lines += ["### Pattern Flags", ""]
            for f in flags:
                lines += [f"> {f}", ""]
            if not flags:
                lines += ["*No clear systematic pattern identified.*", ""]

            if learnings:
                lines += ["### Actionable Learnings", ""]
                for lr in learnings:
                    lines.append(f"- {lr}")
                lines += [
                    "",
                    "### Retraining Flag",
                    "",
                    f"**Flag for model weight review:** {'YES' if analysis.get('learnable') else 'NO'}  ",
                    f"**Confidence this is learnable:** {'HIGH' if confidence >= 0.50 else 'MEDIUM'}",
                    "",
                ]
            else:
                lines += [
                    "### Verdict: Genuine Upset — No Actionable Learning",
                    "",
                    "Wrong at meaningful confidence but no clear systematic driver. "
                    "The model had the right signals; the match went the other way.",
                    "",
                ]

    # --- Model detail section ---
    lines += [
        "## Model Details",
        "",
        "### Ensemble Components",
        "",
        f"| Base Model | P(home wins) |",
        f"|---|---|",
    ]
    for k, v in base.items():
        lines.append(f"| {k} | {_fmt_float(v)} |")
    lines.append(f"| **ensemble** | **{p_home:.2f}** |")
    lines += [""]

    lines += [
        "### Key Features",
        "",
        f"| Feature | {home} (t1) | {away} (t2) |",
        f"|---|---|---|",
        f"| ELO rating | {_fmt_float(feats.get('t1_elo_pre'),',.0f')} | {_fmt_float(feats.get('t2_elo_pre'),',.0f')} |",
        f"| ELO diff (t1-t2) | {_fmt_float(feats.get('elo_diff_pre'),'+.1f')} | — |",
        f"| Last-5 win% | {_fmt_float(feats.get('t1_last5'),'.0%')} | {_fmt_float(feats.get('t2_last5'),'.0%')} |",
        f"| Last-10 win% | {_fmt_float(feats.get('t1_last10'),'.0%')} | {_fmt_float(feats.get('t2_last10'),'.0%')} |",
        f"| H2H win% | {_fmt_float(feats.get('h2h_t1_winpct'),'.0%')} | — |",
        f"| H2H meetings | {feats.get('h2h_n_prior','—')} | — |",
        f"| Batting form SR | {_fmt_float(feats.get('t1_bat_form_sr'),'.1f')} | {_fmt_float(feats.get('t2_bat_form_sr'),'.1f')} |",
        f"| Bowl career econ | {_fmt_float(feats.get('t1_bowl_career_econ'),'.2f')} | {_fmt_float(feats.get('t2_bowl_career_econ'),'.2f')} |",
        f"| Venue toss win% | {_fmt_float(feats.get('venue_toss_winner_winpct'),'.0%')} | — |",
        "",
    ]

    if mvb:
        edge_home = mvb.get("edge_home_pp")
        edge_away = mvb.get("edge_away_pp")
        n_books   = odds.get("n_books", 0)
        if n_books:
            cons = odds.get("consensus", {})
            lines += [
                "### Model vs Bookmakers",
                "",
                f"| | Value |",
                f"|---|---|",
                f"| Books sampled | {n_books} |",
                f"| Market P(home) | {_fmt_float(cons.get('p_home'),'.1%')} |",
                f"| Market P(away) | {_fmt_float(cons.get('p_away'),'.1%')} |",
                f"| Model edge vs market (home) | {_fmt_float(edge_home,'+.1f')}pp |",
                f"| Model edge vs market (away) | {_fmt_float(edge_away,'+.1f')}pp |",
                "",
            ]

    if totals:
        p50 = totals.get("first_innings_p50")
        p10 = totals.get("first_innings_p10")
        p90 = totals.get("first_innings_p90")
        if p50:
            lines += [
                "### Projected Scores (Monte Carlo)",
                "",
                f"First innings: p10={p10:.0f}, median={p50:.0f}, p90={p90:.0f}",
                "",
            ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _update_summary(reviews: list[dict]) -> None:
    LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LEARNINGS_DIR / "SUMMARY.md"

    existing_lines: list[str] = []
    if summary_path.exists():
        existing_lines = summary_path.read_text(encoding="utf-8").splitlines()

    today  = date.today().isoformat()
    done   = [r for r in reviews if r["analysis"].get("status") not in ("pending", "no_model")]
    total  = len(done)
    correct   = sum(1 for r in done if r["analysis"].get("correct"))
    wrong     = sum(1 for r in done if not r["analysis"].get("correct"))
    luck      = sum(1 for r in done if r["analysis"].get("is_luck"))
    learnable = sum(1 for r in done if r["analysis"].get("learnable"))
    pending   = sum(1 for r in reviews if r["analysis"].get("status") == "pending")

    all_flags: list[str] = []
    for r in done:
        all_flags.extend(r["analysis"].get("pattern_flags", []))

    elo_upsets    = sum(1 for f in all_flags if "ELO upset" in f)
    bl_disagree   = sum(1 for f in all_flags if "disagreement" in f.lower())
    book_diverge  = sum(1 for f in all_flags if "bookmakers" in f.lower())
    form_conflict = sum(1 for f in all_flags if "Form" in f)
    allcomp_miss  = sum(1 for f in all_flags if "all-model" in f.lower())

    lines: list[str] = [
        "# Prediction Learning Summary",
        "",
        f"*Last updated: {today}*",
        "",
        "## This Batch",
        "",
        f"| Metric | Count |",
        f"|---|---|",
        f"| Matches with results | {total} |",
        f"| Correct predictions | {correct} ({correct/total:.0%} accuracy)" if total else f"| Correct predictions | {correct} |",
        f"| Wrong predictions | {wrong} |",
        f"| Wrong due to luck (narrow margin) | {luck} |",
        f"| Learnable errors flagged | {learnable} |",
        f"| Results still pending | {pending} |",
        "",
    ]

    if wrong > 0:
        lines += [
            "## Pattern Frequency (Wrong Predictions)",
            "",
            f"| Pattern | Occurrences |",
            f"|---|---|",
            f"| ELO upset (underdog won despite large gap) | {elo_upsets} |",
            f"| Base-learner internal disagreement | {bl_disagree} |",
            f"| Bookmaker vs model divergence | {book_diverge} |",
            f"| Batting form vs bowling quality conflict | {form_conflict} |",
            f"| All-model high-confidence miss | {allcomp_miss} |",
            "",
        ]

        retrain_flags: list[str] = []
        if elo_upsets >= 2:
            retrain_flags.append(
                "**ELO weighting:** Multiple ELO-based upsets — ELO may be over-weighted "
                "or poorly calibrated for current competition. Review ELO K-factor."
            )
        if bl_disagree >= 2:
            retrain_flags.append(
                "**Ensemble calibration:** High-disagreement predictions repeatedly wrong — "
                "add calibration penalty or widen confidence interval when base models diverge."
            )
        if book_diverge >= 2:
            retrain_flags.append(
                "**Market alignment:** Model consistently disagrees with bookmakers and loses — "
                "consider using bookmaker consensus as an additional feature or prior."
            )
        if retrain_flags:
            lines += [
                "## Recurring Patterns — Retraining Recommendations",
                "",
            ]
            for flag in retrain_flags:
                lines += [f"- {flag}", ""]

    lines += [
        "## Match-Level Log",
        "",
        f"| Date | Match | Predicted | Actual | Edge | OK? | Notes |",
        f"|---|---|---|---|---|---|---|",
    ]

    for r in sorted(reviews, key=lambda x: x["analysis"].get("date", ""), reverse=True):
        an  = r["analysis"]
        doc = r.get("doc_path", "")
        rel = Path(doc).name if doc else ""
        date_s  = an.get("date", "")
        matchup = f"{an.get('home','')} vs {an.get('away','')}"
        pred_w  = an.get("predicted_winner", "—")
        act_w   = an.get("actual_winner", "—") or "pending"
        edge    = f"{an.get('edge_pct',0):.1f}pp"
        st      = an.get("status", "?")
        ok      = ("ok" if an.get("correct")
                   else ("--" if st in ("pending", "no_model") else "NO"))
        note = ""
        if st == "pending":
            note = "result pending"
        elif an.get("is_luck"):
            note = "luck (narrow margin)"
        elif an.get("learnable"):
            note = "**learnable error flagged**"
        elif st == "wrong" and not an.get("interesting"):
            note = "low confidence / noise"

        link = f"[{matchup}]({rel})" if rel else matchup
        lines.append(f"| {date_s} | {link} | {pred_w} | {act_w} | {edge} | {ok} | {note} |")

    lines += [""]

    hist_start = next(
        (i for i, l in enumerate(existing_lines)
         if l.startswith("## Historical")), None,
    )
    if hist_start is not None:
        lines += [""] + existing_lines[hist_start:]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_review(days_back: int = 7) -> int:
    """Scan prediction files for completed matches; analyse and write docs."""
    cutoff_date = date.today() - timedelta(days=days_back)
    today_str   = date.today().isoformat()

    all_preds = _load_pred_files()
    if not all_preds:
        print(f"  No prediction files found in {PREDICTIONS_DIR}")
        return 0

    reviews: list[dict] = []
    skipped = 0

    for pred in all_preds:
        m        = pred.get("match", {})
        home     = m.get("home", "")
        away     = m.get("away", "")
        date_s   = str(m.get("date", ""))

        if not date_s or not home or not away:
            continue

        try:
            match_date = datetime.strptime(date_s, "%Y-%m-%d").date()
        except ValueError:
            continue

        # Only review matches within the window AND in the past
        if match_date > date.today():
            continue
        if match_date < cutoff_date:
            continue

        if _already_reviewed(date_s, home, away):
            skipped += 1
            continue

        result = pred.get("result") or {}

        # If pred file has no result, try DB fallback
        if not result.get("winner"):
            db_res = _db_result(home, away, date_s)
            if db_res:
                result = {
                    "status":  "complete",
                    "winner":  db_res["winner"],
                    "margin_runs":    db_res.get("margin_runs"),
                    "margin_wickets": db_res.get("margin_wickets"),
                }
                pred = {**pred, "result": result}

        analysis = _analyse(pred)

        doc_path = _write_match_doc(pred, analysis)
        print(f"\n  [{date_s}] {home} vs {away}")
        print(f"    venue:  {m.get('venue','?')}")
        actual_w = analysis.get("actual_winner") or "pending"
        print(f"    result: {actual_w}")
        print(f"    -> {doc_path.relative_to(doc_path.parent.parent)}")

        st = analysis.get("status", "?")
        if st == "correct":
            print(f"    [OK] Correct  (edge {analysis.get('edge_pct',0):.1f}pp)")
        elif st == "wrong":
            tag = "LUCK" if analysis.get("is_luck") else (
                  "LEARNABLE" if analysis.get("learnable") else "miss")
            print(f"    [X]  Wrong -- {tag}  (edge {analysis.get('edge_pct',0):.1f}pp)")
        elif st == "pending":
            print("    [?]  Result not yet available")

        reviews.append({
            "analysis": analysis,
            "doc_path": str(doc_path),
        })

    if reviews:
        _update_summary(reviews)
        sp = LEARNINGS_DIR / "SUMMARY.md"
        print(f"\n  Updated {sp.relative_to(sp.parent.parent)}")

    if skipped:
        print(f"\n  Skipped {skipped} already-reviewed match(es).")

    return len(reviews)


# ---------------------------------------------------------------------------
# Per-match phase-aware review (called by orchestrator's phase_loop on
# COMPLETE.review action firing). Performs structured per-version error
# attribution and appends one line to learnings/post_match_log.jsonl.
# ---------------------------------------------------------------------------

LEDGER_PATH = LEARNINGS_DIR / "post_match_log.jsonl"


def _analyse_version(version: dict, match: dict, result: dict) -> dict:
    """Run the standard `_analyse` against a single phase-prediction version.

    Synthesises a pred-shaped dict so we can reuse the existing analysis,
    then folds the version tag and timestamp back in."""
    synth = {
        "match":          match,
        "result":         result,
        "prediction":     version.get("prediction") or {},
        "base_learners":  version.get("base_learners") or {},
        "features":       version.get("features") or {},
        "model_vs_book":  version.get("model_vs_book") or {},
        "xi":             version.get("xi") or {},
    }
    out = _analyse(synth)
    out["tag"] = version.get("tag")
    out["at"]  = version.get("at")
    return out


def _version_delta(prev: dict, cur: dict) -> dict:
    """What changed between two consecutive versions of the same match.

    Reports:
      - p_home_change: how much the home win prob shifted (in pp)
      - edge_change_pp: shift in best-side market edge
      - moved_correctness: did this version flip the call from wrong→right
        (or right→wrong) — the leading indicator that new info actually
        mattered
      - trigger: human-readable hint of what new info likely caused the move
    """
    triggers = {
        "pre_match_v0":  None,
        "pre_start_v1":  "announced XI",
        "toss_aware_v2": "toss winner + decision",
    }
    return {
        "from": prev.get("tag"),
        "to":   cur.get("tag"),
        "p_home_change_pp":   round(((cur.get("p_home_wins") or 0)
                                     - (prev.get("p_home_wins") or 0)) * 100, 2),
        "edge_change_pp":     round((cur.get("edge_pct") or 0)
                                     - (prev.get("edge_pct") or 0), 2),
        "moved_correctness":  bool(prev.get("correct") != cur.get("correct")),
        "trigger":            triggers.get(cur.get("tag")) or "version update",
    }


def _attribution(versions_analysis: list[dict], match: dict, result: dict) -> dict:
    """Roll up the per-version analyses into an "attribution" verdict that
    answers: was the final call right? If not, which phase was responsible
    for the miss — and did any phase add real signal that helped?"""
    if not versions_analysis:
        return {}
    final = versions_analysis[-1]
    final_correct = bool(final.get("correct"))

    # What did each phase add (or take away)?
    phase_contribution: dict[str, str] = {}
    for prev, cur in zip(versions_analysis, versions_analysis[1:]):
        d = _version_delta(prev, cur)
        if d["moved_correctness"]:
            phase_contribution[cur.get("tag")] = (
                "FLIPPED CALL: "
                + ("wrong → right" if cur.get("correct") else "right → wrong")
            )
        elif abs(d["p_home_change_pp"]) >= 3:
            direction = "shifted toward " + (
                match.get("home") if d["p_home_change_pp"] > 0 else match.get("away")
            )
            phase_contribution[cur.get("tag")] = (
                f"{direction} by {abs(d['p_home_change_pp']):.1f}pp"
                + (" (helpful)" if (d["p_home_change_pp"] > 0) == (result.get("winner") == match.get("home")) else " (unhelpful)")
            )
        else:
            phase_contribution[cur.get("tag")] = "no meaningful update"

    # Primary error factor — when the FINAL call was wrong, what's the
    # likely systemic cause? Read the existing _analyse pattern flags
    # from the final version (they already encode ELO upset / base-model
    # disagreement / form-vs-book-disagreement etc.).
    primary_error = None
    if not final_correct:
        flags = final.get("pattern_flags") or []
        if flags:
            primary_error = flags[0].split(":")[0].lower().strip()
        elif final.get("is_luck"):
            primary_error = "narrow_margin_luck"
        else:
            primary_error = "low_signal"

    # Which version got it right (if any)?
    correct_versions = [v.get("tag") for v in versions_analysis if v.get("correct")]

    return {
        "final_version":         final.get("tag"),
        "final_correct":         final_correct,
        "primary_error_factor":  primary_error,
        "phase_contribution":    phase_contribution,
        "correct_versions":      correct_versions,
        "luck_signal":           final.get("is_luck", False),
    }


def review_one(home: str, away: str, date: str) -> Optional[dict]:
    """Phase-aware post-match review for a single fixture.

    Reads the prediction file (with `versions[]` if phase_loop has been
    running), runs per-version analysis, computes deltas + attribution,
    and appends a structured line to `learnings/post_match_log.jsonl`.
    Idempotent: re-running for the same fixture is a no-op.

    Called by orchestrator's phase_loop on the `COMPLETE.review` action.
    """
    LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    fname_stem = f"{_slug(home)}_vs_{_slug(away)}_{date}"
    pred_path = PREDICTIONS_DIR / f"{fname_stem}.json"
    if not pred_path.exists():
        return None
    try:
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    match  = pred.get("match", {}) or {}
    result = pred.get("result") or {}
    if not result.get("winner"):
        # Try the alt copy in data/preds/ which is updated by export
        alt = PREDS_ALT_DIR / pred_path.name
        if alt.exists():
            try:
                alt_d = json.loads(alt.read_text(encoding="utf-8"))
                if (alt_d.get("result") or {}).get("winner"):
                    result = alt_d["result"]
            except Exception:
                pass
    if not result.get("winner"):
        # Fall back to the matches DB
        db_res = _db_result(home, away, date)
        if db_res:
            result = {**result, "winner": db_res["winner"],
                       "margin_runs": db_res.get("margin_runs"),
                       "margin_wickets": db_res.get("margin_wickets")}
    if not result.get("winner"):
        return None  # still no result; review will retry on next phase tick

    # Idempotency: if we already wrote a ledger entry for this fixture,
    # don't append a duplicate. (Use the .md file as the canonical signal —
    # _already_reviewed already exists.)
    if _already_reviewed(date, home, away):
        return None

    # Build the per-version analysis; fall back to the legacy single-prediction
    # shape if the file pre-dates the phase-versioning rework.
    versions = pred.get("versions") or [{
        "tag": "legacy", "at": None,
        "prediction":    pred.get("prediction"),
        "base_learners": pred.get("base_learners"),
        "features":      pred.get("features"),
        "model_vs_book": pred.get("model_vs_book"),
    }]
    versions_analysis = [_analyse_version(v, match, result) for v in versions]

    deltas = []
    for prev, cur in zip(versions_analysis, versions_analysis[1:]):
        deltas.append(_version_delta(prev, cur))

    attribution = _attribution(versions_analysis, match, result)

    ledger_entry = {
        "match": {
            "home":   home,
            "away":   away,
            "date":   date,
            "venue":  match.get("venue"),
            "format": match.get("format"),
        },
        "actual": {
            "winner":      result.get("winner"),
            "live_status": result.get("live_status"),
        },
        "versions":     versions_analysis,
        "deltas":       deltas,
        "attribution":  attribution,
        "reviewed_at":  datetime.now(timezone.utc).isoformat(),
    }

    # Append to ledger (one line per match, JSONL)
    try:
        with LEDGER_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(ledger_entry, default=str) + "\n")
    except Exception:
        pass

    # Also write the per-match .md doc the batch run_review writes, so the
    # existing learnings/ folder stays consistent. Reuse the existing
    # _write_match_doc renderer; it expects a (pred, analysis) pair.
    try:
        final = versions_analysis[-1] if versions_analysis else {}
        if final and final.get("status") in ("correct", "wrong"):
            # Compose a synthetic pred dict that _write_match_doc can render,
            # but stash the version trajectory + attribution under a known
            # key so the renderer (or a downstream reader) can surface it.
            pred_for_doc = {
                **pred,
                "result":              {**(pred.get("result") or {}), **result},
                "_version_trajectory": [
                    {"tag":              v.get("tag"),
                     "predicted_winner": v.get("predicted_winner"),
                     "p_home":           v.get("p_home_wins"),
                     "correct":          v.get("correct")}
                    for v in versions_analysis
                ],
                "_phase_attribution":  attribution,
            }
            _write_match_doc(pred_for_doc, final)
    except Exception:
        pass    # md doc is best-effort

    return ledger_entry
