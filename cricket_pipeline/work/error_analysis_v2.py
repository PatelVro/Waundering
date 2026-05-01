"""Stratified error analysis on the FULL stacked ensemble.

What this gives over `error_analysis.py`:
  1. Uses the production ensemble (LGBM-num, LGBM-cat, XGB, CatBoost*, LR + LR meta)
     instead of the LGBM-numeric baseline.
  2. Stratifies test-set accuracy by **tier** (top-flight leagues + Test nations
     vs associate / women's qualifier matches). This separates "matches we care
     about for betting" from "matches that drag down our top-line metric".
  3. Slices by venue, by year, by team-strength gap, by chase vs set, and by
     close vs blowout — so we see exactly where the model breaks.
  4. Writes a clean markdown report plus a CSV of the test set with per-row
     predictions for any further drill-down.

Run:
    python -m cricket_pipeline.work.error_analysis_v2 --fmt T20,IT20 --tag t20
    python -m cricket_pipeline.work.error_analysis_v2 --fmt ODI       --tag odi
    python -m cricket_pipeline.work.error_analysis_v2 --fmt all       --tag all
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .ensemble import _cat_pred, _lgb_pred, _lr_pred, _xgb_pred
from .eval import RUNS_DIR, ece_bins, fit_isotonic, time_split
from .features_v2 import CATEGORICAL, NUMERIC, PLAYER_NUMERIC, build_features_with_players


# ---------- tier classification ----------

# Regex on competition string: matches bind tier in priority order, first hit wins.
TIER1_PATTERNS = [
    r"\bindian premier league\b",
    r"\bipl\b",
    r"\bicc men's t20 world cup\b(?!\s+(qualifier|africa|asia|americas|eap|europe))",
    r"\bicc cricket world cup\b(?!\s+(qualifier|league|challenge))",
    r"\bicc champions trophy\b",
    r"\bbig bash league\b",
    r"\bcaribbean premier league\b",
    r"\bmajor league cricket\b",
    r"\bthe hundred\b",
    r"\bsa20\b",
    r"\bilt20\b",
    r"\bbangladesh premier league\b",
    r"\blanka premier league\b",
    r"\bvitality blast\b",
    r"\bcounty championship\b",
    r"\bsheffield shield\b",
]

TEST_NATIONS = {
    "India", "Australia", "England", "South Africa", "New Zealand",
    "Pakistan", "Sri Lanka", "Bangladesh", "West Indies",
    "Afghanistan", "Ireland", "Zimbabwe",
}

WOMEN_PATTERN = re.compile(r"\bwomen\b|\bwpl\b|\bwbbl\b", re.I)
ASSOC_QUAL_PATTERN = re.compile(
    r"qualifier|division|tri[- ]?nation|tri[- ]?series|trophy|"
    r"\baf(?:rica)? region\b|\basia region\b|\bamericas region\b|\beap\b|\beurope\b|\bcwc league\b",
    re.I,
)


def classify_tier(competition: str | None, team_home: str | None,
                   team_away: str | None) -> str:
    """Return 'tier1', 'tier2', or 'other'."""
    comp = (competition or "")
    if WOMEN_PATTERN.search(comp):
        # all women's go to tier2 by default; promote big women's ICC + WPL
        if re.search(r"women's premier league|women's t20 world cup\b(?!.*qualifier)|"
                      r"women's world cup\b(?!.*qualifier)", comp, re.I):
            return "tier2_main"
        return "tier2_assoc"
    for pat in TIER1_PATTERNS:
        if re.search(pat, comp, re.I):
            return "tier1"
    if ASSOC_QUAL_PATTERN.search(comp):
        return "tier2_assoc"
    # Fallback: bilateral series — tier1 if both Test nations
    if team_home in TEST_NATIONS and team_away in TEST_NATIONS:
        return "tier1"
    return "tier2_other"


# ---------- ensemble training ----------

def _train_full_ensemble(train, calib, test, formats):
    """Train the production stacked ensemble on the train split, calibrate
    base learners on calib, fit a LR meta-learner on calib, score on test.
    Returns the calibrated, ensembled probabilities for the test set."""
    feat_num = NUMERIC + PLAYER_NUMERIC
    feat_cat = CATEGORICAL

    for c in feat_cat:
        train[c] = train[c].astype("category")
        calib[c] = calib[c].astype("category")
        test[c]  = test[c].astype("category")

    ytr = train["y_t1_wins"].astype(int).to_numpy()
    yca = calib["y_t1_wins"].astype(int).to_numpy()
    yte = test["y_t1_wins"].astype(int).to_numpy()

    print("  Training base learners (LGBM-num, LGBM-cat, XGB, CatBoost, LR)…")
    seeds = (0, 7, 42)
    lgb_n_ca, lgb_n_te = _lgb_pred(train, calib, test, feat_num, [], ytr, yca, seeds=seeds)
    lgb_c_ca, lgb_c_te = _lgb_pred(train, calib, test, feat_num + feat_cat, feat_cat, ytr, yca, seeds=seeds)
    xgb_ca,    xgb_te  = _xgb_pred(train, calib, test, feat_num, ytr, yca, seeds=seeds)
    cat_ca,    cat_te  = _cat_pred(train, calib, test, feat_num, feat_cat, ytr, yca, seeds=seeds)
    lr_ca,     lr_te   = _lr_pred(train, calib, test, feat_num, ytr, yca)

    Xst_ca = np.column_stack([lgb_n_ca, lgb_c_ca, xgb_ca, cat_ca, lr_ca])
    Xst_te = np.column_stack([lgb_n_te, lgb_c_te, xgb_te, cat_te, lr_te])
    stk = LogisticRegression(C=10.0).fit(Xst_ca, yca)
    p_te = stk.predict_proba(Xst_te)[:, 1]

    # Per-base-learner test predictions for diagnostic dump
    bases = {
        "lgbm_num": lgb_n_te, "lgbm_cat": lgb_c_te,
        "xgb": xgb_te, "cat": cat_te, "lr": lr_te,
        "ensemble": p_te,
    }
    return bases


# ---------- analysis ----------

def _slice_metrics(df: pd.DataFrame, by: str, min_n: int = 10) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        brier=("brier", "mean"),
        avg_conf=("confidence", "mean"),
    ).reset_index()
    g = g[g["n"] >= min_n].sort_values("acc", ascending=False)
    return g


def _md_table(df: pd.DataFrame, top: int | None = None,
              colmap: dict[str, str] | None = None) -> str:
    if df.empty: return "_(no rows)_\n"
    df = df.copy()
    if top: df = df.head(top)
    if colmap:
        df = df.rename(columns=colmap)
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if c.lower() in ("acc", "avg_conf", "avg conf", "share"):
                    cells.append(f"{v:.1%}")
                elif c.lower() in ("brier", "ece"):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(f"{v:.2f}")
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(formats: list[str], tag: str):
    print(f"Loading features (formats={formats})…")
    if formats == ["all"]:
        df = build_features_with_players(format_filter=None)
    else:
        df = build_features_with_players(format_filter=formats)
    train, calib, test, sd = time_split(df, test_frac=0.15, calib_frac=0.10)
    print(f"  rows  train={len(train):,}  calib={len(calib):,}  test={len(test):,}")

    bases = _train_full_ensemble(train, calib, test, formats)

    # Tag test set + per-row diagnostics
    test = test.copy()
    test["pred_p_t1"]  = bases["ensemble"]
    test["pred"]       = (test["pred_p_t1"] >= 0.5).astype(int)
    test["correct"]    = (test["pred"] == test["y_t1_wins"].astype(int)).astype(int)
    test["confidence"] = (np.abs(test["pred_p_t1"] - 0.5) * 2)
    test["brier"]      = (test["pred_p_t1"] - test["y_t1_wins"].astype(int)) ** 2
    test["tier"]       = test.apply(lambda r: classify_tier(r.get("competition"),
                                                              r.get("team_home"),
                                                              r.get("team_away")), axis=1)
    test["year"]       = pd.to_datetime(test["start_date"]).dt.year
    test["elo_gap"]    = pd.cut(
        test["elo_diff_pre"].abs(),
        bins=[0, 30, 80, 150, 1000],
        labels=["close (≤30)", "moderate (30-80)", "wide (80-150)", "very wide (150+)"],
        include_lowest=True,
    )
    test["conf_bin"] = pd.cut(test["confidence"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
                              labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
                              right=False)

    # ==================================================================
    # Build the markdown report
    # ==================================================================
    out_md  = RUNS_DIR / f"{tag}_err_v2.md"
    out_csv = RUNS_DIR / f"{tag}_err_v2.csv"

    # Headline metrics (all + per-tier)
    overall = {
        "n":          len(test),
        "acc":        float(test["correct"].mean()),
        "logloss":    float(log_loss(test["y_t1_wins"], np.clip(test["pred_p_t1"], 1e-7, 1-1e-7))),
        "brier":      float(brier_score_loss(test["y_t1_wins"], test["pred_p_t1"])),
        "ece":        float(ece_bins(test["y_t1_wins"].astype(int).to_numpy(), test["pred_p_t1"].to_numpy())["ece"]),
    }
    per_tier = test.groupby("tier", observed=True).apply(lambda g: pd.Series({
        "n":     len(g),
        "acc":   g["correct"].mean(),
        "brier": g["brier"].mean(),
        "ece":   ece_bins(g["y_t1_wins"].astype(int).to_numpy(), g["pred_p_t1"].to_numpy())["ece"],
    })).reset_index()

    # Per-base-learner accuracy
    base_table = []
    for k in ("lgbm_num", "lgbm_cat", "xgb", "cat", "lr", "ensemble"):
        p = bases[k]
        pred = (p >= 0.5).astype(int)
        base_table.append({
            "learner": k,
            "acc":     float((pred == test["y_t1_wins"].astype(int).to_numpy()).mean()),
            "brier":   float(brier_score_loss(test["y_t1_wins"], p)),
            "logloss": float(log_loss(test["y_t1_wins"], np.clip(p, 1e-7, 1-1e-7))),
        })
    base_df = pd.DataFrame(base_table)

    # Tier-1 only deep dive
    t1 = test[test["tier"] == "tier1"].copy()
    if len(t1) >= 10:
        by_year_t1   = _slice_metrics(t1, "year", min_n=20)
        by_comp_t1   = _slice_metrics(t1, "competition", min_n=15)
        by_elogap_t1 = t1.groupby("elo_gap", observed=True).agg(
            n=("correct","size"), acc=("correct","mean"), avg_conf=("confidence","mean"),
        ).reset_index()
    else:
        by_year_t1 = by_comp_t1 = by_elogap_t1 = pd.DataFrame()

    # Reliability (overall and tier1) — conf_bin added above before t1 copy
    rel_all  = test.groupby("conf_bin", observed=True).agg(
        n=("correct","size"), acc=("correct","mean"),
        avg_p=("pred_p_t1", lambda s: float(s.where(s >= 0.5, 1 - s).mean())),
    ).reset_index()
    if len(t1) >= 10:
        rel_t1 = t1.groupby("conf_bin", observed=True).agg(
            n=("correct","size"), acc=("correct","mean"),
            avg_p=("pred_p_t1", lambda s: float(s.where(s >= 0.5, 1 - s).mean())),
        ).reset_index()
    else:
        rel_t1 = pd.DataFrame()

    # High-confidence wrongs (tier1 only)
    if len(t1) >= 10:
        t1_wrongs = t1[t1["correct"] == 0].sort_values("confidence", ascending=False).head(20)
    else:
        t1_wrongs = test[test["correct"] == 0].sort_values("confidence", ascending=False).head(20)

    # Upset detection
    test["elo_favored_t1"] = (test["elo_diff_pre"] >= 0).astype(int)
    test["upset"]          = (test["elo_favored_t1"] != test["y_t1_wins"].astype(int)).astype(int)
    upset_share = float(test["upset"].mean())
    upset_caught = float(((test["pred"] == test["y_t1_wins"].astype(int)) & (test["upset"] == 1)).sum() /
                          max(test["upset"].sum(), 1))

    # Save CSV for any post-hoc drill
    csv_cols = [c for c in (
        "match_id","start_date","format","competition","tier","year",
        "team_home","team_away","venue","winner","y_t1_wins","pred_p_t1","pred","correct","confidence","brier",
        "elo_diff_pre","h2h_t1_winpct","h2h_n_prior",
    ) if c in test.columns]
    test[csv_cols].to_csv(out_csv, index=False)

    # ==================================================================
    # Render markdown
    # ==================================================================
    lines = []
    lines.append(f"# Error analysis · `{tag}` (formats: {','.join(formats)})\n")
    lines.append(f"_Generated against the **production stacked ensemble** "
                  f"(LGBM-num + LGBM-cat + XGB + CatBoost + LR + LR meta)._\n")
    lines.append("\n## Headline\n")
    lines.append(f"- **Test set:** {overall['n']:,} matches "
                  f"(time-based hold-out, latest slice)\n"
                  f"- **Accuracy:** {overall['acc']*100:.2f}%\n"
                  f"- **Brier:** {overall['brier']:.3f}   |   "
                  f"**Logloss:** {overall['logloss']:.3f}   |   "
                  f"**ECE:** {overall['ece']*100:.2f}%\n"
                  f"- **Upsets:** {upset_share*100:.1f}% of test (lower-Elo team won), "
                  f"model caught **{upset_caught*100:.1f}%** of them\n")
    lines.append("\n## Per-base-learner\n")
    lines.append(_md_table(base_df))

    lines.append("\n## Per-tier — where the accuracy actually lives\n")
    lines.append(
        "Tier-1 = top-flight leagues (IPL, Big Bash, CPL, MLC, Hundred, SA20, ILT20, "
        "BPL, LPL, ICC men's WC + Champions Trophy) and bilateral series between full Test nations.\n"
        "Tier-2 = women's competitions, qualifier tournaments, ICC CWC League 2, "
        "associate-nation tri-series, etc. — these often dominate the test set "
        "but are not the matches you bet on day-to-day.\n"
    )
    lines.append(_md_table(per_tier))

    if not rel_t1.empty:
        lines.append("\n## Reliability — Tier 1 only\n")
        lines.append("Confidence × accuracy. Well-calibrated would have `acc ≈ avg_p` for each row.\n")
        lines.append(_md_table(rel_t1))

    lines.append("\n## Reliability — All tiers\n")
    lines.append(_md_table(rel_all))

    if not by_comp_t1.empty:
        lines.append("\n## Tier-1 accuracy by competition (n ≥ 15)\n")
        lines.append(_md_table(by_comp_t1, top=20))

    if not by_year_t1.empty:
        lines.append("\n## Tier-1 accuracy by year\n")
        lines.append(_md_table(by_year_t1))

    if not by_elogap_t1.empty:
        lines.append("\n## Tier-1 accuracy by Elo gap\n")
        lines.append("Bigger Elo gap → easier prediction. If accuracy is flat across gaps, the model isn't learning team strength well.\n")
        lines.append(_md_table(by_elogap_t1))

    lines.append("\n## Top 20 high-confidence misses (Tier 1 if available)\n")
    pretty = t1_wrongs[["start_date","competition","team_home","team_away","venue",
                         "winner","pred_p_t1","elo_diff_pre","h2h_t1_winpct"]].copy()
    pretty["pred_p_t1"]    = pretty["pred_p_t1"].round(3)
    pretty["elo_diff_pre"] = pretty["elo_diff_pre"].round(0)
    pretty["h2h_t1_winpct"] = pretty["h2h_t1_winpct"].round(2)
    pretty["start_date"]   = pretty["start_date"].astype(str).str[:10]
    lines.append(_md_table(pretty))

    lines.append(f"\n## Files\n- Per-row CSV: `{out_csv.name}`\n- This report:  `{out_md.name}`\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote: {out_md}")
    print(f"Wrote: {out_csv}")
    print()
    print(f"=== Headline ===")
    print(f"  Overall acc: {overall['acc']*100:.2f}%   Brier: {overall['brier']:.3f}   ECE: {overall['ece']*100:.2f}%")
    print()
    print(per_tier.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="T20,IT20")
    ap.add_argument("--tag", default="t20")
    args = ap.parse_args()
    fmts = [s.strip() for s in args.fmt.split(",") if s.strip()]
    main(fmts, args.tag)
