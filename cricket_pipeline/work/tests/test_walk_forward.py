"""Regression tests for the Wave 1 walk-forward harness, leakage audits,
and reproducibility manifest. These cover the structural correctness of
the new infrastructure without requiring the full DuckDB / cricsheet
data layer — they use synthetic frames.
"""
from __future__ import annotations

import os

# Pre-set BET_MODE so importing siblings (which import bet_engine) doesn't
# trip the live-money validation gate during test collection.
os.environ.setdefault("BET_MODE", "paper")
os.environ.setdefault("BANKROLL", "1000")

import numpy as np
import pandas as pd
import pytest

from cricket_pipeline.work import features_v2 as F
from cricket_pipeline.work import walk_forward as WF


# ---------------------------------------------------------------------------
# walk_forward.WalkForwardWindow
# ---------------------------------------------------------------------------


class TestWalkForwardWindow:
    def _df(self, n=200, start="2024-01-01"):
        dates = pd.date_range(start=start, periods=n, freq="D")
        return pd.DataFrame({
            "match_id": [f"m{i}" for i in range(n)],
            "start_date": dates,
            "season": dates.year.astype(str),
            "y_t1_wins": np.random.RandomState(0).randint(0, 2, size=n),
        })

    def test_slice_basic_no_meta(self):
        df = self._df(n=200)
        w = WF.WalkForwardWindow(
            train_cutoff="2024-05-01",
            calib_start="2024-05-01",
            calib_end="2024-05-15",
            test_start="2024-05-15",
            test_end="2024-06-01",
        )
        train, calib, meta, test = w.slice(df)
        assert len(train) > 0
        assert train["start_date"].max() < pd.Timestamp("2024-05-01")
        assert calib["start_date"].min() >= pd.Timestamp("2024-05-01")
        assert calib["start_date"].max() < pd.Timestamp("2024-05-15")
        assert test["start_date"].min() >= pd.Timestamp("2024-05-15")
        assert meta is None

    def test_slice_with_meta_test(self):
        df = self._df(n=200)
        w = WF.WalkForwardWindow(
            train_cutoff="2024-04-15",
            meta_test_start="2024-04-15",
            meta_test_end="2024-05-01",
            calib_start="2024-05-01",
            calib_end="2024-05-15",
            test_start="2024-05-15",
            test_end="2024-06-01",
        )
        train, calib, meta, test = w.slice(df)
        assert meta is not None
        # Disjointness: meta < calib < test
        assert meta["start_date"].max() < calib["start_date"].min()
        assert calib["start_date"].max() < test["start_date"].min()
        # Train doesn't leak into meta
        assert train["start_date"].max() < meta["start_date"].min()

    def test_season_exclusion(self):
        df = self._df(n=300, start="2023-01-01")
        # Test slice in 2024; exclude any train rows from 2024
        w = WF.WalkForwardWindow(
            train_cutoff="2024-09-01",
            calib_start="2024-09-01",
            calib_end="2024-09-15",
            test_start="2024-09-15",
            test_end="2024-10-01",
        )
        train, _, _, _ = w.slice(df, exclude_seasons=["2024"])
        assert (train["season"] != "2024").all()

    def test_train_leak_guard(self):
        # Construct a frame where train would leak into test if guard fails
        df = pd.DataFrame({
            "match_id": ["a", "b"],
            "start_date": pd.to_datetime(["2024-06-01", "2024-05-01"]),
            "season": ["2024", "2024"],
            "y_t1_wins": [1, 0],
        })
        w = WF.WalkForwardWindow(
            train_cutoff="2024-07-01",
            calib_start="2024-07-01",
            calib_end="2024-07-10",
            test_start="2024-05-15",
            test_end="2024-06-15",
        )
        # The guard inside slice() asserts train.max < test.min; this case
        # has train ending 2024-06-01 but test starting 2024-05-15 → must fail
        with pytest.raises(AssertionError):
            w.slice(df)


# ---------------------------------------------------------------------------
# Reproducibility manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_has_required_fields(self):
        df = pd.DataFrame({
            "match_id": ["a", "b"],
            "start_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "y_t1_wins": [1, 0],
        })
        manifest = WF.make_manifest(tag="test", df=df, seeds={"numpy": 42})
        for key in ("tag", "created_at", "data_hash", "row_count",
                    "seeds", "platform", "python_version", "library_versions"):
            assert key in manifest, f"manifest missing {key}"
        assert manifest["row_count"] == 2
        assert manifest["seeds"]["numpy"] == 42

    def test_data_hash_stable_across_calls(self):
        df = pd.DataFrame({
            "match_id": ["a", "b"],
            "start_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "y_t1_wins": [1, 0],
        })
        h1 = WF.make_manifest(tag="t", df=df, seeds={})["data_hash"]
        h2 = WF.make_manifest(tag="t", df=df, seeds={})["data_hash"]
        assert h1 == h2

    def test_data_hash_changes_when_rows_change(self):
        df1 = pd.DataFrame({
            "match_id": ["a"],
            "start_date": pd.to_datetime(["2024-01-01"]),
            "y_t1_wins": [1],
        })
        df2 = pd.DataFrame({
            "match_id": ["a", "b"],
            "start_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "y_t1_wins": [1, 0],
        })
        h1 = WF.make_manifest(tag="t", df=df1, seeds={})["data_hash"]
        h2 = WF.make_manifest(tag="t", df=df2, seeds={})["data_hash"]
        assert h1 != h2


class TestSeeds:
    def test_set_global_seeds_returns_table(self):
        seeds = WF.set_global_seeds(42)
        assert seeds["numpy"] == 42
        assert seeds["python_random"] == 42

    def test_seeded_numpy_is_reproducible(self):
        WF.set_global_seeds(42)
        a = np.random.rand(5)
        WF.set_global_seeds(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Leakage audits
# ---------------------------------------------------------------------------


class TestAudits:
    def _clean_frame(self, n=200):
        rng = np.random.RandomState(0)
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        h2h_n = rng.randint(0, 20, size=n)
        h2h_wp = rng.uniform(0.3, 0.7, size=n)
        # Honest leakage-free frame: h2h_t1_winpct must be NaN when n_prior == 0
        # (no prior matches → no win pct to compute).
        h2h_wp_clean = np.where(h2h_n == 0, np.nan, h2h_wp)
        return pd.DataFrame({
            "match_id": [f"m{i}" for i in range(n)],
            "start_date": dates,
            "venue": (["MCG", "Lord's"] * (n // 2 + 1))[:n],
            "format": ["T20"] * n,
            "y_t1_wins": rng.randint(0, 2, size=n),
            "t1_elo_pre": 1500 + rng.randn(n) * 100,
            "t1_last5": rng.uniform(0.3, 0.7, size=n),
            "t2_last5": rng.uniform(0.3, 0.7, size=n),
            "venue_n_prior": np.concatenate([[0, 0], rng.randint(1, 100, size=n - 2)]),
            "h2h_n_prior": h2h_n,
            "h2h_t1_winpct": h2h_wp_clean,
        })

    def test_audit_clean_frame_passes(self):
        df = self._clean_frame()
        # Tighten: first MCG and first Lord's are venue_n_prior=0 by construction
        report = F.audit_all(df, strict=False)
        assert report["ok"] or report["n_issues"] == 0

    def test_form_out_of_bounds_caught(self):
        df = self._clean_frame()
        df.loc[5, "t1_last5"] = 1.5  # impossible win pct
        with pytest.raises(AssertionError, match="form leakage"):
            F.audit_form_leakage(df, strict=True)

    def test_h2h_leakage_caught(self):
        df = self._clean_frame()
        # Force a row to have winpct populated with n_prior=0 → leak
        df.loc[10, "h2h_n_prior"] = 0
        df.loc[10, "h2h_t1_winpct"] = 0.5
        with pytest.raises(AssertionError, match="h2h leakage"):
            F.audit_h2h_leakage(df, strict=True)

    def test_audit_all_non_strict_returns_report(self):
        df = self._clean_frame()
        df.loc[5, "t1_last5"] = 1.5
        report = F.audit_all(df, strict=False)
        assert report["ok"] is False
        assert report["n_issues"] >= 1


# ---------------------------------------------------------------------------
# Quarterly window helper
# ---------------------------------------------------------------------------


class TestQuarterlyWindows:
    def test_generates_4_per_year(self):
        windows = WF.quarterly_windows(2024, 2024)
        assert len(windows) == 4
        labels = [w.label for w in windows]
        assert labels == ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]

    def test_meta_test_disjoint_from_calib(self):
        windows = WF.quarterly_windows(2024, 2024)
        for w in windows:
            assert w.meta_test_end == w.calib_start  # touch but don't overlap
            assert w.calib_end == w.test_start
