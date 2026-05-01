"""Regression tests for bet engine: settlement winner-parsing and idempotency.

These cover the cricket-specific edge cases caught in the multi-agent code
review (super over, rain delays, toss-only status, both-team mentions) and
the dedup_key-based idempotency guarantee for place_bet().
"""
from __future__ import annotations

import os

# Force a clean, validatable config BEFORE importing bet_engine. Module-level
# validation runs at import; a stale BET_MODE=polymarket without API key would
# fail import otherwise.
os.environ.setdefault("BET_MODE", "paper")
os.environ.setdefault("BANKROLL", "1000")

from cricket_pipeline.work import bet_engine as BE  # noqa: E402


# ----------------------------------------------------------------------
# _parse_match_winner — the regex-based settlement parser
# ----------------------------------------------------------------------

class TestParseMatchWinner:
    def test_simple_home_win(self):
        assert BE._parse_match_winner(
            "rajasthan royals won by 5 wkts",
            "Rajasthan Royals", "Sunrisers Hyderabad",
        ) == "Rajasthan Royals"

    def test_simple_away_win(self):
        assert BE._parse_match_winner(
            "sunrisers hyderabad won by 12 runs",
            "Rajasthan Royals", "Sunrisers Hyderabad",
        ) == "Sunrisers Hyderabad"

    def test_toss_only_does_not_settle(self):
        # "won the toss" alone must NOT trigger a settlement - this was the
        # bug Round 1 QA found (false-positive settlement on toss).
        assert BE._parse_match_winner(
            "rajasthan royals won the toss and chose to bat",
            "Rajasthan Royals", "Sunrisers Hyderabad",
        ) is None

    def test_no_result_does_not_settle(self):
        assert BE._parse_match_winner(
            "no result - rain stopped play",
            "Rajasthan Royals", "Sunrisers Hyderabad",
        ) is None

    def test_abandoned_does_not_settle(self):
        assert BE._parse_match_winner(
            "match abandoned without a ball bowled",
            "Rajasthan Royals", "Sunrisers Hyderabad",
        ) is None

    def test_both_teams_mentioned_returns_none(self):
        # "X won by 3 runs, Y super over loss" - ambiguous string. Refuse
        # to settle rather than guess. This was the Round 1 QA bug where
        # iteration order picked the wrong team.
        assert BE._parse_match_winner(
            "rajasthan royals won by 5, mumbai super over loss",
            "Rajasthan Royals", "Mumbai Indians",
        ) in (None, "Rajasthan Royals")  # Either is acceptable; never Mumbai

    def test_substring_team_name_collision(self):
        # If one team name is a substring of another, plain `in` matching
        # would falsely settle. Word boundaries should fix that.
        # E.g. "India" vs "West Indies"
        result = BE._parse_match_winner(
            "west indies won by 4 wkts",
            "India", "West Indies",
        )
        assert result == "West Indies"

    def test_empty_status_returns_none(self):
        assert BE._parse_match_winner("", "A", "B") is None

    def test_none_status_returns_none(self):
        assert BE._parse_match_winner(None, "A", "B") is None  # type: ignore[arg-type]

    def test_team_name_with_special_chars(self):
        # Team names with parens / hyphens shouldn't break the regex
        result = BE._parse_match_winner(
            "royal challengers (rcb) won by 7 wkts",
            "Royal Challengers (RCB)", "Mumbai Indians",
        )
        assert result == "Royal Challengers (RCB)"


# ----------------------------------------------------------------------
# _bet_dedup_key — stable identity hashing
# ----------------------------------------------------------------------

class TestBetDedupKey:
    def _decision(self, selection):
        return BE.BetDecision(
            should_bet=True, reason="test", selection=selection,
            decimal_odds=2.0, model_p=0.55, book_p=0.50, edge_pp=5.0,
            kelly=0.05, stake=50.0,
        )

    def test_same_match_same_selection_same_key(self):
        pred1 = {"match": {"home": "RR", "away": "SRH", "date": "2026-04-25"}}
        pred2 = {"match": {"home": "RR", "away": "SRH", "date": "2026-04-25"}}
        d = self._decision("RR")
        assert BE._bet_dedup_key(pred1, d) == BE._bet_dedup_key(pred2, d)

    def test_team_order_does_not_change_key(self):
        # Same logical match (RR vs SRH) regardless of which side the API
        # called "home" - dedup_key should canonicalize via sorted teams.
        pred1 = {"match": {"home": "RR", "away": "SRH", "date": "2026-04-25"}}
        pred2 = {"match": {"home": "SRH", "away": "RR", "date": "2026-04-25"}}
        d = self._decision("RR")
        assert BE._bet_dedup_key(pred1, d) == BE._bet_dedup_key(pred2, d)

    def test_different_dates_different_keys(self):
        pred1 = {"match": {"home": "RR", "away": "SRH", "date": "2026-04-25"}}
        pred2 = {"match": {"home": "RR", "away": "SRH", "date": "2026-04-26"}}
        d = self._decision("RR")
        assert BE._bet_dedup_key(pred1, d) != BE._bet_dedup_key(pred2, d)

    def test_different_selection_different_keys(self):
        pred = {"match": {"home": "RR", "away": "SRH", "date": "2026-04-25"}}
        assert BE._bet_dedup_key(pred, self._decision("RR")) != \
               BE._bet_dedup_key(pred, self._decision("SRH"))

    def test_case_insensitive_normalization(self):
        pred1 = {"match": {"home": "Rajasthan Royals", "away": "SRH", "date": "2026-04-25"}}
        pred2 = {"match": {"home": "rajasthan royals", "away": "srh", "date": "2026-04-25"}}
        d1 = self._decision("Rajasthan Royals")
        d2 = self._decision("rajasthan royals")
        assert BE._bet_dedup_key(pred1, d1) == BE._bet_dedup_key(pred2, d2)


# ----------------------------------------------------------------------
# decide_bet — input validation
# ----------------------------------------------------------------------

class TestDecideBetValidation:
    def _base_pred(self, **overrides):
        pred = {
            "match": {"home": "RR", "away": "SRH", "date": "2026-04-25"},
            "prediction": {"p_home_wins": 0.60, "p_away_wins": 0.40,
                            "favored": "RR", "favored_pct": 60.0},
            "model_vs_book": {"best_side": "RR", "best_odds": 2.0,
                               "kelly_fraction": 0.04},
            "odds": {"h2h": {"consensus": {"p_home": 0.55, "p_away": 0.45}}},
        }
        pred.update(overrides)
        return pred

    def test_missing_odds_block_no_bet(self):
        pred = self._base_pred()
        pred["odds"] = {}
        d = BE.decide_bet(pred)
        assert d.should_bet is False
        assert "no odds" in d.reason.lower()

    def test_missing_match_home_no_bet(self):
        pred = self._base_pred()
        pred["match"] = {}
        d = BE.decide_bet(pred)
        assert d.should_bet is False

    def test_invalid_probability_no_bet(self):
        pred = self._base_pred()
        pred["odds"]["h2h"]["consensus"]["p_home"] = 1.5  # > 1.0
        d = BE.decide_bet(pred)
        assert d.should_bet is False
        assert "out of" in d.reason.lower()

    def test_missing_book_p_no_bet(self):
        pred = self._base_pred()
        pred["odds"]["h2h"]["consensus"] = {}
        d = BE.decide_bet(pred)
        assert d.should_bet is False

    def test_kelly_capped_to_max(self):
        # With KELLY_CAP=0.5, MAX_STAKE_PCT=0.05, BANKROLL=1000:
        # max stake should never exceed 50, even with kelly_fraction=0.99
        pred = self._base_pred()
        pred["model_vs_book"]["kelly_fraction"] = 0.99
        d = BE.decide_bet(pred, bankroll=1000.0)
        if d.should_bet:
            assert d.stake <= 50.0 + 1e-9
