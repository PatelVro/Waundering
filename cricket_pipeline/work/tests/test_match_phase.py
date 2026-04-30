"""Tests for the phase state machine.

Exercises the pure phase classifier with frozen-clock fixtures covering:
  - clean upcoming → live → complete progression
  - abandoned-no-toss short-circuit
  - super-over completion
  - rescheduled match (action rewind)
  - mid-day discovery (no start_ts initially, then derived)
  - tied-without-super-over (no winner)

Run from project root:
    .venv/Scripts/python -m unittest cricket_pipeline.work.tests.test_match_phase
"""
from __future__ import annotations

import unittest

from .. import match_phase as mp


# Fixed reference time: 2026-04-30 14:00:00 UTC
# (= what `parse_start_ts_from_status("Match starts at Apr 30, 14:00 GMT", year_hint=2026)` returns)
T0 = 1777557600
DAY = 86400


def make_entry(**overrides) -> dict:
    """Default per-match entry with optional overrides."""
    base = {
        "slug": "fixture-test",
        "last_state": {
            "home": "Bangladesh",
            "away": "New Zealand",
            "status": "Match starts at Apr 30, 14:00 GMT",
            "is_complete": False,
        },
        "phase": mp.Phase.DISCOVERED.value,
        "start_ts": T0,
        "actions_fired": {},
    }
    base.update(overrides)
    return base


class StatusParserTests(unittest.TestCase):
    def test_parse_starts_at_gmt(self):
        ts = mp.parse_start_ts_from_status(
            "Match starts at Apr 30, 14:00 GMT", year_hint=2026,
            now_func=lambda: T0 - DAY,
        )
        self.assertIsNotNone(ts)
        # 2026-04-30 14:00 UTC
        self.assertEqual(ts, T0)

    def test_parse_starts_at_no_tz(self):
        ts = mp.parse_start_ts_from_status(
            "Match starts at May 02, 09:30",
            year_hint=2026,
            now_func=lambda: T0,
        )
        self.assertIsNotNone(ts)

    def test_parse_returns_none_on_garbage(self):
        self.assertIsNone(mp.parse_start_ts_from_status(""))
        self.assertIsNone(mp.parse_start_ts_from_status("Match in progress"))
        self.assertIsNone(mp.parse_start_ts_from_status(None))

    def test_year_rollover(self):
        # Status says "Jan 02" but it's currently late December
        ts = mp.parse_start_ts_from_status(
            "Match starts at Jan 02, 09:00 GMT",
            year_hint=2026,
            now_func=lambda: 1798675200,  # 2026-12-30
        )
        self.assertIsNotNone(ts)
        # Should land in 2027, not 2026
        from datetime import datetime, timezone
        self.assertEqual(datetime.fromtimestamp(ts, tz=timezone.utc).year, 2027)


class TossDetectionTests(unittest.TestCase):
    def test_opt_to_bowl(self):
        w, d = mp.detect_toss({"status": "Bangladesh opt to bowl"})
        self.assertEqual((w, d), ("Bangladesh", "bowl"))

    def test_elected_to_bat(self):
        w, d = mp.detect_toss({"status": "New Zealand elected to bat"})
        self.assertEqual((w, d), ("New Zealand", "bat"))

    def test_won_the_toss_chose_to_field(self):
        w, d = mp.detect_toss({"status": "India won the toss and chose to field"})
        self.assertEqual((w, d), ("India", "field"))

    def test_no_toss_in_status(self):
        w, d = mp.detect_toss({"status": "Match starts at Apr 30, 14:00 GMT"})
        self.assertEqual((w, d), (None, None))


class PhaseClassifierTests(unittest.TestCase):
    def test_pure_discovered_when_no_state(self):
        e = make_entry(last_state={}, start_ts=None, phase=mp.Phase.DISCOVERED.value)
        self.assertEqual(mp.compute_next_phase(e, now=T0 - DAY),
                         mp.Phase.DISCOVERED.value)

    def test_scheduled_well_before_kickoff(self):
        e = make_entry()
        # 2 hours before
        self.assertEqual(mp.compute_next_phase(e, now=T0 - 2 * 3600),
                         mp.Phase.SCHEDULED.value)

    def test_pre_start_in_window(self):
        e = make_entry()
        # 10 min before kickoff (inside T-30m window, before T-0)
        self.assertEqual(mp.compute_next_phase(e, now=T0 - 600),
                         mp.Phase.PRE_START.value)

    def test_live_at_kickoff(self):
        e = make_entry()
        self.assertEqual(mp.compute_next_phase(e, now=T0 + 60),
                         mp.Phase.LIVE.value)

    def test_live_when_status_has_play_markers(self):
        # Even if start_ts is None, in-play status flips to LIVE
        e = make_entry(start_ts=None,
                        last_state={"home": "Bangladesh", "away": "New Zealand",
                                    "status": "Bangladesh need 56 runs in 6.4 overs",
                                    "is_complete": False})
        self.assertEqual(mp.compute_next_phase(e, now=T0),
                         mp.Phase.LIVE.value)

    def test_complete_when_winner_text(self):
        e = make_entry(last_state={"home": "Bangladesh", "away": "New Zealand",
                                    "status": "Bangladesh won by 6 wkts",
                                    "is_complete": True})
        self.assertEqual(mp.compute_next_phase(e, now=T0 + 4 * 3600),
                         mp.Phase.COMPLETE.value)

    def test_complete_via_super_over(self):
        e = make_entry(last_state={"home": "KKR", "away": "LSG",
                                    "status": "Match tied (KKR won the Super Over)",
                                    "is_complete": True})
        self.assertEqual(mp.compute_next_phase(e, now=T0 + 4 * 3600),
                         mp.Phase.COMPLETE.value)

    def test_abandoned_short_circuit(self):
        # Abandoned-no-toss: even before kickoff, jump straight to ABANDONED
        e = make_entry(last_state={"home": "Bangladesh", "away": "New Zealand",
                                    "status": "Match abandoned due to rain (no toss)",
                                    "is_complete": False})
        self.assertEqual(mp.compute_next_phase(e, now=T0 - 1800),
                         mp.Phase.ABANDONED.value)

    def test_abandoned_after_play_starts(self):
        e = make_entry(last_state={"home": "Bangladesh", "away": "New Zealand",
                                    "status": "Match abandoned due to rain after 2 overs",
                                    "is_complete": True})
        self.assertEqual(mp.compute_next_phase(e, now=T0 + 3600),
                         mp.Phase.ABANDONED.value)


class DueActionsTests(unittest.TestCase):
    def test_no_actions_well_before_kickoff(self):
        e = make_entry(phase=mp.Phase.SCHEDULED.value)
        # T-2h: no actions due yet (T-30m window not entered)
        self.assertEqual(mp.due_actions(e, now=T0 - 2 * 3600), [])

    def test_pre_match_actions_at_t_minus_30(self):
        e = make_entry(phase=mp.Phase.SCHEDULED.value)
        due = mp.due_actions(e, now=T0 - 30 * 60)
        self.assertEqual(set(due), {mp.A_PITCH_WEATHER, mp.A_PRE_MATCH_PRED})

    def test_pre_match_actions_idempotent(self):
        e = make_entry(phase=mp.Phase.SCHEDULED.value,
                        actions_fired={mp.A_PRE_MATCH_PRED: "2026-04-30T13:30:00Z"})
        due = mp.due_actions(e, now=T0 - 25 * 60)
        # PRE_MATCH_PRED already fired; only PITCH_WEATHER should be left
        self.assertEqual(due, [mp.A_PITCH_WEATHER])

    def test_pre_start_pred_at_t_minus_5(self):
        e = make_entry(phase=mp.Phase.PRE_START.value)
        due = mp.due_actions(e, now=T0 - 4 * 60)
        self.assertIn(mp.A_PRE_START_PRED, due)

    def test_toss_aware_fires_after_toss_seen(self):
        e = make_entry(phase=mp.Phase.PRE_START.value, toss_seen_at=T0 - 8 * 60)
        due = mp.due_actions(e, now=T0 - 7 * 60)
        self.assertIn(mp.A_TOSS_AWARE_PRED, due)

    def test_toss_aware_can_fire_even_in_live_phase(self):
        # Sometimes we miss PRE_START (orchestrator restart) and the match
        # is already LIVE before we observe toss. The toss-aware version
        # should still fire.
        e = make_entry(phase=mp.Phase.LIVE.value, toss_seen_at=T0 + 30)
        self.assertIn(mp.A_TOSS_AWARE_PRED, mp.due_actions(e, now=T0 + 90))

    def test_settle_fires_on_complete(self):
        e = make_entry(phase=mp.Phase.COMPLETE.value, completed_at=T0 + 4 * 3600)
        self.assertIn(mp.A_SETTLE, mp.due_actions(e, now=T0 + 4 * 3600 + 60))

    def test_review_waits_an_hour_after_complete(self):
        e = make_entry(phase=mp.Phase.COMPLETE.value, completed_at=T0 + 4 * 3600)
        # 30 min after complete: review NOT due
        self.assertNotIn(mp.A_REVIEW,
                          mp.due_actions(e, now=T0 + 4 * 3600 + 30 * 60))
        # 70 min after complete: review IS due
        self.assertIn(mp.A_REVIEW,
                       mp.due_actions(e, now=T0 + 4 * 3600 + 70 * 60))


class RescheduleTests(unittest.TestCase):
    def test_meaningful_reschedule(self):
        self.assertTrue(mp.is_meaningful_reschedule(T0, T0 + 7200))      # +2h
        self.assertTrue(mp.is_meaningful_reschedule(T0, T0 - 86400))    # -1d
        self.assertFalse(mp.is_meaningful_reschedule(T0, T0 + 60))      # +1m: noise
        self.assertFalse(mp.is_meaningful_reschedule(T0, None))
        self.assertFalse(mp.is_meaningful_reschedule(None, T0))

    def test_reset_actions_for_scheduled_clears_pre_match_and_pre_start(self):
        entry = {"actions_fired": {
            mp.A_PRE_MATCH_PRED:  "x",
            mp.A_PITCH_WEATHER:   "x",
            mp.A_PRE_START_PRED:  "x",
            mp.A_TOSS_AWARE_PRED: "x",
            mp.A_SETTLE:          "x",   # COMPLETE — must be preserved
        }}
        mp.reset_actions_for_phase_and_after(entry, mp.Phase.SCHEDULED.value)
        self.assertEqual(set(entry["actions_fired"].keys()), {mp.A_SETTLE})

    def test_reset_actions_for_pre_start_keeps_scheduled(self):
        entry = {"actions_fired": {
            mp.A_PRE_MATCH_PRED:  "x",   # SCHEDULED — must be preserved
            mp.A_PRE_START_PRED:  "x",
            mp.A_TOSS_AWARE_PRED: "x",
        }}
        mp.reset_actions_for_phase_and_after(entry, mp.Phase.PRE_START.value)
        self.assertEqual(set(entry["actions_fired"].keys()), {mp.A_PRE_MATCH_PRED})


class FullProgressionScenarioTest(unittest.TestCase):
    """Walk one match through every phase to verify the action firing order."""

    def test_full_progression(self):
        # T-2h: scheduled, no actions
        e = make_entry(phase=mp.Phase.DISCOVERED.value)
        self.assertEqual(mp.compute_next_phase(e, now=T0 - 7200),
                         mp.Phase.SCHEDULED.value)
        e["phase"] = mp.Phase.SCHEDULED.value
        self.assertEqual(mp.due_actions(e, now=T0 - 7200), [])

        # T-30m: pre-match snapshot fires
        due = mp.due_actions(e, now=T0 - 30 * 60)
        self.assertEqual(set(due), {mp.A_PITCH_WEATHER, mp.A_PRE_MATCH_PRED})
        for k in due:
            e["actions_fired"][k] = "fired"

        # T-10m: phase advances to PRE_START; no new actions until T-5m
        self.assertEqual(mp.compute_next_phase(e, now=T0 - 10 * 60),
                         mp.Phase.PRE_START.value)
        e["phase"] = mp.Phase.PRE_START.value
        self.assertEqual(mp.due_actions(e, now=T0 - 10 * 60), [])

        # T-5m: pre-start prediction
        due = mp.due_actions(e, now=T0 - 5 * 60)
        self.assertEqual(due, [mp.A_PRE_START_PRED])
        e["actions_fired"][mp.A_PRE_START_PRED] = "fired"

        # T-3m: toss happens
        e["last_state"]["status"] = "New Zealand opt to bowl"
        e["toss_seen_at"] = T0 - 3 * 60
        due = mp.due_actions(e, now=T0 - 2 * 60)
        self.assertEqual(due, [mp.A_TOSS_AWARE_PRED])
        e["actions_fired"][mp.A_TOSS_AWARE_PRED] = "fired"

        # T+1m: in-play
        e["last_state"]["status"] = "Bangladesh need 156 in 20 ov"
        self.assertEqual(mp.compute_next_phase(e, now=T0 + 60),
                         mp.Phase.LIVE.value)

        # T+4h: match complete
        e["last_state"]["status"] = "Bangladesh won by 6 wkts"
        e["last_state"]["is_complete"] = True
        self.assertEqual(mp.compute_next_phase(e, now=T0 + 4 * 3600),
                         mp.Phase.COMPLETE.value)
        e["phase"] = mp.Phase.COMPLETE.value
        e["completed_at"] = T0 + 4 * 3600

        # Settle fires immediately, review waits 1h
        due_now = mp.due_actions(e, now=T0 + 4 * 3600 + 30)
        self.assertEqual(due_now, [mp.A_SETTLE])

        e["actions_fired"][mp.A_SETTLE] = "fired"
        # T+complete+90m: review due
        due_later = mp.due_actions(e, now=T0 + 5 * 3600 + 30 * 60)
        self.assertEqual(due_later, [mp.A_REVIEW])


if __name__ == "__main__":
    unittest.main(verbosity=2)
