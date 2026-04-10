"""
Tests for sanctuary/revelation.py.

Covers: deterministic 5-day revelation lag, scheduler queue mechanics.
"""

import pytest

from sanctuary.economics import REVELATION_LAG_DAYS
from sanctuary.revelation import RevelationScheduler, RevelationEvent


class TestRevelationLag:
    def test_lag_is_five_days(self):
        assert REVELATION_LAG_DAYS == 5

    def test_revelation_day_is_transaction_day_plus_five(self):
        scheduler = RevelationScheduler()
        rev_day = scheduler.schedule(
            transaction_id="tx1", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=3,
        )
        assert rev_day == 8

    def test_revelation_fires_on_correct_day(self):
        scheduler = RevelationScheduler()
        scheduler.schedule(
            transaction_id="tx1", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=1,
        )
        # Should not fire on days 1-5
        for day in range(1, 6):
            events = scheduler.fire(day)
            assert events == [], f"Should not fire on day {day}"

        # Should fire on day 6
        events = scheduler.fire(6)
        assert len(events) == 1
        assert events[0].transaction_id == "tx1"
        assert events[0].revelation_day == 6

    def test_multiple_revelations_same_day(self):
        scheduler = RevelationScheduler()
        scheduler.schedule(
            transaction_id="tx1", seller="S1", buyer="B1",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=5,
        )
        scheduler.schedule(
            transaction_id="tx2", seller="S2", buyer="B2",
            claimed_quality="Poor", true_quality="Poor",
            quantity=2, transaction_day=5,
        )
        events = scheduler.fire(10)
        assert len(events) == 2

    def test_revelations_fire_in_order(self):
        scheduler = RevelationScheduler()
        scheduler.schedule(
            transaction_id="tx_early", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=1,
        )
        scheduler.schedule(
            transaction_id="tx_late", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Excellent",
            quantity=1, transaction_day=3,
        )
        # Day 6: only tx_early fires
        events = scheduler.fire(6)
        assert len(events) == 1
        assert events[0].transaction_id == "tx_early"

        # Day 8: tx_late fires
        events = scheduler.fire(8)
        assert len(events) == 1
        assert events[0].transaction_id == "tx_late"

    def test_pending_count(self):
        scheduler = RevelationScheduler()
        assert scheduler.pending_count() == 0
        scheduler.schedule(
            transaction_id="tx1", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=1,
        )
        assert scheduler.pending_count() == 1
        scheduler.fire(6)
        assert scheduler.pending_count() == 0

    def test_fire_removes_from_queue(self):
        scheduler = RevelationScheduler()
        scheduler.schedule(
            transaction_id="tx1", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=1,
        )
        scheduler.fire(6)
        # Firing again on the same day should return nothing
        events = scheduler.fire(6)
        assert events == []

    def test_late_fire_catches_overdue(self):
        """If fire() is called late, it still catches overdue revelations."""
        scheduler = RevelationScheduler()
        scheduler.schedule(
            transaction_id="tx1", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=1,
        )
        # Skip days 2-9, fire on day 10
        events = scheduler.fire(10)
        assert len(events) == 1

    def test_misrepresented_property(self):
        event = RevelationEvent(
            transaction_id="tx1", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=1, revelation_day=6,
        )
        assert event.misrepresented is True

    def test_honest_transaction_not_misrepresented(self):
        event = RevelationEvent(
            transaction_id="tx1", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Excellent",
            quantity=1, transaction_day=1, revelation_day=6,
        )
        assert event.misrepresented is False

    def test_all_pending_returns_copy(self):
        scheduler = RevelationScheduler()
        scheduler.schedule(
            transaction_id="tx1", seller="S", buyer="B",
            claimed_quality="Excellent", true_quality="Poor",
            quantity=1, transaction_day=1,
        )
        pending = scheduler.all_pending()
        assert len(pending) == 1
        assert pending[0].transaction_id == "tx1"
