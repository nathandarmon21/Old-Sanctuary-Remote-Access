"""
Tests for sanctuary/revelation.py.

Covers: PMF properties, delay sampling distribution, correct firing day,
scheduler determinism.
"""

import numpy as np
import pytest

from sanctuary.revelation import (
    REVELATION_DELAYS,
    REVELATION_PMF,
    RevelationScheduler,
    RevelationEvent,
)


# ── PMF sanity ────────────────────────────────────────────────────────────────

class TestPMFProperties:
    def test_sums_to_one(self):
        assert abs(sum(REVELATION_PMF) - 1.0) < 1e-9

    def test_all_non_negative(self):
        assert all(p >= 0 for p in REVELATION_PMF)

    def test_delays_are_positive(self):
        assert all(d > 0 for d in REVELATION_DELAYS)

    def test_mode_is_day_4(self):
        # PMF should have its maximum at delay=4
        max_prob = max(REVELATION_PMF)
        max_delay = REVELATION_DELAYS[REVELATION_PMF.index(max_prob)]
        assert max_delay == 4

    def test_median_is_day_4(self):
        # CDF at day 4: 0.05+0.15+0.20+0.30 = 0.70 ≥ 0.5; at day 3: 0.40 < 0.5
        cumsum = 0.0
        for delay, prob in zip(REVELATION_DELAYS, REVELATION_PMF):
            cumsum += prob
            if cumsum >= 0.5:
                assert delay == 4
                break


# ── Distribution accuracy ─────────────────────────────────────────────────────

class TestDelayDistribution:
    def test_empirical_distribution_matches_pmf(self):
        """
        Over 100,000 samples, the empirical distribution should be close
        to the specified PMF (within 1% for each bucket).
        """
        n = 100_000
        rng = np.random.default_rng(0)
        scheduler = RevelationScheduler(rng)

        counts = {d: 0 for d in REVELATION_DELAYS}
        for i in range(n):
            day = scheduler.schedule(
                transaction_id=f"tx_{i}",
                seller="S",
                buyer="B",
                claimed_quality="Excellent",
                true_quality="Poor",
                quantity=1,
                transaction_day=0,
            )
            delay = day  # transaction_day=0 so delay == revelation_day
            counts[delay] += 1

        for delay, prob in zip(REVELATION_DELAYS, REVELATION_PMF):
            empirical = counts[delay] / n
            assert abs(empirical - prob) < 0.01, (
                f"Delay {delay}: expected ~{prob:.2f}, got {empirical:.4f}"
            )


# ── Firing mechanics ──────────────────────────────────────────────────────────

class TestRevelationFiring:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.scheduler = RevelationScheduler(self.rng)

    def _schedule_one(self, transaction_day: int) -> tuple[str, int]:
        tx_id = "tx_test"
        revelation_day = self.scheduler.schedule(
            transaction_id=tx_id,
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            claimed_quality="Excellent",
            true_quality="Poor",
            quantity=2,
            transaction_day=transaction_day,
        )
        return tx_id, revelation_day

    def test_fires_on_correct_day(self):
        _, rev_day = self._schedule_one(transaction_day=5)

        # Should not fire before the revelation day
        for day in range(5, rev_day):
            events = self.scheduler.fire(day)
            assert len(events) == 0, f"Fired early on day {day} (expected {rev_day})"

        # Should fire on revelation day
        events = self.scheduler.fire(rev_day)
        assert len(events) == 1
        assert events[0].transaction_id == "tx_test"
        assert events[0].seller == "Meridian Manufacturing"
        assert events[0].buyer == "Halcyon Assembly"
        assert events[0].claimed_quality == "Excellent"
        assert events[0].true_quality == "Poor"
        assert events[0].misrepresented is True

    def test_removed_from_queue_after_firing(self):
        _, rev_day = self._schedule_one(transaction_day=1)
        assert self.scheduler.pending_count() == 1
        self.scheduler.fire(rev_day)
        assert self.scheduler.pending_count() == 0

    def test_fires_late_if_day_skipped(self):
        """fire() with day > revelation_day should still return the event."""
        rng2 = np.random.default_rng(7)
        s2 = RevelationScheduler(rng2)
        rev_day = s2.schedule("tx", "S", "B", "Poor", "Poor", 1, transaction_day=1)
        # Fire on a day much later than revelation_day
        events = s2.fire(rev_day + 10)
        assert len(events) == 1

    def test_multiple_revelations_same_day(self):
        rng3 = np.random.default_rng(99)
        s3 = RevelationScheduler(rng3)
        # Schedule many transactions on day 1, collect their revelation days
        tx_ids = []
        rev_days = []
        for i in range(20):
            tx_id = f"tx_{i}"
            rd = s3.schedule(tx_id, "S", "B", "Excellent", "Excellent", 1, transaction_day=1)
            tx_ids.append(tx_id)
            rev_days.append(rd)

        # Fire for every possible revelation day
        all_fired = []
        for day in range(2, 20):
            all_fired.extend(s3.fire(day))

        # Every transaction should have fired exactly once
        assert len(all_fired) == 20
        fired_ids = {e.transaction_id for e in all_fired}
        assert fired_ids == set(tx_ids)

    def test_honest_transaction_misrepresented_false(self):
        rng4 = np.random.default_rng(1)
        s4 = RevelationScheduler(rng4)
        rev_day = s4.schedule("tx", "S", "B", "Excellent", "Excellent", 1, transaction_day=1)
        events = s4.fire(rev_day)
        assert events[0].misrepresented is False

    def test_pending_count_accurate(self):
        rng5 = np.random.default_rng(5)
        s5 = RevelationScheduler(rng5)
        assert s5.pending_count() == 0
        s5.schedule("t1", "S", "B", "Excellent", "Poor", 1, transaction_day=1)
        assert s5.pending_count() == 1
        s5.schedule("t2", "S", "B", "Poor", "Poor", 1, transaction_day=1)
        assert s5.pending_count() == 2


# ── Determinism ───────────────────────────────────────────────────────────────

class TestSchedulerDeterminism:
    def test_same_seed_same_revelation_days(self):
        """Two schedulers initialised with the same seed must produce
        identical revelation day sequences."""
        def run(seed: int) -> list[int]:
            rng = np.random.default_rng(seed)
            s = RevelationScheduler(rng)
            days = []
            for i in range(50):
                rd = s.schedule(f"tx_{i}", "S", "B", "Excellent", "Poor", 1, transaction_day=i)
                days.append(rd)
            return days

        assert run(42) == run(42)

    def test_different_seeds_different_revelation_days(self):
        def run(seed: int) -> list[int]:
            rng = np.random.default_rng(seed)
            s = RevelationScheduler(rng)
            return [
                s.schedule(f"tx_{i}", "S", "B", "Excellent", "Poor", 1, transaction_day=i)
                for i in range(30)
            ]

        # Different seeds should almost certainly produce different sequences
        assert run(42) != run(43)
