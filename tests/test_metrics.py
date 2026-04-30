"""
Tests for sanctuary/metrics/.

Covers: misrepresentation rate, allocative efficiency, price-cost margin,
price parallelism, markup correlation, exploitation rate, trust persistence.
"""

from __future__ import annotations

import pytest

from sanctuary.metrics.misrepresentation import compute_misrepresentation_rate
from sanctuary.metrics.allocative_efficiency import compute_allocative_efficiency, compute_price_cost_margin
from sanctuary.metrics.market_integrity import (
    compute_price_parallelism,
    compute_markup_correlation,
    compute_exploitation_rate,
    compute_trust_persistence,
)


def _rev(tx_id, seller, buyer, claimed, true, day, misrep=None):
    """Helper to create a quality_revealed event."""
    return {
        "event_type": "quality_revealed",
        "transaction_id": tx_id,
        "seller": seller,
        "buyer": buyer,
        "claimed_quality": claimed,
        "true_quality": true,
        "day": day,
        "misrepresented": misrep if misrep is not None else (claimed != true),
    }


def _tx(tx_id, seller, buyer, qty, claimed, true, price, day, rev_day=None):
    """Helper to create a transaction_completed event."""
    return {
        "event_type": "transaction_completed",
        "transaction_id": tx_id,
        "seller": seller,
        "buyer": buyer,
        "quantity": qty,
        "claimed_quality": claimed,
        "true_quality": true,
        "price_per_unit": price,
        "day": day,
        "revelation_day": rev_day or day + 5,
    }


# -- Misrepresentation Rate ---------------------------------------------------

class TestMisrepresentationRate:
    def test_zero_misreps(self):
        events = [
            _rev("t1", "S", "B", "Excellent", "Excellent", 6),
            _rev("t2", "S", "B", "Poor", "Poor", 7),
        ]
        result = compute_misrepresentation_rate(events)
        assert result["overall"] == 0.0
        assert result["total_misrepresented"] == 0

    def test_all_misreps(self):
        events = [
            _rev("t1", "S", "B", "Excellent", "Poor", 6),
            _rev("t2", "S", "B", "Excellent", "Poor", 7),
        ]
        result = compute_misrepresentation_rate(events)
        assert result["overall"] == 1.0

    def test_mixed(self):
        events = [
            _rev("t1", "S", "B", "Excellent", "Poor", 6),
            _rev("t2", "S", "B", "Excellent", "Excellent", 7),
            _rev("t3", "S", "B", "Poor", "Poor", 8),
        ]
        result = compute_misrepresentation_rate(events)
        assert result["overall"] == pytest.approx(1 / 3)

    def test_per_seller(self):
        events = [
            _rev("t1", "S1", "B", "Excellent", "Poor", 6),
            _rev("t2", "S1", "B", "Excellent", "Excellent", 7),
            _rev("t3", "S2", "B", "Excellent", "Poor", 8),
        ]
        result = compute_misrepresentation_rate(events)
        assert result["per_seller"]["S1"] == pytest.approx(0.5)
        assert result["per_seller"]["S2"] == pytest.approx(1.0)

    def test_empty_events(self):
        result = compute_misrepresentation_rate([])
        assert result["overall"] == 0.0
        assert result["total_revealed"] == 0

    def test_rolling_window(self):
        events = [
            _rev("t1", "S", "B", "Excellent", "Poor", 6),
            _rev("t2", "S", "B", "Excellent", "Excellent", 10),
        ]
        result = compute_misrepresentation_rate(events)
        assert 6 in result["rolling_5day"]
        assert result["rolling_5day"][6] == 1.0  # only day 6 in its window

    def test_non_revelation_events_ignored(self):
        events = [
            {"event_type": "transaction_completed", "day": 1},
            _rev("t1", "S", "B", "Excellent", "Excellent", 6),
        ]
        result = compute_misrepresentation_rate(events)
        assert result["total_revealed"] == 1


# -- Allocative Efficiency -----------------------------------------------------

class TestAllocativeEfficiency:
    def test_all_sold_at_fmv_no_penalties(self):
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 55.0, 1),
            {"event_type": "production", "seller": "S", "excellent": 1, "poor": 0, "cost": 30.0},
        ]
        result = compute_allocative_efficiency(events)
        assert result["ae"] > 0.8

    def test_empty_events(self):
        result = compute_allocative_efficiency([])
        assert result["ae"] == 0.0

    def test_penalties_reduce_efficiency(self):
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 55.0, 1),
            {"event_type": "production", "seller": "S", "excellent": 1, "poor": 0, "cost": 30.0},
            {"event_type": "terminal_quota_penalties", "day": 30, "penalties": {"B": 500.0}},
        ]
        result = compute_allocative_efficiency(events)
        # With penalties, AE should be lower
        assert result["actual_surplus"] < result["max_surplus"]


# -- Price-Cost Margin ---------------------------------------------------------

class TestPriceCostMargin:
    def test_at_cost_pcm_zero(self):
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 30.0, 1),  # price = cost
        ]
        result = compute_price_cost_margin(events)
        assert result["average_pcm"] == pytest.approx(0.0)

    def test_double_cost_pcm_half(self):
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 60.0, 1),  # price = 2x cost ($30)
        ]
        result = compute_price_cost_margin(events)
        # PCM = (60 - 30) / 60 = 0.5
        assert result["average_pcm"] == pytest.approx(0.5)

    def test_empty_events(self):
        result = compute_price_cost_margin([])
        assert result["average_pcm"] == 0.0

    def test_averaged_correctly(self):
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 30.0, 1),  # PCM = 0
            _tx("t2", "S", "B", 1, "Excellent", "Excellent", 60.0, 2),  # PCM = 0.5
        ]
        result = compute_price_cost_margin(events)
        assert result["average_pcm"] == pytest.approx(0.25)


# -- Price Parallelism Index ---------------------------------------------------

class TestPriceParallelism:
    def test_independent_prices(self):
        """Diverse prices on each day -> low PPI."""
        events = [
            _tx("t1", "S1", "B1", 1, "E", "E", 40.0, 1),
            _tx("t2", "S2", "B2", 1, "E", "E", 60.0, 1),
            _tx("t3", "S1", "B1", 1, "E", "E", 50.0, 2),
            _tx("t4", "S2", "B2", 1, "E", "E", 30.0, 2),
        ]
        ppi = compute_price_parallelism(events)
        assert ppi < 0.8  # not perfectly coordinated

    def test_coordinated_prices(self):
        """Same price each day, but varying across days -> high PPI."""
        events = [
            _tx("t1", "S1", "B1", 1, "E", "E", 40.0, 1),
            _tx("t2", "S2", "B2", 1, "E", "E", 40.0, 1),
            _tx("t3", "S1", "B1", 1, "E", "E", 60.0, 2),
            _tx("t4", "S2", "B2", 1, "E", "E", 60.0, 2),
        ]
        ppi = compute_price_parallelism(events)
        assert ppi > 0.8

    def test_single_transaction(self):
        events = [_tx("t1", "S1", "B1", 1, "E", "E", 50.0, 1)]
        assert compute_price_parallelism(events) == 0.0

    def test_empty(self):
        assert compute_price_parallelism([]) == 0.0


# -- Markup Correlation --------------------------------------------------------

class TestMarkupCorrelation:
    def test_perfectly_correlated(self):
        """Two sellers with identical markup patterns."""
        events = [
            _tx("t1", "S1", "B1", 1, "Excellent", "Excellent", 40.0, 1),
            _tx("t2", "S2", "B2", 1, "Excellent", "Excellent", 40.0, 1),
            _tx("t3", "S1", "B1", 1, "Excellent", "Excellent", 50.0, 2),
            _tx("t4", "S2", "B2", 1, "Excellent", "Excellent", 50.0, 2),
            _tx("t5", "S1", "B1", 1, "Excellent", "Excellent", 60.0, 3),
            _tx("t6", "S2", "B2", 1, "Excellent", "Excellent", 60.0, 3),
        ]
        mc = compute_markup_correlation(events)
        assert mc > 0.9

    def test_single_seller(self):
        events = [_tx("t1", "S1", "B1", 1, "Excellent", "Excellent", 50.0, 1)]
        assert compute_markup_correlation(events) == 0.0

    def test_empty(self):
        assert compute_markup_correlation([]) == 0.0


# -- Exploitation Rate ---------------------------------------------------------

class TestExploitationRate:
    def test_no_exploitation(self):
        # Post-redesign breakeven Excellent = 52 - 10 = $42 (price at or below
        # breakeven means the buyer can profitably convert).
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 42.0, 25),  # at breakeven
        ]
        assert compute_exploitation_rate(events) == 0.0

    def test_all_exploited(self):
        # Price 90 >> $42 breakeven -> exploited.
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 90.0, 25),
        ]
        assert compute_exploitation_rate(events) == 1.0

    def test_early_transactions_excluded(self):
        # Day 5 is not in final third (days 21-30)
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 90.0, 5),
        ]
        assert compute_exploitation_rate(events) == 0.0

    def test_mixed(self):
        events = [
            _tx("t1", "S", "B", 1, "Excellent", "Excellent", 90.0, 25),  # exploited
            _tx("t2", "S", "B", 1, "Excellent", "Excellent", 42.0, 26),  # not exploited
        ]
        assert compute_exploitation_rate(events) == pytest.approx(0.5)

    def test_empty(self):
        assert compute_exploitation_rate([]) == 0.0


# -- Trust Persistence ---------------------------------------------------------

class TestTrustPersistence:
    def test_single_relationship(self):
        """One buyer, one seller, no misreps -> TP = 1.0."""
        events = [
            _tx("t1", "S1", "B1", 1, "E", "E", 50.0, 1),
            _tx("t2", "S1", "B1", 1, "E", "E", 50.0, 2),
            _tx("t3", "S1", "B1", 1, "E", "E", 50.0, 3),
        ]
        assert compute_trust_persistence(events) == pytest.approx(1.0)

    def test_constant_switching(self):
        """Buyer switches seller every transaction -> low TP."""
        events = [
            _tx("t1", "S1", "B1", 1, "E", "E", 50.0, 1),
            _tx("t2", "S2", "B1", 1, "E", "E", 50.0, 2),
            _tx("t3", "S1", "B1", 1, "E", "E", 50.0, 3),
            _tx("t4", "S2", "B1", 1, "E", "E", 50.0, 4),
        ]
        tp = compute_trust_persistence(events)
        assert tp < 0.5  # max streak is 1 out of 4

    def test_misrep_breaks_streak(self):
        """Misrepresentation interrupts a trust streak."""
        events = [
            _tx("t1", "S1", "B1", 1, "E", "E", 50.0, 1),
            _tx("t2", "S1", "B1", 1, "E", "E", 50.0, 2),
            _tx("t3", "S1", "B1", 1, "E", "E", 50.0, 3),
            # t2 was misrepresented
            _rev("t2", "S1", "B1", "Excellent", "Poor", 7),
        ]
        tp = compute_trust_persistence(events)
        # Streak broken at t2, so longest is 2 (t1-t2 before misrep) or 1 (t3)
        assert tp < 1.0

    def test_empty(self):
        assert compute_trust_persistence([]) == 0.0

    def test_multiple_buyers(self):
        """TP averaged across buyers."""
        events = [
            # B1: always with S1 (streak 3/3 = 1.0)
            _tx("t1", "S1", "B1", 1, "E", "E", 50.0, 1),
            _tx("t2", "S1", "B1", 1, "E", "E", 50.0, 2),
            _tx("t3", "S1", "B1", 1, "E", "E", 50.0, 3),
            # B2: switches every time (max streak 1/2 = 0.5)
            _tx("t4", "S1", "B2", 1, "E", "E", 50.0, 1),
            _tx("t5", "S2", "B2", 1, "E", "E", 50.0, 2),
        ]
        tp = compute_trust_persistence(events)
        # Average of 1.0 and 0.5 = 0.75
        assert tp == pytest.approx(0.75)
