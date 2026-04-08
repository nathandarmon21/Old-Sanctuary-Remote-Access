"""
Tests for sanctuary/economics.py.

Covers: production cost formula, holding costs, revenue adjustment.
"""

import pytest

from sanctuary.economics import (
    BASE_PRODUCTION_COSTS,
    COST_MULTIPLIER_FLOOR,
    production_cost,
    total_holding_cost,
    holding_cost_per_unit_per_day,
    revenue_adjustment,
    apply_price_walk,
    end_of_run_write_off,
    factory_daily_capacity,
)
import numpy as np


# ── Production cost ───────────────────────────────────────────────────────────

class TestProductionCost:
    def test_one_factory_excellent(self):
        # multiplier = max(0.76, 1 - 0.08*(1-1)) = 1.0
        assert production_cost("Excellent", 1) == pytest.approx(25.0)

    def test_one_factory_poor(self):
        assert production_cost("Poor", 1) == pytest.approx(15.0)

    def test_two_factories_excellent(self):
        # multiplier = max(0.76, 1 - 0.08*1) = 0.92
        assert production_cost("Excellent", 2) == pytest.approx(25.0 * 0.92)

    def test_two_factories_poor(self):
        assert production_cost("Poor", 2) == pytest.approx(15.0 * 0.92)

    def test_three_factories(self):
        # multiplier = max(0.76, 1 - 0.08*2) = 0.84
        assert production_cost("Excellent", 3) == pytest.approx(25.0 * 0.84)
        assert production_cost("Poor", 3) == pytest.approx(15.0 * 0.84)

    def test_four_factories_hits_floor(self):
        # multiplier = max(0.76, 1 - 0.08*3) = max(0.76, 0.76) = 0.76 — floor reached here
        assert production_cost("Excellent", 4) == pytest.approx(25.0 * COST_MULTIPLIER_FLOOR)
        assert production_cost("Poor", 4) == pytest.approx(15.0 * COST_MULTIPLIER_FLOOR)

    def test_floor_holds_above_four_factories(self):
        # Above 4 factories the multiplier would go below 0.76 without the floor
        # multiplier = max(0.76, 1 - 0.08*4) = max(0.76, 0.68) = 0.76
        assert production_cost("Excellent", 5) == pytest.approx(25.0 * COST_MULTIPLIER_FLOOR)
        assert production_cost("Poor", 5) == pytest.approx(15.0 * COST_MULTIPLIER_FLOOR)

    def test_large_factory_count_stays_at_floor(self):
        assert production_cost("Excellent", 100) == production_cost("Excellent", 4)
        assert production_cost("Poor", 100) == production_cost("Poor", 4)

    def test_cost_decreases_monotonically(self):
        # More factories → lower (or equal) cost
        costs = [production_cost("Excellent", f) for f in range(1, 12)]
        for i in range(len(costs) - 1):
            assert costs[i] >= costs[i + 1], f"Cost increased at factory count {i+2}"

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown quality"):
            production_cost("Mediocre", 1)

    def test_zero_factories_raises(self):
        with pytest.raises(ValueError, match="factories must be >= 1"):
            production_cost("Excellent", 0)

    def test_negative_factories_raises(self):
        with pytest.raises(ValueError, match="factories must be >= 1"):
            production_cost("Excellent", -1)


# ── Factory capacity ──────────────────────────────────────────────────────────

class TestFactoryCapacity:
    def test_one_factory(self):
        assert factory_daily_capacity(1) == 1

    def test_three_factories(self):
        assert factory_daily_capacity(3) == 3


# ── Holding costs ─────────────────────────────────────────────────────────────

class TestHoldingCosts:
    def test_excellent_unit_cost(self):
        assert holding_cost_per_unit_per_day("Excellent") == pytest.approx(0.125)

    def test_poor_unit_cost(self):
        assert holding_cost_per_unit_per_day("Poor") == pytest.approx(0.075)

    def test_total_holding_cost_mixed_inventory(self):
        inv = {"Excellent": 2, "Poor": 3}
        expected = 2 * 0.125 + 3 * 0.075
        assert total_holding_cost(inv) == pytest.approx(expected)

    def test_total_holding_cost_empty_inventory(self):
        assert total_holding_cost({"Excellent": 0, "Poor": 0}) == pytest.approx(0.0)

    def test_total_holding_cost_only_excellent(self):
        assert total_holding_cost({"Excellent": 5, "Poor": 0}) == pytest.approx(5 * 0.125)

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown quality"):
            holding_cost_per_unit_per_day("Medium")


# ── Revenue adjustment ────────────────────────────────────────────────────────

class TestRevenueAdjustment:
    def test_no_misrepresentation_no_adjustment(self):
        assert revenue_adjustment("Excellent", "Excellent", 80.0, 45.0, 5) == pytest.approx(0.0)
        assert revenue_adjustment("Poor", "Poor", 80.0, 45.0, 3) == pytest.approx(0.0)

    def test_cheated_buyer_negative_adjustment(self):
        # Seller claimed Excellent ($80), true quality is Poor ($45)
        # Buyer was over-credited by (80 - 45) per unit
        adj = revenue_adjustment("Excellent", "Poor", 80.0, 45.0, 3)
        assert adj == pytest.approx((45.0 - 80.0) * 3)
        assert adj < 0  # buyer loses money on revelation

    def test_pleasant_surprise_positive_adjustment(self):
        # Seller claimed Poor ($45), true quality is Excellent ($80)
        adj = revenue_adjustment("Poor", "Excellent", 80.0, 45.0, 2)
        assert adj == pytest.approx((80.0 - 45.0) * 2)
        assert adj > 0  # buyer gains money on revelation

    def test_adjustment_scales_with_quantity(self):
        adj1 = revenue_adjustment("Excellent", "Poor", 80.0, 45.0, 1)
        adj10 = revenue_adjustment("Excellent", "Poor", 80.0, 45.0, 10)
        assert adj10 == pytest.approx(10 * adj1)

    def test_adjustment_uses_production_day_prices(self):
        # When prices drift, the adjustment should use the stored prices
        # from the production day, not current prices.
        # Here we test with different Excellent/Poor prices.
        adj = revenue_adjustment("Excellent", "Poor", 85.0, 50.0, 2)
        assert adj == pytest.approx((50.0 - 85.0) * 2)

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError):
            revenue_adjustment("Excellent", "Mediocre", 80.0, 45.0, 1)


# ── End-of-run write-off ──────────────────────────────────────────────────────

class TestWriteOff:
    def test_write_off_at_production_cost(self):
        inv = {"Excellent": 2, "Poor": 1}
        # 2 Excellent × $25 + 1 Poor × $15 = $65 (1 factory)
        assert end_of_run_write_off(inv, 1) == pytest.approx(2 * 25.0 + 1 * 15.0)

    def test_empty_inventory_no_write_off(self):
        assert end_of_run_write_off({"Excellent": 0, "Poor": 0}, 1) == pytest.approx(0.0)

    def test_write_off_reflects_economies_of_scale(self):
        inv = {"Excellent": 1}
        cost_1f = end_of_run_write_off(inv, 1)
        cost_3f = end_of_run_write_off(inv, 3)
        assert cost_3f < cost_1f  # more factories → lower production cost


# ── Price walk ────────────────────────────────────────────────────────────────

class TestPriceWalk:
    def test_price_walk_stays_positive(self):
        rng = np.random.default_rng(42)
        price = 1.0
        for _ in range(1000):
            price = apply_price_walk(price, rng)
            assert price >= 1.0

    def test_price_walk_is_deterministic(self):
        p1 = 80.0
        p2 = 80.0
        for _ in range(20):
            rng1 = np.random.default_rng(99)
            rng2 = np.random.default_rng(99)
            p1 = apply_price_walk(p1, rng1)
            p2 = apply_price_walk(p2, rng2)
        assert p1 == p2

    def test_price_walk_drifts_from_seed(self):
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        after_a = apply_price_walk(80.0, rng_a)
        after_b = apply_price_walk(80.0, rng_b)
        # Different seeds should produce different outcomes
        # (Not guaranteed but astronomically likely)
        assert after_a != after_b
