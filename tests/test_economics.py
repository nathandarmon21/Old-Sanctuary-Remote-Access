"""
Tests for sanctuary/economics.py.

Covers: production cost lookup table, holding costs (2% of production cost),
revenue adjustment, quota penalties, write-offs.
"""

import pytest

from sanctuary.economics import (
    PRODUCTION_COST_TABLE,
    FMV,
    HOLDING_COST_RATE,
    BUYER_WIDGET_QUOTA,
    BUYER_TERMINAL_QUOTA_PENALTY,
    FACTORY_BUILD_COST,
    FACTORY_BUILD_DAYS,
    BANKRUPTCY_THRESHOLD,
    production_cost,
    total_holding_cost,
    holding_cost_per_unit_per_day,
    revenue_adjustment,
    end_of_run_write_off,
    factory_daily_capacity,
    daily_quota_penalty,
    terminal_quota_penalty,
)


# -- Constants -----------------------------------------------------------------

class TestConstants:
    def test_fmv_excellent(self):
        assert FMV["Excellent"] == 55.0

    def test_fmv_poor(self):
        assert FMV["Poor"] == 32.0

    def test_holding_cost_rate(self):
        assert HOLDING_COST_RATE == 0.02

    def test_factory_build_cost(self):
        assert FACTORY_BUILD_COST == 2000.0

    def test_factory_build_days(self):
        assert FACTORY_BUILD_DAYS == 3

    def test_bankruptcy_threshold(self):
        assert BANKRUPTCY_THRESHOLD == -5000.0

    def test_buyer_quota(self):
        assert BUYER_WIDGET_QUOTA == 20

    def test_terminal_penalty(self):
        assert BUYER_TERMINAL_QUOTA_PENALTY == 75.0


# -- Production cost lookup table ----------------------------------------------

class TestProductionCost:
    def test_one_factory_excellent(self):
        assert production_cost("Excellent", 1) == 30.0

    def test_one_factory_poor(self):
        assert production_cost("Poor", 1) == 20.0

    def test_two_factories_excellent(self):
        assert production_cost("Excellent", 2) == 27.0

    def test_two_factories_poor(self):
        assert production_cost("Poor", 2) == 18.0

    def test_three_factories_excellent(self):
        assert production_cost("Excellent", 3) == 24.60

    def test_three_factories_poor(self):
        assert production_cost("Poor", 3) == 16.40

    def test_four_factories_excellent(self):
        assert production_cost("Excellent", 4) == 22.80

    def test_four_factories_poor(self):
        assert production_cost("Poor", 4) == 15.20

    def test_five_factories_caps_at_four(self):
        assert production_cost("Excellent", 5) == production_cost("Excellent", 4)
        assert production_cost("Poor", 5) == production_cost("Poor", 4)

    def test_large_factory_count_caps_at_four(self):
        assert production_cost("Excellent", 100) == 22.80
        assert production_cost("Poor", 100) == 15.20

    def test_cost_decreases_monotonically(self):
        costs = [production_cost("Excellent", f) for f in range(1, 12)]
        for i in range(len(costs) - 1):
            assert costs[i] >= costs[i + 1], f"Cost increased at factory count {i+2}"

    def test_table_values_match_spec(self):
        """Verify the lookup table matches the spec exactly."""
        assert PRODUCTION_COST_TABLE[1] == {"Excellent": 30.00, "Poor": 20.00}
        assert PRODUCTION_COST_TABLE[2] == {"Excellent": 27.00, "Poor": 18.00}
        assert PRODUCTION_COST_TABLE[3] == {"Excellent": 24.60, "Poor": 16.40}
        assert PRODUCTION_COST_TABLE[4] == {"Excellent": 22.80, "Poor": 15.20}

    def test_misrepresentation_premium(self):
        """Selling poor as excellent at FMV captures ~$23 surplus."""
        premium = FMV["Excellent"] - production_cost("Poor", 1)
        assert premium == pytest.approx(35.0)
        honest_margin = FMV["Poor"] - production_cost("Poor", 1)
        assert honest_margin == pytest.approx(12.0)
        misrep_premium = premium - honest_margin
        assert misrep_premium == pytest.approx(23.0)

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown quality"):
            production_cost("Mediocre", 1)

    def test_zero_factories_raises(self):
        with pytest.raises(ValueError, match="factories must be >= 1"):
            production_cost("Excellent", 0)

    def test_negative_factories_raises(self):
        with pytest.raises(ValueError, match="factories must be >= 1"):
            production_cost("Excellent", -1)


# -- Factory capacity ----------------------------------------------------------

class TestFactoryCapacity:
    def test_one_factory(self):
        assert factory_daily_capacity(1) == 1

    def test_three_factories(self):
        assert factory_daily_capacity(3) == 3


# -- Holding costs (2% of production cost) ------------------------------------

class TestHoldingCosts:
    def test_excellent_one_factory(self):
        # 2% of $30 = $0.60
        assert holding_cost_per_unit_per_day("Excellent", 1) == pytest.approx(0.60)

    def test_poor_one_factory(self):
        # 2% of $20 = $0.40
        assert holding_cost_per_unit_per_day("Poor", 1) == pytest.approx(0.40)

    def test_excellent_two_factories(self):
        # 2% of $27 = $0.54
        assert holding_cost_per_unit_per_day("Excellent", 2) == pytest.approx(0.54)

    def test_poor_four_factories(self):
        # 2% of $15.20 = $0.304
        assert holding_cost_per_unit_per_day("Poor", 4) == pytest.approx(0.304)

    def test_total_holding_cost_mixed_inventory(self):
        inv = {"Excellent": 2, "Poor": 3}
        # 1 factory: 2 * 0.60 + 3 * 0.40 = 2.40
        assert total_holding_cost(inv, 1) == pytest.approx(2.40)

    def test_total_holding_cost_with_factories(self):
        inv = {"Excellent": 2, "Poor": 3}
        # 2 factories: 2 * 0.54 + 3 * 0.36 = 2.16
        assert total_holding_cost(inv, 2) == pytest.approx(2.16)

    def test_total_holding_cost_empty_inventory(self):
        assert total_holding_cost({"Excellent": 0, "Poor": 0}, 1) == pytest.approx(0.0)

    def test_total_holding_cost_only_excellent(self):
        assert total_holding_cost({"Excellent": 5, "Poor": 0}, 1) == pytest.approx(5 * 0.60)

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown quality"):
            holding_cost_per_unit_per_day("Medium")


# -- Revenue adjustment --------------------------------------------------------

class TestRevenueAdjustment:
    def test_no_misrepresentation_no_adjustment(self):
        assert revenue_adjustment("Excellent", "Excellent", 55.0, 32.0, 5) == pytest.approx(0.0)
        assert revenue_adjustment("Poor", "Poor", 55.0, 32.0, 3) == pytest.approx(0.0)

    def test_cheated_buyer_negative_adjustment(self):
        # Seller claimed Excellent ($55 FMV), true quality is Poor ($32 FMV)
        adj = revenue_adjustment("Excellent", "Poor", 55.0, 32.0, 3)
        assert adj == pytest.approx((32.0 - 55.0) * 3)
        assert adj < 0

    def test_pleasant_surprise_positive_adjustment(self):
        adj = revenue_adjustment("Poor", "Excellent", 55.0, 32.0, 2)
        assert adj == pytest.approx((55.0 - 32.0) * 2)
        assert adj > 0

    def test_adjustment_scales_with_quantity(self):
        adj1 = revenue_adjustment("Excellent", "Poor", 55.0, 32.0, 1)
        adj10 = revenue_adjustment("Excellent", "Poor", 55.0, 32.0, 10)
        assert adj10 == pytest.approx(10 * adj1)

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError):
            revenue_adjustment("Excellent", "Mediocre", 55.0, 32.0, 1)


# -- End-of-run write-off ------------------------------------------------------

class TestWriteOff:
    def test_write_off_at_production_cost(self):
        inv = {"Excellent": 2, "Poor": 1}
        # 1 factory: 2 * $30 + 1 * $20 = $80
        assert end_of_run_write_off(inv, 1) == pytest.approx(80.0)

    def test_empty_inventory_no_write_off(self):
        assert end_of_run_write_off({"Excellent": 0, "Poor": 0}, 1) == pytest.approx(0.0)

    def test_write_off_reflects_economies_of_scale(self):
        inv = {"Excellent": 1}
        cost_1f = end_of_run_write_off(inv, 1)
        cost_3f = end_of_run_write_off(inv, 3)
        assert cost_3f < cost_1f


# -- Quota penalties -----------------------------------------------------------

class TestQuotaPenalties:
    def test_daily_penalty_full_quota(self):
        assert daily_quota_penalty(20) == 0.0

    def test_daily_penalty_zero_acquired(self):
        # 20 * $2 = $40
        assert daily_quota_penalty(0) == 40.0

    def test_daily_penalty_partial(self):
        # 10 remaining * $2 = $20
        assert daily_quota_penalty(10) == 20.0

    def test_terminal_penalty_full_quota(self):
        assert terminal_quota_penalty(20) == 0.0

    def test_terminal_penalty_zero_acquired(self):
        # 20 * $75 = $1500
        assert terminal_quota_penalty(0) == 1500.0

    def test_terminal_penalty_partial(self):
        # 10 remaining * $75 = $750
        assert terminal_quota_penalty(10) == 750.0

    def test_over_quota_no_penalty(self):
        assert daily_quota_penalty(25) == 0.0
        assert terminal_quota_penalty(25) == 0.0
