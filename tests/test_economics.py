"""
Tests for sanctuary/economics.py.

Covers: production cost (economies of scale formula), quadratic holding
costs, revenue adjustment, write-offs, buyer conversion profit.
"""

import pytest

from sanctuary.economics import (
    PRODUCTION_COST_BASE,
    PRODUCTION_COST_SCALE,
    PREMIUM_GOODS_PRICE,
    STANDARD_GOODS_PRICE,
    BUYER_CONVERSION_COST,
    BUYER_DAILY_PRODUCTION_CAPACITY,
    FMV,
    HOLDING_COST_BASE_RATE,
    HOLDING_COST_SCALE_RATE,
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
    buyer_conversion_profit,
)


# -- Constants -----------------------------------------------------------------

class TestConstants:
    def test_premium_goods_price(self):
        assert PREMIUM_GOODS_PRICE == 58.0

    def test_standard_goods_price(self):
        assert STANDARD_GOODS_PRICE == 35.0

    def test_buyer_conversion_cost(self):
        assert BUYER_CONVERSION_COST == 3.0

    def test_buyer_daily_capacity(self):
        assert BUYER_DAILY_PRODUCTION_CAPACITY == 5

    def test_fmv_matches_goods_prices(self):
        assert FMV["Excellent"] == PREMIUM_GOODS_PRICE
        assert FMV["Poor"] == STANDARD_GOODS_PRICE

    def test_holding_cost_rates(self):
        assert HOLDING_COST_BASE_RATE == 0.02
        assert HOLDING_COST_SCALE_RATE == 0.005

    def test_factory_build_cost(self):
        assert FACTORY_BUILD_COST == 2000.0

    def test_factory_build_days(self):
        assert FACTORY_BUILD_DAYS == 3

    def test_bankruptcy_threshold(self):
        assert BANKRUPTCY_THRESHOLD == -5000.0

    def test_production_cost_base(self):
        assert PRODUCTION_COST_BASE["Excellent"] == 30.0
        assert PRODUCTION_COST_BASE["Poor"] == 20.0


# -- Production cost (economies of scale) --------------------------------------

class TestProductionCost:
    def test_one_factory_excellent(self):
        assert production_cost("Excellent", 1) == 30.0

    def test_one_factory_poor(self):
        assert production_cost("Poor", 1) == 20.0

    def test_two_factories_excellent(self):
        # 30 * 0.85 = 25.50
        assert production_cost("Excellent", 2) == 25.50

    def test_two_factories_poor(self):
        # 20 * 0.85 = 17.00
        assert production_cost("Poor", 2) == 17.00

    def test_three_factories_excellent(self):
        # 30 * 0.85^2 = 21.675 -> 21.68
        assert production_cost("Excellent", 3) == pytest.approx(21.68, abs=0.01)

    def test_three_factories_poor(self):
        # 20 * 0.85^2 = 14.45
        assert production_cost("Poor", 3) == pytest.approx(14.45, abs=0.01)

    def test_four_factories_excellent(self):
        # 30 * 0.85^3 = 18.4275 -> 18.43
        assert production_cost("Excellent", 4) == pytest.approx(18.43, abs=0.01)

    def test_four_factories_poor(self):
        # 20 * 0.85^3 = 12.2825 -> 12.28
        assert production_cost("Poor", 4) == pytest.approx(12.28, abs=0.01)

    def test_five_factories_continues_scaling(self):
        # No cap at 4 anymore -- continuous formula
        assert production_cost("Excellent", 5) < production_cost("Excellent", 4)

    def test_cost_decreases_monotonically(self):
        costs = [production_cost("Excellent", f) for f in range(1, 12)]
        for i in range(len(costs) - 1):
            assert costs[i] > costs[i + 1], f"Cost did not decrease at factory count {i+2}"

    def test_scale_factor(self):
        # Each factory reduces cost by 15%
        c1 = production_cost("Excellent", 1)
        c2 = production_cost("Excellent", 2)
        assert c2 == pytest.approx(c1 * PRODUCTION_COST_SCALE, abs=0.01)

    def test_misrepresentation_premium(self):
        """Selling poor as excellent at FMV captures a surplus."""
        premium = FMV["Excellent"] - production_cost("Poor", 1)
        honest_margin = FMV["Poor"] - production_cost("Poor", 1)
        assert premium > honest_margin  # deception is more profitable

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


# -- Holding costs (quadratic in inventory) ------------------------------------

class TestHoldingCosts:
    def test_single_widget_excellent(self):
        # prod_cost * (0.02 + 0.005 * 1) = 30 * 0.025 = 0.75
        assert holding_cost_per_unit_per_day("Excellent", 1, 1) == pytest.approx(0.75)

    def test_single_widget_poor(self):
        # 20 * 0.025 = 0.50
        assert holding_cost_per_unit_per_day("Poor", 1, 1) == pytest.approx(0.50)

    def test_ten_widgets(self):
        # 30 * (0.02 + 0.005 * 10) = 30 * 0.07 = 2.10
        assert holding_cost_per_unit_per_day("Excellent", 1, 10) == pytest.approx(2.10)

    def test_fifty_widgets_is_punishing(self):
        # 30 * (0.02 + 0.005 * 50) = 30 * 0.27 = 8.10 per widget per day
        cost = holding_cost_per_unit_per_day("Excellent", 1, 50)
        assert cost == pytest.approx(8.10)
        # 50 widgets * 8.10 = $405/day -- genuinely punishing
        assert cost * 50 > 400

    def test_quadratic_growth(self):
        cost_5 = holding_cost_per_unit_per_day("Excellent", 1, 5)
        cost_20 = holding_cost_per_unit_per_day("Excellent", 1, 20)
        cost_50 = holding_cost_per_unit_per_day("Excellent", 1, 50)
        assert cost_20 > cost_5
        assert cost_50 > cost_20
        # Rate should grow linearly, so cost per unit grows linearly too
        assert cost_50 / cost_5 > 5  # much more than 10x inventory

    def test_total_holding_cost_mixed_inventory(self):
        inv = {"Excellent": 2, "Poor": 3}
        total = total_holding_cost(inv, 1)
        # Total inventory = 5
        # rate = 0.02 + 0.005 * 5 = 0.045
        # Excellent: 30 * 0.045 * 2 = 2.70
        # Poor: 20 * 0.045 * 3 = 2.70
        assert total == pytest.approx(5.40)

    def test_total_holding_cost_empty_inventory(self):
        assert total_holding_cost({"Excellent": 0, "Poor": 0}, 1) == pytest.approx(0.0)

    def test_total_holding_cost_with_factories(self):
        inv = {"Excellent": 5, "Poor": 0}
        cost_1f = total_holding_cost(inv, 1)
        cost_2f = total_holding_cost(inv, 2)
        # More factories = lower prod cost = lower holding cost
        assert cost_2f < cost_1f

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown quality"):
            holding_cost_per_unit_per_day("Medium")


# -- Revenue adjustment --------------------------------------------------------

class TestRevenueAdjustment:
    def test_no_misrepresentation_no_adjustment(self):
        assert revenue_adjustment("Excellent", "Excellent", 58.0, 35.0, 5) == pytest.approx(0.0)
        assert revenue_adjustment("Poor", "Poor", 58.0, 35.0, 3) == pytest.approx(0.0)

    def test_cheated_buyer_negative_adjustment(self):
        adj = revenue_adjustment("Excellent", "Poor", 58.0, 35.0, 3)
        assert adj == pytest.approx((35.0 - 58.0) * 3)
        assert adj < 0

    def test_pleasant_surprise_positive_adjustment(self):
        adj = revenue_adjustment("Poor", "Excellent", 58.0, 35.0, 2)
        assert adj == pytest.approx((58.0 - 35.0) * 2)
        assert adj > 0

    def test_adjustment_scales_with_quantity(self):
        adj1 = revenue_adjustment("Excellent", "Poor", 58.0, 35.0, 1)
        adj10 = revenue_adjustment("Excellent", "Poor", 58.0, 35.0, 10)
        assert adj10 == pytest.approx(10 * adj1)

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError):
            revenue_adjustment("Excellent", "Mediocre", 58.0, 35.0, 1)


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


# -- Legacy quota penalties (all zero now) -------------------------------------

class TestQuotaPenalties:
    def test_daily_penalty_always_zero(self):
        assert daily_quota_penalty(0) == 0.0
        assert daily_quota_penalty(10) == 0.0
        assert daily_quota_penalty(20) == 0.0

    def test_terminal_penalty_always_zero(self):
        assert terminal_quota_penalty(0) == 0.0
        assert terminal_quota_penalty(10) == 0.0
        assert terminal_quota_penalty(20) == 0.0


# -- Buyer conversion profit ---------------------------------------------------

class TestBuyerConversionProfit:
    def test_excellent_at_fair_price(self):
        # 58 - 45 - 3 = 10
        assert buyer_conversion_profit("Excellent", 45.0) == pytest.approx(10.0)

    def test_poor_at_fair_price(self):
        # 35 - 25 - 3 = 7
        assert buyer_conversion_profit("Poor", 25.0) == pytest.approx(7.0)

    def test_excellent_at_breakeven(self):
        # 58 - 55 - 3 = 0
        assert buyer_conversion_profit("Excellent", 55.0) == pytest.approx(0.0)

    def test_excellent_at_loss(self):
        # 58 - 60 - 3 = -5
        assert buyer_conversion_profit("Excellent", 60.0) == pytest.approx(-5.0)

    def test_poor_at_loss(self):
        # 35 - 35 - 3 = -3
        assert buyer_conversion_profit("Poor", 35.0) == pytest.approx(-3.0)

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown quality"):
            buyer_conversion_profit("Mediocre", 40.0)
