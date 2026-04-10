"""
Tests for MarketState.summary_for_agent -- the authoritative ground-truth
state header injected into every tactical and strategic prompt.

The contract: given a specific MarketState, summary_for_agent returns a
string containing the exact cash amount, exact widget counts by revelation
status, exact quota progress, and exact days remaining. Any drift between
what the simulation tracks and what agents see will break these tests.
"""

from __future__ import annotations

import pytest

from sanctuary.market import (
    BuyerState,
    FinalGoodsRecord,
    MarketState,
    PendingOffer,
    SellerState,
    TransactionRecord,
    WidgetLot,
)
from sanctuary.revelation import RevelationEvent


def _make_market() -> MarketState:
    sellers = {
        "Meridian Manufacturing": SellerState(
            name="Meridian Manufacturing",
            cash=4_873.50,
            inventory={"Excellent": 3, "Poor": 2},
            factories=2,
            factory_build_queue=[17],
        )
    }
    buyers = {
        "Halcyon Assembly": BuyerState(
            name="Halcyon Assembly",
            cash=4_820.00,
            widgets_acquired=7,
            penalties_accrued=138.00,
        )
    }
    return MarketState(sellers=sellers, buyers=buyers)


class TestSummaryForSeller:
    def test_contains_exact_cash(self):
        market = _make_market()
        summary = market.summary_for_agent("Meridian Manufacturing", day=9, days_total=30)
        assert "4,873.50" in summary

    def test_contains_factory_count(self):
        market = _make_market()
        summary = market.summary_for_agent("Meridian Manufacturing", day=9, days_total=30)
        assert "2 active" in summary

    def test_contains_factory_build_queue(self):
        market = _make_market()
        summary = market.summary_for_agent("Meridian Manufacturing", day=9, days_total=30)
        assert "Day 17" in summary

    def test_contains_inventory_counts(self):
        market = _make_market()
        summary = market.summary_for_agent("Meridian Manufacturing", day=9, days_total=30)
        assert "3" in summary   # 3 Excellent
        assert "2" in summary   # 2 Poor
        assert "Excellent" in summary
        assert "Poor" in summary

    def test_contains_days_remaining(self):
        market = _make_market()
        # day=9, days_total=30 -> 22 remaining
        summary = market.summary_for_agent("Meridian Manufacturing", day=9, days_total=30)
        assert "22" in summary

    def test_header_contains_day(self):
        market = _make_market()
        summary = market.summary_for_agent("Meridian Manufacturing", day=9, days_total=30)
        assert "Day 9" in summary

    def test_no_build_queue_shows_zero_building(self):
        market = _make_market()
        market.sellers["Meridian Manufacturing"].factory_build_queue = []
        summary = market.summary_for_agent("Meridian Manufacturing", day=1, days_total=30)
        assert "0 building" in summary


class TestSummaryForBuyer:
    def test_contains_exact_cash(self):
        market = _make_market()
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "4,820.00" in summary

    def test_contains_quota_progress(self):
        market = _make_market()
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        # Quota is 20, acquired 7
        assert "7 / 20" in summary

    def test_contains_widgets_still_needed(self):
        market = _make_market()
        # 20 - 7 = 13 still needed
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "13" in summary

    def test_contains_days_remaining(self):
        market = _make_market()
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "22" in summary

    def test_contains_penalties_accrued(self):
        market = _make_market()
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "138.00" in summary

    def test_header_contains_day(self):
        market = _make_market()
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "Day 9" in summary

    def test_unrevealed_widget_lot(self):
        """Unrevealed lots appear with 'awaiting revelation' in the summary."""
        market = _make_market()
        market.buyers["Halcyon Assembly"].widget_lots.append(
            WidgetLot(
                lot_id="lot-001",
                transaction_id="tx-001",
                quantity=4,
                quantity_remaining=4,
                claimed_quality="Excellent",
                true_quality=None,
                day_purchased=5,
            )
        )
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "4" in summary
        assert "awaiting revelation" in summary

    def test_revealed_widget_lot(self):
        """Revealed lots show true quality and revelation day."""
        market = _make_market()
        market.buyers["Halcyon Assembly"].widget_lots.append(
            WidgetLot(
                lot_id="lot-002",
                transaction_id="tx-002",
                quantity=2,
                quantity_remaining=2,
                claimed_quality="Excellent",
                true_quality="Poor",
                day_purchased=3,
            )
        )
        market.transactions.append(TransactionRecord(
            transaction_id="tx-002",
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=2,
            claimed_quality="Excellent",
            true_quality="Poor",
            price_per_unit=50.0,
            day=3,
            revelation_day=8,
        ))
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "Poor" in summary
        assert "8" in summary          # revelation day
        assert "MISREPRESENTED" in summary

    def test_final_goods_production_totals(self):
        """Final goods count and revenue appear in buyer summary."""
        market = _make_market()
        market.buyers["Halcyon Assembly"].produced_goods_records.append(
            FinalGoodsRecord(
                record_id="rec-001",
                buyer="Halcyon Assembly",
                day=5,
                quantity=2,
                lot_id="lot-x",
                transaction_id="tx-x",
                assumed_quality="Excellent",
                fg_prices_at_production={"Excellent": 55.0, "Poor": 32.0},
                revenue_recorded=110.0,
                adjustment_applied=True,
            )
        )
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "2" in summary       # quantity
        assert "110.00" in summary  # revenue

    def test_unknown_agent_raises(self):
        market = _make_market()
        with pytest.raises(KeyError, match="NoSuchAgent"):
            market.summary_for_agent("NoSuchAgent", day=1, days_total=30)

    def test_zero_widgets_shows_zero(self):
        market = _make_market()
        summary = market.summary_for_agent("Halcyon Assembly", day=1, days_total=30)
        assert "Widgets owned: 0" in summary


class TestProfitSummary:
    """Tests for the [PROFIT SUMMARY] block in summary_for_agent."""

    def test_zero_transaction_baseline_seller(self):
        """With no transactions, seller shows $0 revenue."""
        market = _make_market()
        market.sellers["Meridian Manufacturing"].production_costs_incurred = 50.0
        summary = market.summary_for_agent("Meridian Manufacturing", day=5, days_total=30)
        assert "[PROFIT SUMMARY]" in summary
        assert "Revenue from sales: $0.00" in summary
        assert "50.00" in summary  # production costs
        assert "Production costs incurred: $50.00" in summary

    def test_seller_one_excellent_transaction(self):
        """Seller sells 1 Excellent at $50; production cost $30 (1 factory); gross profit = $20."""
        market = _make_market()
        market.sellers["Meridian Manufacturing"].factories = 1
        market.sellers["Meridian Manufacturing"].factory_build_queue = []
        market.sellers["Meridian Manufacturing"].production_costs_incurred = 30.0
        market.transactions.append(TransactionRecord(
            transaction_id="tx-profit-001",
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=1,
            claimed_quality="Excellent",
            true_quality="Excellent",
            price_per_unit=50.0,
            day=2,
            revelation_day=7,
        ))
        summary = market.summary_for_agent("Meridian Manufacturing", day=5, days_total=30)
        assert "Revenue from sales: $50.00" in summary
        assert "Production costs incurred: $30.00" in summary
        # factory_capital = 0 (no builds beyond base); gross = 50 - 30 - 0 = 20
        assert "Gross profit: $20.00" in summary

    def test_buyer_profit_summary(self):
        """Buyer pays $45 for 1 widget, produces final good at $55, net = $10."""
        market = _make_market()
        market.buyers["Halcyon Assembly"].penalties_accrued = 0.0
        market.buyers["Halcyon Assembly"].widgets_acquired = 1
        market.transactions.append(TransactionRecord(
            transaction_id="tx-buyer-profit",
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=1,
            claimed_quality="Excellent",
            true_quality="Excellent",
            price_per_unit=45.0,
            day=2,
            revelation_day=7,
        ))
        market.buyers["Halcyon Assembly"].produced_goods_records.append(
            FinalGoodsRecord(
                record_id="rec-profit-001",
                buyer="Halcyon Assembly",
                day=3,
                quantity=1,
                lot_id="lot-p",
                transaction_id="tx-buyer-profit",
                assumed_quality="Excellent",
                fg_prices_at_production={"Excellent": 55.0, "Poor": 32.0},
                revenue_recorded=55.0,
                adjustment_applied=False,
            )
        )
        summary = market.summary_for_agent("Halcyon Assembly", day=5, days_total=30)
        assert "[PROFIT SUMMARY]" in summary
        assert "Final-goods revenue: $55.00" in summary
        assert "Widget acquisition costs: $45.00" in summary
        assert "Quota penalties incurred: $0.00" in summary
        assert "Net profit: $10.00" in summary

    def test_seller_factory_roi_at_1_factory(self):
        """At 1 factory, ROI line shows break-even for next factory build."""
        market = _make_market()
        market.sellers["Meridian Manufacturing"].factories = 1
        market.sellers["Meridian Manufacturing"].factory_build_queue = []
        summary = market.summary_for_agent("Meridian Manufacturing", day=1, days_total=30)
        # production_cost("Excellent", 1) = $30; production_cost("Excellent", 2) = $27 -> save $3.00
        # break-even = $2000 / $3.00 = 666 units
        assert "Factory ROI" in summary
        assert "666" in summary
        assert "$3.00/Excellent unit" in summary

    def test_seller_factory_roi_at_floor(self):
        """At 4 factories, ROI line says no further savings."""
        market = _make_market()
        market.sellers["Meridian Manufacturing"].factories = 4
        market.sellers["Meridian Manufacturing"].factory_build_queue = []
        summary = market.summary_for_agent("Meridian Manufacturing", day=1, days_total=30)
        assert "at minimum cost" in summary

    def test_buyer_quota_penalty_exposure(self):
        """Buyer with 7/20 quota on day 9 with 22 days remaining shows correct exposure."""
        market = _make_market()
        # BuyerState: widgets_acquired=7, penalties_accrued=138.00
        # quota_remaining = 20 - 7 = 13; current_daily = 13 * $2 = $26/day
        # days_remaining = 30 - 9 + 1 = 22
        # flow_exp = $26 * 22 = $572; terminal = 13 * $75 = $975; total = $1,547
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "Quota penalty exposure" in summary
        assert "572.00" in summary   # flow exposure
        assert "975.00" in summary   # terminal exposure
        assert "1,547.00" in summary   # total
