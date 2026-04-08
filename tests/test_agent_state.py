"""
Tests for MarketState.summary_for_agent — the authoritative ground-truth
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
    MarketState,
    PendingOffer,
    SellerState,
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
        # day=9, days_total=30 → 22 remaining
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
        assert "7 / 30" in summary

    def test_contains_widgets_still_needed(self):
        market = _make_market()
        # 30 - 7 = 23 still needed
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "23" in summary

    def test_contains_days_remaining(self):
        market = _make_market()
        # day=9, days_total=30 → 22 remaining
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
                true_quality=None,   # not yet revealed
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
                true_quality="Poor",   # misrepresented, revealed
                day_purchased=3,
            )
        )
        # Add a transaction record so revelation_day can be looked up
        from sanctuary.market import TransactionRecord
        market.transactions.append(TransactionRecord(
            transaction_id="tx-002",
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=2,
            claimed_quality="Excellent",
            true_quality="Poor",
            price_per_unit=50.0,
            day=3,
            revelation_day=7,
        ))
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "Poor" in summary
        assert "7" in summary          # revelation day
        assert "MISREPRESENTED" in summary

    def test_final_goods_production_totals(self):
        """Final goods count and revenue appear in buyer summary."""
        from sanctuary.market import FinalGoodsRecord
        market = _make_market()
        # Manually add production records
        market.buyers["Halcyon Assembly"].produced_goods_records.append(
            FinalGoodsRecord(
                record_id="rec-001",
                buyer="Halcyon Assembly",
                day=5,
                quantity=2,
                lot_id="lot-x",
                transaction_id="tx-x",
                assumed_quality="Excellent",
                fg_prices_at_production={"Excellent": 90.0, "Poor": 52.0},
                revenue_recorded=180.0,
                adjustment_applied=True,
            )
        )
        summary = market.summary_for_agent("Halcyon Assembly", day=9, days_total=30)
        assert "2" in summary       # quantity
        assert "180.00" in summary  # revenue

    def test_unknown_agent_raises(self):
        market = _make_market()
        with pytest.raises(KeyError, match="NoSuchAgent"):
            market.summary_for_agent("NoSuchAgent", day=1, days_total=30)

    def test_zero_widgets_shows_zero(self):
        market = _make_market()
        summary = market.summary_for_agent("Halcyon Assembly", day=1, days_total=30)
        assert "Widgets owned: 0" in summary
