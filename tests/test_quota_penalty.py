"""
Tests for the quota-penalty buyer model.

Covers:
  - daily_quota_penalty and terminal_quota_penalty math
  - MarketState.apply_buyer_quota_penalties deducts correct amounts
  - MarketState.apply_terminal_quota_penalties fires at end of run
  - widgets_acquired increments correctly on accept_offer
  - Buyer who meets quota pays no penalty
"""

from __future__ import annotations

import pytest

from sanctuary.economics import (
    BUYER_DAILY_QUOTA_PENALTY,
    BUYER_TERMINAL_QUOTA_PENALTY,
    BUYER_WIDGET_QUOTA,
    daily_quota_penalty,
    terminal_quota_penalty,
)
from sanctuary.market import BuyerState, SellerState, MarketState


# -- economics.py function tests -----------------------------------------------

class TestDailyQuotaPenalty:
    def test_zero_acquired_pays_full_penalty(self):
        # 20 unfulfilled x $2 = $40/day
        assert daily_quota_penalty(0) == pytest.approx(40.0)

    def test_quota_met_pays_nothing(self):
        assert daily_quota_penalty(20) == pytest.approx(0.0)

    def test_quota_exceeded_pays_nothing(self):
        assert daily_quota_penalty(25) == pytest.approx(0.0)

    def test_ten_widgets(self):
        # 10 unfulfilled x $2 = $20/day
        assert daily_quota_penalty(10) == pytest.approx(20.0)

    def test_nineteen_acquired(self):
        # 1 unfulfilled x $2 = $2/day
        assert daily_quota_penalty(19) == pytest.approx(2.0)


class TestTerminalQuotaPenalty:
    def test_zero_acquired(self):
        # 20 unfulfilled x $75 = $1,500
        assert terminal_quota_penalty(0) == pytest.approx(1500.0)

    def test_quota_met_no_penalty(self):
        assert terminal_quota_penalty(20) == pytest.approx(0.0)

    def test_ten_acquired(self):
        # 10 unfulfilled x $75 = $750
        assert terminal_quota_penalty(10) == pytest.approx(750.0)

    def test_worst_case_math(self):
        # Total worst case: 30 days x $40/day + $1,500 terminal = $2,700
        total = sum(daily_quota_penalty(0) for _ in range(30)) + terminal_quota_penalty(0)
        assert total == pytest.approx(2700.0)


# -- market.py quota mechanics -------------------------------------------------

def _make_buyer_state(name: str, cash: float = 6000.0, acquired: int = 0) -> BuyerState:
    return BuyerState(name=name, cash=cash, widgets_acquired=acquired)


def _simple_market(buyer_acquired: int = 0) -> MarketState:
    buyers = {"Halcyon Assembly": _make_buyer_state("Halcyon Assembly", acquired=buyer_acquired)}
    sellers = {
        "Meridian Manufacturing": SellerState(
            name="Meridian Manufacturing", cash=5000.0,
            inventory={"Excellent": 5, "Poor": 5}, factories=1,
        )
    }
    return MarketState(sellers=sellers, buyers=buyers)


class TestApplyBuyerQuotaPenalties:
    def test_idle_buyer_charged_daily_penalty(self):
        market = _simple_market(buyer_acquired=0)
        initial_cash = market.buyers["Halcyon Assembly"].cash
        penalties = market.apply_buyer_quota_penalties()
        expected = daily_quota_penalty(0)  # $40/day for 0 of 20 widgets
        assert penalties["Halcyon Assembly"] == pytest.approx(expected)
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash - expected)

    def test_partial_quota_charged_proportionally(self):
        market = _simple_market(buyer_acquired=10)
        initial_cash = market.buyers["Halcyon Assembly"].cash
        market.apply_buyer_quota_penalties()
        expected = daily_quota_penalty(10)  # $20/day
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash - expected)

    def test_full_quota_no_penalty(self):
        market = _simple_market(buyer_acquired=20)
        initial_cash = market.buyers["Halcyon Assembly"].cash
        penalties = market.apply_buyer_quota_penalties()
        assert penalties["Halcyon Assembly"] == pytest.approx(0.0)
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash)

    def test_bankrupt_buyer_skipped(self):
        market = _simple_market(buyer_acquired=0)
        market.buyers["Halcyon Assembly"].bankrupt = True
        initial_cash = market.buyers["Halcyon Assembly"].cash
        market.apply_buyer_quota_penalties()
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash)


class TestApplyTerminalQuotaPenalties:
    def test_terminal_penalty_fires_at_zero_acquired(self):
        market = _simple_market(buyer_acquired=0)
        initial_cash = market.buyers["Halcyon Assembly"].cash
        penalties = market.apply_terminal_quota_penalties()
        expected = terminal_quota_penalty(0)  # $1,500
        assert penalties["Halcyon Assembly"] == pytest.approx(expected)
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash - expected)

    def test_terminal_no_penalty_when_quota_met(self):
        market = _simple_market(buyer_acquired=20)
        initial_cash = market.buyers["Halcyon Assembly"].cash
        penalties = market.apply_terminal_quota_penalties()
        assert penalties["Halcyon Assembly"] == pytest.approx(0.0)
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash)

    def test_terminal_partial_penalty(self):
        market = _simple_market(buyer_acquired=15)
        initial_cash = market.buyers["Halcyon Assembly"].cash
        market.apply_terminal_quota_penalties()
        expected = terminal_quota_penalty(15)  # 5 unfulfilled x $75 = $375
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash - expected)


class TestWidgetsAcquiredTracking:
    def test_accept_offer_increments_widgets_acquired(self):
        """Accepting an offer must increment widgets_acquired on the buyer."""
        market = _simple_market(buyer_acquired=0)

        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=3,
            claimed_quality="Excellent",
            quality_to_send="Excellent",
            price_per_unit=50.0,
            day=1,
        )
        market.accept_offer(offer.offer_id, revelation_day=6, day=1)

        assert market.buyers["Halcyon Assembly"].widgets_acquired == 3

    def test_multiple_transactions_accumulate(self):
        """widgets_acquired sums across multiple accepted offers."""
        market = _simple_market(buyer_acquired=0)

        for qty in [2, 3]:
            offer = market.place_offer(
                seller="Meridian Manufacturing",
                buyer="Halcyon Assembly",
                quantity=qty,
                claimed_quality="Poor",
                quality_to_send="Poor",
                price_per_unit=20.0,
                day=1,
            )
            market.accept_offer(offer.offer_id, revelation_day=6, day=1)

        assert market.buyers["Halcyon Assembly"].widgets_acquired == 5
