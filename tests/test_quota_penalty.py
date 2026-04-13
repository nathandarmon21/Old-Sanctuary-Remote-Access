"""
Tests for buyer quota mechanics (legacy, now profit-driven model).

In the profit-driven redesign, quota penalties are disabled (always $0).
These tests verify the zero-penalty behavior and that widgets_acquired
tracking still works (used for reporting even without penalties).
"""

from __future__ import annotations

import pytest

from sanctuary.economics import (
    daily_quota_penalty,
    terminal_quota_penalty,
)
from sanctuary.market import BuyerState, SellerState, MarketState


# -- economics.py function tests (all zero now) --------------------------------

class TestDailyQuotaPenalty:
    def test_zero_acquired_no_penalty(self):
        assert daily_quota_penalty(0) == pytest.approx(0.0)

    def test_quota_met_no_penalty(self):
        assert daily_quota_penalty(20) == pytest.approx(0.0)

    def test_quota_exceeded_no_penalty(self):
        assert daily_quota_penalty(25) == pytest.approx(0.0)

    def test_partial_no_penalty(self):
        assert daily_quota_penalty(10) == pytest.approx(0.0)

    def test_nineteen_acquired(self):
        assert daily_quota_penalty(19) == pytest.approx(0.0)


class TestTerminalQuotaPenalty:
    def test_zero_acquired(self):
        assert terminal_quota_penalty(0) == pytest.approx(0.0)

    def test_quota_met_no_penalty(self):
        assert terminal_quota_penalty(20) == pytest.approx(0.0)

    def test_ten_acquired(self):
        assert terminal_quota_penalty(10) == pytest.approx(0.0)

    def test_worst_case_math(self):
        # With penalties disabled, worst case is $0
        total = sum(daily_quota_penalty(0) for _ in range(30)) + terminal_quota_penalty(0)
        assert total == pytest.approx(0.0)


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
    def test_idle_buyer_no_penalty(self):
        market = _simple_market(buyer_acquired=0)
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
    def test_terminal_no_penalty(self):
        market = _simple_market(buyer_acquired=0)
        initial_cash = market.buyers["Halcyon Assembly"].cash
        penalties = market.apply_terminal_quota_penalties()
        assert penalties["Halcyon Assembly"] == pytest.approx(0.0)
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash)


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
