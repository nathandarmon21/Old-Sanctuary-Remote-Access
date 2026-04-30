"""Tests for sanctuary/living_ledger.py — the structured per-agent scratchpad
injected into every tactical prompt to prevent inventory confabulation."""

from __future__ import annotations

import pytest

from sanctuary.living_ledger import build_living_ledger
from sanctuary.market import MarketState, build_initial_market


def _make_market(seller_cash: float = 3500.0, buyer_cash: float = 4500.0) -> MarketState:
    config = {
        "economics": {
            "seller_starting_cash": [seller_cash],
            "seller_starting_factories": 1,
            "buyer_starting_cash": buyer_cash,
            "starting_widgets_per_seller": 0,
        },
        "agents": {
            "sellers": [{"name": "Aldridge"}],
            "buyers": [{"name": "Halcyon"}],
        },
    }
    return build_initial_market(config)


def _evt(**kwargs) -> dict:
    return kwargs


# ─── Seller ledger ────────────────────────────────────────────────────────────


class TestSellerLedger:
    def test_empty_market_renders(self):
        market = _make_market()
        text = build_living_ledger(
            "Aldridge", market, day=1, daily_events={},
        )
        assert "LIVING LEDGER" in text
        assert "CASH TRAJECTORY" in text
        assert "$3,500.00" in text
        assert "CURRENT INVENTORY" in text
        assert "(empty)" in text  # no widgets yet
        assert "(no production events" in text or "no offers" in text

    def test_inventory_lists_widget_ids_by_quality(self):
        market = _make_market()
        s = market.sellers["Aldridge"]
        s.mint_widget("Excellent", 30.0, 1)
        s.mint_widget("Excellent", 30.0, 1)
        s.mint_widget("Poor", 10.0, 2)

        text = build_living_ledger(
            "Aldridge", market, day=5, daily_events={},
        )
        # Widget IDs should appear in the ledger
        assert "W0001" in text
        assert "W0002" in text
        assert "W0003" in text
        assert "Excellent: W0001, W0002" in text
        assert "Poor: W0003" in text

    def test_production_history_shows_defects(self):
        market = _make_market()
        events = {
            5: [
                _evt(
                    event_type="production",
                    agent_id="Aldridge",
                    intended_quality="Excellent",
                    quantity=3,
                    actual_excellent=2,
                    actual_poor=1,
                ),
            ],
        }
        text = build_living_ledger("Aldridge", market, day=10, daily_events=events)
        assert "Day 5: produced 3 Excellent" in text
        assert "(defect)" in text
        assert "2 Excellent + 1 Poor" in text

    def test_offers_table_shows_status_and_reveal(self):
        market = _make_market()
        events = {
            10: [
                _evt(
                    event_type="transaction_proposed",
                    seller="Aldridge", buyer="Halcyon",
                    offer_id="OID-abc12345",
                    claimed_quality="Excellent", quantity=2,
                    price_per_unit=42.0, day=10,
                    widget_ids=["W0001", "W0002"],
                ),
            ],
            11: [
                _evt(
                    event_type="transaction_completed",
                    offer_id="OID-abc12345",
                    seller="Aldridge", buyer="Halcyon",
                    transaction_id="OID-abc12345",
                ),
            ],
            16: [
                _evt(
                    event_type="quality_revealed",
                    transaction_id="OID-abc12345",
                    seller="Aldridge", buyer="Halcyon",
                    claimed_quality="Excellent", true_quality="Poor",
                    misrepresented=True,
                ),
            ],
        }
        text = build_living_ledger("Aldridge", market, day=20, daily_events=events)
        assert "Day 10: offer_OID-abc1" in text or "offer_OID-abc1" in text
        assert "Halcyon" in text
        assert "ACCEPTED" in text
        assert "you misrepresented" in text

    def test_cash_trajectory_shows_runway(self):
        market = _make_market(seller_cash=400.0)
        text = build_living_ledger("Aldridge", market, day=10, daily_events={})
        # $400 / $80 burn = 5 days runway
        assert "Days to bankruptcy" in text
        assert "5" in text or "4" in text  # rounding tolerance


# ─── Buyer ledger ─────────────────────────────────────────────────────────────


class TestBuyerLedger:
    def test_empty_market_renders(self):
        market = _make_market()
        text = build_living_ledger("Halcyon", market, day=1, daily_events={})
        assert "LIVING LEDGER" in text
        assert "$4,500.00" in text
        assert "(empty)" in text

    def test_purchases_table_with_misrep_callout(self):
        market = _make_market()
        events = {
            12: [
                _evt(
                    event_type="transaction_completed",
                    transaction_id="T1",
                    seller="Vector", buyer="Halcyon",
                    claimed_quality="Excellent", quantity=1,
                    price_per_unit=44.0, day=12,
                ),
            ],
            18: [
                _evt(
                    event_type="quality_revealed",
                    transaction_id="T1",
                    seller="Vector", buyer="Halcyon",
                    claimed_quality="Excellent", true_quality="Poor",
                    misrepresented=True,
                ),
            ],
        }
        text = build_living_ledger("Halcyon", market, day=25, daily_events=events)
        assert "Day 12: bought 1x from Vector" in text
        assert "Vector misrepresented" in text


# ─── Unknown agent ────────────────────────────────────────────────────────────


def test_unknown_agent_returns_empty():
    market = _make_market()
    assert build_living_ledger("ghost", market, day=5, daily_events={}) == ""
