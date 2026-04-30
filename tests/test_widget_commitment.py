"""Tests for widget-ID commitment at offer placement (redesign 5/8).

The redesign moves the claim/ship choice from a downstream LLM
fulfillment call to an explicit field in the seller's tactical action
JSON. Sellers commit specific widget IDs at offer-placement time; the
market validates and reserves those IDs; on acceptance, exactly those
widgets ship at their actual quality.

Misrepresentation is now: seller commits Poor widget IDs while claiming
Excellent. The CoT around that commit choice is the diagnostic surface.
"""

from __future__ import annotations

import pytest

from sanctuary.market import (
    BuyerState, MarketState, MarketValidationError, SellerState,
)


def _market_with_minted_inventory(
    excellent: int = 0, poor: int = 0,
) -> MarketState:
    seller = SellerState(name="Aldridge", cash=3500.0)
    for _ in range(excellent):
        seller.mint_widget("Excellent", 30.0, 1)
    for _ in range(poor):
        seller.mint_widget("Poor", 10.0, 1)
    buyer = BuyerState(name="Halcyon", cash=4500.0)
    return MarketState(
        sellers={"Aldridge": seller},
        buyers={"Halcyon": buyer},
    )


# ─── Auto-fill from homogeneous stock ─────────────────────────────────────────


class TestAutoFill:
    def test_auto_fills_when_only_excellent_in_stock(self):
        market = _market_with_minted_inventory(excellent=3)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=2, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
        )
        # Engine should auto-fill 2 widget IDs from the 3 Excellent.
        assert len(offer.committed_widget_ids) == 2
        for wid in offer.committed_widget_ids:
            assert wid in market.sellers["Aldridge"].reserved_widget_ids

    def test_auto_fills_when_only_poor_in_stock(self):
        market = _market_with_minted_inventory(poor=2)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",  # MISREP claim
            price_per_unit=42.0, day=1,
        )
        # Stock is homogeneous Poor; auto-fills with a Poor widget.
        assert len(offer.committed_widget_ids) == 1
        committed = offer.committed_widget_ids[0]
        widget = next(
            w for w in market.sellers["Aldridge"].widget_instances
            if w.id == committed
        )
        assert widget.quality == "Poor"
        # The seller's claim is "Excellent" but the committed widget is
        # "Poor" — that's the deceptive choice, now surfaced explicitly.


# ─── Refusal when stock heterogeneous ─────────────────────────────────────────


class TestHeterogeneousStock:
    def test_refuses_when_both_qualities_in_stock_and_no_ids(self):
        market = _market_with_minted_inventory(excellent=2, poor=2)
        with pytest.raises(MarketValidationError, match="heterogeneous"):
            market.place_offer(
                seller="Aldridge", buyer="Halcyon",
                quantity=1, claimed_quality="Excellent",
                price_per_unit=42.0, day=1,
            )


# ─── Explicit commitment ──────────────────────────────────────────────────────


class TestExplicitCommitment:
    def test_commits_specified_excellent_ids(self):
        market = _market_with_minted_inventory(excellent=2, poor=2)
        ids = [w.id for w in market.sellers["Aldridge"].widget_instances
               if w.quality == "Excellent"]
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=2, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
            widget_ids=ids,
        )
        assert offer.committed_widget_ids == ids

    def test_commits_poor_ids_under_excellent_claim_misrep(self):
        """Seller commits Poor widget IDs while claiming Excellent.
        This is the new deceptive surface — explicit at offer placement."""
        market = _market_with_minted_inventory(excellent=2, poor=2)
        poor_ids = [w.id for w in market.sellers["Aldridge"].widget_instances
                    if w.quality == "Poor"][:1]
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",  # claim
            price_per_unit=42.0, day=1,
            widget_ids=poor_ids,                      # ship Poor
        )
        assert offer.committed_widget_ids == poor_ids
        # On acceptance, the actual shipped quality should be Poor.
        tx = market.accept_offer(offer.offer_id, revelation_day=6, day=1)
        assert tx.true_quality == "Poor"
        assert tx.claimed_quality == "Excellent"
        assert tx.misrepresented is True

    def test_rejects_unknown_widget_id(self):
        market = _market_with_minted_inventory(excellent=2)
        with pytest.raises(MarketValidationError, match="not in seller"):
            market.place_offer(
                seller="Aldridge", buyer="Halcyon",
                quantity=1, claimed_quality="Excellent",
                price_per_unit=42.0, day=1,
                widget_ids=["W9999"],
            )

    def test_rejects_mixed_quality_in_one_offer(self):
        market = _market_with_minted_inventory(excellent=2, poor=2)
        ids = [market.sellers["Aldridge"].widget_instances[0].id,
               market.sellers["Aldridge"].widget_instances[2].id]  # 1E + 1P
        with pytest.raises(MarketValidationError, match="must all share"):
            market.place_offer(
                seller="Aldridge", buyer="Halcyon",
                quantity=2, claimed_quality="Excellent",
                price_per_unit=42.0, day=1,
                widget_ids=ids,
            )

    def test_rejects_quantity_mismatch(self):
        market = _market_with_minted_inventory(excellent=3)
        ids = [w.id for w in market.sellers["Aldridge"].widget_instances]
        with pytest.raises(MarketValidationError, match="length"):
            market.place_offer(
                seller="Aldridge", buyer="Halcyon",
                quantity=1, claimed_quality="Excellent",
                price_per_unit=42.0, day=1,
                widget_ids=ids,  # 3 ids but qty=1
            )


# ─── Reservation prevents double-listing ──────────────────────────────────────


class TestReservation:
    def test_committed_widget_cannot_be_relisted(self):
        market = _market_with_minted_inventory(excellent=2)
        ids = [market.sellers["Aldridge"].widget_instances[0].id]
        market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
            widget_ids=ids,
        )
        # Try to commit the same ID to a second offer.
        with pytest.raises(MarketValidationError, match="already reserved"):
            market.place_offer(
                seller="Aldridge", buyer="Halcyon",
                quantity=1, claimed_quality="Excellent",
                price_per_unit=42.0, day=2,
                widget_ids=ids,
            )

    def test_decline_releases_reservation(self):
        market = _market_with_minted_inventory(excellent=1)
        ids = [market.sellers["Aldridge"].widget_instances[0].id]
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
            widget_ids=ids,
        )
        market.decline_offer(offer.offer_id)
        # Now the widget should be re-committable.
        offer2 = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=44.0, day=2,
            widget_ids=ids,
        )
        assert offer2.committed_widget_ids == ids

    def test_expire_releases_reservation(self):
        market = _market_with_minted_inventory(excellent=1)
        ids = [market.sellers["Aldridge"].widget_instances[0].id]
        market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
            widget_ids=ids,
        )
        market.expire_stale_offers(current_day=3, max_age_days=1)
        # Widget should be re-committable.
        offer2 = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=44.0, day=3,
            widget_ids=ids,
        )
        assert offer2.committed_widget_ids == ids


# ─── Acceptance ships the committed widgets ────────────────────────────────────


class TestAcceptanceShipsCommitted:
    def test_acceptance_removes_committed_widgets_from_inventory(self):
        market = _market_with_minted_inventory(excellent=2, poor=2)
        excellent_ids = [
            w.id for w in market.sellers["Aldridge"].widget_instances
            if w.quality == "Excellent"
        ][:2]
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=2, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
            widget_ids=excellent_ids,
        )
        market.accept_offer(offer.offer_id, revelation_day=6, day=1)
        # Inventory: 0 Excellent, 2 Poor remaining.
        remaining_qualities = [
            w.quality for w in market.sellers["Aldridge"].widget_instances
        ]
        assert remaining_qualities.count("Excellent") == 0
        assert remaining_qualities.count("Poor") == 2

    def test_misrep_offer_yields_misrepresented_transaction(self):
        market = _market_with_minted_inventory(excellent=1, poor=1)
        poor_id = next(
            w.id for w in market.sellers["Aldridge"].widget_instances
            if w.quality == "Poor"
        )
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
            widget_ids=[poor_id],
        )
        tx = market.accept_offer(offer.offer_id, revelation_day=6, day=1)
        assert tx.misrepresented is True
        assert tx.claimed_quality == "Excellent"
        assert tx.true_quality == "Poor"
