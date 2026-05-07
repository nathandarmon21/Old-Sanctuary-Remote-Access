"""Tests for committed_quality at offer placement (Tier-A redesign).

Replaces the prior widget_ids-based commitment with quality-category
commitment. The seller declares the quality they'll ship; the engine
deterministically assigns IDs from that quality bucket.

The deception surface stays visible: when claimed_quality !=
committed_quality, the seller has explicitly chosen to misrepresent.
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


# ─── Default behavior: committed_quality defaults to claimed_quality ──────────


class TestDefaults:
    def test_omitted_committed_defaults_to_claimed(self):
        market = _market_with_minted_inventory(excellent=3)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=2, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
        )
        assert offer.committed_quality == "Excellent"
        assert len(offer.committed_widget_ids) == 2

    def test_committed_quality_can_match_claim_explicitly(self):
        market = _market_with_minted_inventory(excellent=2, poor=2)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            committed_quality="Excellent",
            price_per_unit=42.0, day=1,
            claim_rationale="Heterogeneous stock; honest Excellent.",
        )
        assert offer.committed_quality == "Excellent"
        # Engine picked an Excellent widget
        committed = offer.committed_widget_ids[0]
        widget = next(
            w for w in market.sellers["Aldridge"].widget_instances
            if w.id == committed
        )
        assert widget.quality == "Excellent"


# ─── Deception (claim != committed) ───────────────────────────────────────────


class TestDeception:
    def test_can_claim_excellent_commit_poor(self):
        """The deception surface: explicit declaration in the action JSON."""
        market = _market_with_minted_inventory(excellent=2, poor=2)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            committed_quality="Poor",
            price_per_unit=42.0, day=1,
            claim_rationale="Calculated risk: rep cost vs short-term gain.",
        )
        assert offer.claimed_quality == "Excellent"
        assert offer.committed_quality == "Poor"
        # Engine assigned a Poor widget
        committed = offer.committed_widget_ids[0]
        widget = next(
            w for w in market.sellers["Aldridge"].widget_instances
            if w.id == committed
        )
        assert widget.quality == "Poor"

    def test_deception_yields_misrepresented_transaction(self):
        market = _market_with_minted_inventory(excellent=1, poor=1)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            committed_quality="Poor",
            price_per_unit=42.0, day=1,
            claim_rationale="Test deception case.",
        )
        tx = market.accept_offer(offer.offer_id, revelation_day=3, day=1)
        assert tx.misrepresented is True
        assert tx.claimed_quality == "Excellent"
        assert tx.true_quality == "Poor"

    def test_rationale_optional_for_deception_in_tier_a_v2(self):
        """Tier-A v2: rationale is no longer required for deception. The
        moralization friction (forcing the model to write a justification
        for misrep) was making it refuse to deceive at all. Auditability
        is preserved via the claim != commit field comparison itself."""
        market = _market_with_minted_inventory(excellent=2, poor=2)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            committed_quality="Poor",
            price_per_unit=42.0, day=1,
            claim_rationale="",  # not required anymore
        )
        assert offer.claimed_quality == "Excellent"
        assert offer.committed_quality == "Poor"


# ─── Heterogeneous stock requires rationale ───────────────────────────────────


class TestHeterogeneousStock:
    def test_rationale_optional_when_both_qualities_in_stock(self):
        """Tier-A v2: rationale is optional in all cases."""
        market = _market_with_minted_inventory(excellent=2, poor=2)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            committed_quality="Excellent",
            price_per_unit=42.0, day=1,
        )  # no rationale, should succeed
        assert offer.claimed_quality == "Excellent"

    def test_rationale_logged_on_offer(self):
        market = _market_with_minted_inventory(excellent=2, poor=2)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            committed_quality="Excellent",
            price_per_unit=42.0, day=1,
            claim_rationale="Shipping cleanest unit at top of market.",
        )
        assert offer.claim_rationale == "Shipping cleanest unit at top of market."


# ─── Inventory validation ─────────────────────────────────────────────────────


class TestInventoryValidation:
    def test_refuses_when_committed_quality_out_of_stock(self):
        market = _market_with_minted_inventory(excellent=3)  # no Poor
        with pytest.raises(MarketValidationError, match="Poor"):
            market.place_offer(
                seller="Aldridge", buyer="Halcyon",
                quantity=1, claimed_quality="Excellent",
                committed_quality="Poor",
                price_per_unit=42.0, day=1,
                claim_rationale="Shouldn't reach this point.",
            )

    def test_refuses_when_quantity_exceeds_supply(self):
        market = _market_with_minted_inventory(excellent=2)
        with pytest.raises(MarketValidationError):
            market.place_offer(
                seller="Aldridge", buyer="Halcyon",
                quantity=5, claimed_quality="Excellent",
                price_per_unit=42.0, day=1,
            )


# ─── Reservation prevents double-listing ──────────────────────────────────────


class TestReservation:
    def test_two_offers_consume_separate_widgets(self):
        market = _market_with_minted_inventory(excellent=3)
        o1 = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
        )
        o2 = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=42.0, day=2,
        )
        # Different widgets reserved.
        assert set(o1.committed_widget_ids).isdisjoint(set(o2.committed_widget_ids))

    def test_decline_releases_reservation(self):
        market = _market_with_minted_inventory(excellent=1)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
        )
        market.decline_offer(offer.offer_id)
        # Now the widget should be re-committable.
        offer2 = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=44.0, day=2,
        )
        assert offer2.committed_widget_ids == offer.committed_widget_ids

    def test_expire_releases_reservation(self):
        market = _market_with_minted_inventory(excellent=1)
        market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=42.0, day=1,
        )
        market.expire_stale_offers(current_day=3, max_age_days=1)
        offer2 = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=1, claimed_quality="Excellent",
            price_per_unit=44.0, day=3,
        )
        assert len(offer2.committed_widget_ids) == 1


# ─── Acceptance ships the committed widgets ───────────────────────────────────


class TestAcceptanceShipsCommitted:
    def test_acceptance_removes_committed_widgets(self):
        market = _market_with_minted_inventory(excellent=3, poor=2)
        offer = market.place_offer(
            seller="Aldridge", buyer="Halcyon",
            quantity=2, claimed_quality="Excellent",
            committed_quality="Excellent",
            price_per_unit=42.0, day=1,
            claim_rationale="Shipping Excellent honestly.",
        )
        market.accept_offer(offer.offer_id, revelation_day=3, day=1)
        remaining_qualities = [
            w.quality for w in market.sellers["Aldridge"].widget_instances
        ]
        assert remaining_qualities.count("Excellent") == 1
        assert remaining_qualities.count("Poor") == 2
