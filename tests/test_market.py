"""
Tests for sanctuary/market.py.

CRITICAL: This file contains the inventory visibility invariant tests.
These must pass after any refactor of MarketState.

Covers:
  - Transaction validation (insufficient funds, inventory, self-trade)
  - Bankruptcy triggering
  - Inventory visibility (seller sees true quality; buyer sees claimed only)
  - Retroactive revenue adjustment on revelation
  - Production, factory build, factory completion
  - Buyer final goods production
  - End-of-run write-off
"""

import pytest
import numpy as np

from sanctuary.market import (
    BuyerState,
    MarketState,
    MarketValidationError,
    PendingOffer,
    SellerState,
    WidgetLot,
    build_initial_market,
)
from sanctuary.revelation import RevelationEvent


# ── Offer ID resolution ───────────────────────────────────────────────────────

class TestResolveOfferId:
    """
    Tests for MarketState.resolve_offer_id — the prefix-match fallback that
    allows agents who copy truncated UUIDs to still have accepts resolved.

    Four cases are required by the fix specification:
    (a) full UUID → exact match succeeds
    (b) prefix of a unique offer → prefix match succeeds
    (c) ambiguous prefix → returns (None, error) — transparent failure
    (d) non-existent ID → returns (None, error) — transparent failure
    """

    def _offer(self, offer_id: str, seller: str = "S", buyer: str = "B") -> PendingOffer:
        return PendingOffer(
            offer_id=offer_id,
            seller=seller,
            buyer=buyer,
            quantity=1,
            claimed_quality="Excellent",
            quality_to_send="Excellent",
            price_per_unit=50.0,
            day_made=1,
            status="pending",
        )

    def _market_with_offers(self, offer_ids: list[str]) -> MarketState:
        sellers = {"S": SellerState(name="S", cash=5000.0, inventory={"Excellent": 5, "Poor": 5})}
        buyers = {"B": BuyerState(name="B", cash=6000.0)}
        market = MarketState(sellers=sellers, buyers=buyers)
        for oid in offer_ids:
            market.pending_offers[oid] = self._offer(oid)
        return market

    def test_full_uuid_exact_match(self):
        """(a) Full UUID in accept_offers resolves to itself."""
        oid = "aabbccdd-1234-5678-abcd-ef0123456789"
        market = self._market_with_offers([oid])
        resolved, err = market.resolve_offer_id(oid)
        assert resolved == oid
        assert err is None

    def test_prefix_unique_match(self):
        """(b) Unique 8-char prefix resolves to the full UUID."""
        oid = "aabbccdd-1234-5678-abcd-ef0123456789"
        market = self._market_with_offers([oid])
        resolved, err = market.resolve_offer_id("aabbccdd")
        assert resolved == oid
        assert err is None

    def test_ambiguous_prefix_returns_none(self):
        """(c) Prefix matching two offers returns (None, error) — no silent success."""
        oid1 = "aabbccdd-1111-0000-0000-000000000001"
        oid2 = "aabbccdd-2222-0000-0000-000000000002"
        market = self._market_with_offers([oid1, oid2])
        resolved, err = market.resolve_offer_id("aabbccdd")
        assert resolved is None
        assert err is not None
        assert "ambiguous" in err.lower()

    def test_nonexistent_id_returns_none(self):
        """(d) ID or prefix matching nothing returns (None, error)."""
        oid = "aabbccdd-1234-5678-abcd-ef0123456789"
        market = self._market_with_offers([oid])
        resolved, err = market.resolve_offer_id("deadbeef")
        assert resolved is None
        assert err is not None

    def test_prefix_ignores_expired_offers(self):
        """Prefix match only considers pending offers, not expired/accepted ones."""
        oid = "aabbccdd-1234-5678-abcd-ef0123456789"
        market = self._market_with_offers([oid])
        market.pending_offers[oid].status = "expired"
        resolved, err = market.resolve_offer_id("aabbccdd")
        assert resolved is None  # expired offer not eligible


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_simple_market(
    seller_cash: float = 5_000.0,
    buyer_cash: float = 6_000.0,
    seller_excellent: int = 5,
    seller_poor: int = 5,
    seller_factories: int = 1,
) -> MarketState:
    """Minimal two-agent market for focused unit tests."""
    sellers = {
        "Meridian Manufacturing": SellerState(
            name="Meridian Manufacturing",
            cash=seller_cash,
            inventory={"Excellent": seller_excellent, "Poor": seller_poor},
            factories=seller_factories,
        )
    }
    buyers = {
        "Halcyon Assembly": BuyerState(
            name="Halcyon Assembly",
            cash=buyer_cash,
        )
    }
    return MarketState(sellers=sellers, buyers=buyers)


def make_full_market() -> MarketState:
    """Eight-agent market matching the spec roster."""
    sellers = {
        name: SellerState(
            name=name,
            cash=5_000.0,
            inventory={"Excellent": 3, "Poor": 3},
        )
        for name in [
            "Meridian Manufacturing",
            "Aldridge Industrial",
            "Crestline Components",
            "Vector Works",
        ]
    }
    buyers = {
        name: BuyerState(name=name, cash=6_000.0)
        for name in [
            "Halcyon Assembly",
            "Pinnacle Goods",
            "Coastal Fabrication",
            "Northgate Systems",
        ]
    }
    return MarketState(sellers=sellers, buyers=buyers)


# ── CRITICAL: Inventory visibility invariant ───────────────────────────────────

class TestInventoryVisibility:
    """
    The single most important invariant in the experiment:
    buyers must never see true quality until revelation fires.

    These tests must pass after every refactor.
    """

    def test_seller_sees_true_quality(self):
        market = make_simple_market(seller_excellent=3, seller_poor=2)
        view = market.view_inventory_for("Meridian Manufacturing")

        assert view["agent_type"] == "seller"
        assert view["true_quality_visible"] is True
        assert view["excellent"] == 3
        assert view["poor"] == 2

    def test_buyer_does_not_see_true_quality_before_revelation(self):
        """A buyer who purchased widgets sees claimed quality only."""
        market = make_simple_market()

        # Place and accept an offer with deception (send Poor, claim Excellent)
        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=2,
            claimed_quality="Excellent",
            quality_to_send="Poor",   # ← deception
            price_per_unit=50.0,
            day=1,
        )
        market.accept_offer(offer.offer_id, revelation_day=5, day=1)

        view = market.view_inventory_for("Halcyon Assembly")

        assert view["agent_type"] == "buyer"
        assert view["true_quality_visible"] is False

        # Buyer should see exactly one lot
        assert len(view["lots"]) == 1
        lot = view["lots"][0]

        # Buyer sees claimed quality (Excellent), NOT true quality (Poor)
        assert lot["claimed_quality"] == "Excellent"
        assert "true_quality" not in lot, (
            "true_quality must not appear in buyer view before revelation"
        )
        assert lot["revealed"] is False

    def test_buyer_sees_true_quality_after_revelation(self):
        """After apply_revelation(), buyer's lot view includes true_quality."""
        market = make_simple_market()

        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=2,
            claimed_quality="Excellent",
            quality_to_send="Poor",
            price_per_unit=50.0,
            day=1,
        )
        tx = market.accept_offer(offer.offer_id, revelation_day=5, day=1)

        event = RevelationEvent(
            transaction_id=tx.transaction_id,
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            claimed_quality="Excellent",
            true_quality="Poor",
            quantity=2,
            transaction_day=1,
            revelation_day=5,
        )
        market.apply_revelation(event)

        view = market.view_inventory_for("Halcyon Assembly")
        lot = view["lots"][0]

        assert lot["revealed"] is True
        assert lot["true_quality"] == "Poor"
        assert lot["claimed_quality"] == "Excellent"

    def test_buyer_lot_consumed_disappears_from_view(self):
        """Widgets consumed for final goods production are removed from buyer view."""
        market = make_simple_market()

        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=3,
            claimed_quality="Excellent",
            quality_to_send="Excellent",
            price_per_unit=50.0,
            day=1,
        )
        market.accept_offer(offer.offer_id, revelation_day=5, day=1)
        market.execute_buyer_production("Halcyon Assembly", 3, current_day=1)

        view = market.view_inventory_for("Halcyon Assembly")
        # All widgets consumed; no lots with remaining quantity
        assert len(view["lots"]) == 0

    def test_unknown_agent_raises_key_error(self):
        market = make_simple_market()
        with pytest.raises(KeyError):
            market.view_inventory_for("NoSuchAgent")


# ── Transaction validation ────────────────────────────────────────────────────

class TestTransactionValidation:
    def test_self_trade_rejected(self):
        """An agent cannot trade with itself."""
        market = make_simple_market()
        with pytest.raises(MarketValidationError, match="Self-trade"):
            market.place_offer(
                seller="Meridian Manufacturing",
                buyer="Meridian Manufacturing",  # same agent
                quantity=1,
                claimed_quality="Excellent",
                quality_to_send="Excellent",
                price_per_unit=50.0,
                day=1,
            )

    def test_insufficient_inventory_rejected(self):
        market = make_simple_market(seller_excellent=1)
        with pytest.raises(MarketValidationError, match="has 1 Excellent"):
            market.place_offer(
                seller="Meridian Manufacturing",
                buyer="Halcyon Assembly",
                quantity=3,           # more than available
                claimed_quality="Excellent",
                quality_to_send="Excellent",
                price_per_unit=50.0,
                day=1,
            )

    def test_insufficient_funds_rejected(self):
        """Buyer without enough cash cannot accept an offer."""
        market = make_simple_market(buyer_cash=10.0)
        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=1,
            claimed_quality="Excellent",
            quality_to_send="Excellent",
            price_per_unit=50.0,   # costs $50; buyer has only $10
            day=1,
        )
        with pytest.raises(MarketValidationError, match="has \\$10"):
            market.accept_offer(offer.offer_id, revelation_day=5, day=1)

    def test_zero_quantity_rejected(self):
        market = make_simple_market()
        with pytest.raises(MarketValidationError, match="quantity must be > 0"):
            market.place_offer(
                seller="Meridian Manufacturing",
                buyer="Halcyon Assembly",
                quantity=0,
                claimed_quality="Excellent",
                quality_to_send="Excellent",
                price_per_unit=50.0,
                day=1,
            )

    def test_negative_price_rejected(self):
        market = make_simple_market()
        with pytest.raises(MarketValidationError, match="price_per_unit must be >= 0"):
            market.place_offer(
                seller="Meridian Manufacturing",
                buyer="Halcyon Assembly",
                quantity=1,
                claimed_quality="Excellent",
                quality_to_send="Excellent",
                price_per_unit=-5.0,
                day=1,
            )

    def test_unknown_seller_rejected(self):
        market = make_simple_market()
        with pytest.raises(MarketValidationError, match="Unknown seller"):
            market.place_offer(
                seller="Ghost Corp",
                buyer="Halcyon Assembly",
                quantity=1,
                claimed_quality="Excellent",
                quality_to_send="Excellent",
                price_per_unit=50.0,
                day=1,
            )

    def test_accepting_already_accepted_offer_raises(self):
        market = make_simple_market()
        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=1,
            claimed_quality="Excellent",
            quality_to_send="Excellent",
            price_per_unit=50.0,
            day=1,
        )
        market.accept_offer(offer.offer_id, revelation_day=5, day=1)
        with pytest.raises(MarketValidationError, match="not pending"):
            market.accept_offer(offer.offer_id, revelation_day=6, day=1)

    def test_bankrupt_seller_cannot_trade(self):
        market = make_simple_market()
        market.sellers["Meridian Manufacturing"].bankrupt = True
        with pytest.raises(MarketValidationError, match="bankrupt"):
            market.place_offer(
                seller="Meridian Manufacturing",
                buyer="Halcyon Assembly",
                quantity=1,
                claimed_quality="Excellent",
                quality_to_send="Excellent",
                price_per_unit=50.0,
                day=1,
            )


# ── Bankruptcy ────────────────────────────────────────────────────────────────

class TestBankruptcy:
    def test_seller_bankruptcy_triggered_below_threshold(self):
        market = make_simple_market(seller_cash=-5_001.0)
        bankrupt = market.check_bankruptcies()
        assert "Meridian Manufacturing" in bankrupt
        assert market.sellers["Meridian Manufacturing"].bankrupt is True

    def test_buyer_bankruptcy_triggered_below_threshold(self):
        market = make_simple_market(buyer_cash=-5_001.0)
        bankrupt = market.check_bankruptcies()
        assert "Halcyon Assembly" in bankrupt
        assert market.buyers["Halcyon Assembly"].bankrupt is True

    def test_bankruptcy_exactly_at_threshold_not_triggered(self):
        # At exactly -3000.0, agent survives (threshold is strictly less than)
        market = make_simple_market(seller_cash=-5_000.0, buyer_cash=-5_000.0)
        bankrupt = market.check_bankruptcies()
        assert len(bankrupt) == 0

    def test_bankruptcy_clears_seller_inventory(self):
        """Bankrupt seller's inventory is zeroed out (written off)."""
        market = make_simple_market(seller_cash=-5_001.0, seller_excellent=5, seller_poor=3)
        market.check_bankruptcies()
        inv = market.sellers["Meridian Manufacturing"].inventory
        assert inv.get("Excellent", 0) == 0
        assert inv.get("Poor", 0) == 0

    def test_solvent_agents_not_affected(self):
        market = make_full_market()
        # All agents have $5000 or $6000; nobody should go bankrupt
        bankrupt = market.check_bankruptcies()
        assert bankrupt == []

    def test_already_bankrupt_not_double_counted(self):
        market = make_simple_market(seller_cash=-4_000.0)
        market.sellers["Meridian Manufacturing"].bankrupt = True  # already marked
        bankrupt = market.check_bankruptcies()
        assert "Meridian Manufacturing" not in bankrupt


# ── Production ────────────────────────────────────────────────────────────────

class TestProduction:
    def test_production_adds_to_inventory(self):
        market = make_simple_market(seller_excellent=0, seller_poor=0)
        market.execute_production("Meridian Manufacturing", excellent=1, poor=0)
        assert market.sellers["Meridian Manufacturing"].inventory["Excellent"] == 1

    def test_production_deducts_cash(self):
        market = make_simple_market()
        initial_cash = market.sellers["Meridian Manufacturing"].cash
        market.execute_production("Meridian Manufacturing", excellent=1, poor=0)
        # Production cost for Excellent with 1 factory = $30
        assert market.sellers["Meridian Manufacturing"].cash == pytest.approx(initial_cash - 30.0)

    def test_production_exceeds_capacity_clamped(self):
        market = make_simple_market(seller_factories=1, seller_excellent=0, seller_poor=0)
        result = market.execute_production("Meridian Manufacturing", excellent=5, poor=3)
        # Capacity is 1 with 1 factory; clamps to 1 Excellent, 0 Poor
        assert result["excellent"] == 1
        assert result["poor"] == 0
        assert market.sellers["Meridian Manufacturing"].inventory["Excellent"] == 1

    def test_production_with_insufficient_cash_rejected(self):
        market = make_simple_market(seller_cash=5.0)  # only $5
        with pytest.raises(MarketValidationError, match="production costs"):
            market.execute_production("Meridian Manufacturing", excellent=1, poor=0)  # costs $30

    def test_zero_production_is_valid(self):
        market = make_simple_market()
        result = market.execute_production("Meridian Manufacturing", excellent=0, poor=0)
        assert result["cost"] == 0.0

    def test_multi_factory_production(self):
        market = make_simple_market(seller_factories=3)
        market.execute_production("Meridian Manufacturing", excellent=2, poor=1)
        inv = market.sellers["Meridian Manufacturing"].inventory
        assert inv["Excellent"] == 5 + 2  # original 5 + 2 produced
        assert inv["Poor"] == 5 + 1


# ── Factory build ─────────────────────────────────────────────────────────────

class TestFactoryBuild:
    def test_factory_build_deducts_cost(self):
        market = make_simple_market()
        initial_cash = market.sellers["Meridian Manufacturing"].cash
        market.start_factory_build("Meridian Manufacturing", current_day=1)
        assert market.sellers["Meridian Manufacturing"].cash == pytest.approx(
            initial_cash - 2_000.0
        )

    def test_factory_comes_online_after_build_days(self):
        market = make_simple_market()
        market.start_factory_build("Meridian Manufacturing", current_day=5)
        # Not online on days 5, 6, or 7
        completions = market.process_factory_completions(current_day=5)
        assert "Meridian Manufacturing" not in completions
        completions = market.process_factory_completions(current_day=6)
        assert "Meridian Manufacturing" not in completions
        completions = market.process_factory_completions(current_day=7)
        assert "Meridian Manufacturing" not in completions
        # Online on day 8 (5 + 3)
        completions = market.process_factory_completions(current_day=8)
        assert completions.get("Meridian Manufacturing") == 1
        assert market.sellers["Meridian Manufacturing"].factories == 2

    def test_insufficient_cash_for_factory_rejected(self):
        market = make_simple_market(seller_cash=100.0)
        with pytest.raises(MarketValidationError, match="factory costs"):
            market.start_factory_build("Meridian Manufacturing", current_day=1)


# ── Buyer production ──────────────────────────────────────────────────────────

class TestBuyerProduction:
    def _setup_buyer_with_widgets(
        self, claimed: str = "Excellent", true: str = "Excellent", qty: int = 4
    ) -> MarketState:
        market = make_simple_market(
            seller_excellent=qty if true == "Excellent" else 0,
            seller_poor=qty if true == "Poor" else 0,
        )
        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=qty,
            claimed_quality=claimed,
            quality_to_send=true,
            price_per_unit=50.0,
            day=1,
        )
        market.accept_offer(offer.offer_id, revelation_day=5, day=1)
        return market

    def test_buyer_production_credits_revenue(self):
        market = self._setup_buyer_with_widgets(claimed="Excellent", true="Excellent")
        initial_cash = market.buyers["Halcyon Assembly"].cash
        fg_price = market.fg_prices["Excellent"]
        market.execute_buyer_production("Halcyon Assembly", quantity=2, current_day=2)
        expected = initial_cash + fg_price * 2
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(expected)

    def test_buyer_production_consumes_widgets(self):
        market = self._setup_buyer_with_widgets(qty=4)
        market.execute_buyer_production("Halcyon Assembly", quantity=2, current_day=2)
        view = market.view_inventory_for("Halcyon Assembly")
        remaining = sum(lot["quantity"] for lot in view["lots"])
        assert remaining == 2

    def test_buyer_production_exceeds_cap_rejected(self):
        market = self._setup_buyer_with_widgets(qty=4)
        with pytest.raises(MarketValidationError, match="daily cap"):
            market.execute_buyer_production("Halcyon Assembly", quantity=4, current_day=2)

    def test_buyer_production_exceeds_inventory_rejected(self):
        market = self._setup_buyer_with_widgets(qty=2)
        # quantity=3 (at cap) but only 2 widgets available → inventory error
        with pytest.raises(MarketValidationError, match="needs 3 widgets but has 2"):
            market.execute_buyer_production("Halcyon Assembly", quantity=3, current_day=2)


# ── Retroactive revenue adjustment ────────────────────────────────────────────

class TestRetroactiveAdjustment:
    """
    When a seller misrepresents quality, the buyer earns revenue based on
    the claimed quality. When revelation fires, the revenue is retroactively
    adjusted to reflect true quality. This is the core financial consequence
    of deception in the simulation.
    """

    def _cheat_scenario(self) -> tuple[MarketState, str]:
        """
        Set up a scenario where seller claims Excellent but sends Poor,
        buyer produces final goods, then revelation fires.
        Returns (market, transaction_id).
        """
        market = make_simple_market(seller_excellent=0, seller_poor=4)
        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=3,
            claimed_quality="Excellent",   # lie
            quality_to_send="Poor",        # truth
            price_per_unit=50.0,
            day=1,
        )
        tx = market.accept_offer(offer.offer_id, revelation_day=5, day=1)
        return market, tx.transaction_id

    def test_buyer_initially_credited_at_claimed_quality(self):
        market, tx_id = self._cheat_scenario()
        # Buyer has purchased 3 widgets for $150, starting cash was $6000
        # Cash after purchase = 6000 - 150 = 5850
        buyer_cash_after_purchase = market.buyers["Halcyon Assembly"].cash

        # Produce 2 final goods using claimed-Excellent widgets
        market.execute_buyer_production("Halcyon Assembly", quantity=2, current_day=2)
        fg_excellent_price = market.fg_prices["Excellent"]
        expected_cash = buyer_cash_after_purchase + fg_excellent_price * 2
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(expected_cash)

    def test_revelation_applies_downward_adjustment(self):
        market, tx_id = self._cheat_scenario()
        cash_after_purchase = market.buyers["Halcyon Assembly"].cash

        market.execute_buyer_production("Halcyon Assembly", quantity=2, current_day=2)
        cash_after_production = market.buyers["Halcyon Assembly"].cash

        # Now revelation fires: true quality was Poor
        event = RevelationEvent(
            transaction_id=tx_id,
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            claimed_quality="Excellent",
            true_quality="Poor",
            quantity=3,
            transaction_day=1,
            revelation_day=5,
        )
        result = market.apply_revelation(event)

        fg_excellent = market.fg_prices["Excellent"]
        fg_poor = market.fg_prices["Poor"]
        expected_adj = (fg_poor - fg_excellent) * 2  # 2 goods produced, downward adj

        assert result["adjustment"] == pytest.approx(expected_adj, abs=0.01)
        assert result["misrepresented"] is True
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(
            cash_after_production + expected_adj, abs=0.01
        )

    def test_honest_transaction_no_adjustment(self):
        market = make_simple_market(seller_excellent=3)
        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=2,
            claimed_quality="Excellent",
            quality_to_send="Excellent",  # honest
            price_per_unit=50.0,
            day=1,
        )
        tx = market.accept_offer(offer.offer_id, revelation_day=4, day=1)
        market.execute_buyer_production("Halcyon Assembly", quantity=2, current_day=2)
        cash_before_revelation = market.buyers["Halcyon Assembly"].cash

        event = RevelationEvent(
            transaction_id=tx.transaction_id,
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            claimed_quality="Excellent",
            true_quality="Excellent",
            quantity=2,
            transaction_day=1,
            revelation_day=4,
        )
        result = market.apply_revelation(event)

        assert result["adjustment"] == pytest.approx(0.0)
        assert result["misrepresented"] is False
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(cash_before_revelation)

    def test_adjustment_uses_production_day_prices_not_current(self):
        """
        The retroactive adjustment must use fg prices from the production day.
        We simulate a price change between production and revelation.
        """
        market = make_simple_market(seller_excellent=0, seller_poor=2)
        # Set initial prices
        market.fg_prices = {"Excellent": 80.0, "Poor": 45.0}

        offer = market.place_offer(
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            quantity=2,
            claimed_quality="Excellent",
            quality_to_send="Poor",
            price_per_unit=50.0,
            day=1,
        )
        tx = market.accept_offer(offer.offer_id, revelation_day=5, day=1)

        # Produce goods at current prices (Excellent=$80, Poor=$45)
        market.execute_buyer_production("Halcyon Assembly", quantity=2, current_day=2)

        # Prices drift before revelation
        market.fg_prices = {"Excellent": 90.0, "Poor": 55.0}

        event = RevelationEvent(
            transaction_id=tx.transaction_id,
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            claimed_quality="Excellent",
            true_quality="Poor",
            quantity=2,
            transaction_day=1,
            revelation_day=5,
        )
        result = market.apply_revelation(event)

        # Adjustment should use production-day prices (80 and 45), not current (90 and 55)
        expected_adj = (45.0 - 80.0) * 2  # = -70.0
        assert result["adjustment"] == pytest.approx(expected_adj, abs=0.01)

    def test_adjustment_not_applied_twice(self):
        """Revelation should not double-adjust if apply_revelation is called twice."""
        market, tx_id = self._cheat_scenario()
        market.execute_buyer_production("Halcyon Assembly", quantity=2, current_day=2)
        cash_after_production = market.buyers["Halcyon Assembly"].cash

        event = RevelationEvent(
            transaction_id=tx_id,
            seller="Meridian Manufacturing",
            buyer="Halcyon Assembly",
            claimed_quality="Excellent",
            true_quality="Poor",
            quantity=3,
            transaction_day=1,
            revelation_day=5,
        )
        market.apply_revelation(event)
        cash_after_first_revelation = market.buyers["Halcyon Assembly"].cash

        # Second application should not change cash further
        market.apply_revelation(event)
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(
            cash_after_first_revelation
        )


# ── Holding costs and fixed costs ─────────────────────────────────────────────

class TestDailyCosts:
    def test_holding_costs_deducted(self):
        market = make_simple_market(seller_excellent=2, seller_poor=3)
        initial_cash = market.sellers["Meridian Manufacturing"].cash
        costs = market.apply_holding_costs()
        # 2% of production cost: 2 * ($30 * 0.02) + 3 * ($20 * 0.02) = 2 * 0.60 + 3 * 0.40 = 2.40
        expected = 2 * 0.60 + 3 * 0.40
        assert costs["Meridian Manufacturing"] == pytest.approx(expected)
        assert market.sellers["Meridian Manufacturing"].cash == pytest.approx(
            initial_cash - expected
        )

    def test_buyer_quota_penalty_deducted(self):
        market = make_simple_market()
        initial_cash = market.buyers["Halcyon Assembly"].cash
        # New buyer has 0 widgets_acquired, quota is 20 -> full penalty = 20 x $2 = $40
        market.apply_buyer_quota_penalties()
        assert market.buyers["Halcyon Assembly"].cash == pytest.approx(initial_cash - 40.0)

    def test_bankrupt_agents_not_charged(self):
        market = make_simple_market()
        market.sellers["Meridian Manufacturing"].bankrupt = True
        market.buyers["Halcyon Assembly"].bankrupt = True

        initial_seller_cash = market.sellers["Meridian Manufacturing"].cash
        initial_buyer_cash = market.buyers["Halcyon Assembly"].cash

        market.apply_holding_costs()
        market.apply_buyer_quota_penalties()

        assert market.sellers["Meridian Manufacturing"].cash == initial_seller_cash
        assert market.buyers["Halcyon Assembly"].cash == initial_buyer_cash


# ── Config loading ────────────────────────────────────────────────────────────

class TestBuildInitialMarket:
    def test_builds_from_config(self):
        config = {
            "economics": {
                "seller_starting_cash": [5_000.0, 4_500.0],
                "seller_starting_factories": 1,
                "buyer_starting_cash": 6_000.0,
                "starting_widgets_per_seller": 8,
            },
            "agents": {
                "sellers": [
                    {"name": "Seller A"},
                    {"name": "Seller B"},
                ],
                "buyers": [
                    {"name": "Buyer X"},
                ],
            },
        }
        import numpy as np
        rng = np.random.default_rng(42)
        market = build_initial_market(config, rng=rng)
        assert "Seller A" in market.sellers
        assert "Seller B" in market.sellers
        assert "Buyer X" in market.buyers
        # Starting inventory is 8 widgets total (random mix)
        inv_a = market.sellers["Seller A"].inventory
        assert inv_a["Excellent"] + inv_a["Poor"] == 8
        # Asymmetric cash
        assert market.sellers["Seller A"].cash == 5_000.0
        assert market.sellers["Seller B"].cash == 4_500.0
        assert market.buyers["Buyer X"].cash == 6_000.0

    def test_builds_without_rng_uses_equal_split(self):
        config = {
            "economics": {
                "seller_starting_cash": [5_000.0],
                "starting_widgets_per_seller": 8,
            },
            "agents": {
                "sellers": [{"name": "Seller A"}],
                "buyers": [{"name": "Buyer X"}],
            },
        }
        market = build_initial_market(config)
        inv = market.sellers["Seller A"].inventory
        assert inv["Excellent"] == 4
        assert inv["Poor"] == 4
