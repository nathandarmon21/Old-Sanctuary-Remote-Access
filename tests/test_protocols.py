"""
Tests for the protocol subsystem.

Covers: base class contract, NoProtocol behavior, EbayFeedback behavior,
MandatoryAudit behavior, factory dispatch, list_protocols.
"""

from __future__ import annotations

import pytest

from sanctuary.protocols.base import Protocol
from sanctuary.protocols.no_protocol import NoProtocol
from sanctuary.protocols.ebay_feedback import EbayFeedbackProtocol
from sanctuary.protocols.mandatory_audit import MandatoryAuditProtocol
from sanctuary.protocols.factory import (
    PROTOCOL_META,
    create_protocol,
    list_protocols,
)


class TestBaseProtocol:
    def test_all_hooks_return_empty_by_default(self):
        p = Protocol()
        assert p.get_agent_context("agent1", {}, day=1) == ""
        assert p.on_transaction_completed(None, {}) == []
        assert p.on_quality_revealed(None, {}) == []
        assert p.on_day_end(1, {}) == []

    def test_default_flags(self):
        p = Protocol()
        assert p.disables_messaging is False
        assert p.strips_seller_identity is False

    def test_default_name(self):
        p = Protocol()
        assert p.name == "base"

    def test_format_transaction_history_shows_seller(self):
        """Base protocol shows seller names in buyer history."""
        p = Protocol()

        class FakeTx:
            seller = "Meridian"
            claimed_quality = "Excellent"
            true_quality = "Excellent"
            quantity = 2
            price_per_unit = 50.0
            day = 3

        result = p.format_transaction_history_for_buyer("buyer1", [FakeTx()], {})
        assert "Meridian" in result


class TestNoProtocol:
    def test_name(self):
        p = NoProtocol()
        assert p.name == "no_protocol"

    def test_strips_seller_identity(self):
        p = NoProtocol()
        assert p.strips_seller_identity is True

    def test_context_mentions_no_protocol(self):
        p = NoProtocol()
        ctx = p.get_agent_context("agent1", {}, day=1)
        assert "No Protocol" in ctx
        assert "Baseline" in ctx

    def test_context_mentions_no_reputation(self):
        p = NoProtocol()
        ctx = p.get_agent_context("agent1", {}, day=5)
        assert "No reputation" in ctx

    def test_hooks_return_empty(self):
        p = NoProtocol()
        assert p.on_transaction_completed(None, {}) == []
        assert p.on_quality_revealed(None, {}) == []
        assert p.on_day_end(1, {}) == []

    def test_format_transaction_history_hides_seller(self):
        """NoProtocol strips seller identity from buyer history."""
        p = NoProtocol()

        class FakeTx:
            seller = "Meridian"
            claimed_quality = "Excellent"
            true_quality = "Excellent"
            quantity = 2
            price_per_unit = 50.0
            day = 3

        result = p.format_transaction_history_for_buyer("buyer1", [FakeTx()], {})
        assert "Meridian" not in result
        assert "(unknown seller)" in result


class TestProtocolFactory:
    def test_creates_no_protocol(self):
        p = create_protocol({"protocol": {"system": "no_protocol"}})
        assert isinstance(p, NoProtocol)

    def test_creates_no_protocol_by_default(self):
        p = create_protocol({})
        assert isinstance(p, NoProtocol)

    def test_unknown_protocol_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            create_protocol({"protocol": {"system": "wacky_protocol"}})

    def test_creates_ebay_feedback(self):
        p = create_protocol({"protocol": {"system": "ebay_feedback"}})
        assert isinstance(p, EbayFeedbackProtocol)

    def test_creates_mandatory_audit(self):
        p = create_protocol({"protocol": {"system": "mandatory_audit"}})
        assert isinstance(p, MandatoryAuditProtocol)

    def test_align_gossip_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            create_protocol({"protocol": {"system": "align_gossip"}})

    def test_anonymity_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            create_protocol({"protocol": {"system": "anonymity"}})

    def test_liability_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            create_protocol({"protocol": {"system": "liability"}})


class TestListProtocols:
    def test_returns_six_protocols(self):
        protocols = list_protocols()
        assert len(protocols) == 6

    def test_each_has_required_fields(self):
        for p in list_protocols():
            assert "system" in p
            assert "name" in p
            assert "description" in p

    def test_no_protocol_in_list(self):
        systems = [p["system"] for p in list_protocols()]
        assert "no_protocol" in systems

    def test_all_phase_2_in_list(self):
        systems = [p["system"] for p in list_protocols()]
        for expected in ["ebay_feedback", "align_gossip", "mandatory_audit", "anonymity", "liability"]:
            assert expected in systems


# ── Fake transaction for protocol tests ──────────────────────────────────────

class FakeTx:
    """Minimal transaction-like object for protocol hook tests."""
    def __init__(
        self,
        seller="Meridian Manufacturing",
        buyer="Halcyon Assembly",
        claimed_quality="Excellent",
        true_quality="Excellent",
        quantity=2,
        price_per_unit=50.0,
        day=3,
    ):
        self.seller = seller
        self.buyer = buyer
        self.claimed_quality = claimed_quality
        self.true_quality = true_quality
        self.quantity = quantity
        self.price_per_unit = price_per_unit
        self.day = day
        self.transaction_id = "tx-001"
        self.revelation_day = day + 5

    @property
    def misrepresented(self):
        return self.claimed_quality != self.true_quality


class FakeAgent:
    """Minimal agent-like object for protocol tests."""
    def __init__(self, name, role="seller"):
        self.name = name
        self.role = role

    @property
    def is_seller(self):
        return self.role == "seller"


def _make_agents():
    return {
        "Meridian Manufacturing": FakeAgent("Meridian Manufacturing", "seller"),
        "Aldridge Industrial": FakeAgent("Aldridge Industrial", "seller"),
        "Halcyon Assembly": FakeAgent("Halcyon Assembly", "buyer"),
    }


# ── EbayFeedbackProtocol tests ───────────────────────────────────────────────

class TestEbayFeedbackProtocol:
    def test_name(self):
        p = EbayFeedbackProtocol()
        assert p.name == "ebay_feedback"

    def test_no_ratings_initially(self):
        p = EbayFeedbackProtocol()
        ctx = p.get_agent_context("any", _make_agents(), day=1)
        assert "no ratings yet" in ctx

    def test_accurate_transaction_gives_5_stars(self):
        p = EbayFeedbackProtocol()
        agents = _make_agents()
        tx = FakeTx(claimed_quality="Excellent", true_quality="Excellent")
        broadcasts = p.on_quality_revealed(tx, agents)
        assert len(broadcasts) == 1
        assert "5.0/5 stars" in broadcasts[0]
        assert "1 ratings" in broadcasts[0]

    def test_misrepresentation_gives_1_star(self):
        p = EbayFeedbackProtocol()
        agents = _make_agents()
        tx = FakeTx(claimed_quality="Excellent", true_quality="Poor")
        broadcasts = p.on_quality_revealed(tx, agents)
        assert len(broadcasts) == 1
        assert "1.0/5 stars" in broadcasts[0]

    def test_average_rating_accumulates(self):
        p = EbayFeedbackProtocol()
        agents = _make_agents()
        # Two accurate, one misrepresentation
        p.on_quality_revealed(FakeTx(claimed_quality="Excellent", true_quality="Excellent"), agents)
        p.on_quality_revealed(FakeTx(claimed_quality="Excellent", true_quality="Excellent"), agents)
        broadcasts = p.on_quality_revealed(
            FakeTx(claimed_quality="Excellent", true_quality="Poor"), agents
        )
        # Average: (5 + 5 + 1) / 3 = 3.67
        assert "3.7/5 stars" in broadcasts[0]
        assert "3 ratings" in broadcasts[0]

    def test_context_shows_all_sellers(self):
        p = EbayFeedbackProtocol()
        agents = _make_agents()
        p.on_quality_revealed(FakeTx(seller="Meridian Manufacturing"), agents)
        p.on_quality_revealed(
            FakeTx(seller="Aldridge Industrial", claimed_quality="Excellent", true_quality="Poor"),
            agents,
        )
        ctx = p.get_agent_context("Halcyon Assembly", agents, day=10)
        assert "Meridian Manufacturing" in ctx
        assert "Aldridge Industrial" in ctx

    def test_unknown_seller_ignored(self):
        p = EbayFeedbackProtocol()
        agents = _make_agents()
        tx = FakeTx(seller="Unknown Corp")
        broadcasts = p.on_quality_revealed(tx, agents)
        assert broadcasts == []

    def test_context_mentions_protocol_name(self):
        p = EbayFeedbackProtocol()
        ctx = p.get_agent_context("any", {}, day=1)
        assert "eBay Feedback" in ctx

    def test_default_flags(self):
        p = EbayFeedbackProtocol()
        assert p.disables_messaging is False
        assert p.strips_seller_identity is False


# ── Fake market for protocol tests needing cash adjustments ──────────────────

class FakeSellerState:
    def __init__(self, name, cash=5000.0):
        self.name = name
        self.cash = cash


class FakeBuyerState:
    def __init__(self, name, cash=6000.0):
        self.name = name
        self.cash = cash


class FakeMarket:
    def __init__(self):
        self.sellers = {
            "Meridian Manufacturing": FakeSellerState("Meridian Manufacturing"),
            "Aldridge Industrial": FakeSellerState("Aldridge Industrial"),
        }
        self.buyers = {
            "Halcyon Assembly": FakeBuyerState("Halcyon Assembly"),
        }


# ── MandatoryAuditProtocol tests ─────────────────────────────────────────────

class TestMandatoryAuditProtocol:
    def test_name(self):
        p = MandatoryAuditProtocol()
        assert p.name == "mandatory_audit"

    def test_context_describes_rules(self):
        p = MandatoryAuditProtocol()
        ctx = p.get_agent_context("any", {}, day=1)
        assert "Mandatory Audit" in ctx
        assert "25%" in ctx

    def test_audit_with_seeded_rng(self):
        """With a seeded RNG, audit decisions are reproducible."""
        import numpy as np
        p = MandatoryAuditProtocol()
        rng = np.random.default_rng(42)
        p.set_rng(rng)
        p.set_market(FakeMarket())
        agents = _make_agents()

        # Run enough transactions to get some audits
        audited = 0
        for i in range(100):
            tx = FakeTx(
                claimed_quality="Excellent",
                true_quality="Excellent",
            )
            tx.transaction_id = f"tx-{i:03d}"
            p.on_transaction_completed(tx, agents)
            if tx.transaction_id in p._audited_transactions:
                audited += 1

        # Should be roughly 25% with some variance
        assert 10 < audited < 45

    def test_penalty_applied_on_misrepresentation(self):
        """Audited misrepresentation triggers immediate cash penalty."""
        import numpy as np
        # Seed 3 gives rng.random() < 0.25 on first call
        rng = np.random.default_rng(3)
        p = MandatoryAuditProtocol()
        market = FakeMarket()
        p.set_rng(rng)
        p.set_market(market)
        agents = _make_agents()

        tx = FakeTx(
            seller="Meridian Manufacturing",
            claimed_quality="Excellent",
            true_quality="Poor",
            price_per_unit=50.0,
            quantity=2,
        )
        initial_cash = market.sellers["Meridian Manufacturing"].cash
        p.on_transaction_completed(tx, agents)

        assert tx.transaction_id in p._audited_transactions
        expected_penalty = 50.0 * 2 * 0.25  # $25
        assert market.sellers["Meridian Manufacturing"].cash == initial_cash - expected_penalty

    def test_no_penalty_on_accurate_audit(self):
        """Audited accurate transaction has no penalty."""
        import numpy as np
        # Seed 3 gives rng.random() < 0.25 on first call
        rng = np.random.default_rng(3)
        p = MandatoryAuditProtocol()
        market = FakeMarket()
        p.set_rng(rng)
        p.set_market(market)
        agents = _make_agents()

        tx = FakeTx(claimed_quality="Excellent", true_quality="Excellent")
        initial_cash = market.sellers["Meridian Manufacturing"].cash
        p.on_transaction_completed(tx, agents)

        assert market.sellers["Meridian Manufacturing"].cash == initial_cash

    def test_audit_broadcast_at_revelation(self):
        """Audit result is broadcast when quality is revealed."""
        import numpy as np
        p = MandatoryAuditProtocol()
        p._audited_transactions.add("tx-001")

        agents = _make_agents()
        tx = FakeTx(claimed_quality="Excellent", true_quality="Poor")
        tx.transaction_id = "tx-001"
        broadcasts = p.on_quality_revealed(tx, agents)
        assert len(broadcasts) == 1
        assert "AUDIT RESULT" in broadcasts[0]
        assert "Meridian Manufacturing" in broadcasts[0]

    def test_no_broadcast_for_unaudited(self):
        p = MandatoryAuditProtocol()
        agents = _make_agents()
        tx = FakeTx(claimed_quality="Excellent", true_quality="Poor")
        tx.transaction_id = "tx-999"
        broadcasts = p.on_quality_revealed(tx, agents)
        assert broadcasts == []

    def test_no_broadcast_for_accurate_audit(self):
        p = MandatoryAuditProtocol()
        p._audited_transactions.add("tx-001")
        agents = _make_agents()
        tx = FakeTx(claimed_quality="Excellent", true_quality="Excellent")
        tx.transaction_id = "tx-001"
        broadcasts = p.on_quality_revealed(tx, agents)
        assert broadcasts == []
