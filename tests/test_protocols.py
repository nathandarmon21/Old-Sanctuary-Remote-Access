"""
Tests for the protocol subsystem.

Covers: base class contract, NoProtocol behavior, factory dispatch,
Phase 2 protocol rejection, list_protocols.
"""

from __future__ import annotations

import pytest

from sanctuary.protocols.base import Protocol
from sanctuary.protocols.no_protocol import NoProtocol
from sanctuary.protocols.peer_ratings import PeerRatingsProtocol
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

    def test_creates_peer_ratings(self):
        p = create_protocol({"protocol": {"system": "peer_ratings"}})
        assert isinstance(p, PeerRatingsProtocol)

    def test_credit_bureau_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            create_protocol({"protocol": {"system": "credit_bureau"}})

    def test_mandatory_audit_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            create_protocol({"protocol": {"system": "mandatory_audit"}})

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
        for expected in ["peer_ratings", "credit_bureau", "mandatory_audit", "anonymity", "liability"]:
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


# ── PeerRatingsProtocol tests ────────────────────────────────────────────────

class TestPeerRatingsProtocol:
    def test_name(self):
        p = PeerRatingsProtocol()
        assert p.name == "peer_ratings"

    def test_no_ratings_initially(self):
        p = PeerRatingsProtocol()
        ctx = p.get_agent_context("any", _make_agents(), day=1)
        assert "no ratings yet" in ctx

    def test_accurate_transaction_gives_5_stars(self):
        p = PeerRatingsProtocol()
        agents = _make_agents()
        tx = FakeTx(claimed_quality="Excellent", true_quality="Excellent")
        broadcasts = p.on_quality_revealed(tx, agents)
        assert len(broadcasts) == 1
        assert "5.0/5 stars" in broadcasts[0]
        assert "1 ratings" in broadcasts[0]

    def test_misrepresentation_gives_1_star(self):
        p = PeerRatingsProtocol()
        agents = _make_agents()
        tx = FakeTx(claimed_quality="Excellent", true_quality="Poor")
        broadcasts = p.on_quality_revealed(tx, agents)
        assert len(broadcasts) == 1
        assert "1.0/5 stars" in broadcasts[0]

    def test_average_rating_accumulates(self):
        p = PeerRatingsProtocol()
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
        p = PeerRatingsProtocol()
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
        p = PeerRatingsProtocol()
        agents = _make_agents()
        tx = FakeTx(seller="Unknown Corp")
        broadcasts = p.on_quality_revealed(tx, agents)
        assert broadcasts == []

    def test_context_mentions_protocol_name(self):
        p = PeerRatingsProtocol()
        ctx = p.get_agent_context("any", {}, day=1)
        assert "Peer Ratings" in ctx

    def test_default_flags(self):
        p = PeerRatingsProtocol()
        assert p.disables_messaging is False
        assert p.strips_seller_identity is False
