"""
Tests for the protocol subsystem.

Covers: base class contract, NoProtocol behavior, factory dispatch,
Phase 2 protocol rejection, list_protocols.
"""

from __future__ import annotations

import pytest

from sanctuary.protocols.base import Protocol
from sanctuary.protocols.no_protocol import NoProtocol
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

    def test_peer_ratings_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            create_protocol({"protocol": {"system": "peer_ratings"}})

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
