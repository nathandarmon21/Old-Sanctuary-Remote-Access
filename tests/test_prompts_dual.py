"""
Tests for the dual-tier prompt templates.

Covers: tactical and strategic prompts render without error,
no em-dashes, correct content inclusion/exclusion.
"""

from __future__ import annotations

import re

import pytest

from sanctuary.prompts.tactical import (
    SELLER_TACTICAL_SYSTEM,
    BUYER_TACTICAL_SYSTEM,
    build_seller_tactical_system,
    build_buyer_tactical_system,
)
from sanctuary.prompts.strategic import (
    SELLER_STRATEGIC_SYSTEM,
    BUYER_STRATEGIC_SYSTEM,
    build_seller_strategic_system,
    build_buyer_strategic_system,
)
from sanctuary.prompts.sub_round import SUB_ROUND_PROMPT
from sanctuary.prompts.common import (
    INACTIVITY_NUDGE,
    format_inventory_for_seller,
    format_inventory_for_buyer,
    format_pending_offers_for_buyer,
    format_pending_offers_for_seller,
    format_prev_outcomes,
    format_messages_received,
)

# Em-dash characters to reject
EM_DASH = "\u2014"
EN_DASH = "\u2013"


def _seller_tactical():
    return build_seller_tactical_system(
        company_name="Meridian Manufacturing",
        days_total=30,
        factory_cost=2000.0,
        factory_days=3,
        revelation_days=5,
        protocol_rules="",
    )


def _buyer_tactical():
    return build_buyer_tactical_system(
        company_name="Halcyon Assembly",
        days_total=30,
        widget_quota=20,
        daily_penalty=2.0,
        terminal_penalty=75.0,
        revelation_days=5,
        fmv_excellent=55.0,
        fmv_poor=32.0,
        daily_prod_cap=3,
        protocol_rules="",
    )


def _seller_strategic():
    return build_seller_strategic_system(
        company_name="Meridian Manufacturing",
        days_total=30,
        day=7,
        fmv_excellent=55.0,
        fmv_poor=32.0,
        cost_1e=30.0, cost_1p=20.0,
        cost_2e=27.0, cost_2p=18.0,
        cost_3e=24.60, cost_3p=16.40,
        cost_4e=22.80, cost_4p=15.20,
        factory_cost=2000.0,
        factory_days=3,
        revelation_days=5,
        buyer_quota=20,
        protocol_rules="",
    )


def _buyer_strategic():
    return build_buyer_strategic_system(
        company_name="Halcyon Assembly",
        days_total=30,
        day=7,
        widget_quota=20,
        daily_penalty=2.0,
        terminal_penalty=75.0,
        fmv_excellent=55.0,
        fmv_poor=32.0,
        revelation_days=5,
        protocol_rules="",
    )


class TestNoEmDashes:
    """No em-dashes or en-dashes in any prompt template."""

    def test_seller_tactical_no_em_dash(self):
        text = _seller_tactical()
        assert EM_DASH not in text, f"Em-dash found in seller tactical prompt"
        assert EN_DASH not in text, f"En-dash found in seller tactical prompt"

    def test_buyer_tactical_no_em_dash(self):
        text = _buyer_tactical()
        assert EM_DASH not in text
        assert EN_DASH not in text

    def test_seller_strategic_no_em_dash(self):
        text = _seller_strategic()
        assert EM_DASH not in text
        assert EN_DASH not in text

    def test_buyer_strategic_no_em_dash(self):
        text = _buyer_strategic()
        assert EM_DASH not in text
        assert EN_DASH not in text

    def test_sub_round_no_em_dash(self):
        assert EM_DASH not in SUB_ROUND_PROMPT
        assert EN_DASH not in SUB_ROUND_PROMPT

    def test_inactivity_nudge_no_em_dash(self):
        assert EM_DASH not in INACTIVITY_NUDGE
        assert EN_DASH not in INACTIVITY_NUDGE


class TestTacticalPrompts:
    def test_seller_renders_without_error(self):
        text = _seller_tactical()
        assert len(text) > 100

    def test_buyer_renders_without_error(self):
        text = _buyer_tactical()
        assert len(text) > 100

    def test_seller_contains_company_name(self):
        text = _seller_tactical()
        assert "Meridian Manufacturing" in text

    def test_buyer_contains_quota(self):
        text = _buyer_tactical()
        assert "20 widgets" in text

    def test_buyer_contains_penalties(self):
        text = _buyer_tactical()
        assert "$2.00" in text
        assert "$75.00" in text

    def test_seller_contains_factory_info(self):
        text = _seller_tactical()
        assert "$2,000" in text
        assert "3 days" in text

    def test_tactical_does_not_ask_for_strategy(self):
        """Tactical prompt should NOT instruct long-horizon strategy planning."""
        seller = _seller_tactical()
        buyer = _buyer_tactical()
        for text in [seller, buyer]:
            # Should not contain strategy-setting language
            assert "set strategy" not in text.lower()
            assert "long-term plan" not in text.lower()
            assert "strategic memo" not in text.lower()

    def test_tactical_requests_actions_first(self):
        """Tactical prompt should request <actions> block first."""
        seller = _seller_tactical()
        buyer = _buyer_tactical()
        for text in [seller, buyer]:
            assert "<actions>" in text
            actions_pos = text.index("<actions>")
            reasoning_pos = text.lower().index("reasoning")
            assert actions_pos < reasoning_pos

    def test_seller_contains_revelation_days(self):
        text = _seller_tactical()
        assert "5 days" in text

    def test_buyer_contains_fmv(self):
        text = _buyer_tactical()
        assert "$55.00" in text
        assert "$32.00" in text


class TestStrategicPrompts:
    def test_seller_renders_without_error(self):
        text = _seller_strategic()
        assert len(text) > 100

    def test_buyer_renders_without_error(self):
        text = _buyer_strategic()
        assert len(text) > 100

    def test_seller_contains_cost_table(self):
        text = _seller_strategic()
        assert "$30.00" in text  # 1 factory Excellent
        assert "$27.00" in text  # 2 factories Excellent
        assert "$22.80" in text  # 4+ factories Excellent

    def test_buyer_contains_quota(self):
        text = _buyer_strategic()
        assert "20 widgets" in text

    def test_seller_strategic_contains_reassess(self):
        """Strategic prompt must instruct re-evaluation."""
        text = _seller_strategic()
        assert "re-evaluate" in text.lower() or "reassess" in text.lower()

    def test_buyer_strategic_contains_reassess(self):
        text = _buyer_strategic()
        assert "re-evaluate" in text.lower() or "reassess" in text.lower()

    def test_strategic_requests_policy_first(self):
        """Strategic prompt should instruct <policy> block FIRST before the memo text."""
        seller = _seller_strategic()
        buyer = _buyer_strategic()
        for text in [seller, buyer]:
            assert "<policy>" in text
            # The instruction to respond with <policy> first should appear
            assert "policy> block FIRST" in text or "policy> block first" in text.lower()

    def test_strategic_contains_day(self):
        text = _seller_strategic()
        assert "day 7" in text.lower()


class TestSubRoundPrompt:
    def test_renders(self):
        text = SUB_ROUND_PROMPT.format(
            company_name="Test Co",
            sub_round=1,
            day=5,
            pending_offers="Offer X: 2x Excellent at $50",
            policy_summary="Be aggressive.",
        )
        assert "Test Co" in text
        assert "sub-round 1" in text

    def test_accept_decline_only(self):
        """Sub-round prompt should only request accept/decline actions."""
        text = SUB_ROUND_PROMPT
        assert "accept_offers" in text
        assert "decline_offers" in text
        assert "produce" not in text.lower()
        assert "build_factory" not in text.lower()


class TestCommonFormatters:
    def test_format_inventory_for_seller_empty(self):
        result = format_inventory_for_seller({"Excellent": 0, "Poor": 0}, 1)
        assert "empty" in result

    def test_format_inventory_for_seller_with_items(self):
        result = format_inventory_for_seller({"Excellent": 3, "Poor": 2}, 1)
        assert "3x Excellent" in result
        assert "2x Poor" in result
        assert "$30.00" in result  # production cost at 1 factory

    def test_format_inventory_for_buyer_empty(self):
        result = format_inventory_for_buyer([])
        assert "Widgets owned: 0" in result

    def test_format_prev_outcomes_empty(self):
        assert format_prev_outcomes([]) == ""

    def test_format_prev_outcomes_with_items(self):
        result = format_prev_outcomes(["Offer accepted", "Message sent"])
        assert "Offer accepted" in result
        assert "Message sent" in result

    def test_format_messages_received_empty(self):
        result = format_messages_received([])
        assert "No messages" in result

    def test_format_messages_received_with_items(self):
        msgs = [
            {"from": "Aldridge", "body": "Let's talk pricing", "public": False},
            {"from": "Crestline", "body": "Public announcement", "public": True},
        ]
        result = format_messages_received(msgs)
        assert "Aldridge" in result
        assert "[private]" in result
        assert "[public]" in result
