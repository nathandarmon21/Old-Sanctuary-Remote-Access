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
    build_buyer_quota_urgency_header,
)
from sanctuary.prompts.strategic import (
    SELLER_STRATEGIC_SYSTEM,
    BUYER_STRATEGIC_SYSTEM,
    build_seller_strategic_system,
    build_buyer_strategic_system,
)
from sanctuary.prompts.sub_round import SUB_ROUND_PROMPT
from sanctuary.context_manager import build_outcomes_review, build_repetition_awareness, ContextManager
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


class TestBuyerQuotaUrgencyHeader:
    """Buyer quota urgency header renders correctly."""

    def test_renders_all_fields(self):
        result = build_buyer_quota_urgency_header(
            days_remaining=5, days_total=10,
            quota_remaining=15, original_quota=20,
            terminal_penalty_per_unit=75.0,
            daily_penalty_per_unit=2.0,
        )
        assert "QUOTA STATUS" in result
        assert "Days remaining: 5 / 10" in result
        assert "Quota remaining: 15 / 20" in result
        assert "$1,125.00" in result  # 15 * 75
        assert "$30.00/day" in result  # 15 * 2

    def test_zero_remaining(self):
        result = build_buyer_quota_urgency_header(
            days_remaining=3, days_total=10,
            quota_remaining=0, original_quota=20,
            terminal_penalty_per_unit=75.0,
            daily_penalty_per_unit=2.0,
        )
        assert "Quota remaining: 0 / 20" in result
        assert "$0.00" in result

    def test_full_quota_remaining(self):
        result = build_buyer_quota_urgency_header(
            days_remaining=9, days_total=10,
            quota_remaining=20, original_quota=20,
            terminal_penalty_per_unit=75.0,
            daily_penalty_per_unit=2.0,
        )
        assert "Quota remaining: 20 / 20" in result
        assert "$1,500.00" in result  # 20 * 75

    def test_no_em_dashes(self):
        result = build_buyer_quota_urgency_header(
            days_remaining=5, days_total=10,
            quota_remaining=15, original_quota=20,
            terminal_penalty_per_unit=75.0,
            daily_penalty_per_unit=2.0,
        )
        assert "\u2014" not in result
        assert "\u2013" not in result


class TestOutcomeComparisonInStrategicPrompts:
    """Outcome comparison section renders for day 7+ strategic prompts."""

    def test_outcome_review_renders_with_transactions(self):
        events = [
            {"event_type": "transaction_completed", "seller": "Meridian Manufacturing",
             "buyer": "Halcyon Assembly", "claimed_quality": "Excellent", "price_per_unit": 50.0},
        ]
        result = build_outcomes_review("Week 1: Focus on premium pricing.", events)
        assert "OUTCOME COMPARISON" in result
        assert "Focus on premium pricing" in result
        assert "Transactions completed: 1" in result

    def test_outcome_review_asks_strategy_question(self):
        result = build_outcomes_review("Prior strategy memo.", [])
        assert "Was your previous strategy effective" in result

    def test_outcome_review_no_em_dashes(self):
        result = build_outcomes_review("Prior memo.", [
            {"event_type": "transaction_completed", "seller": "A", "buyer": "B",
             "claimed_quality": "Excellent", "price_per_unit": 50.0},
        ])
        assert "\u2014" not in result
        assert "\u2013" not in result


class TestRepetitionAwarenessInPrompts:
    """Repetition awareness section renders correctly for tactical prompts."""

    def test_repetition_section_renders(self):
        log = [
            {"day": 4, "counterparty": "Halcyon Assembly", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon Assembly", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon Assembly", "type": "offer_made"},
        ]
        result = build_repetition_awareness(log, current_day=6)
        assert "RECENT INTERACTION PATTERNS" in result
        assert "Halcyon Assembly" in result

    def test_repetition_section_flags_no_response(self):
        log = [
            {"day": 3, "counterparty": "Halcyon Assembly", "type": "message_sent"},
            {"day": 4, "counterparty": "Halcyon Assembly", "type": "message_sent"},
            {"day": 4, "counterparty": "Halcyon Assembly", "type": "offer_made"},
        ]
        result = build_repetition_awareness(log, current_day=5)
        assert "NOTE:" in result
        assert "without successful transaction" in result

    def test_repetition_section_no_em_dashes(self):
        log = [
            {"day": 4, "counterparty": "X", "type": "message_sent"},
            {"day": 5, "counterparty": "X", "type": "message_sent"},
        ]
        result = build_repetition_awareness(log, current_day=6)
        assert "\u2014" not in result
        assert "\u2013" not in result


class TestSingleTierFallback:
    """When current_policy is None, prompts use autonomous framing."""

    def test_seller_autonomous_framing(self):
        text = build_seller_tactical_system(
            company_name="Meridian Manufacturing",
            days_total=15,
            factory_cost=2000.0,
            factory_days=3,
            revelation_days=5,
            current_policy=None,
        )
        assert "running this firm" in text
        assert "making all decisions yourself" in text
        assert "YOUR CEO" not in text
        assert "CEO directive" not in text.lower()
        assert "strategic memo" not in text.lower()

    def test_buyer_autonomous_framing(self):
        text = build_buyer_tactical_system(
            company_name="Halcyon Assembly",
            days_total=15,
            widget_quota=20,
            daily_penalty=2.0,
            terminal_penalty=75.0,
            revelation_days=5,
            fmv_excellent=55.0,
            fmv_poor=32.0,
            daily_prod_cap=3,
            current_policy=None,
        )
        assert "running this firm" in text
        assert "making all decisions yourself" in text
        assert "YOUR CEO" not in text
        assert "CEO directive" not in text.lower()
        assert "strategic memo" not in text.lower()

    def test_seller_ceo_framing_with_policy(self):
        text = build_seller_tactical_system(
            company_name="Meridian Manufacturing",
            days_total=15,
            factory_cost=2000.0,
            factory_days=3,
            revelation_days=5,
            current_policy="Focus on premium pricing.",
        )
        assert "YOUR CEO" in text
        assert "running this firm" not in text

    def test_buyer_ceo_framing_with_policy(self):
        text = build_buyer_tactical_system(
            company_name="Halcyon Assembly",
            days_total=15,
            widget_quota=20,
            daily_penalty=2.0,
            terminal_penalty=75.0,
            revelation_days=5,
            fmv_excellent=55.0,
            fmv_poor=32.0,
            daily_prod_cap=3,
            current_policy="Be aggressive on pricing.",
        )
        assert "YOUR CEO" in text
        assert "running this firm" not in text

    def test_autonomous_framing_no_em_dashes(self):
        for text in [
            build_seller_tactical_system(
                company_name="X", days_total=15, factory_cost=2000.0,
                factory_days=3, revelation_days=5, current_policy=None,
            ),
            build_buyer_tactical_system(
                company_name="X", days_total=15, widget_quota=20,
                daily_penalty=2.0, terminal_penalty=75.0, revelation_days=5,
                fmv_excellent=55.0, fmv_poor=32.0, daily_prod_cap=3,
                current_policy=None,
            ),
        ]:
            assert EM_DASH not in text, "Em-dash found in autonomous framing"
            assert EN_DASH not in text, "En-dash found in autonomous framing"

    def test_context_manager_omits_ceo_section_when_none(self):
        cm = ContextManager()
        result = cm.build_tactical_context(
            state_header="Day 1 state",
            current_policy_memo=None,
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "CEO DIRECTIVE" not in result

    def test_context_manager_includes_ceo_section_when_empty_string(self):
        cm = ContextManager()
        result = cm.build_tactical_context(
            state_header="Day 1 state",
            current_policy_memo="",
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "CEO DIRECTIVE" in result
        assert "No strategic directive yet" in result

    def test_context_manager_includes_ceo_section_when_present(self):
        cm = ContextManager()
        result = cm.build_tactical_context(
            state_header="Day 1 state",
            current_policy_memo="Focus on premium.",
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "CEO DIRECTIVE" in result
        assert "Focus on premium." in result


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
