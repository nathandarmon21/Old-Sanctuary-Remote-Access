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
        revelation_days=5,
        premium_price=58.0,
        standard_price=35.0,
        conversion_cost=3.0,
        daily_prod_cap=5,
        protocol_rules="",
    )


def _seller_strategic():
    return build_seller_strategic_system(
        company_name="Meridian Manufacturing",
        days_total=30,
        day=7,
        fmv_excellent=58.0,
        fmv_poor=35.0,
        cost_1e=30.0, cost_1p=20.0,
        cost_2e=25.50, cost_2p=17.0,
        cost_3e=21.68, cost_3p=14.45,
        cost_4e=18.43, cost_4p=12.28,
        factory_cost=2000.0,
        factory_days=3,
        revelation_days=5,
        protocol_rules="",
    )


def _buyer_strategic():
    return build_buyer_strategic_system(
        company_name="Halcyon Assembly",
        days_total=30,
        day=7,
        fmv_excellent=58.0,
        fmv_poor=35.0,
        revelation_days=5,
        premium_price=58.0,
        standard_price=35.0,
        conversion_cost=3.0,
        daily_prod_cap=5,
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

    def test_buyer_contains_conversion_economics(self):
        text = _buyer_tactical()
        assert "$58.00" in text  # premium goods price
        assert "$35.00" in text  # standard goods price
        assert "$3.00" in text   # conversion cost

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

    def test_tactical_requires_rationale_before_actions(self):
        """Tactical prompt requires <rationale> before <actions> for CoT-monitoring fix."""
        seller = _seller_tactical()
        buyer = _buyer_tactical()
        for text in [seller, buyer]:
            # Sanity-check that the PRE-ACTION RATIONALE instruction is present
            assert "PRE-ACTION RATIONALE" in text
            # The rationale opening tag must appear before the <actions>
            # JSON example template (the closing `</actions>` proves the
            # template is the JSON block, not a prose mention).
            rationale_pos = text.index("<rationale>")
            actions_close_pos = text.index("</actions>")
            assert rationale_pos < actions_close_pos

    def test_seller_contains_revelation_days(self):
        text = _seller_tactical()
        assert "5 days" in text

    def test_buyer_contains_spread_analysis(self):
        text = _buyer_tactical()
        assert "Breakeven" in text or "breakeven" in text
        assert "$55.00" in text  # excellent breakeven (58-3)


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
        assert "$25.50" in text  # 2 factories Excellent (new formula)
        assert "$18.43" in text  # 4 factories Excellent

    def test_buyer_contains_conversion_prices(self):
        text = _buyer_strategic()
        assert "$58.00" in text  # premium goods price
        assert "$35.00" in text  # standard goods price

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


class TestBuyerConversionEconomics:
    """Buyer prompt contains conversion economics information."""

    def test_buyer_shows_breakeven(self):
        text = _buyer_tactical()
        assert "Breakeven" in text or "breakeven" in text

    def test_buyer_shows_spread_examples(self):
        text = _buyer_tactical()
        assert "$10.00" in text  # spread at $45: 58 - 45 - 3 = 10

    def test_buyer_no_quota_requirement(self):
        text = _buyer_tactical()
        # Should NOT have quota requirements, but may mention "no quota"
        assert "must acquire" not in text.lower()
        assert "quota remaining" not in text.lower()


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
            revelation_days=5,
            premium_price=58.0,
            standard_price=35.0,
            conversion_cost=3.0,
            daily_prod_cap=5,
            current_policy=None,
        )
        assert "running this firm" in text
        assert "making all decisions yourself" in text
        assert "YOUR CEO" not in text

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
            revelation_days=5,
            premium_price=58.0,
            standard_price=35.0,
            conversion_cost=3.0,
            daily_prod_cap=5,
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
                company_name="X", days_total=15, revelation_days=5,
                premium_price=58.0, standard_price=35.0,
                conversion_cost=3.0, daily_prod_cap=5,
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


class TestNewPromptSections:
    """Tests for restored prompt sections: engagement, adversarial, reasoning, forecasting."""

    def test_tactical_prompt_contains_mandatory_engagement(self):
        for text in [_seller_tactical(), _buyer_tactical()]:
            assert "MANDATORY ENGAGEMENT" in text
            assert "detailed reasoning at least 3 sentences" in text
            assert "Send at least one message" in text

    def test_tactical_prompt_contains_strategy_permissions(self):
        # Seller has full adversarial permissions
        seller = _seller_tactical()
        assert "YOU ARE FREE TO USE ANY STRATEGY" in seller
        assert "deceptive" in seller.lower()
        # Buyer has cooperative/competitive permissions
        buyer = _buyer_tactical()
        assert "YOU ARE FREE TO USE ANY STRATEGY" in buyer

    def test_seller_tactical_contains_reasoning_prompts(self):
        text = _seller_tactical()
        assert "USE YOUR REASONING FIELD" in text
        assert "financial trajectory" in text

    def test_strategic_prompt_contains_forecasting_section(self):
        for text in [_seller_strategic(), _buyer_strategic()]:
            assert "YOUR FORECASTING RESPONSIBILITY" in text
            assert "predict the future" in text or "predictions" in text

    def test_strategic_prompt_contains_price_forecast_block(self):
        for text in [_seller_strategic(), _buyer_strategic()]:
            assert "PRICE FORECAST (REQUIRED):" in text
            assert "Current observed avg price" in text

    def test_strategic_prompt_contains_action_authorization(self):
        # Sellers now get CEO-framing + policy authority; buyers retain the
        # "YOU ARE FREE TO AUTHORIZE" phrase. Both should communicate that
        # strategy space is open.
        seller = _seller_strategic()
        assert "YOUR POLICY OUTPUT" in seller or "YOU ARE FREE" in seller
        assert "operating strategy" in seller or "Strategy space" in seller
        assert "YOU ARE FREE TO AUTHORIZE ANY STRATEGY" in _buyer_strategic()

    def test_inactivity_escalation_triggers_at_2_days(self):
        cm = ContextManager()
        result = cm.build_tactical_context(
            state_header="Day 5 state",
            current_policy_memo="some memo",
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
            inactivity_days=2,
        )
        assert "URGENT INACTIVITY ESCALATION" in result
        assert "2 consecutive turns" in result

    def test_inactivity_escalation_does_not_trigger_below_2(self):
        cm = ContextManager()
        result = cm.build_tactical_context(
            state_header="Day 5 state",
            current_policy_memo="some memo",
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
            inactivity_days=1,
        )
        assert "INACTIVITY ESCALATION" not in result

    def test_no_em_dashes_in_new_sections(self):
        for text in [_seller_tactical(), _buyer_tactical(),
                     _seller_strategic(), _buyer_strategic()]:
            assert EM_DASH not in text, "Em-dash found in prompt"
            assert EN_DASH not in text, "En-dash found in prompt"


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
