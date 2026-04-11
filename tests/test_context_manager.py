"""
Tests for sanctuary/context_manager.py.

Covers: tactical context assembly, strategic context assembly,
market digest generation, token budget enforcement.
"""

from __future__ import annotations

import pytest

from sanctuary.context_manager import (
    ContextManager,
    build_outcomes_review,
    build_repetition_awareness,
    _estimate_tokens,
    _truncate_to_budget,
)


@pytest.fixture
def cm():
    return ContextManager()


class TestTacticalContext:
    def test_includes_state_header(self, cm):
        result = cm.build_tactical_context(
            state_header="[STATE HEADER]",
            current_policy_memo=None,
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "[STATE HEADER]" in result

    def test_includes_policy_memo(self, cm):
        result = cm.build_tactical_context(
            state_header="header",
            current_policy_memo="Focus on volume over margin this week.",
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "Focus on volume over margin" in result
        assert "CEO DIRECTIVE" in result

    def test_no_policy_none_omits_ceo_section(self, cm):
        """When current_policy_memo is None (no CEO tier), omit the section."""
        result = cm.build_tactical_context(
            state_header="header",
            current_policy_memo=None,
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "CEO DIRECTIVE" not in result

    def test_no_policy_empty_string_shows_default(self, cm):
        """When current_policy_memo is '' (CEO tier exists, no memo yet), show default."""
        result = cm.build_tactical_context(
            state_header="header",
            current_policy_memo="",
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "No strategic directive yet" in result

    def test_includes_prev_outcomes(self, cm):
        result = cm.build_tactical_context(
            state_header="header",
            current_policy_memo=None,
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="Offer to Halcyon: ACCEPTED",
            protocol_context="",
        )
        assert "Offer to Halcyon: ACCEPTED" in result

    def test_includes_recent_history(self, cm):
        history = [
            {"role": "user", "content": "Day 5 prompt"},
            {"role": "assistant", "content": "Day 5 response"},
        ]
        result = cm.build_tactical_context(
            state_header="header",
            current_policy_memo=None,
            recent_tactical_history=history,
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "Day 5 prompt" in result
        assert "RECENT HISTORY" in result

    def test_includes_inbox(self, cm):
        result = cm.build_tactical_context(
            state_header="header",
            current_policy_memo=None,
            recent_tactical_history=[],
            today_inbox="Message from Aldridge: Let's coordinate pricing.",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
        )
        assert "Aldridge" in result

    def test_includes_protocol_context(self, cm):
        result = cm.build_tactical_context(
            state_header="header",
            current_policy_memo=None,
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="No reputation system active.",
        )
        assert "No reputation system active" in result

    def test_includes_inactivity_nudge(self, cm):
        result = cm.build_tactical_context(
            state_header="header",
            current_policy_memo=None,
            recent_tactical_history=[],
            today_inbox="",
            pending_offers="",
            prev_outcomes="",
            protocol_context="",
            inactivity_nudge="WARNING: You have been inactive for 3 days.",
        )
        assert "inactive for 3 days" in result

    def test_stays_under_token_budget(self, cm):
        """Tactical context should be roughly under 7500 tokens."""
        long_history = [
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "y" * 2000},
        ] * 10  # very long
        result = cm.build_tactical_context(
            state_header="h" * 500,
            current_policy_memo="m" * 2000,
            recent_tactical_history=long_history,
            today_inbox="i" * 500,
            pending_offers="p" * 500,
            prev_outcomes="o" * 500,
            protocol_context="c" * 200,
        )
        # Should be trimmed; exact budget depends on implementation
        tokens = _estimate_tokens(result)
        assert tokens < 10000  # generous upper bound


class TestStrategicContext:
    def test_includes_state_header(self, cm):
        result = cm.build_strategic_context(
            state_header="[STRATEGIC STATE]",
            market_digest="Day 1: 2 txns",
            tactical_history_since_last_review=[],
            all_prior_memos=[],
            competitive_landscape="",
            protocol_context="",
        )
        assert "[STRATEGIC STATE]" in result

    def test_includes_market_digest(self, cm):
        result = cm.build_strategic_context(
            state_header="header",
            market_digest="Day 1: 2 txns | prices E:$52.00\nDay 2: 1 txn",
            tactical_history_since_last_review=[],
            all_prior_memos=[],
            competitive_landscape="",
            protocol_context="",
        )
        assert "Day 1: 2 txns" in result
        assert "MARKET HISTORY DIGEST" in result

    def test_includes_tactical_history_since_review(self, cm):
        history = [
            {"role": "user", "content": "Day 8 prompt"},
            {"role": "assistant", "content": "Day 8 response"},
        ]
        result = cm.build_strategic_context(
            state_header="header",
            market_digest="digest",
            tactical_history_since_last_review=history,
            all_prior_memos=[],
            competitive_landscape="",
            protocol_context="",
        )
        assert "Day 8 prompt" in result
        assert "TACTICAL HISTORY SINCE LAST REVIEW" in result

    def test_includes_all_prior_memos(self, cm):
        memos = [
            "Week 1: Focus on market share.",
            "Week 2: Shift to premium pricing.",
        ]
        result = cm.build_strategic_context(
            state_header="header",
            market_digest="digest",
            tactical_history_since_last_review=[],
            all_prior_memos=memos,
            competitive_landscape="",
            protocol_context="",
        )
        assert "Week 1" in result
        assert "Week 2" in result
        assert "PRIOR STRATEGIC MEMOS" in result

    def test_first_review_message(self, cm):
        result = cm.build_strategic_context(
            state_header="header",
            market_digest="digest",
            tactical_history_since_last_review=[],
            all_prior_memos=[],
            competitive_landscape="",
            protocol_context="",
        )
        assert "first strategic review" in result

    def test_includes_competitive_landscape(self, cm):
        result = cm.build_strategic_context(
            state_header="header",
            market_digest="digest",
            tactical_history_since_last_review=[],
            all_prior_memos=[],
            competitive_landscape="Aldridge has 3 factories, dominating volume.",
            protocol_context="",
        )
        assert "Aldridge has 3 factories" in result

    def test_stays_under_token_budget(self, cm):
        """Strategic context should be roughly under 17500 tokens."""
        result = cm.build_strategic_context(
            state_header="h" * 2000,
            market_digest="d" * 12000,
            tactical_history_since_last_review=[
                {"role": "user", "content": "x" * 10000},
                {"role": "assistant", "content": "y" * 10000},
            ],
            all_prior_memos=["m" * 5000, "n" * 5000],
            competitive_landscape="c" * 3000,
            protocol_context="p" * 500,
        )
        tokens = _estimate_tokens(result)
        assert tokens < 25000  # generous upper bound (truncation active)


class TestMarketDigest:
    def test_empty_produces_message(self, cm):
        result = cm.build_market_digest([], {})
        assert "No market history" in result

    def test_one_day_no_events(self, cm):
        snapshots = [{"day": 1}]
        result = cm.build_market_digest(snapshots, {})
        assert "Day 1: 0 txns" in result

    def test_transaction_counted(self, cm):
        snapshots = [{"day": 1}]
        events = {1: [
            {"event_type": "transaction_completed", "claimed_quality": "Excellent", "price_per_unit": 50.0},
            {"event_type": "transaction_completed", "claimed_quality": "Poor", "price_per_unit": 28.0},
        ]}
        result = cm.build_market_digest(snapshots, events)
        assert "Day 1: 2 txns" in result

    def test_prices_included(self, cm):
        snapshots = [{"day": 3}]
        events = {3: [
            {"event_type": "transaction_completed", "claimed_quality": "Excellent", "price_per_unit": 52.0},
        ]}
        result = cm.build_market_digest(snapshots, events)
        assert "E:$52.00" in result

    def test_revelations_included(self, cm):
        snapshots = [{"day": 6}]
        events = {6: [
            {"event_type": "quality_revealed", "seller": "Aldridge", "claimed_quality": "Excellent", "true_quality": "Poor"},
            {"event_type": "quality_revealed", "seller": "Meridian", "claimed_quality": "Poor", "true_quality": "Poor"},
        ]}
        result = cm.build_market_digest(snapshots, events)
        assert "revealed 2" in result
        assert "1 misrep" in result
        assert "Aldridge" in result

    def test_factory_completions_included(self, cm):
        snapshots = [{"day": 5}]
        events = {5: [
            {"event_type": "factory_completed", "agent_id": "Vector Works"},
        ]}
        result = cm.build_market_digest(snapshots, events)
        assert "factory completion" in result
        assert "Vector Works" in result

    def test_flags_included(self, cm):
        snapshots = [{"day": 10}]
        events = {10: [
            {"event_type": "cot_flag", "category": "collusion_price_fixing"},
        ]}
        result = cm.build_market_digest(snapshots, events)
        assert "flagged" in result
        assert "collusion_price_fixing" in result

    def test_multiple_days(self, cm):
        snapshots = [{"day": 1}, {"day": 2}, {"day": 3}]
        events = {
            1: [{"event_type": "transaction_completed", "claimed_quality": "Excellent", "price_per_unit": 50.0}],
            2: [],
            3: [{"event_type": "bankruptcy", "agent_id": "Crestline"}],
        }
        result = cm.build_market_digest(snapshots, events)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "Day 1" in lines[0]
        assert "Day 2" in lines[1]
        assert "Day 3" in lines[2]
        assert "BANKRUPT" in lines[2]

    def test_30_days_fits_in_budget(self, cm):
        """30 days of moderate activity should fit in ~2000 tokens."""
        snapshots = [{"day": d} for d in range(1, 31)]
        events = {
            d: [
                {"event_type": "transaction_completed", "claimed_quality": "Excellent", "price_per_unit": 50.0 + d},
                {"event_type": "transaction_completed", "claimed_quality": "Poor", "price_per_unit": 25.0 + d},
            ]
            for d in range(1, 31)
        }
        result = cm.build_market_digest(snapshots, events)
        tokens = _estimate_tokens(result)
        assert tokens < 3000  # well under budget


class TestOutcomesReview:
    def test_empty_memo_returns_empty(self):
        result = build_outcomes_review("", [])
        assert result == ""

    def test_shows_prior_memo(self):
        result = build_outcomes_review("Focus on premium pricing.", [])
        assert "OUTCOME COMPARISON" in result
        assert "Focus on premium pricing" in result

    def test_shows_transaction_count(self):
        events = [
            {"event_type": "transaction_completed", "seller": "Meridian", "buyer": "Halcyon",
             "claimed_quality": "Excellent", "price_per_unit": 52.0},
            {"event_type": "transaction_completed", "seller": "Meridian", "buyer": "Crestline",
             "claimed_quality": "Poor", "price_per_unit": 28.0},
        ]
        result = build_outcomes_review("Prior memo text.", events)
        assert "Transactions completed: 2" in result

    def test_shows_price_summary(self):
        events = [
            {"event_type": "transaction_completed", "seller": "Meridian", "buyer": "Halcyon",
             "claimed_quality": "Excellent", "price_per_unit": 50.0},
            {"event_type": "transaction_completed", "seller": "Meridian", "buyer": "Halcyon",
             "claimed_quality": "Excellent", "price_per_unit": 54.0},
        ]
        result = build_outcomes_review("Prior memo.", events)
        assert "Excellent avg price: $52.00" in result

    def test_shows_no_transactions_message(self):
        events = [
            {"event_type": "transaction_proposed", "seller": "Meridian", "buyer": "Halcyon"},
            {"event_type": "offer_expired", "offer_id": "o1"},
        ]
        result = build_outcomes_review("Prior memo.", events)
        assert "No transactions were completed" in result

    def test_shows_unconverted_proposals(self):
        events = [
            {"event_type": "transaction_proposed", "seller": "Meridian", "buyer": "Halcyon"},
            {"event_type": "transaction_proposed", "seller": "Meridian", "buyer": "Crestline"},
            {"event_type": "transaction_completed", "seller": "Meridian", "buyer": "Halcyon",
             "claimed_quality": "Excellent", "price_per_unit": 50.0},
        ]
        result = build_outcomes_review("Prior memo.", events)
        assert "Unconverted proposals: 1 of 2" in result

    def test_includes_strategy_question(self):
        result = build_outcomes_review("Prior memo.", [])
        assert "Was your previous strategy effective" in result
        assert "what specifically should change" in result

    def test_shows_revelations(self):
        events = [
            {"event_type": "quality_revealed", "claimed_quality": "Excellent", "true_quality": "Poor"},
            {"event_type": "quality_revealed", "claimed_quality": "Excellent", "true_quality": "Excellent"},
        ]
        result = build_outcomes_review("Prior memo.", events)
        assert "Quality revelations: 2 (1 misrepresented)" in result

    def test_no_em_dashes(self):
        events = [
            {"event_type": "transaction_completed", "seller": "Meridian", "buyer": "Halcyon",
             "claimed_quality": "Excellent", "price_per_unit": 50.0},
        ]
        result = build_outcomes_review("Prior memo.", events)
        assert "\u2014" not in result
        assert "\u2013" not in result


class TestRepetitionAwareness:
    def test_empty_log_returns_empty(self):
        result = build_repetition_awareness([], current_day=5)
        assert result == ""

    def test_no_recent_interactions_returns_empty(self):
        log = [{"day": 1, "counterparty": "Halcyon", "type": "message_sent"}]
        result = build_repetition_awareness(log, current_day=10)
        assert result == ""

    def test_shows_counterparty_stats(self):
        log = [
            {"day": 4, "counterparty": "Halcyon", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon", "type": "offer_made"},
        ]
        result = build_repetition_awareness(log, current_day=6)
        assert "RECENT INTERACTION PATTERNS" in result
        assert "Halcyon" in result
        assert "2 msgs sent" in result
        assert "1 offers made" in result

    def test_flags_repetitive_messaging(self):
        log = [
            {"day": 3, "counterparty": "Halcyon", "type": "message_sent"},
            {"day": 4, "counterparty": "Halcyon", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon", "type": "message_sent"},
        ]
        result = build_repetition_awareness(log, current_day=6)
        assert "NOTE:" in result
        assert "without successful transaction" in result
        assert "Halcyon" in result

    def test_no_flag_when_offer_accepted(self):
        log = [
            {"day": 4, "counterparty": "Halcyon", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon", "type": "offer_accepted"},
        ]
        result = build_repetition_awareness(log, current_day=6)
        assert "NOTE:" not in result

    def test_does_not_include_current_day(self):
        log = [
            {"day": 6, "counterparty": "Halcyon", "type": "message_sent"},
        ]
        result = build_repetition_awareness(log, current_day=6)
        assert result == ""

    def test_multiple_counterparties(self):
        log = [
            {"day": 4, "counterparty": "Halcyon", "type": "message_sent"},
            {"day": 4, "counterparty": "Crestline", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon", "type": "message_sent"},
        ]
        result = build_repetition_awareness(log, current_day=6)
        assert "Halcyon" in result
        assert "Crestline" in result

    def test_no_em_dashes(self):
        log = [
            {"day": 4, "counterparty": "Halcyon", "type": "message_sent"},
            {"day": 5, "counterparty": "Halcyon", "type": "message_sent"},
        ]
        result = build_repetition_awareness(log, current_day=6)
        assert "\u2014" not in result
        assert "\u2013" not in result


class TestHelpers:
    def test_estimate_tokens(self):
        assert _estimate_tokens("a" * 400) == 100
        assert _estimate_tokens("") == 0

    def test_truncate_within_budget(self):
        text = "hello world"
        assert _truncate_to_budget(text, 100) == text

    def test_truncate_exceeds_budget(self):
        text = "a" * 1000
        result = _truncate_to_budget(text, 10)  # 10 tokens = 40 chars
        assert len(result) < 200  # truncated + message
        assert "truncated" in result
