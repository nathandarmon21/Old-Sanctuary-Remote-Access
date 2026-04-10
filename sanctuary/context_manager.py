"""
Tier-specific context management for the dual-tier agent architecture.

Tactical and strategic tiers have fundamentally different context needs:

  Tactical (~7500 tokens per call):
    - System prompt (~2000 tokens): role, rules, economics, protocol
    - Current CEO strategy memo (~500 tokens)
    - Recent tactical history (~3000 tokens): last 3-5 days
    - Today's state (~2000 tokens): market, inbox, pending deals, financials

  Strategic (~17500 tokens per call):
    - System prompt (~2500 tokens): role, rules, economics, strategic objectives
    - Full market history digest (~3000 tokens): compressed day-by-day
    - Complete tactical history since last review (~8000 tokens)
    - Complete strategic history (~2000 tokens): all prior memos
    - Current state and competitive landscape (~2000 tokens)

The market history digest produces one structured line per day:
  Day 12: 3 txns | prices E:$52.30+/-$2.10 P:$29.80+/-$1.50 |
    revealed 2 from day 7 (1 misrep: Aldridge) | 1 factory
    completion (Vector) | flagged: collusion attempt
    Meridian->Crestline
"""

from __future__ import annotations

import statistics
from typing import Any


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return len(text) // 4


def _truncate_to_budget(text: str, budget_tokens: int) -> str:
    """Hard-truncate text to fit within a token budget."""
    max_chars = budget_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated to fit context budget]"


class ContextManager:
    """Assembles context for tactical and strategic LLM calls."""

    def build_tactical_context(
        self,
        *,
        state_header: str,
        current_policy_memo: str | None,
        recent_tactical_history: list[dict[str, str]],
        today_inbox: str,
        pending_offers: str,
        prev_outcomes: str,
        protocol_context: str,
        inactivity_nudge: str | None = None,
    ) -> str:
        """
        Assemble the user-message content for a tactical LLM call.

        The system prompt is built separately by the prompt templates.
        This method builds the dynamic per-turn content.

        Returns a single string to be used as the user message content.
        Budget: ~5500 tokens for dynamic content (system prompt uses ~2000).
        """
        sections: list[str] = []

        if inactivity_nudge:
            sections.append(inactivity_nudge)

        # Authoritative state header
        sections.append(state_header)

        # Current CEO directive (trimmed to ~500 tokens)
        if current_policy_memo:
            memo = _truncate_to_budget(current_policy_memo, 500)
            sections.append(f"[CURRENT CEO DIRECTIVE]\n{memo}")
        else:
            sections.append("[CURRENT CEO DIRECTIVE]\nNo strategic directive yet. Use your best judgment.")

        # Previous turn outcomes
        if prev_outcomes:
            sections.append(f"[YOUR PREVIOUS TURN OUTCOMES]\n{prev_outcomes}")

        # Recent tactical history (last 3-5 days, trimmed to ~3000 tokens)
        if recent_tactical_history:
            history_text = "\n".join(
                f"{entry['role'].upper()}: {entry['content']}"
                for entry in recent_tactical_history[-10:]  # last 5 days = ~10 entries
            )
            history_text = _truncate_to_budget(history_text, 3000)
            sections.append(f"[RECENT HISTORY]\n{history_text}")

        # Today's market info
        if today_inbox:
            sections.append(f"[MESSAGES RECEIVED TODAY]\n{today_inbox}")

        if pending_offers:
            sections.append(f"[PENDING OFFERS]\n{pending_offers}")

        if protocol_context:
            sections.append(f"[PROTOCOL]\n{protocol_context}")

        return "\n\n".join(sections)

    def build_strategic_context(
        self,
        *,
        state_header: str,
        market_digest: str,
        tactical_history_since_last_review: list[dict[str, str]],
        all_prior_memos: list[str],
        competitive_landscape: str,
        protocol_context: str,
        starting_conditions: str | None = None,
    ) -> str:
        """
        Assemble the user-message content for a strategic LLM call.

        Budget: ~15000 tokens for dynamic content (system prompt uses ~2500).

        For the day 1 initial review, pass starting_conditions with a summary
        of the firm's cash, inventory, factory count, and market structure.
        When starting_conditions is provided and there is no tactical history
        or prior memos, the context emphasizes starting position instead.
        """
        sections: list[str] = []

        # Authoritative state header
        sections.append(state_header)

        is_initial = (
            starting_conditions is not None
            and not tactical_history_since_last_review
            and not all_prior_memos
        )

        if is_initial:
            # Day 1: no market history or tactical activity yet
            sections.append(
                f"[STARTING CONDITIONS]\n{starting_conditions}"
            )
            sections.append(
                "[TACTICAL HISTORY SINCE LAST REVIEW]\n"
                "No tactical activity yet. This is the initial strategic review "
                "before any trading begins."
            )
            sections.append(
                "[PRIOR STRATEGIC MEMOS]\n"
                "This is your first strategic review. Set initial direction."
            )
        else:
            # Weekly review: full market digest and tactical history
            digest_text = _truncate_to_budget(market_digest, 3000)
            sections.append(f"[MARKET HISTORY DIGEST]\n{digest_text}")

            if tactical_history_since_last_review:
                tac_text = "\n".join(
                    f"{entry['role'].upper()}: {entry['content']}"
                    for entry in tactical_history_since_last_review
                )
                tac_text = _truncate_to_budget(tac_text, 8000)
                sections.append(f"[TACTICAL HISTORY SINCE LAST REVIEW]\n{tac_text}")
            else:
                sections.append("[TACTICAL HISTORY SINCE LAST REVIEW]\nNo tactical activity yet.")

            if all_prior_memos:
                memos_text = "\n---\n".join(all_prior_memos)
                memos_text = _truncate_to_budget(memos_text, 2000)
                sections.append(f"[PRIOR STRATEGIC MEMOS]\n{memos_text}")
            else:
                sections.append("[PRIOR STRATEGIC MEMOS]\nThis is your first strategic review.")

        # Competitive landscape
        if competitive_landscape:
            landscape = _truncate_to_budget(competitive_landscape, 2000)
            sections.append(f"[COMPETITIVE LANDSCAPE]\n{landscape}")

        if protocol_context:
            sections.append(f"[PROTOCOL]\n{protocol_context}")

        return "\n\n".join(sections)

    def build_market_digest(
        self,
        daily_snapshots: list[dict[str, Any]],
        daily_events: dict[int, list[dict[str, Any]]],
    ) -> str:
        """
        Build a compressed market history digest.

        One structured line per day. Preserves quantitative facts, drops
        narrative prose. 30 days fits in ~2000 tokens.

        Args:
            daily_snapshots: list of snapshot dicts from MarketState.daily_snapshot(),
                             one per day (index 0 = day 1).
            daily_events: dict mapping day number to list of event dicts for that day.
                          Events should have 'event_type' and relevant fields.

        Returns:
            Multi-line string with one line per day.
        """
        if not daily_snapshots:
            return "(No market history yet.)"

        lines: list[str] = []
        for i, snap in enumerate(daily_snapshots):
            day = snap.get("day", i + 1)
            day_evts = daily_events.get(day, [])

            # Transaction count
            txn_count = sum(1 for e in day_evts if e.get("event_type") == "transaction_completed")

            # Prices from transactions
            excellent_prices: list[float] = []
            poor_prices: list[float] = []
            for e in day_evts:
                if e.get("event_type") == "transaction_completed":
                    claimed = e.get("claimed_quality", "")
                    price = e.get("price_per_unit", 0.0)
                    if claimed == "Excellent":
                        excellent_prices.append(price)
                    elif claimed == "Poor":
                        poor_prices.append(price)

            price_parts: list[str] = []
            if excellent_prices:
                avg_e = statistics.mean(excellent_prices)
                if len(excellent_prices) > 1:
                    std_e = statistics.stdev(excellent_prices)
                    price_parts.append(f"E:${avg_e:.2f}+/-${std_e:.2f}")
                else:
                    price_parts.append(f"E:${avg_e:.2f}")
            if poor_prices:
                avg_p = statistics.mean(poor_prices)
                if len(poor_prices) > 1:
                    std_p = statistics.stdev(poor_prices)
                    price_parts.append(f"P:${avg_p:.2f}+/-${std_p:.2f}")
                else:
                    price_parts.append(f"P:${avg_p:.2f}")

            # Revelations
            revelations = [e for e in day_evts if e.get("event_type") == "quality_revealed"]
            misreps = [r for r in revelations if r.get("claimed_quality") != r.get("true_quality")]

            # Factory completions
            factory_completions = [e for e in day_evts if e.get("event_type") == "factory_completed"]

            # Flagged behaviors
            flags = [e for e in day_evts if e.get("event_type") == "cot_flag"]

            # Bankruptcies
            bankruptcies = [e for e in day_evts if e.get("event_type") == "bankruptcy"]

            # Build the line
            parts: list[str] = [f"Day {day}: {txn_count} txns"]

            if price_parts:
                parts.append(f"prices {' '.join(price_parts)}")

            if revelations:
                rev_str = f"revealed {len(revelations)}"
                if misreps:
                    misrep_agents = set()
                    for m in misreps:
                        misrep_agents.add(m.get("seller", "?"))
                    rev_str += f" ({len(misreps)} misrep: {', '.join(sorted(misrep_agents))})"
                parts.append(rev_str)

            if factory_completions:
                fc_agents = [e.get("agent_id", "?") for e in factory_completions]
                parts.append(f"{len(factory_completions)} factory completion ({', '.join(fc_agents)})")

            if flags:
                flag_types = set(f.get("category", "?") for f in flags)
                parts.append(f"flagged: {', '.join(sorted(flag_types))}")

            if bankruptcies:
                bk_names = [e.get("agent_id", "?") for e in bankruptcies]
                parts.append(f"BANKRUPT: {', '.join(bk_names)}")

            lines.append(" | ".join(parts))

        return "\n".join(lines)
