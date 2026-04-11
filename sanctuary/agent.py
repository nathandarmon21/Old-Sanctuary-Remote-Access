"""
Agent module for the Sanctuary simulation.

Each agent wraps one company (seller or buyer) and owns:
  - Its full conversation history (never trimmed)
  - Its policy history (one entry per strategic call)
  - The dual-tier dispatch logic (strategic / tactical)
  - Action parsing from LLM completions

Prompt construction pulls data through MarketState.view_inventory_for()
to enforce the information asymmetry invariant. The agent never accesses
internal market state directly.

Action parsing is intentionally robust: a missing or malformed <actions>
or <policy> tag logs a parse_error event and returns a no-op action set,
rather than crashing the run.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sanctuary.market import MarketState, PendingOffer
from sanctuary.messaging import MessageRouter
from sanctuary.providers.base import ModelProvider, ModelResponse
from sanctuary.prompts.common import (
    INACTIVITY_NUDGE,
    format_inventory_for_buyer,
    format_inventory_for_seller,
    format_pending_offers_for_buyer,
    format_pending_offers_for_seller,
    format_prev_outcomes,
)
from sanctuary.context_manager import build_outcomes_review, build_repetition_awareness
from sanctuary.prompts.sub_round import SUB_ROUND_PROMPT


# ── Parsed action types ───────────────────────────────────────────────────────

@dataclass
class ParsedMessage:
    to: str
    public: bool
    body: str


@dataclass
class ParsedOffer:
    """An offer a seller wants to place."""
    to: str
    qty: int
    claimed_quality: str
    quality_to_send: str   # defaults to claimed_quality if not specified
    price_per_unit: float


@dataclass
class ParsedBuyerOffer:
    """An offer a buyer wants to place (counter-offer to a seller)."""
    to: str
    qty: int
    claimed_quality: str   # quality the buyer is asking for
    price_per_unit: float


@dataclass
class TacticalActions:
    """Parsed output of a tactical LLM call."""
    messages: list[ParsedMessage] = field(default_factory=list)
    seller_offers: list[ParsedOffer] = field(default_factory=list)
    buyer_offers: list[ParsedBuyerOffer] = field(default_factory=list)
    accept_offers: list[str] = field(default_factory=list)
    decline_offers: list[str] = field(default_factory=list)
    # Seller-only
    produce_excellent: int = 0
    produce_poor: int = 0
    build_factory: bool = False
    # Buyer-only
    produce_final_goods: int = 0
    # Metadata: parse_error means hard failure (no action taken).
    # parse_recovery means soft fallback succeeded (action taken, but format was wrong).
    parse_error: str | None = None
    parse_recovery: str | None = None


@dataclass
class SubRoundActions:
    """Parsed output of a sub-round (offer-response-only) call."""
    accept_offers: list[str] = field(default_factory=list)
    decline_offers: list[str] = field(default_factory=list)
    parse_error: str | None = None
    parse_recovery: str | None = None


@dataclass
class PolicyRecord:
    """One strategic policy, as produced by a strategic-tier call."""
    day: int
    week: int
    raw_memo: str       # full LLM completion (the memo text)
    policy_json: dict   # parsed <policy> block
    model_response: ModelResponse


# ── Agent class ───────────────────────────────────────────────────────────────

class Agent:
    """
    Wraps one company (seller or buyer) and drives its LLM-based decision making.

    The agent owns its conversation history. The simulation loop calls
    strategic_call() on strategic days and tactical_call() every day.
    """

    def __init__(
        self,
        name: str,
        role: str,                      # "seller" or "buyer"
        strategic_provider: ModelProvider,
        tactical_provider: ModelProvider,
        strategic_max_tokens: int = 2048,
        tactical_max_tokens: int = 1024,
        days_total: int = 30,
        seller_names: list[str] | None = None,
        buyer_names: list[str] | None = None,
    ) -> None:
        if role not in ("seller", "buyer"):
            raise ValueError(f"role must be 'seller' or 'buyer', got {role!r}")

        self.name = name
        self.role = role
        self.seller_names = seller_names or []
        self.buyer_names = buyer_names or []
        self._strategic_provider = strategic_provider
        self._tactical_provider = tactical_provider
        self._strategic_max_tokens = strategic_max_tokens
        self._tactical_max_tokens = tactical_max_tokens
        self.days_total = days_total

        # Separate conversation histories per tier
        self.tactical_history: list[dict[str, str]] = []
        self.strategic_history: list[dict[str, str]] = []
        # Unified history kept for backward compatibility during transition
        self.history: list[dict[str, str]] = []
        self.policy_history: list[PolicyRecord] = []
        self.current_policy: PolicyRecord | None = None

        # Interaction tracking for repetition awareness
        self.interaction_log: list[dict[str, Any]] = []

        # Call counters
        self.tactical_call_count: int = 0
        self.strategic_call_count: int = 0

    def record_interaction(self, day: int, counterparty: str, interaction_type: str) -> None:
        """Record an interaction event for repetition awareness tracking."""
        self.interaction_log.append({
            "day": day,
            "counterparty": counterparty,
            "type": interaction_type,
        })

    @property
    def is_seller(self) -> bool:
        return self.role == "seller"

    @property
    def is_buyer(self) -> bool:
        return self.role == "buyer"

    # ── Strategic tier ───────────────────────────────────────────────────────

    def strategic_call(
        self,
        day: int,
        week: int,
        market: MarketState,
        market_summary: str,
        transaction_summary: str,
        events_since_last_review: list[dict[str, Any]] | None = None,
    ) -> tuple[PolicyRecord, ModelResponse]:
        """
        Fire the strategic-tier LLM call for this agent.

        Builds the full strategic prompt, runs inference, parses the <policy>
        block, appends the exchange to conversation history, and stores the
        policy for use by tactical calls this week.

        Returns (PolicyRecord, ModelResponse) for logging.
        """
        system_prompt = self._build_strategic_system_prompt(
            day=day,
            week=week,
            market=market,
            market_summary=market_summary,
            transaction_summary=transaction_summary,
        )

        # The user turn for a strategic call is just the prompt itself.
        # We pass the full history so the model has context.
        user_content = (
            f"Day {day} -- Week {week} strategic planning session. "
            f"Please write your strategic memo and <policy> block."
        )

        # For day 7+ reviews, add outcome comparison
        if day >= 7 and self.current_policy is not None:
            outcomes_text = build_outcomes_review(
                prior_memo=self.current_policy.raw_memo,
                events_since_last_review=events_since_last_review or [],
            )
            if outcomes_text:
                user_content = f"{outcomes_text}\n\n{user_content}"

        self.history.append({"role": "user", "content": user_content})
        self.strategic_history.append({"role": "user", "content": user_content})

        response = self._strategic_provider.complete(
            system_prompt=system_prompt,
            history=self.history,
            max_tokens=self._strategic_max_tokens,
        )

        self.history.append({"role": "assistant", "content": response.completion})
        self.strategic_history.append({"role": "assistant", "content": response.completion})
        self.strategic_call_count += 1

        policy_json = _parse_policy_block(response.completion)
        record = PolicyRecord(
            day=day,
            week=week,
            raw_memo=response.completion,
            policy_json=policy_json,
            model_response=response,
        )
        self.policy_history.append(record)
        self.current_policy = record
        return record, response

    # ── Tactical tier ────────────────────────────────────────────────────────

    def tactical_call(
        self,
        day: int,
        market: MarketState,
        router: MessageRouter,
        pending_offers_for_me: list[PendingOffer],
        my_pending_offers: list[PendingOffer],
        inactivity_days: int = 0,
        prev_outcomes: list[str] | None = None,
    ) -> tuple[TacticalActions, ModelResponse]:
        """
        Fire the tactical-tier LLM call for this agent.

        Returns (TacticalActions, ModelResponse) for execution and logging.
        """
        system_prompt = self._build_tactical_system_prompt(
            day=day,
            market=market,
            router=router,
            pending_offers_for_me=pending_offers_for_me,
            my_pending_offers=my_pending_offers,
            inactivity_days=inactivity_days,
            prev_outcomes=prev_outcomes or [],
        )

        # Build user message with contextual sections
        sections: list[str] = []

        # Buyer quota urgency header (at the top)
        if self.is_buyer:
            from sanctuary.economics import (
                BUYER_DAILY_QUOTA_PENALTY as _bdqp,
                BUYER_TERMINAL_QUOTA_PENALTY as _btqp,
                BUYER_WIDGET_QUOTA as _bwq,
            )
            from sanctuary.prompts.tactical import build_buyer_quota_urgency_header
            buyer_state = market.buyers.get(self.name)
            if buyer_state:
                quota_remaining = max(0, _bwq - buyer_state.widgets_acquired)
                days_remaining = max(0, self.days_total - day)
                sections.append(build_buyer_quota_urgency_header(
                    days_remaining=days_remaining,
                    days_total=self.days_total,
                    quota_remaining=quota_remaining,
                    original_quota=_bwq,
                    terminal_penalty_per_unit=_btqp,
                    daily_penalty_per_unit=_bdqp,
                ))

        # Repetition awareness
        rep_awareness = build_repetition_awareness(self.interaction_log, day)
        if rep_awareness:
            sections.append(rep_awareness)

        sections.append(f"Day {day} -- please make your tactical decisions.")
        user_content = "\n\n".join(sections)
        self.history.append({"role": "user", "content": user_content})
        self.tactical_history.append({"role": "user", "content": user_content})

        response = self._tactical_provider.complete(
            system_prompt=system_prompt,
            history=self.history,
            max_tokens=self._tactical_max_tokens,
        )

        self.history.append({"role": "assistant", "content": response.completion})
        self.tactical_history.append({"role": "assistant", "content": response.completion})
        self.tactical_call_count += 1

        actions = _parse_tactical_actions(response.completion, agent_role=self.role)
        return actions, response

    def sub_round_call(
        self,
        day: int,
        sub_round: int,
        pending_offers_for_me: list[PendingOffer],
        market: MarketState,
    ) -> tuple[SubRoundActions, ModelResponse]:
        """
        Fire a sub-round call for offer response only.
        Appends to conversation history so the agent has full context.
        """
        policy_summary = (
            self.current_policy.policy_json.get("notes", "No policy summary available.")
            if self.current_policy
            else "No strategic policy yet."
        )

        prompt = SUB_ROUND_PROMPT.format(
            company_name=self.name,
            sub_round=sub_round,
            day=day,
            pending_offers=format_pending_offers_for_buyer(pending_offers_for_me),
            policy_summary=policy_summary,
        )

        user_content = (
            f"Day {day} sub-round {sub_round} -- "
            f"please respond to the pending offers."
        )
        self.history.append({"role": "user", "content": user_content})
        self.tactical_history.append({"role": "user", "content": user_content})

        response = self._tactical_provider.complete(
            system_prompt=prompt,
            history=self.history,
            max_tokens=600,
        )

        self.history.append({"role": "assistant", "content": response.completion})
        self.tactical_history.append({"role": "assistant", "content": response.completion})
        self.tactical_call_count += 1

        actions = _parse_sub_round_actions(response.completion)
        return actions, response

    # ── Prompt construction ──────────────────────────────────────────────────

    def _build_strategic_system_prompt(
        self,
        day: int,
        week: int,
        market: MarketState,
        market_summary: str,
        transaction_summary: str,
    ) -> str:
        from sanctuary.economics import (
            BUYER_DAILY_QUOTA_PENALTY,
            BUYER_TERMINAL_QUOTA_PENALTY,
            BUYER_WIDGET_QUOTA,
            FACTORY_BUILD_COST,
            FACTORY_BUILD_DAYS,
            FMV,
            REVELATION_LAG_DAYS,
            production_cost,
        )
        from sanctuary.prompts.strategic import (
            build_buyer_strategic_system,
            build_seller_strategic_system,
        )

        if self.is_seller:
            return build_seller_strategic_system(
                company_name=self.name,
                days_total=self.days_total,
                day=day,
                fmv_excellent=FMV["Excellent"],
                fmv_poor=FMV["Poor"],
                cost_1e=production_cost("Excellent", 1),
                cost_1p=production_cost("Poor", 1),
                cost_2e=production_cost("Excellent", 2),
                cost_2p=production_cost("Poor", 2),
                cost_3e=production_cost("Excellent", 3),
                cost_3p=production_cost("Poor", 3),
                cost_4e=production_cost("Excellent", 4),
                cost_4p=production_cost("Poor", 4),
                factory_cost=FACTORY_BUILD_COST,
                factory_days=FACTORY_BUILD_DAYS,
                revelation_days=REVELATION_LAG_DAYS,
                buyer_quota=BUYER_WIDGET_QUOTA,
                seller_names=self.seller_names,
                buyer_names=self.buyer_names,
            )
        else:
            return build_buyer_strategic_system(
                company_name=self.name,
                days_total=self.days_total,
                day=day,
                widget_quota=BUYER_WIDGET_QUOTA,
                daily_penalty=BUYER_DAILY_QUOTA_PENALTY,
                terminal_penalty=BUYER_TERMINAL_QUOTA_PENALTY,
                fmv_excellent=FMV["Excellent"],
                fmv_poor=FMV["Poor"],
                revelation_days=REVELATION_LAG_DAYS,
                seller_names=self.seller_names,
                buyer_names=self.buyer_names,
            )

    def _build_tactical_system_prompt(
        self,
        day: int,
        market: MarketState,
        router: MessageRouter,
        pending_offers_for_me: list[PendingOffer],
        my_pending_offers: list[PendingOffer],
        inactivity_days: int,
        prev_outcomes: list[str] | None = None,
    ) -> str:
        from sanctuary.economics import (
            BUYER_DAILY_QUOTA_PENALTY,
            BUYER_MAX_DAILY_PRODUCTION,
            BUYER_TERMINAL_QUOTA_PENALTY,
            BUYER_WIDGET_QUOTA,
            FACTORY_BUILD_COST,
            FACTORY_BUILD_DAYS,
            FMV,
            REVELATION_LAG_DAYS,
        )
        from sanctuary.prompts.tactical import (
            build_buyer_tactical_system,
            build_seller_tactical_system,
        )

        # Collect pending offer IDs relevant to this agent
        offer_ids = []
        for o in pending_offers_for_me:
            offer_ids.append(o.offer_id)
        for o in my_pending_offers:
            offer_ids.append(o.offer_id)

        if self.is_seller:
            return build_seller_tactical_system(
                company_name=self.name,
                days_total=self.days_total,
                factory_cost=FACTORY_BUILD_COST,
                factory_days=FACTORY_BUILD_DAYS,
                revelation_days=REVELATION_LAG_DAYS,
                seller_names=self.seller_names,
                buyer_names=self.buyer_names,
                pending_offer_ids=offer_ids,
            )
        else:
            return build_buyer_tactical_system(
                company_name=self.name,
                days_total=self.days_total,
                widget_quota=BUYER_WIDGET_QUOTA,
                daily_penalty=BUYER_DAILY_QUOTA_PENALTY,
                terminal_penalty=BUYER_TERMINAL_QUOTA_PENALTY,
                revelation_days=REVELATION_LAG_DAYS,
                fmv_excellent=FMV["Excellent"],
                fmv_poor=FMV["Poor"],
                daily_prod_cap=BUYER_MAX_DAILY_PRODUCTION,
                seller_names=self.seller_names,
                buyer_names=self.buyer_names,
                pending_offer_ids=offer_ids,
            )

    def _format_policy_history(self) -> str:
        if not self.policy_history:
            return "(no prior strategic memos)"
        parts = []
        for rec in self.policy_history:
            parts.append(
                f"--- Week {rec.week} / Day {rec.day} ---\n{rec.raw_memo}"
            )
        return "\n\n".join(parts)


# ── Parsing helpers ───────────────────────────────────────────────────────────

# Canonical field sets used for step-3 recovery (any one present = likely correct block)
_POLICY_FIELDS = frozenset([
    "excellent_price_floor", "max_price_excellent", "honesty_stance",
    "daily_excellent_target", "daily_production_target",
])
_ACTIONS_FIELDS = frozenset([
    "messages", "offers", "accept_offers", "produce", "produce_final_goods",
])
_SUB_ROUND_FIELDS = frozenset(["accept_offers", "decline_offers"])


def _find_all_json_objects(text: str) -> list[str]:
    """
    Find all top-level {...} blocks in text, respecting string literals.
    Returns them in document order. Each element is the raw substring.
    """
    objects: list[str] = []
    depth = 0
    start = -1
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and in_string:
            i += 2  # skip escaped character
            continue
        if c == '"':
            in_string = not in_string
        elif not in_string:
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(text[start : i + 1])
                    start = -1
        i += 1
    return objects


def _extract_json_robust(
    text: str,
    tag: str,
    required_fields: frozenset[str] | None = None,
) -> tuple[dict | None, str | None]:
    """
    Extract a JSON object from an LLM response using a 4-step fallback.

    Step 1: Strict — parse content between <tag>...</tag> tags.
    Step 2: Last balanced {...} block in the response.
    Step 3: Any {...} block containing one of required_fields.
    Step 4: Hard failure — return (None, error_message).

    Returns (data, note):
      (dict, None)  → clean parse (step 1)
      (dict, str)   → recovered (steps 2–3); note describes fallback used
      (None, str)   → hard failure; note has the error message
    """
    tag_re = re.compile(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        re.DOTALL | re.IGNORECASE,
    )

    # Step 1: strict tag extraction
    m = tag_re.search(text)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                return data, None
        except json.JSONDecodeError:
            pass  # tag found but contents invalid — fall through

    all_blocks = _find_all_json_objects(text)

    # Step 2: last balanced {...} block
    if all_blocks:
        try:
            data = json.loads(all_blocks[-1])
            if isinstance(data, dict):
                return data, f"recovered: no <{tag}> tag — used last JSON block"
        except json.JSONDecodeError:
            pass

    # Step 3: any block with a recognisable field
    if required_fields and len(all_blocks) > 1:
        for block in reversed(all_blocks[:-1]):
            try:
                data = json.loads(block)
                if isinstance(data, dict) and (required_fields & data.keys()):
                    return data, f"recovered: used JSON block with known fields"
            except json.JSONDecodeError:
                pass

    snippet = text[:300].replace("\n", " ")
    return None, (
        f"No parseable JSON found (tried {len(all_blocks)} block(s)). "
        f"Response snippet: {snippet!r}"
    )


def _parse_policy_block(text: str) -> dict:
    """
    Extract and parse the <policy>...</policy> JSON block from an LLM completion.

    Uses 4-step robust extraction. On hard failure returns a dict with a
    "parse_error" key. On soft recovery adds a "_parse_recovery" note.
    """
    data, note = _extract_json_robust(text, "policy", _POLICY_FIELDS)
    if data is None:
        return {"parse_error": note}
    if note:
        data["_parse_recovery"] = note
    return data


def _parse_tactical_actions(text: str, agent_role: str) -> TacticalActions:
    """
    Extract and parse the <actions>...</actions> JSON block from a tactical completion.

    Uses 4-step robust extraction. On hard failure returns a no-op TacticalActions
    with parse_error set.
    """
    data, note = _extract_json_robust(text, "actions", _ACTIONS_FIELDS)
    if data is None:
        return TacticalActions(parse_error=note)

    actions = TacticalActions(parse_recovery=note)  # note is None on clean parse

    # Messages
    for m in data.get("messages", []):
        try:
            actions.messages.append(ParsedMessage(
                to=str(m.get("to", "all")),
                public=bool(m.get("public", True)),
                body=str(m.get("body", "")),
            ))
        except (KeyError, TypeError):
            pass

    # Accept / decline
    actions.accept_offers = [str(x) for x in data.get("accept_offers", [])]
    actions.decline_offers = [str(x) for x in data.get("decline_offers", [])]

    if agent_role == "seller":
        # Offers to place
        for o in data.get("offers", []):
            try:
                cq = str(o.get("claimed_quality", "Excellent"))
                actions.seller_offers.append(ParsedOffer(
                    to=str(o["to"]),
                    qty=int(o.get("qty", 0)),
                    claimed_quality=cq,
                    quality_to_send=str(o.get("quality_to_send", cq)),
                    price_per_unit=float(o.get("price_per_unit", 0.0)),
                ))
            except (KeyError, TypeError, ValueError):
                pass

        # Production
        produce = data.get("produce", {})
        actions.produce_excellent = int(produce.get("excellent", 0))
        actions.produce_poor = int(produce.get("poor", 0))
        actions.build_factory = bool(data.get("build_factory", False))

    else:  # buyer
        # Counter-offers to sellers
        for o in data.get("offers", []):
            try:
                actions.buyer_offers.append(ParsedBuyerOffer(
                    to=str(o["to"]),
                    qty=int(o.get("qty", 0)),
                    claimed_quality=str(o.get("claimed_quality", "Excellent")),
                    price_per_unit=float(o.get("price_per_unit", 0.0)),
                ))
            except (KeyError, TypeError, ValueError):
                pass

        actions.produce_final_goods = int(data.get("produce_final_goods", 0))

    return actions


def _parse_sub_round_actions(text: str) -> SubRoundActions:
    """Parse the <actions> block from a sub-round completion (accept/decline only)."""
    data, note = _extract_json_robust(text, "actions", _SUB_ROUND_FIELDS)
    if data is None:
        return SubRoundActions(parse_error=note)
    return SubRoundActions(
        accept_offers=[str(x) for x in data.get("accept_offers", [])],
        decline_offers=[str(x) for x in data.get("decline_offers", [])],
        parse_recovery=note,  # None on clean parse
    )


# ── Market state formatting for prompts ──────────────────────────────────────

def _format_market_state(market: MarketState, viewer: str) -> str:
    """Build a readable market state summary for a tactical prompt."""
    lines = [
        f"  Final-good prices: Excellent → ${market.fg_prices['Excellent']:.2f} | "
        f"Poor → ${market.fg_prices['Poor']:.2f}",
        "",
        "  Seller status:",
    ]
    for name, seller in market.sellers.items():
        if seller.bankrupt:
            lines.append(f"    {name}: BANKRUPT")
        else:
            # Buyers see seller inventory as unknown; sellers can see their own
            if name == viewer:
                inv_str = (
                    f"Excellent={seller.inventory.get('Excellent', 0)}, "
                    f"Poor={seller.inventory.get('Poor', 0)}"
                )
            else:
                inv_str = "(inventory unknown)"
            lines.append(
                f"    {name}: cash=${seller.cash:.0f}, "
                f"factories={seller.factories}, inventory={inv_str}"
            )

    from sanctuary.economics import BUYER_WIDGET_QUOTA
    lines.append("")
    lines.append("  Buyer status (quota=30 widgets each):")
    for name, buyer in market.buyers.items():
        if buyer.bankrupt:
            lines.append(f"    {name}: BANKRUPT")
        else:
            total_widgets = sum(lot.quantity_remaining for lot in buyer.widget_lots)
            quota_remaining = max(0, BUYER_WIDGET_QUOTA - buyer.widgets_acquired)
            lines.append(
                f"    {name}: cash=${buyer.cash:.0f}, "
                f"acquired={buyer.widgets_acquired}/30 (need {quota_remaining} more), "
                f"on_hand={total_widgets}"
            )

    return "\n".join(lines)


def _format_seller_inv_summary(inv_view: dict) -> str:
    excellent = inv_view.get("excellent", 0)
    poor = inv_view.get("poor", 0)
    return f"Excellent: {excellent}, Poor: {poor}"


def _format_incoming(
    messages: list,
    pending_offers: list[PendingOffer],
    agent_role: str,
) -> str:
    """Format incoming messages and offers for a tactical prompt."""
    parts: list[str] = []

    if messages:
        parts.append("Messages received today:")
        for m in messages:
            prefix = "[PUBLIC]" if m.is_public else "[PRIVATE]"
            parts.append(f"  {prefix} From {m.sender}: {m.content}")
    else:
        parts.append("  (no messages received today)")

    parts.append("")

    if pending_offers:
        parts.append("Offers awaiting your response:")
        if agent_role == "buyer":
            parts.append(format_pending_offers_for_buyer(pending_offers))
        else:
            parts.append(format_pending_offers_for_seller(pending_offers))
    else:
        parts.append("  (no pending offers)")

    return "\n".join(parts)
