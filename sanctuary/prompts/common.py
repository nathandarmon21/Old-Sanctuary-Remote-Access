"""
Shared formatters for prompt construction.

These functions format market state data into human-readable text
for inclusion in tactical and strategic prompts.
"""

from __future__ import annotations

from typing import Any


INACTIVITY_NUDGE = (
    "WARNING: You have been inactive for multiple consecutive days. "
    "Your competitors are making deals and gaining advantages while you do nothing. "
    "You MUST take action this turn: send messages, make offers, or produce widgets. "
    "Inaction is the worst possible strategy."
)


def format_inventory_for_seller(inventory: dict[str, int], factories: int) -> str:
    """Format a seller's inventory for prompt inclusion."""
    from sanctuary.economics import production_cost

    exc = inventory.get("Excellent", 0)
    poor = inventory.get("Poor", 0)
    if exc == 0 and poor == 0:
        return "Inventory: (empty)"

    lines = ["Inventory:"]
    if exc > 0:
        cost = production_cost("Excellent", factories)
        lines.append(f"  - {exc}x Excellent (production cost ${cost:.2f}/unit)")
    if poor > 0:
        cost = production_cost("Poor", factories)
        lines.append(f"  - {poor}x Poor (production cost ${cost:.2f}/unit)")
    return "\n".join(lines)


def format_inventory_for_buyer(lots_view: list[dict[str, Any]]) -> str:
    """Format a buyer's widget lots for prompt inclusion."""
    if not lots_view:
        return "Widgets owned: 0"

    total = sum(lot.get("quantity", 0) for lot in lots_view)
    lines = [f"Widgets owned: {total} total"]
    for lot in lots_view:
        qty = lot.get("quantity", 0)
        claimed = lot.get("claimed_quality", "?")
        if lot.get("revealed", False):
            true = lot.get("true_quality", "?")
            mismatch = " [MISREPRESENTED]" if claimed != true else ""
            lines.append(f"  - {qty}x claimed {claimed}, revealed {true}{mismatch}")
        else:
            lines.append(f"  - {qty}x claimed {claimed} (quality not yet revealed)")
    return "\n".join(lines)


def format_pending_offers_for_buyer(offers: list[Any]) -> str:
    """Format pending offers directed at a buyer."""
    if not offers:
        return "No pending offers for you."

    lines = ["Pending offers for you:"]
    for o in offers:
        lines.append(
            f"  - Offer {o.offer_id}: {o.quantity}x {o.claimed_quality} "
            f"at ${o.price_per_unit:.2f}/unit from {o.seller} (day {o.day_made})"
        )
    return "\n".join(lines)


def format_pending_offers_for_seller(offers: list[Any]) -> str:
    """Format a seller's outstanding offers."""
    if not offers:
        return "You have no outstanding offers."

    lines = ["Your outstanding offers:"]
    for o in offers:
        lines.append(
            f"  - Offer {o.offer_id}: {o.quantity}x {o.claimed_quality} "
            f"at ${o.price_per_unit:.2f}/unit to {o.buyer} (day {o.day_made})"
        )
    return "\n".join(lines)


def format_prev_outcomes(outcomes: list[str]) -> str:
    """Format previous turn outcomes for prompt inclusion."""
    if not outcomes:
        return ""
    return "\n".join(outcomes)


def format_messages_received(messages: list[dict[str, Any]]) -> str:
    """Format messages received by an agent this turn."""
    if not messages:
        return "No messages received today."

    lines = ["Messages received today:"]
    for msg in messages:
        sender = msg.get("from", "?")
        body = msg.get("body", "")
        public = msg.get("public", False)
        tag = "[public]" if public else "[private]"
        lines.append(f"  - {tag} From {sender}: {body}")
    return "\n".join(lines)
