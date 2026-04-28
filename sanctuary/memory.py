"""Memory consolidation helpers (spec §7).

Three consolidation surfaces feed into the tactical prompt for long-horizon
runs:

1. digest_recent_memos: a compressed view of the last K strategic memos
   (excluding the most recent, which the tactical prompt already shows
   in full via current_policy).
2. build_metric_ledger: a deterministic per-agent performance summary
   computed from market state + the engine's per-day events log. The
   agent cannot hallucinate this — it's derived from the same state the
   engine actually tracks.
3. build_per_day_summary: a 1-2 sentence factual recap of one day for
   one agent. Injected as "[YESTERDAY'S SUMMARY]" the next day.

All helpers are pure functions: they take read-only inputs and return
strings. The engine wires the right inputs into each tactical call.
"""

from __future__ import annotations

from typing import Any


def digest_recent_memos(policy_history: list, k: int = 10) -> str:
    """Compress the last K strategic memos (excluding the most recent)
    into one short line per memo.

    The most recent memo is omitted because the tactical prompt already
    shows it in full via the `current_policy` framing.
    """
    if not policy_history:
        return ""
    recent = policy_history[-k:]
    if len(recent) <= 1:
        return ""
    lines: list[str] = []
    for rec in recent[:-1]:
        keys = (
            "price_floor_excellent",
            "price_ceiling_excellent",
            "max_price_excellent",
            "quality_stance",
            "urgency",
        )
        kvs = []
        for key in keys:
            if key in rec.policy_json:
                kvs.append(f"{key}={rec.policy_json[key]}")
        notes = rec.policy_json.get("notes", "")
        if isinstance(notes, str) and len(notes) > 80:
            notes = notes[:77] + "..."
        kv_part = ", ".join(kvs) if kvs else ""
        notes_part = f" — {notes}" if notes else ""
        lines.append(f"- Week {rec.week} (day {rec.day}): {kv_part}{notes_part}")
    return "\n".join(lines)


def build_per_day_summary(
    name: str,
    day: int,
    daily_events: list[dict[str, Any]],
) -> str:
    """One-or-two sentence factual recap of `day` for agent `name`.

    Walks the engine's `_daily_events[day]` list (events written during
    that day's tick) and summarizes counts and notable transactions.
    """
    deals: list[str] = []
    n_msgs_received = 0
    n_offers_placed = 0
    n_offers_accepted = 0
    quality_revealed: list[str] = []

    for e in daily_events:
        et = e.get("event_type")
        if et == "transaction_completed":
            if e.get("seller") == name:
                deals.append(
                    f"sold {e['quantity']}x {e['claimed_quality']} to "
                    f"{e['buyer']} @${e['price_per_unit']:.2f}"
                )
                n_offers_accepted += 1
            elif e.get("buyer") == name:
                deals.append(
                    f"bought {e['quantity']}x {e['claimed_quality']} from "
                    f"{e['seller']} @${e['price_per_unit']:.2f}"
                )
                n_offers_accepted += 1
        elif et == "transaction_proposed":
            if e.get("seller") == name or e.get("buyer") == name:
                n_offers_placed += 1
        elif et == "message_sent":
            if e.get("public") or e.get("to_agent") == name:
                if e.get("from_agent") != name:
                    n_msgs_received += 1
        elif et == "quality_revealed":
            if e.get("seller") == name or e.get("buyer") == name:
                if e.get("misrepresented"):
                    quality_revealed.append(
                        f"revelation on tx {e['transaction_id']}: "
                        f"claimed {e['claimed_quality']}, true {e['true_quality']}"
                    )

    if not deals and not n_msgs_received and not n_offers_placed and not quality_revealed:
        return f"Day {day}: no activity."

    parts: list[str] = []
    if deals:
        parts.append("transactions: " + "; ".join(deals))
    elif n_offers_placed:
        parts.append(f"placed {n_offers_placed} offer(s), none closed")
    if quality_revealed:
        parts.append("; ".join(quality_revealed))
    if n_msgs_received:
        parts.append(f"received {n_msgs_received} message(s)")

    return f"Day {day}: " + "; ".join(parts) + "."


def build_metric_ledger(
    name: str,
    is_seller: bool,
    state,
    transactions: list,
    daily_events: dict[int, list[dict[str, Any]]],
    current_day: int,
) -> str:
    """Build a deterministic per-agent performance ledger (spec §7).

    Inputs:
        name: agent name.
        is_seller: True if seller, False if buyer.
        state: SellerState or BuyerState — for cash and starting_cash.
        transactions: list of TransactionRecord (closed deals).
        daily_events: dict[day -> list of event dicts] from the engine.
        current_day: current simulation day (for the Days header).

    Returns a multi-line string suitable for direct injection into the
    tactical prompt. Empty if state is None.
    """
    if state is None:
        return ""

    starting = float(getattr(state, "starting_cash", 0.0))
    current = float(state.cash)

    offers_placed = 0
    offers_declined = 0
    for d in sorted(daily_events.keys()):
        for e in daily_events[d]:
            et = e.get("event_type")
            if et == "transaction_proposed":
                if is_seller and e.get("seller") == name:
                    offers_placed += 1
                elif (not is_seller) and e.get("buyer") == name:
                    offers_placed += 1
            elif et == "offer_declined":
                if e.get("agent_id") == name:
                    offers_declined += 1

    # Closed deals from market.transactions.
    my_txs = [
        tx for tx in transactions
        if (is_seller and tx.seller == name) or ((not is_seller) and tx.buyer == name)
    ]
    closed = len(my_txs)
    avg_price = (
        sum(tx.price_per_unit for tx in my_txs) / closed if closed else 0.0
    )

    delta = current - starting
    delta_sign = "+" if delta >= 0 else "-"
    lines = [
        f"Days simulated: {current_day}",
        f"Cash: ${current:,.2f} (started ${starting:,.2f}, "
        f"change {delta_sign}${abs(delta):,.2f})",
        f"Offers placed: {offers_placed}, closed: {closed}, declined: {offers_declined}",
    ]
    if closed:
        lines.append(f"Closed-deal avg price: ${avg_price:.2f}")

    # Per-week summary (last 4 weeks max).
    weeks: dict[int, dict[str, Any]] = {}
    for tx in my_txs:
        wk = (tx.day - 1) // 7 + 1
        w = weeks.setdefault(wk, {"closed": 0, "prices": []})
        w["closed"] += 1
        w["prices"].append(tx.price_per_unit)
    if weeks:
        lines.append("Recent weeks:")
        for wk in sorted(weeks.keys())[-4:]:
            w = weeks[wk]
            prices = w["prices"]
            lines.append(
                f"  Week {wk}: {w['closed']} deal(s), "
                f"price ${min(prices):.2f}-${max(prices):.2f} "
                f"(avg ${sum(prices)/len(prices):.2f})"
            )

    return "\n".join(lines)
