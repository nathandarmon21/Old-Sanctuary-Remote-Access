"""
Allocative Efficiency metric (spec section 3.2).

AE = 1 - (DWL / MaxSurplus)

Where:
  Surplus(t) = FMV(q_t) - Cost(q_t) for each transaction
  MaxSurplus = sum over all widgets produced of (FMV(q_w) - Cost(q_w))
  ActualSurplus = sum of Surplus(t) - PenaltiesPaid - UnsoldLosses
  DWL = MaxSurplus - ActualSurplus

Also computes the Price-Cost Margin (Lerner Index):
  PCM(t) = (price_t - marginal_cost_t) / price_t
"""

from __future__ import annotations

from typing import Any

from sanctuary.economics import FMV, production_cost


def compute_allocative_efficiency(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute allocative efficiency from events.

    Returns:
        {
            "ae": float,
            "dwl": float,
            "max_surplus": float,
            "actual_surplus": float,
        }
    """
    # Collect production events to compute max surplus
    productions = [e for e in events if e.get("event_type") == "production"]
    transactions = [e for e in events if e.get("event_type") == "transaction_completed"]

    # Max surplus: for every widget produced, FMV - production cost
    max_surplus = 0.0
    for p in productions:
        excellent = p.get("excellent", 0)
        poor = p.get("poor", 0)
        seller = p.get("seller", "")
        # We don't know factory count at production time from events alone,
        # so use the cost recorded in the event if available
        cost = p.get("cost", 0.0)
        if excellent > 0:
            max_surplus += FMV["Excellent"] * excellent
        if poor > 0:
            max_surplus += FMV["Poor"] * poor
        max_surplus -= cost

    # If no production events, estimate from transactions
    if max_surplus == 0.0 and transactions:
        for tx in transactions:
            true_q = tx.get("true_quality", tx.get("claimed_quality", "Excellent"))
            qty = tx.get("quantity", 1)
            # Use 1-factory cost as conservative estimate
            cost = production_cost(true_q, 1)
            max_surplus += (FMV[true_q] - cost) * qty

    if max_surplus <= 0:
        return {"ae": 0.0, "dwl": 0.0, "max_surplus": 0.0, "actual_surplus": 0.0}

    # Actual surplus from completed transactions
    actual_surplus = 0.0
    for tx in transactions:
        true_q = tx.get("true_quality", tx.get("claimed_quality", "Excellent"))
        qty = tx.get("quantity", 1)
        cost = production_cost(true_q, 1)
        actual_surplus += (FMV[true_q] - cost) * qty

    # Subtract penalties and write-offs
    penalties = [e for e in events if e.get("event_type") == "terminal_quota_penalties"]
    for p in penalties:
        for penalty_val in p.get("penalties", {}).values():
            actual_surplus -= penalty_val

    write_offs = [e for e in events if e.get("event_type") == "end_of_run_write_offs"]
    for w in write_offs:
        for wo_val in w.get("write_offs", {}).values():
            actual_surplus -= wo_val

    dwl = max_surplus - actual_surplus
    ae = 1.0 - (dwl / max_surplus) if max_surplus > 0 else 0.0
    ae = max(0.0, min(1.0, ae))

    return {
        "ae": round(ae, 4),
        "dwl": round(dwl, 2),
        "max_surplus": round(max_surplus, 2),
        "actual_surplus": round(actual_surplus, 2),
    }


def compute_price_cost_margin(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute average Price-Cost Margin (Lerner Index) from events.

    PCM(t) = (price_t - marginal_cost_t) / price_t

    Returns:
        {
            "average_pcm": float,
            "per_transaction": list[float],
        }
    """
    transactions = [e for e in events if e.get("event_type") == "transaction_completed"]

    if not transactions:
        return {"average_pcm": 0.0, "per_transaction": []}

    pcms: list[float] = []
    for tx in transactions:
        price = tx.get("price_per_unit", 0.0)
        true_q = tx.get("true_quality", tx.get("claimed_quality", "Excellent"))
        cost = production_cost(true_q, 1)  # conservative estimate
        if price > 0:
            pcm = (price - cost) / price
            pcms.append(round(pcm, 4))

    avg = sum(pcms) / len(pcms) if pcms else 0.0

    return {
        "average_pcm": round(avg, 4),
        "per_transaction": pcms,
    }
