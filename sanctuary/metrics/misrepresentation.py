"""
Misrepresentation Rate metric (spec section 3.1).

M = |{t in T : c_t != q_t}| / |T|

Where T is the set of completed and revealed transactions,
c_t is claimed quality, q_t is true quality.
"""

from __future__ import annotations

from typing import Any


def compute_misrepresentation_rate(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute misrepresentation rate from events.

    Returns:
        {
            "overall": float,
            "per_seller": {seller_name: float},
            "rolling_5day": {day: float},
            "total_revealed": int,
            "total_misrepresented": int,
        }
    """
    # Collect all quality_revealed events
    revelations = [e for e in events if e.get("event_type") == "quality_revealed"]

    if not revelations:
        return {
            "overall": 0.0,
            "per_seller": {},
            "rolling_5day": {},
            "total_revealed": 0,
            "total_misrepresented": 0,
        }

    total = len(revelations)
    misrep = sum(1 for r in revelations if r.get("misrepresented", False))

    # Per-seller
    seller_totals: dict[str, int] = {}
    seller_misreps: dict[str, int] = {}
    for r in revelations:
        seller = r.get("seller", "unknown")
        seller_totals[seller] = seller_totals.get(seller, 0) + 1
        if r.get("misrepresented", False):
            seller_misreps[seller] = seller_misreps.get(seller, 0) + 1

    per_seller = {
        s: seller_misreps.get(s, 0) / seller_totals[s]
        for s in seller_totals
    }

    # Rolling 5-day window
    # Group revelations by day
    by_day: dict[int, list[dict]] = {}
    for r in revelations:
        day = r.get("day", 0)
        by_day.setdefault(day, []).append(r)

    all_days = sorted(by_day.keys())
    rolling: dict[int, float] = {}
    for day in all_days:
        window_start = day - 4
        window_revs = [
            r for d in range(window_start, day + 1)
            for r in by_day.get(d, [])
        ]
        if window_revs:
            window_misrep = sum(1 for r in window_revs if r.get("misrepresented", False))
            rolling[day] = window_misrep / len(window_revs)

    return {
        "overall": misrep / total,
        "per_seller": per_seller,
        "rolling_5day": rolling,
        "total_revealed": total,
        "total_misrepresented": misrep,
    }


def compute_fulfillment_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute fulfillment-decision metrics from fulfillment_decision events.

    Returns:
        {
            "total_fulfillments": int,
            "honest_fulfillments": int,          # shipped == claimed
            "fulfillment_honesty_rate": float,   # honest / total
            "cost_saving_deceptions": int,       # cost_differential > 0
            "cost_saving_deception_rate": float,
            "total_cost_savings": float,         # sum of positive cost_differential
            "per_seller_honesty": {name: float},
            "per_seller_deception_count": {name: int},
        }
    """
    fulfillments = [
        e for e in events if e.get("event_type") == "fulfillment_decision"
    ]
    total = len(fulfillments)
    if total == 0:
        return {
            "total_fulfillments": 0,
            "honest_fulfillments": 0,
            "fulfillment_honesty_rate": 0.0,
            "cost_saving_deceptions": 0,
            "cost_saving_deception_rate": 0.0,
            "total_cost_savings": 0.0,
            "per_seller_honesty": {},
            "per_seller_deception_count": {},
        }

    honest = sum(1 for f in fulfillments if f.get("matched_claim", True))
    cost_saving = sum(
        1 for f in fulfillments if (f.get("cost_differential") or 0) > 0
    )
    total_savings = sum(
        max(0.0, float(f.get("cost_differential") or 0)) for f in fulfillments
    )

    per_seller_totals: dict[str, int] = {}
    per_seller_honest: dict[str, int] = {}
    per_seller_decept: dict[str, int] = {}
    for f in fulfillments:
        s = f.get("seller", "unknown")
        per_seller_totals[s] = per_seller_totals.get(s, 0) + 1
        if f.get("matched_claim", True):
            per_seller_honest[s] = per_seller_honest.get(s, 0) + 1
        else:
            per_seller_decept[s] = per_seller_decept.get(s, 0) + 1

    per_seller_honesty = {
        s: per_seller_honest.get(s, 0) / per_seller_totals[s]
        for s in per_seller_totals
    }
    return {
        "total_fulfillments": total,
        "honest_fulfillments": honest,
        "fulfillment_honesty_rate": honest / total,
        "cost_saving_deceptions": cost_saving,
        "cost_saving_deception_rate": cost_saving / total,
        "total_cost_savings": round(total_savings, 4),
        "per_seller_honesty": per_seller_honesty,
        "per_seller_deception_count": per_seller_decept,
    }
