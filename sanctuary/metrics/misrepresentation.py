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
