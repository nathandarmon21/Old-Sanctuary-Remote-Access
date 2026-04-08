"""
Pure economic functions for the Sanctuary simulation.

All functions here are stateless and deterministic — no side effects,
no external dependencies. This makes them straightforward to test and
to reason about formally.

The formula for per-unit production cost with economies of scale:

    cost(quality, factories) = base_cost(quality) × max(0.76, 1 − 0.08 × (factories − 1))

The floor of 0.76 is reached at exactly 4 factories (1 − 0.08 × 3 = 0.76).
With a 30-day game and a $1,500 / 2-day factory build, this makes the
cost-reduction curve meaningful: a seller who commits early can reach
minimum cost by day ~10, creating a real strategic choice.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Quality = Literal["Excellent", "Poor"]

# ── Base constants ────────────────────────────────────────────────────────────

BASE_PRODUCTION_COSTS: dict[str, float] = {
    "Excellent": 25.0,
    "Poor": 15.0,
}

HOLDING_COSTS_PER_DAY: dict[str, float] = {
    "Excellent": 0.125,   # 0.5% of $25 production cost
    "Poor": 0.075,        # 0.5% of $15 production cost
}

FINAL_GOOD_BASE_PRICES: dict[str, float] = {
    "Excellent": 90.0,
    "Poor": 52.0,
}

COST_MULTIPLIER_FLOOR: float = 0.76
FACTORY_BUILD_COST: float = 1_500.0
FACTORY_BUILD_DAYS: int = 2          # days until a new factory is operational
BANKRUPTCY_THRESHOLD: float = -3_000.0
BUYER_MAX_DAILY_PRODUCTION: int = 3  # final goods per buyer per day
PRICE_WALK_SIGMA: float = 1.0        # Brownian step size (dollars/day)

# ── Buyer quota constants ─────────────────────────────────────────────────────
# Buyers must acquire BUYER_WIDGET_QUOTA widgets over the run.
# Not meeting this quota costs BUYER_DAILY_QUOTA_PENALTY per unfulfilled unit
# per day, plus BUYER_TERMINAL_QUOTA_PENALTY per unit left at end of day 30.
BUYER_WIDGET_QUOTA: int = 30
BUYER_DAILY_QUOTA_PENALTY: float = 2.0    # $/day per unfulfilled quota unit
BUYER_TERMINAL_QUOTA_PENALTY: float = 60.0  # $/unit at end of run


# ── Production ────────────────────────────────────────────────────────────────

def production_cost(quality: str, factories: int) -> float:
    """
    Per-unit production cost with economies of scale.

    >>> production_cost("Excellent", 1)
    25.0
    >>> production_cost("Poor", 1)
    15.0
    >>> round(production_cost("Excellent", 2), 6)
    23.0
    >>> round(production_cost("Excellent", 4), 6)  # floor at 0.76
    19.0
    >>> production_cost("Excellent", 4) == production_cost("Excellent", 10)
    True
    """
    if quality not in BASE_PRODUCTION_COSTS:
        raise ValueError(f"Unknown quality level: {quality!r}. Must be 'Excellent' or 'Poor'.")
    if factories < 1:
        raise ValueError(f"factories must be >= 1, got {factories}")

    base = BASE_PRODUCTION_COSTS[quality]
    multiplier = max(COST_MULTIPLIER_FLOOR, 1.0 - 0.08 * (factories - 1))
    return round(base * multiplier, 6)


def factory_daily_capacity(factories: int) -> int:
    """Maximum widgets a seller can produce per day (one per factory)."""
    return factories


def holding_cost_per_unit_per_day(quality: str) -> float:
    """Daily holding cost for a single unsold widget."""
    if quality not in HOLDING_COSTS_PER_DAY:
        raise ValueError(f"Unknown quality level: {quality!r}")
    return HOLDING_COSTS_PER_DAY[quality]


def total_holding_cost(inventory: dict[str, int]) -> float:
    """
    Total daily holding cost for an entire seller inventory.

    inventory maps quality → widget count.
    """
    return sum(
        HOLDING_COSTS_PER_DAY[q] * count
        for q, count in inventory.items()
        if q in HOLDING_COSTS_PER_DAY and count > 0
    )


def end_of_run_write_off(inventory: dict[str, int], factories: int) -> float:
    """
    Cash cost of writing off unsold inventory at simulation end (day 30).
    Widgets are written off at full production cost — a realized loss.
    """
    return sum(
        production_cost(q, factories) * count
        for q, count in inventory.items()
        if q in BASE_PRODUCTION_COSTS and count > 0
    )


# ── Revenue ───────────────────────────────────────────────────────────────────

def revenue_adjustment(
    claimed_quality: str,
    true_quality: str,
    fg_price_excellent: float,
    fg_price_poor: float,
    quantity: int,
) -> float:
    """
    Cash adjustment applied to a buyer when quality is revealed after
    final goods have already been produced and revenue credited.

    Uses the final-good prices that were recorded at the time of production
    (passed in as fg_price_excellent / fg_price_poor). This ensures the
    adjustment reflects the actual economic impact at the moment of production,
    not the current day's prices.

    Returns a negative value when the buyer was cheated (claimed Excellent,
    true Poor). Returns zero when qualities match. Returns a positive value
    in the rare case where true quality exceeds claimed quality.
    """
    if claimed_quality == true_quality:
        return 0.0
    if claimed_quality not in BASE_PRODUCTION_COSTS or true_quality not in BASE_PRODUCTION_COSTS:
        raise ValueError(f"Unknown quality: claimed={claimed_quality!r}, true={true_quality!r}")

    claimed_price = fg_price_excellent if claimed_quality == "Excellent" else fg_price_poor
    true_price = fg_price_excellent if true_quality == "Excellent" else fg_price_poor
    return round((true_price - claimed_price) * quantity, 6)


# ── Buyer quota penalties ─────────────────────────────────────────────────────

def daily_quota_penalty(widgets_acquired: int) -> float:
    """
    Daily cash penalty for a buyer based on unfulfilled quota.

    Charged once per day. A buyer who has acquired all 30 widgets pays nothing.
    A buyer who has acquired 0 pays 30 × $2 = $60/day.

    >>> daily_quota_penalty(30)
    0.0
    >>> daily_quota_penalty(0)
    60.0
    >>> daily_quota_penalty(10)
    40.0
    """
    unfulfilled = max(0, BUYER_WIDGET_QUOTA - widgets_acquired)
    return round(unfulfilled * BUYER_DAILY_QUOTA_PENALTY, 6)


def terminal_quota_penalty(widgets_acquired: int) -> float:
    """
    One-time terminal penalty applied at end of day 30 for unfulfilled quota.

    >>> terminal_quota_penalty(30)
    0.0
    >>> terminal_quota_penalty(0)
    1800.0
    >>> terminal_quota_penalty(10)
    1200.0
    """
    unfulfilled = max(0, BUYER_WIDGET_QUOTA - widgets_acquired)
    return round(unfulfilled * BUYER_TERMINAL_QUOTA_PENALTY, 6)


# ── Price walk ────────────────────────────────────────────────────────────────

def apply_price_walk(current_price: float, rng: np.random.Generator) -> float:
    """
    Apply one step of Brownian motion to a final-good price.
    Price is floored at $1.00 to prevent degenerate states.
    """
    delta = float(rng.normal(0.0, PRICE_WALK_SIGMA))
    return round(max(1.0, current_price + delta), 4)
