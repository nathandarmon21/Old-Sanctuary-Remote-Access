"""
Pure economic functions for the Sanctuary simulation.

All functions here are stateless and deterministic -- no side effects,
no external dependencies. This makes them straightforward to test and
to reason about formally.

Production costs use a lookup table indexed by factory count (capped at 4+):

    Factories | Excellent | Poor
    ----------+-----------+------
    1         | $30.00    | $20.00
    2         | $27.00    | $18.00
    3         | $24.60    | $16.40
    4+        | $22.80    | $15.20

Holding cost is 2% of production cost per unit per day.
"""

from __future__ import annotations

from typing import Literal

Quality = Literal["Excellent", "Poor"]

# -- Production cost lookup table (spec section 1.4) --------------------------

PRODUCTION_COST_TABLE: dict[int, dict[str, float]] = {
    1: {"Excellent": 30.00, "Poor": 20.00},
    2: {"Excellent": 27.00, "Poor": 18.00},
    3: {"Excellent": 24.60, "Poor": 16.40},
    4: {"Excellent": 22.80, "Poor": 15.20},
}

# -- Fair market values -------------------------------------------------------

FMV: dict[str, float] = {
    "Excellent": 55.00,
    "Poor": 32.00,
}

# Alias used by downstream code that previously imported FINAL_GOOD_BASE_PRICES.
FINAL_GOOD_BASE_PRICES: dict[str, float] = FMV

# -- Holding cost rate --------------------------------------------------------

HOLDING_COST_RATE: float = 0.02  # 2% of production cost per unit per day

# -- Factory build parameters -------------------------------------------------

FACTORY_BUILD_COST: float = 2_000.0
FACTORY_BUILD_DAYS: int = 3

# -- Bankruptcy ---------------------------------------------------------------

BANKRUPTCY_THRESHOLD: float = -5_000.0

# -- Buyer parameters ---------------------------------------------------------

BUYER_MAX_DAILY_PRODUCTION: int = 3
BUYER_WIDGET_QUOTA: int = 20
BUYER_DAILY_QUOTA_PENALTY: float = 2.0     # $/day per unfulfilled quota unit
BUYER_TERMINAL_QUOTA_PENALTY: float = 75.0  # $/unit at end of run

# -- Seller starting cash (asymmetric, spec section 1.1) ----------------------

SELLER_STARTING_CASH: list[float] = [5_000.0, 4_500.0, 4_000.0, 3_500.0]

# -- Starting inventory -------------------------------------------------------

SELLER_STARTING_WIDGETS: int = 8  # per seller, random quality mix

# -- Revelation ---------------------------------------------------------------

REVELATION_LAG_DAYS: int = 5  # deterministic

# -- Max transactions ---------------------------------------------------------

MAX_TRANSACTIONS_PER_AGENT_PER_DAY: int = 1


# -- Production ----------------------------------------------------------------

def production_cost(quality: str, factories: int) -> float:
    """
    Per-unit production cost with economies of scale.

    Uses a lookup table capped at 4 factories.

    >>> production_cost("Excellent", 1)
    30.0
    >>> production_cost("Poor", 1)
    20.0
    >>> production_cost("Excellent", 2)
    27.0
    >>> production_cost("Excellent", 4)
    22.8
    >>> production_cost("Excellent", 4) == production_cost("Excellent", 10)
    True
    """
    if quality not in ("Excellent", "Poor"):
        raise ValueError(f"Unknown quality level: {quality!r}. Must be 'Excellent' or 'Poor'.")
    if factories < 1:
        raise ValueError(f"factories must be >= 1, got {factories}")

    effective_factories = min(factories, 4)
    return PRODUCTION_COST_TABLE[effective_factories][quality]


def factory_daily_capacity(factories: int) -> int:
    """Maximum widgets a seller can produce per day (one per factory)."""
    return factories


def holding_cost_per_unit_per_day(quality: str, factories: int = 1) -> float:
    """
    Daily holding cost for a single unsold widget.

    Equals 2% of the production cost at the seller's current factory count.
    """
    if quality not in ("Excellent", "Poor"):
        raise ValueError(f"Unknown quality level: {quality!r}")
    return round(production_cost(quality, factories) * HOLDING_COST_RATE, 6)


def total_holding_cost(inventory: dict[str, int], factories: int = 1) -> float:
    """
    Total daily holding cost for an entire seller inventory.

    inventory maps quality -> widget count.
    """
    total = 0.0
    for q, count in inventory.items():
        if q in ("Excellent", "Poor") and count > 0:
            total += holding_cost_per_unit_per_day(q, factories) * count
    return round(total, 6)


def end_of_run_write_off(inventory: dict[str, int], factories: int) -> float:
    """
    Cash cost of writing off unsold inventory at simulation end (day 30).
    Widgets are written off at full production cost -- a realized loss.
    """
    return sum(
        production_cost(q, factories) * count
        for q, count in inventory.items()
        if q in ("Excellent", "Poor") and count > 0
    )


# -- Revenue -------------------------------------------------------------------

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
    if claimed_quality not in ("Excellent", "Poor") or true_quality not in ("Excellent", "Poor"):
        raise ValueError(f"Unknown quality: claimed={claimed_quality!r}, true={true_quality!r}")

    claimed_price = fg_price_excellent if claimed_quality == "Excellent" else fg_price_poor
    true_price = fg_price_excellent if true_quality == "Excellent" else fg_price_poor
    return round((true_price - claimed_price) * quantity, 6)


# -- Buyer quota penalties -----------------------------------------------------

def daily_quota_penalty(widgets_acquired: int) -> float:
    """
    Daily cash penalty for a buyer based on unfulfilled quota.

    Charged once per day. A buyer who has acquired all 20 widgets pays nothing.
    A buyer who has acquired 0 pays 20 x $2 = $40/day.

    >>> daily_quota_penalty(20)
    0.0
    >>> daily_quota_penalty(0)
    40.0
    >>> daily_quota_penalty(10)
    20.0
    """
    unfulfilled = max(0, BUYER_WIDGET_QUOTA - widgets_acquired)
    return round(unfulfilled * BUYER_DAILY_QUOTA_PENALTY, 6)


def terminal_quota_penalty(widgets_acquired: int) -> float:
    """
    One-time terminal penalty applied at end of day 30 for unfulfilled quota.

    >>> terminal_quota_penalty(20)
    0.0
    >>> terminal_quota_penalty(0)
    1500.0
    >>> terminal_quota_penalty(10)
    750.0
    """
    unfulfilled = max(0, BUYER_WIDGET_QUOTA - widgets_acquired)
    return round(unfulfilled * BUYER_TERMINAL_QUOTA_PENALTY, 6)
