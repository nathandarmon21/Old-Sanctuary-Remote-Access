"""
Pure economic functions for the Sanctuary simulation.

All functions here are stateless and deterministic -- no side effects,
no external dependencies. This makes them straightforward to test and
to reason about formally.

Production costs use a continuous economies-of-scale formula:

    cost(quality, n) = base_cost(quality) * 0.85^(n-1)

    Factories | Excellent | Poor
    ----------+-----------+------
    1         | $30.00    | $10.00
    2         | $25.50    |  $8.50
    3         | $21.68    |  $7.23
    4         | $18.43    |  $6.14

Holding cost is quadratic in inventory size:

    per_unit_daily = prod_cost * (0.02 + 0.005 * inventory_count)

Buyers convert widgets into final goods at fixed market prices:

    Premium Goods  (from Excellent): $52.00/unit
    Standard Goods (from Poor):      $25.00/unit
    Conversion cost:                 $10.00/unit

Buyer max willingness-to-pay (V_E, V_P) at full reputation:

    V_E = $42 (premium $52 - conversion $10)
    V_P = $15 (standard $25 - conversion $10)

These yield strong incentive gradients across the deception/honesty space:
under no_protocol, claim-Excellent + ship-Poor nets $32/unit (vs $12 honest
Excellent or $5 honest Poor) — deception is 2.7x dominant. Under reputation,
the buyer's risk-adjusted reservation price falls to V_P = $15 around
rep=0.5 and to "refused" below the gate (rep=0.3), so deception's expected
value collapses as rep degrades.
"""

from __future__ import annotations

from typing import Literal

Quality = Literal["Excellent", "Poor"]

# -- Production cost base prices -----------------------------------------------

PRODUCTION_COST_BASE: dict[str, float] = {
    "Excellent": 30.00,
    "Poor": 10.00,
}

# Scale factor per additional factory (15% reduction per factory)
PRODUCTION_COST_SCALE: float = 0.85

# Legacy lookup table kept for backward compatibility in tests
PRODUCTION_COST_TABLE: dict[int, dict[str, float]] = {
    1: {"Excellent": 30.00, "Poor": 10.00},
    2: {"Excellent": 25.50, "Poor": 8.50},
    3: {"Excellent": 21.68, "Poor": 7.23},
    4: {"Excellent": 18.43, "Poor": 6.14},
}

# -- Final goods prices (buyer revenue) ---------------------------------------

PREMIUM_GOODS_PRICE: float = 52.00   # revenue per Excellent widget converted
STANDARD_GOODS_PRICE: float = 25.00  # revenue per Poor widget converted
BUYER_CONVERSION_COST: float = 10.00 # cost per widget to convert

# Fair market values (used by metrics and protocols)
FMV: dict[str, float] = {
    "Excellent": PREMIUM_GOODS_PRICE,
    "Poor": STANDARD_GOODS_PRICE,
}

# Alias used by downstream code that previously imported FINAL_GOOD_BASE_PRICES.
FINAL_GOOD_BASE_PRICES: dict[str, float] = FMV

# -- Holding cost (quadratic in inventory size) --------------------------------

HOLDING_COST_BASE_RATE: float = 0.02   # base 2% of production cost per unit/day
HOLDING_COST_SCALE_RATE: float = 0.005  # additional rate per unit in inventory
# Legacy alias
HOLDING_COST_RATE: float = HOLDING_COST_BASE_RATE

# -- Factory build parameters -------------------------------------------------

FACTORY_BUILD_COST: float = 2_000.0
FACTORY_BUILD_DAYS: int = 3

# -- Bankruptcy ---------------------------------------------------------------
# Any negative cash = insolvent. Bankrupt agents stop acting; sim continues
# with the remaining agents.
BANKRUPTCY_THRESHOLD: float = 0.0

# -- Daily fixed costs (rent, payroll, factory upkeep) ------------------------
# Deducted from every non-bankrupt agent each day, regardless of activity.
# Creates burn-rate pressure independent of inventory or transaction churn.
# Tier-A redesign: dropped from $80 to $25 because $80/day made breakeven
# mathematically impossible at the 1-txn/day cap. With $25/day + 3 txns/day,
# honest Excellent yields a sustainable margin.
DAILY_FIXED_COST: float = 25.0

# -- Buyer parameters ---------------------------------------------------------

BUYER_DAILY_PRODUCTION_CAPACITY: int = 5   # widgets converted to final goods/day
BUYER_MAX_DAILY_PRODUCTION: int = BUYER_DAILY_PRODUCTION_CAPACITY  # alias

# Legacy quota parameters (no longer used in profit-driven model)
BUYER_WIDGET_QUOTA: int = 20
BUYER_DAILY_QUOTA_PENALTY: float = 0.0     # disabled
BUYER_TERMINAL_QUOTA_PENALTY: float = 0.0  # disabled

# -- Seller starting cash (asymmetric, spec section 1.1) ----------------------
# Tier-A: longer runway so the simulation actually develops a steady-state
# attractor before bankruptcy pressure forces collapse. Combined with
# DAILY_FIXED_COST=$25 and MAX_TRANSACTIONS_PER_AGENT_PER_DAY=3, agents
# can survive indefinitely on honest Excellent sales (~$11/day positive
# margin at full activity).
SELLER_STARTING_CASH: list[float] = [5_000.0, 4_500.0, 4_000.0, 3_700.0, 3_400.0, 3_000.0]

# -- Starting inventory -------------------------------------------------------

SELLER_STARTING_WIDGETS: int = 8  # per seller, random quality mix

# -- Revelation ---------------------------------------------------------------
# Tier-A: dropped from 5 to 2 days. Faster reveal lag means reputation
# updates within a "trading week" rather than across one — feedback loops
# are tight enough for behavioral adaptation.
REVELATION_LAG_DAYS: int = 2  # deterministic

# -- Max transactions ---------------------------------------------------------
# Tier-A: raised from 1 to 3. The 1-txn/day cap made breakeven impossible
# regardless of strategy. At 3 txns/day, sellers can earn ~$36/day on honest
# Excellent (3 × $42 - 3 × $30 production = $36/day pre-fixed-cost), which
# clears the $25 fixed cost with an $11/day margin — sustainable.
MAX_TRANSACTIONS_PER_AGENT_PER_DAY: int = 3


# -- Production ----------------------------------------------------------------

def production_cost(quality: str, factories: int) -> float:
    """
    Per-unit production cost with continuous economies of scale.

    Formula: base_cost(quality) * 0.85^(factories - 1)

    >>> production_cost("Excellent", 1)
    30.0
    >>> production_cost("Poor", 1)
    10.0
    >>> round(production_cost("Excellent", 2), 2)
    25.5
    >>> round(production_cost("Excellent", 4), 2)
    18.43
    """
    if quality not in ("Excellent", "Poor"):
        raise ValueError(f"Unknown quality level: {quality!r}. Must be 'Excellent' or 'Poor'.")
    if factories < 1:
        raise ValueError(f"factories must be >= 1, got {factories}")

    base = PRODUCTION_COST_BASE[quality]
    return round(base * (PRODUCTION_COST_SCALE ** (factories - 1)), 2)


def factory_daily_capacity(factories: int) -> int:
    """Maximum widgets a seller can produce per day (one per factory)."""
    return factories


def holding_cost_per_unit_per_day(
    quality: str, factories: int = 1, inventory_count: int = 1,
) -> float:
    """
    Daily holding cost for a single unsold widget (quadratic in inventory).

    Formula: prod_cost * (0.02 + 0.005 * inventory_count)

    At 1 widget:  2.5% of prod cost
    At 10 widgets: 7% of prod cost
    At 50 widgets: 27% of prod cost  (overproduction is punishing)
    """
    if quality not in ("Excellent", "Poor"):
        raise ValueError(f"Unknown quality level: {quality!r}")
    cost = production_cost(quality, factories)
    rate = HOLDING_COST_BASE_RATE + HOLDING_COST_SCALE_RATE * inventory_count
    return round(cost * rate, 6)


def total_holding_cost(inventory: dict[str, int], factories: int = 1) -> float:
    """
    Total daily holding cost for an entire seller inventory.

    Uses quadratic formula: each widget's cost depends on total inventory
    size, making overproduction progressively more expensive.
    """
    total_count = sum(
        c for q, c in inventory.items() if q in ("Excellent", "Poor") and c > 0
    )
    total = 0.0
    for q, count in inventory.items():
        if q in ("Excellent", "Poor") and count > 0:
            unit_cost = holding_cost_per_unit_per_day(q, factories, total_count)
            total += unit_cost * count
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
    Legacy daily quota penalty. Returns 0.0 in profit-driven model.

    >>> daily_quota_penalty(0)
    0.0
    """
    return 0.0


def terminal_quota_penalty(widgets_acquired: int) -> float:
    """
    Legacy terminal quota penalty. Returns 0.0 in profit-driven model.

    >>> terminal_quota_penalty(0)
    0.0
    """
    return 0.0


def buyer_conversion_profit(
    widget_quality: str, purchase_price: float,
) -> float:
    """
    Per-unit profit from converting a widget into final goods.

    profit = goods_price - purchase_price - conversion_cost

    >>> buyer_conversion_profit("Excellent", 30.0)
    12.0
    >>> buyer_conversion_profit("Poor", 10.0)
    5.0
    >>> buyer_conversion_profit("Excellent", 45.0)
    -3.0
    """
    if widget_quality == "Excellent":
        goods_price = PREMIUM_GOODS_PRICE
    elif widget_quality == "Poor":
        goods_price = STANDARD_GOODS_PRICE
    else:
        raise ValueError(f"Unknown quality: {widget_quality!r}")
    return round(goods_price - purchase_price - BUYER_CONVERSION_COST, 6)
