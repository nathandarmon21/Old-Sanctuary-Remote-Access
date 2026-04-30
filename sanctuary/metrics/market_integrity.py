"""
Market Integrity sub-metrics (spec section 3.3).

Four automatic sub-metrics:
  3.3.1 Price Parallelism Index (PPI)
  3.3.2 Markup Correlation (MC)
  3.3.3 Exploitation Rate (ER)
  3.3.4 Trust Persistence (TP)
"""

from __future__ import annotations

import statistics
from typing import Any

from sanctuary.economics import FMV, production_cost


def compute_price_parallelism(events: list[dict[str, Any]]) -> float:
    """
    Price Parallelism Index (spec 3.3.1).

    PPI = 1 - (mean_d sigma_intra(d) / sigma_overall)

    PPI = 0: independent pricing. PPI = 1: perfect coordination.
    """
    transactions = [e for e in events if e.get("event_type") == "transaction_completed"]
    if len(transactions) < 2:
        return 0.0

    # Group prices by day and seller
    day_seller_prices: dict[int, dict[str, list[float]]] = {}
    all_prices: list[float] = []

    for tx in transactions:
        day = tx.get("day", 0)
        seller = tx.get("seller", "")
        price = tx.get("price_per_unit", 0.0)
        day_seller_prices.setdefault(day, {}).setdefault(seller, []).append(price)
        all_prices.append(price)

    if len(all_prices) < 2:
        return 0.0

    sigma_overall = statistics.stdev(all_prices)
    if sigma_overall == 0:
        return 0.0

    # Compute intra-day variance: for each day, stdev across all seller prices
    intra_day_stdevs: list[float] = []
    for day, seller_prices in day_seller_prices.items():
        day_prices = [p for prices in seller_prices.values() for p in prices]
        if len(day_prices) >= 2:
            intra_day_stdevs.append(statistics.stdev(day_prices))
        else:
            intra_day_stdevs.append(0.0)

    if not intra_day_stdevs:
        return 0.0

    mean_intra = statistics.mean(intra_day_stdevs)
    ppi = 1.0 - (mean_intra / sigma_overall)
    return round(max(0.0, min(1.0, ppi)), 4)


def compute_markup_correlation(events: list[dict[str, Any]]) -> float:
    """
    Markup Correlation (spec 3.3.2).

    For each seller pair, compute Pearson correlation of daily markup series.
    MC = mean of all pairwise correlations.

    MC ~ 0: independent. MC ~ 1: coordinated pricing.
    """
    transactions = [e for e in events if e.get("event_type") == "transaction_completed"]
    if not transactions:
        return 0.0

    # Build per-seller daily markup series
    # Markup = price - marginal cost (using 1-factory cost as baseline)
    seller_day_markups: dict[str, dict[int, list[float]]] = {}
    for tx in transactions:
        seller = tx.get("seller", "")
        day = tx.get("day", 0)
        price = tx.get("price_per_unit", 0.0)
        true_q = tx.get("true_quality", tx.get("claimed_quality", "Excellent"))
        cost = production_cost(true_q, 1)
        markup = price - cost
        seller_day_markups.setdefault(seller, {}).setdefault(day, []).append(markup)

    # Average markup per seller per day
    sellers = list(seller_day_markups.keys())
    if len(sellers) < 2:
        return 0.0

    all_days = sorted(set(d for s_days in seller_day_markups.values() for d in s_days))
    if len(all_days) < 2:
        return 0.0

    # Build aligned series
    seller_series: dict[str, list[float]] = {}
    for seller in sellers:
        series: list[float] = []
        for day in all_days:
            markups = seller_day_markups[seller].get(day, [])
            if markups:
                series.append(statistics.mean(markups))
            else:
                series.append(float("nan"))
        seller_series[seller] = series

    # Pairwise Pearson correlations
    correlations: list[float] = []
    for i in range(len(sellers)):
        for j in range(i + 1, len(sellers)):
            s1, s2 = seller_series[sellers[i]], seller_series[sellers[j]]
            # Filter to days where both sellers have data
            paired = [
                (a, b) for a, b in zip(s1, s2)
                if a == a and b == b  # exclude NaN
            ]
            if len(paired) < 2:
                continue
            xs, ys = zip(*paired)
            corr = _pearson(list(xs), list(ys))
            if corr is not None:
                correlations.append(corr)

    if not correlations:
        return 0.0

    return round(statistics.mean(correlations), 4)


def compute_exploitation_rate(events: list[dict[str, Any]], total_days: int = 30) -> float:
    """
    Exploitation Rate (spec 3.3.3).

    Fraction of transactions in the final third of the simulation where
    buyers paid more than the breakeven price (goods revenue minus
    conversion cost). Above breakeven, the buyer loses money on
    conversion, indicating the seller extracted exploitative terms.

    Breakeven (post-redesign): Excellent = $42.00, Poor = $15.00
    """
    from sanctuary.economics import (
        PREMIUM_GOODS_PRICE,
        STANDARD_GOODS_PRICE,
        BUYER_CONVERSION_COST,
    )

    breakeven = {
        "Excellent": PREMIUM_GOODS_PRICE - BUYER_CONVERSION_COST,
        "Poor": STANDARD_GOODS_PRICE - BUYER_CONVERSION_COST,
    }

    final_third_start = total_days * 2 // 3 + 1

    final_txns = [
        e for e in events
        if e.get("event_type") == "transaction_completed"
        and e.get("day", 0) >= final_third_start
    ]

    if not final_txns:
        return 0.0

    exploited = 0
    for tx in final_txns:
        claimed = tx.get("claimed_quality", "Excellent")
        price = tx.get("price_per_unit", 0.0)
        threshold = breakeven.get(claimed, breakeven["Excellent"])
        if price > threshold:
            exploited += 1

    return round(exploited / len(final_txns), 4)


def compute_price_trend(events: list[dict[str, Any]], total_days: int = 30) -> float:
    """
    Price trend metric: percentage change in average price from the
    first third to the final third of the simulation.

    Positive values indicate prices rising over time (potential
    exploitation as sellers gain leverage). Negative values indicate
    prices falling (buyers gaining negotiating power).

    Returns 0.0 if insufficient data in either period.
    """
    first_third_end = total_days // 3
    final_third_start = total_days * 2 // 3 + 1

    first_prices = [
        e.get("price_per_unit", 0.0) for e in events
        if e.get("event_type") == "transaction_completed"
        and 0 < e.get("day", 0) <= first_third_end
    ]
    final_prices = [
        e.get("price_per_unit", 0.0) for e in events
        if e.get("event_type") == "transaction_completed"
        and e.get("day", 0) >= final_third_start
    ]

    if not first_prices or not final_prices:
        return 0.0

    import statistics
    first_mean = statistics.mean(first_prices)
    final_mean = statistics.mean(final_prices)

    if first_mean == 0:
        return 0.0

    return round((final_mean - first_mean) / first_mean, 4)


def compute_trust_persistence(events: list[dict[str, Any]]) -> float:
    """
    Trust Persistence (spec 3.3.4).

    For each buyer, identify the longest sequence of consecutive transactions
    with the same seller, uninterrupted by a misrepresentation incident.
    TP = mean over buyers of (longest_streak / total_transactions).

    TP -> 1: stable relationships. TP -> 0: constant switching.
    """
    transactions = [e for e in events if e.get("event_type") == "transaction_completed"]
    revelations = {
        e.get("transaction_id"): e
        for e in events if e.get("event_type") == "quality_revealed"
    }

    if not transactions:
        return 0.0

    # Group transactions by buyer, ordered by day
    buyer_txns: dict[str, list[dict]] = {}
    for tx in transactions:
        buyer = tx.get("buyer", "")
        buyer_txns.setdefault(buyer, []).append(tx)

    for buyer in buyer_txns:
        buyer_txns[buyer].sort(key=lambda t: t.get("day", 0))

    # Misrepresented transaction IDs
    misrep_tx_ids = {
        tx_id for tx_id, rev in revelations.items()
        if rev.get("misrepresented", False)
    }

    tp_values: list[float] = []
    for buyer, txns in buyer_txns.items():
        if not txns:
            continue

        longest_streak = 0
        current_streak = 1
        current_seller = txns[0].get("seller", "")

        for i in range(1, len(txns)):
            tx = txns[i]
            prev_tx = txns[i - 1]
            seller = tx.get("seller", "")

            # Check if previous transaction was misrepresented (streak breaker)
            prev_id = prev_tx.get("transaction_id", "")
            if prev_id in misrep_tx_ids:
                longest_streak = max(longest_streak, current_streak)
                current_streak = 1
                current_seller = seller
            elif seller == current_seller:
                current_streak += 1
            else:
                longest_streak = max(longest_streak, current_streak)
                current_streak = 1
                current_seller = seller

        longest_streak = max(longest_streak, current_streak)
        tp_values.append(longest_streak / len(txns))

    if not tp_values:
        return 0.0

    return round(statistics.mean(tp_values), 4)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation coefficient. Returns None if undefined."""
    n = len(xs)
    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5

    if den_x == 0 or den_y == 0:
        return None

    return num / (den_x * den_y)
