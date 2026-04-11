"""
Time series tracker for the Sanctuary simulation (spec section 3.4).

Tracks daily data points and writes series.csv at end of run.
"""

from __future__ import annotations

import csv
import io
from typing import Any


class SeriesTracker:
    """
    Accumulates daily time series data.

    Tracks: transaction count, avg price by quality, rolling misrep rate,
    agent cash balances, inventory levels, quota progress, factory counts,
    flagged behavior counts, net profit (realized and projected).
    """

    def __init__(
        self,
        agent_names: list[str],
        agent_starting_cash: dict[str, float] | None = None,
    ) -> None:
        self._agent_names = sorted(agent_names)
        self._agent_starting_cash = agent_starting_cash or {}
        self._rows: list[dict[str, Any]] = []
        self._misrep_window: list[tuple[int, bool]] = []  # (day, is_misrep)

    def update(
        self,
        day: int,
        day_events: list[dict[str, Any]],
        agent_cash: dict[str, float],
        agent_inventory: dict[str, int],
        agent_quota: dict[str, int],
        agent_factories: dict[str, int],
        agent_net_profit_realized: dict[str, float] | None = None,
        agent_net_profit_projected: dict[str, float] | None = None,
    ) -> None:
        """Record one day's data."""
        # Transaction count
        txns = [e for e in day_events if e.get("event_type") == "transaction_completed"]
        txn_count = len(txns)

        # Average prices by quality
        excellent_prices = [e["price_per_unit"] for e in txns if e.get("claimed_quality") == "Excellent"]
        poor_prices = [e["price_per_unit"] for e in txns if e.get("claimed_quality") == "Poor"]
        avg_price_excellent = sum(excellent_prices) / len(excellent_prices) if excellent_prices else 0.0
        avg_price_poor = sum(poor_prices) / len(poor_prices) if poor_prices else 0.0

        # Revelations and misrep tracking
        revs = [e for e in day_events if e.get("event_type") == "quality_revealed"]
        for r in revs:
            self._misrep_window.append((day, r.get("misrepresented", False)))

        # Rolling 5-day misrep rate
        window_start = day - 4
        window = [(d, m) for d, m in self._misrep_window if d >= window_start]
        if window:
            rolling_misrep = sum(1 for _, m in window if m) / len(window)
        else:
            rolling_misrep = 0.0

        # Flagged behavior counts
        flags = [e for e in day_events if e.get("event_type") == "cot_flag"]
        flag_count = len(flags)

        row: dict[str, Any] = {
            "day": day,
            "txn_count": txn_count,
            "avg_price_excellent": round(avg_price_excellent, 2),
            "avg_price_poor": round(avg_price_poor, 2),
            "rolling_5day_misrep_rate": round(rolling_misrep, 4),
            "flag_count": flag_count,
        }

        realized = agent_net_profit_realized or {}
        projected = agent_net_profit_projected or {}

        for name in self._agent_names:
            safe = name.lower().replace(" ", "_")
            row[f"cash_{safe}"] = round(agent_cash.get(name, 0.0), 2)
            row[f"inventory_{safe}"] = agent_inventory.get(name, 0)
            row[f"quota_{safe}"] = agent_quota.get(name, 0)
            row[f"factories_{safe}"] = agent_factories.get(name, 0)
            row[f"net_profit_realized_{safe}"] = round(realized.get(name, 0.0), 2)
            row[f"net_profit_projected_{safe}"] = round(projected.get(name, 0.0), 2)

        self._rows.append(row)

    def to_csv(self) -> str:
        """Export series data as a CSV string."""
        if not self._rows:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(self._rows[0].keys()))
        writer.writeheader()
        writer.writerows(self._rows)
        return output.getvalue()
