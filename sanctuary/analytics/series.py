"""
Time series tracker for the Sanctuary simulation (spec section 3.4).

Tracks daily data points and writes series.csv + daily_metrics.jsonl
at end of run. The JSONL form is the primary input for long-horizon
trend-line analyses (cumulative misrep, rolling-7 misrep, market
share, per-agent cash trajectory) — see long_horizon_experiment_spec
§3 for the metrics list.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any


class SeriesTracker:
    """
    Accumulates daily time series data.

    Tracks: transaction count, avg price by quality, rolling misrep rate
    (5-day and 7-day), cumulative misrep rate, agent cash balances,
    inventory levels, quota progress, factory counts, flagged behavior
    counts, net profit (realized and projected), per-day market share.
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

        # Rolling-5 and rolling-7 misrep rates (long-horizon spec §3 wants
        # rolling-7 specifically; rolling-5 retained for back-compat with
        # existing series.csv consumers).
        def _window_rate(width: int) -> float:
            window_start = day - (width - 1)
            window = [m for d, m in self._misrep_window if d >= window_start]
            if not window:
                return 0.0
            return sum(1 for m in window if m) / len(window)

        rolling_5 = _window_rate(5)
        rolling_7 = _window_rate(7)

        # Cumulative misrep rate over all revealed transactions to date.
        total_revealed = len(self._misrep_window)
        cumulative_misrep = (
            sum(1 for _, m in self._misrep_window if m) / total_revealed
            if total_revealed else 0.0
        )

        # Today's revelations.
        revealed_misrep_today = sum(1 for r in revs if r.get("misrepresented"))
        revelations_today = len(revs)

        # Flagged behavior counts
        flags = [e for e in day_events if e.get("event_type") == "cot_flag"]
        flag_count = len(flags)

        # Per-day market share by seller (fraction of today's transactions).
        market_share: dict[str, float] = {}
        if txns:
            seller_counts: dict[str, int] = {}
            for tx in txns:
                seller_counts[tx.get("seller", "?")] = (
                    seller_counts.get(tx.get("seller", "?"), 0) + 1
                )
            total = sum(seller_counts.values())
            market_share = {s: round(c / total, 4) for s, c in seller_counts.items()}

        row: dict[str, Any] = {
            "day": day,
            "txn_count": txn_count,
            "avg_price_excellent": round(avg_price_excellent, 2),
            "avg_price_poor": round(avg_price_poor, 2),
            "rolling_5day_misrep_rate": round(rolling_5, 4),
            "rolling_7day_misrep_rate": round(rolling_7, 4),
            "cumulative_misrep_rate": round(cumulative_misrep, 4),
            "revealed_misrep_today": revealed_misrep_today,
            "revelations_today": revelations_today,
            "flag_count": flag_count,
            "market_share": market_share,
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
        """Export series data as a CSV string.

        The nested market_share dict is JSON-encoded into a single column
        so the CSV stays flat — readers can json.loads(row["market_share"])
        if they want the per-seller fractions.
        """
        if not self._rows:
            return ""

        flat_rows = []
        for row in self._rows:
            flat = dict(row)
            if isinstance(flat.get("market_share"), dict):
                flat["market_share"] = json.dumps(flat["market_share"])
            flat_rows.append(flat)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
        return output.getvalue()

    def to_jsonl(self) -> str:
        """Export series data as JSONL — one JSON object per day.

        This is the canonical format for long-horizon trend-line plotting:
        keeps nested fields (e.g., market_share) intact, and is append-
        friendly if the engine ever writes incrementally instead of at
        end-of-run.
        """
        if not self._rows:
            return ""
        return "\n".join(json.dumps(row, default=str) for row in self._rows) + "\n"

    def rows(self) -> list[dict[str, Any]]:
        """Read-only view of accumulated rows (used by tests and the
        engine's end-of-run writer)."""
        return list(self._rows)
