"""
Aggregate metric computation from events.jsonl.

Reads events, computes all metrics, writes metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sanctuary.events import read_events
from sanctuary.metrics.allocative_efficiency import (
    compute_allocative_efficiency,
    compute_price_cost_margin,
)
from sanctuary.metrics.market_integrity import (
    compute_exploitation_rate,
    compute_markup_correlation,
    compute_price_parallelism,
    compute_price_trend,
    compute_trust_persistence,
)
from sanctuary.metrics.misrepresentation import compute_misrepresentation_rate


def compute_all_metrics(
    events_path: Path,
    total_days: int = 30,
    final_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Read events.jsonl and compute all automatic metrics.

    Returns a dict suitable for writing to metrics.json.
    """
    events = read_events(events_path)

    misrep = compute_misrepresentation_rate(events)
    ae = compute_allocative_efficiency(events)
    pcm = compute_price_cost_margin(events)
    ppi = compute_price_parallelism(events)
    mc = compute_markup_correlation(events)
    er = compute_exploitation_rate(events, total_days=total_days)
    tp = compute_trust_persistence(events)
    pt = compute_price_trend(events, total_days=total_days)

    result: dict[str, Any] = {
        "misrepresentation": misrep,
        "allocative_efficiency": ae,
        "price_cost_margin": pcm,
        "market_integrity": {
            "price_parallelism_index": ppi,
            "markup_correlation": mc,
            "exploitation_rate": er,
            "trust_persistence": tp,
            "price_trend": pt,
        },
    }

    # Per-agent net profit from final state snapshot
    if final_state:
        per_agent: dict[str, dict[str, float]] = {}
        for name, data in final_state.get("sellers", {}).items():
            per_agent[name] = {
                "net_profit_realized": data.get("net_profit_realized", 0.0),
                "net_profit_projected": data.get("net_profit_projected", 0.0),
                "role": "seller",
            }
        for name, data in final_state.get("buyers", {}).items():
            per_agent[name] = {
                "net_profit_realized": data.get("net_profit_realized", 0.0),
                "net_profit_projected": data.get("net_profit_projected", 0.0),
                "role": "buyer",
            }
        result["per_agent_net_profit"] = per_agent

    return result


def write_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Write metrics dict to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
