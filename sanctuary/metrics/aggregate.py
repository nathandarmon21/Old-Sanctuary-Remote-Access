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
    compute_trust_persistence,
)
from sanctuary.metrics.misrepresentation import compute_misrepresentation_rate


def compute_all_metrics(
    events_path: Path,
    total_days: int = 30,
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

    return {
        "misrepresentation": misrep,
        "allocative_efficiency": ae,
        "price_cost_margin": pcm,
        "market_integrity": {
            "price_parallelism_index": ppi,
            "markup_correlation": mc,
            "exploitation_rate": er,
            "trust_persistence": tp,
        },
    }


def write_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Write metrics dict to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
