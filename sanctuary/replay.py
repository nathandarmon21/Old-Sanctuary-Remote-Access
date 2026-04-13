"""
Mode 3 CLI entry point: replay from completed run directory.

Usage:
    python -m sanctuary.replay --run runs/<run_id>/ --port 8090
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn

from sanctuary.dashboard.app import app, set_replay_data
from sanctuary.events import read_events, read_events_by_day


def _build_collusion_series(
    events_by_day: dict[int, list[dict[str, Any]]], up_to_day: int,
) -> list[dict[str, Any]]:
    """Build cumulative collusion flag count series from cot_flag events."""
    series: list[dict[str, Any]] = []
    cum = 0
    for d in range(1, up_to_day + 1):
        day_flags = [
            e for e in events_by_day.get(d, [])
            if e.get("event_type") == "cot_flag"
            and e.get("category") == "collusion_price_fixing"
        ]
        cum += len(day_flags)
        series.append({"day": d, "value": cum})
    return series


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sanctuary.replay",
        description="Replay a completed Sanctuary run (Mode 3).",
    )
    parser.add_argument("--run", "-r", required=True, help="Path to completed run directory")
    parser.add_argument("--port", "-p", type=int, default=8090, help="Dashboard port")
    return parser.parse_args(argv)


def _build_agents_dict(
    final_state: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build agents dict (keyed by name) from final_state.json sellers/buyers."""
    agents: dict[str, dict[str, Any]] = {}
    for name, data in final_state.get("sellers", {}).items():
        agents[name] = {
            "id": name,
            "name": name,
            "role": "seller",
            "balance": data.get("cash", 0),
            "factories": data.get("factories", 1),
            "inventory_count": data.get("inventory_excellent", 0) + data.get("inventory_poor", 0),
            "bankrupt": data.get("bankrupt", False),
            "quality_accuracy": 1.0,
            "net_profit": 0,
        }
    for name, data in final_state.get("buyers", {}).items():
        agents[name] = {
            "id": name,
            "name": name,
            "role": "buyer",
            "balance": data.get("cash", 0),
            "acquired": data.get("widgets_acquired", 0),
            "quota": 20,
            "quota_remaining": data.get("quota_remaining", 20),
            "total_penalties": 0,
            "bankrupt": data.get("bankrupt", False),
        }
    return agents


def _build_agents_at_day(
    day: int,
    events_by_day: dict[int, list[dict[str, Any]]],
    final_state: dict[str, Any],
    manifest: dict[str, Any],
    initial_cash: float = 5000.0,
) -> dict[str, dict[str, Any]]:
    """
    Reconstruct approximate agent state at a given day by replaying
    transactions, quality reveals, and offers from events.
    """
    agent_names = manifest.get("agent_names", [])
    sellers = set(final_state.get("sellers", {}).keys())
    buyers = set(final_state.get("buyers", {}).keys())

    # Start with initial state
    agents: dict[str, dict[str, Any]] = {}
    for name in agent_names:
        if name in sellers:
            agents[name] = {
                "id": name, "name": name, "role": "seller",
                "balance": initial_cash, "factories": 1,
                "inventory_count": 0, "bankrupt": False,
                "quality_accuracy": 1.0, "net_profit": 0,
                "balance_history": [],
                "total_true_quality_matches": 0,
                "total_deliveries": 0,
            }
        elif name in buyers:
            agents[name] = {
                "id": name, "name": name, "role": "buyer",
                "balance": initial_cash, "acquired": 0, "quota": 20,
                "quota_remaining": 20, "total_penalties": 0,
                "bankrupt": False, "balance_history": [],
            }

    # Replay events day by day up to target day
    for d in range(0, day + 1):
        day_events = events_by_day.get(d, [])
        for ev in day_events:
            et = ev.get("event_type", "")

            if et == "transaction_completed":
                seller_name = ev.get("seller", "")
                buyer_name = ev.get("buyer", "")
                price = ev.get("price_per_unit", 0) * ev.get("quantity", 1)

                if seller_name in agents:
                    agents[seller_name]["balance"] += price
                if buyer_name in agents:
                    agents[buyer_name]["balance"] -= price
                    agents[buyer_name]["acquired"] = agents[buyer_name].get("acquired", 0) + ev.get("quantity", 1)
                    agents[buyer_name]["quota_remaining"] = max(0, 20 - agents[buyer_name].get("acquired", 0))

            elif et == "quality_revealed":
                seller_name = ev.get("seller", "")
                claimed = ev.get("claimed_quality", "")
                true_q = ev.get("true_quality", "")
                if seller_name in agents and agents[seller_name]["role"] == "seller":
                    agents[seller_name]["total_deliveries"] += 1
                    if claimed == true_q:
                        agents[seller_name]["total_true_quality_matches"] += 1
                    total = agents[seller_name]["total_deliveries"]
                    matches = agents[seller_name]["total_true_quality_matches"]
                    agents[seller_name]["quality_accuracy"] = matches / total if total > 0 else 1.0

        # Record balance at end of each day
        for name, agent in agents.items():
            agent["balance_history"].append({"day": d, "balance": agent["balance"]})

    # Compute net_profit for sellers
    for name, agent in agents.items():
        if agent["role"] == "seller":
            agent["net_profit"] = agent["balance"] - initial_cash

    return agents


def _load_run_data(run_dir: Path) -> dict[str, Any]:
    """Load all data from a completed run directory for replay."""
    # Read manifest
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {run_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Read events
    events_path = run_dir / "events.jsonl"
    events = read_events(events_path) if events_path.exists() else []
    events_by_day = read_events_by_day(events_path) if events_path.exists() else {}

    # Read final_state.json
    final_state_path = run_dir / "final_state.json"
    final_state: dict[str, Any] = {}
    if final_state_path.exists():
        with open(final_state_path) as f:
            final_state = json.load(f)

    # Read metrics
    metrics_path = run_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    max_days = manifest.get("days_total", 30)

    # Pre-build per-day snapshots with full state
    daily_snapshots: dict[int, dict[str, Any]] = {}
    for day in range(1, max_days + 1):
        agents = _build_agents_at_day(day, events_by_day, final_state, manifest)

        # Collect transactions up to this day
        all_tx = []
        for d in range(1, day + 1):
            for ev in events_by_day.get(d, []):
                if ev.get("event_type") == "transaction_completed":
                    all_tx.append(ev)
        recent_tx = all_tx[-15:]

        # Collect messages for this day
        day_messages = [
            ev for ev in events_by_day.get(day, [])
            if ev.get("event_type") == "message_sent"
        ]

        # Build stats
        quality_events = []
        for d in range(1, day + 1):
            for ev in events_by_day.get(d, []):
                if ev.get("event_type") == "quality_revealed":
                    quality_events.append(ev)
        misrep_count = sum(
            1 for qe in quality_events
            if qe.get("claimed_quality") != qe.get("true_quality")
        )
        misrep_rate = misrep_count / len(quality_events) if quality_events else 0

        # Build price series up to this day
        price_series: list[dict[str, Any]] = []
        for d in range(1, day + 1):
            d_txns = [t for t in all_tx if t.get("day") == d]
            if d_txns:
                ex_prices = [t["price_per_unit"] for t in d_txns if t.get("claimed_quality", "").lower() == "excellent"]
                po_prices = [t["price_per_unit"] for t in d_txns if t.get("claimed_quality", "").lower() == "poor"]
                entry: dict[str, Any] = {"day": d}
                if ex_prices:
                    entry["excellent"] = sum(ex_prices) / len(ex_prices)
                if po_prices:
                    entry["poor"] = sum(po_prices) / len(po_prices)
                price_series.append(entry)

        # Build misrep series (cumulative misrep rate at each day)
        misrep_series: list[dict[str, Any]] = []
        cum_reveals = 0
        cum_misrep = 0
        for d in range(1, day + 1):
            for ev in events_by_day.get(d, []):
                if ev.get("event_type") == "quality_revealed":
                    cum_reveals += 1
                    if ev.get("claimed_quality") != ev.get("true_quality"):
                        cum_misrep += 1
            rate = cum_misrep / cum_reveals if cum_reveals > 0 else 0
            misrep_series.append({"day": d, "value": rate})

        # Build deal quality series (fraction of Excellent quality per day)
        deal_quality_series: list[dict[str, Any]] = []
        for d in range(1, day + 1):
            d_txns = [t for t in all_tx if t.get("day") == d]
            if d_txns:
                excellent_count = sum(1 for t in d_txns if t.get("true_quality", "").lower() == "excellent")
                deal_quality_series.append({"day": d, "value": excellent_count / len(d_txns)})

        # Active listings (offers proposed but not yet completed on this day)
        active_listings = [
            {
                "seller_id": ev.get("seller", ""),
                "price": ev.get("price_per_unit", 0),
                "listed_condition": ev.get("claimed_quality", ""),
                "spec_summary": f"Qty {ev.get('quantity', 1)}",
            }
            for ev in events_by_day.get(day, [])
            if ev.get("event_type") == "transaction_proposed"
        ]

        daily_snapshots[day] = {
            "day": day,
            "max_days": max_days,
            "protocol": manifest.get("config", {}).get("protocol", "unknown"),
            "paused": True,
            "completed": day >= max_days,
            "replay_mode": True,
            "agents": agents,
            "recent_transactions": recent_tx,
            "recent_messages": day_messages,
            "active_listings": active_listings,
            "stats": {
                "total_transactions": len(all_tx),
                "misrepresentation_rate": misrep_rate,
            },
            "analytics": {
                "price_series": price_series,
                "misrep_series": misrep_series,
                "deal_quality_series": deal_quality_series,
                "collusion_series": _build_collusion_series(events_by_day, day),
                "info_accuracy_series": [],
            },
        }

    # Current state = final day snapshot
    current_state = daily_snapshots.get(max_days, {
        "day": max_days,
        "max_days": max_days,
        "protocol": "unknown",
        "paused": True,
        "completed": True,
        "replay_mode": True,
    })
    current_state["manifest"] = manifest
    current_state["metrics"] = metrics

    return {
        "manifest": manifest,
        "events": events,
        "events_by_day": events_by_day,
        "daily_snapshots": daily_snapshots,
        "metrics": metrics,
        "current_state": current_state,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_dir = Path(args.run)

    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        raise SystemExit(1)

    print(f"Loading run: {run_dir}")
    data = _load_run_data(run_dir)
    set_replay_data(data)

    manifest = data["manifest"]
    print(f"Run ID: {manifest.get('run_id', 'unknown')}")
    print(f"Status: {manifest.get('status', 'unknown')}")
    print(f"Days: {manifest.get('days_total', '?')}")
    print(f"Events: {len(data['events'])}")
    print(f"Replay: http://localhost:{args.port}")
    print()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
