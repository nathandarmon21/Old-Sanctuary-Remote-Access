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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sanctuary.replay",
        description="Replay a completed Sanctuary run (Mode 3).",
    )
    parser.add_argument("--run", "-r", required=True, help="Path to completed run directory")
    parser.add_argument("--port", "-p", type=int, default=8090, help="Dashboard port")
    return parser.parse_args(argv)


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

    # Build daily snapshots from day_end events or market snapshots
    daily_snapshots: list[dict[str, Any]] = []
    for day in sorted(events_by_day.keys()):
        day_evts = events_by_day[day]
        # Look for the last snapshot-like event of the day
        snapshot: dict[str, Any] = {"day": day}
        for e in day_evts:
            if e.get("event_type") == "day_end":
                snapshot = {**snapshot, **e}
        daily_snapshots.append(snapshot)

    # Read metrics
    metrics_path = run_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Current state = last snapshot
    current_state: dict[str, Any] = {
        "day": manifest.get("days_total", 30),
        "max_days": manifest.get("days_total", 30),
        "protocol": "unknown",
        "paused": False,
        "completed": True,
        "replay_mode": True,
        "manifest": manifest,
        "metrics": metrics,
    }
    if daily_snapshots:
        current_state.update(daily_snapshots[-1])

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
