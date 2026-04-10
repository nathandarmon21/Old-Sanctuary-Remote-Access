"""
Mode 2 CLI entry point: live dashboard with simulation.

Usage:
    python -m sanctuary.dev --config configs/dev_local.yaml --port 8090
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import uvicorn

from sanctuary.config import load_config, config_to_dict
from sanctuary.dashboard.app import app, set_engine
from sanctuary.engine import SimulationEngine
from sanctuary.metrics.aggregate import compute_all_metrics
from sanctuary.protocols.factory import create_protocol
from sanctuary.run_directory import RunDirectory


def _make_run_id(seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_seed{seed}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sanctuary.dev",
        description="Run The Sanctuary with live dashboard (Mode 2).",
    )
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Master random seed")
    parser.add_argument("--port", "-p", type=int, default=8090, help="Dashboard port")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output run directory")
    return parser.parse_args(argv)


async def _run_simulation(engine: SimulationEngine, rd: RunDirectory, config) -> None:
    """Run the simulation as an async task within the server event loop."""
    try:
        # Run in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.run)

        # Compute metrics
        events_path = rd.run_dir / "events.jsonl"
        metrics = compute_all_metrics(events_path, total_days=config.run.days)
        rd.mark_complete(metrics=metrics)
        print(f"\nSimulation complete. Output: {rd.run_dir}")
    except Exception as exc:
        rd.mark_crashed(str(exc))
        print(f"\nSimulation crashed: {exc}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    config = load_config(args.config)
    config_dict = config_to_dict(config)

    run_id = _make_run_id(args.seed)
    if args.output:
        run_dir = Path(args.output)
    else:
        runs_root = Path(__file__).parent.parent / "runs"
        runs_root.mkdir(exist_ok=True)
        run_dir = runs_root / run_id

    agent_names = (
        [s.name for s in config.agents.sellers]
        + [b.name for b in config.agents.buyers]
    )
    protocol = create_protocol(config_dict)

    rd = RunDirectory(run_dir, config_dict, seed=args.seed, agent_names=agent_names)
    engine = SimulationEngine(config=config, seed=args.seed, run_directory=rd, protocol=protocol)
    set_engine(engine)

    print(f"Dashboard: http://localhost:{args.port}")
    print(f"Run directory: {run_dir}")
    print(f"Protocol: {protocol.name}")
    print()

    @app.on_event("startup")
    async def _start_simulation():
        asyncio.create_task(_run_simulation(engine, rd, config))

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
