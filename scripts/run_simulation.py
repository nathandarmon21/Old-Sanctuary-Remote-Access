#!/usr/bin/env python3
"""
CLI entry point for The Sanctuary simulation (legacy script).

Usage:
    python scripts/run_simulation.py --config configs/dev_local.yaml --seed 42

For the new Mode 1 CLI, use:
    python -m sanctuary.run --config configs/dev_local.yaml --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sanctuary.config import load_config, config_to_dict
from sanctuary.engine import SimulationEngine
from sanctuary.protocols.factory import create_protocol
from sanctuary.run_directory import RunDirectory


def _make_run_id(seed: int) -> str:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_seed{seed}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run The Sanctuary multi-agent LLM market simulation."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file (e.g. configs/dev_local.yaml)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Master random seed (default: 42)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output run directory (default: runs/<run_id>/)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    config_dict = config_to_dict(config)

    run_id = _make_run_id(args.seed)
    if args.output:
        run_dir = Path(args.output)
    else:
        runs_root = Path(__file__).parent.parent / "runs"
        runs_root.mkdir(exist_ok=True)
        run_dir = runs_root / run_id

    agent_names = [s.name for s in config.agents.sellers] + [b.name for b in config.agents.buyers]
    protocol = create_protocol(config_dict)

    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Days: {config.run.days}")
    print(f"Strategic model: {config.models.strategic.provider}/{config.models.strategic.model}")
    print(f"Tactical model:  {config.models.tactical.provider}/{config.models.tactical.model}")
    print(f"Protocol: {protocol.name}")
    print()

    with RunDirectory(run_dir, config_dict, seed=args.seed, agent_names=agent_names) as rd:
        engine = SimulationEngine(
            config=config,
            seed=args.seed,
            run_directory=rd,
            protocol=protocol,
        )

        print("Starting simulation...")
        try:
            engine.run()
            rd.mark_complete()
            print(f"\nSimulation complete. Run ID: {run_id}")
        except KeyboardInterrupt:
            rd.mark_crashed("interrupted by user")
            print(f"\nSimulation interrupted. Partial data in: {run_dir}")
            sys.exit(130)
        except Exception as exc:
            rd.mark_crashed(str(exc))
            print(f"\nSimulation crashed: {exc}")
            raise

    print(f"Output: {run_dir}/")


if __name__ == "__main__":
    main()
