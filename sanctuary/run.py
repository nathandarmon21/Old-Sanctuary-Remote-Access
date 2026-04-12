"""
Mode 1 CLI entry point: batch simulation without dashboard.

Usage:
    python -m sanctuary.run --config configs/dev_local.yaml --seed 42
    python -m sanctuary.run --config configs/dev_local.yaml --seed 42 --output runs/my_run/
    python -m sanctuary.run --config configs/dev_local.yaml --seed 42 --resume runs/my_run/
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sanctuary.config import load_config, config_to_dict
from sanctuary.engine import SimulationEngine
from sanctuary.metrics.aggregate import compute_all_metrics
from sanctuary.protocols.factory import create_protocol
from sanctuary.run_directory import RunDirectory


def _make_run_id(seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_seed{seed}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sanctuary.run",
        description="Run The Sanctuary simulation in batch mode (Mode 1).",
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file",
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from an existing run directory's latest checkpoint",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        help="Override the config file's protocol (e.g. ebay_feedback, align_gossip)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

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

    # Override protocol from CLI if specified
    if args.protocol:
        config_dict["protocol"] = {"system": args.protocol}

    agent_names = (
        [s.name for s in config.agents.sellers]
        + [b.name for b in config.agents.buyers]
    )
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

        # SIGINT handler: mark crashed on Ctrl+C
        def _sigint_handler(sig, frame):
            print("\nInterrupted. Marking run as crashed...")
            rd.mark_crashed("interrupted by user (SIGINT)")
            rd.close()
            sys.exit(130)

        signal.signal(signal.SIGINT, _sigint_handler)

        print("Starting simulation...")
        wall_start = time.time()

        try:
            engine.run()

            # Compute metrics
            events_path = run_dir / "events.jsonl"
            final_state = engine.market.daily_snapshot()
            metrics = compute_all_metrics(
                events_path, total_days=config.run.days, final_state=final_state,
            )
            rd.mark_complete(metrics=metrics)

            wall_seconds = time.time() - wall_start
            print(f"\nSimulation complete in {wall_seconds:.1f}s.")
            print(f"Run directory: {run_dir}")

        except KeyboardInterrupt:
            rd.mark_crashed("interrupted by user")
            print(f"\nSimulation interrupted. Partial data in: {run_dir}")
            sys.exit(130)

        except Exception as exc:
            rd.mark_crashed(traceback.format_exc())
            print(f"\nSimulation crashed: {exc}")
            raise


if __name__ == "__main__":
    main()
