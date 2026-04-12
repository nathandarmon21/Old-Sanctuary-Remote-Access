#!/usr/bin/env python3
"""
Submit a Sanctuary Phase 2 sweep as a SLURM job array.

Reads a sweep config YAML, generates a task index file mapping
SLURM array indices to (protocol, seed) pairs, and submits
the job array via sbatch.

Usage:
    python sweeps/submit_sweep.py sweeps/phase2_pilot.yaml
    python sweeps/submit_sweep.py sweeps/phase2_full.yaml
    python sweeps/submit_sweep.py sweeps/phase2_pilot.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_sweep_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_task_index(config: dict) -> list[tuple[str, int]]:
    """
    Generate (protocol, seed) pairs for all tasks in the sweep.
    Order: for each protocol, for each seed.
    """
    tasks = []
    for protocol in config["protocols"]:
        for seed in config["seeds"]:
            tasks.append((protocol, seed))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Submit a Sanctuary sweep as a SLURM job array.",
    )
    parser.add_argument(
        "sweep_config",
        help="Path to sweep config YAML (e.g. sweeps/phase2_pilot.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate task index and print sbatch command without submitting",
    )
    args = parser.parse_args()

    config = load_sweep_config(args.sweep_config)
    sweep_name = config["sweep_name"]
    base_config = config["base_config"]

    # Validate base config exists
    if not Path(base_config).exists():
        print(f"Error: base config not found: {base_config}", file=sys.stderr)
        sys.exit(1)

    # Generate task index
    tasks = generate_task_index(config)
    total = len(tasks)

    print(f"Sweep: {sweep_name}")
    print(f"Protocols: {config['protocols']}")
    print(f"Seeds: {config['seeds']}")
    print(f"Total tasks: {total}")
    print()

    # Create sweep directory for metadata
    sweep_dir = Path(f"sweeps/{sweep_name}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Write task index (1-indexed for SLURM array)
    index_path = sweep_dir / "task_index.tsv"
    with open(index_path, "w") as f:
        for protocol, seed in tasks:
            f.write(f"{protocol}\t{seed}\t{sweep_name}\n")
    print(f"Task index written: {index_path}")

    # Create output directories
    runs_dir = Path(f"runs/{sweep_name}")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Build sbatch command
    script_path = Path("sweeps/run_task.sh")
    sbatch_cmd = [
        "sbatch",
        f"--job-name={sweep_name}",
        f"--array=1-{total}",
        f"--output=sweeps/{sweep_name}/slurm_%A_%a.out",
        f"--error=sweeps/{sweep_name}/slurm_%A_%a.err",
        f"--export=SWEEP_DIR={sweep_dir},CONFIG_PATH={base_config},SWEEP_NAME={sweep_name}",
        str(script_path),
    ]

    print(f"sbatch command:")
    print(f"  {' '.join(sbatch_cmd)}")
    print()

    if args.dry_run:
        print("Dry run -- not submitting.")
        print(f"\nTask index ({total} tasks):")
        for i, (protocol, seed) in enumerate(tasks, 1):
            print(f"  {i:3d}: protocol={protocol}, seed={seed}")
        return

    # Submit
    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        print(f"\nSweep submitted: {total} tasks as SLURM job array.")
        print(f"Monitor: squeue -u {os.environ.get('USER', 'ndarmon')}")
        print(f"Results will be in: runs/{sweep_name}/")
    except FileNotFoundError:
        print("Error: sbatch not found. Are you on the cluster?", file=sys.stderr)
        print("Use --dry-run to preview without submitting.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
