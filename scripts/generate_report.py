#!/usr/bin/env python3
"""
CLI wrapper for the Sanctuary PDF report generator.

Usage:
    python scripts/generate_report.py <run_id_or_path> [--output PATH]
    python scripts/generate_report.py runs/run_20240101_120000_seed42
    python scripts/generate_report.py run_20240101_120000_seed42 --output my_report.pdf

If given a bare run_id (no slashes), the script looks inside ./runs/ for it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_run_dir(run_id_or_path: str) -> Path:
    p = Path(run_id_or_path)
    if p.exists():
        return p.resolve()
    # Try relative to a 'runs' directory next to this script's repo root
    repo_root = Path(__file__).parent.parent
    candidate = repo_root / "runs" / run_id_or_path
    if candidate.exists():
        return candidate.resolve()
    print(f"Error: could not find run directory for '{run_id_or_path}'", file=sys.stderr)
    print(f"  Tried: {p}", file=sys.stderr)
    print(f"  Tried: {candidate}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PDF report for a completed (or partial) Sanctuary run."
    )
    parser.add_argument(
        "run",
        help="Run directory path or run_id (looked up in ./runs/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output PDF path (default: <run_dir>/report.pdf)",
    )
    args = parser.parse_args()

    run_dir = _find_run_dir(args.run)
    output_path = Path(args.output) if args.output else None

    # Late import so that matplotlib backend is already configured before any
    # pyplot call happens elsewhere.
    from sanctuary.report import generate_report

    print(f"Generating report for run: {run_dir}")
    pdf_path = generate_report(run_dir, output_path=output_path)
    print(f"Report written to: {pdf_path}")


if __name__ == "__main__":
    main()
