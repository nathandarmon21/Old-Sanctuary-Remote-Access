#!/usr/bin/env python3
"""
CLI entry point for The Sanctuary simulation.

Usage:
    python scripts/run_simulation.py --config configs/dev_local.yaml --seed 42
    python scripts/run_simulation.py --config configs/dev_local.yaml --seed 42 --profile
"""

from __future__ import annotations

import argparse
import pstats
import sys
from io import StringIO
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sanctuary.config import load_config
from sanctuary.simulation import Simulation


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
        "--profile",
        action="store_true",
        help="Run under cProfile and write perf_profile.prof",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    print(f"Initialising simulation (seed={args.seed})")
    sim = Simulation(config=config, seed=args.seed)

    print(f"Run ID: {sim.run_id}")
    print(f"Run directory: {sim.run_dir}")
    print(f"Days: {config.run.days}")
    print(f"Strategic model: {config.models.strategic.provider}/{config.models.strategic.model}")
    print(f"Tactical model:  {config.models.tactical.provider}/{config.models.tactical.model}")
    print()

    if args.profile:
        print("Starting simulation with cProfile...")
        run_id, stats = sim.run_with_profiler()
        print(f"\nSimulation complete. Run ID: {run_id}")

        # Print top-20 functions to stdout
        stream = StringIO()
        ps = pstats.Stats(stats, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(20)
        print(stream.getvalue())

        # Write perf_notes.md
        _write_perf_notes(sim, stats)
    else:
        print("Starting simulation...")
        run_id = sim.run()
        print(f"\nSimulation complete. Run ID: {run_id}")

    print(f"Logs: {sim.run_dir}/")
    print(f"  transactions.jsonl, revelations.jsonl, messages.jsonl, market_state.jsonl")
    print(f"  agents/*/strategic_calls.jsonl, tactical_calls.jsonl, policy_history.jsonl")

    # Generate PDF report
    print("\nGenerating PDF report...")
    try:
        from sanctuary.report import generate_report
        pdf_path = generate_report(sim.run_dir)
        print(f"Report: {pdf_path}")
        # Auto-open on macOS so the PDF appears on screen immediately
        import subprocess, sys as _sys
        if _sys.platform == "darwin":
            subprocess.Popen(["open", str(pdf_path)])
    except Exception as exc:
        print(f"WARNING: PDF report generation failed: {exc}")
        print("Simulation data is intact; report can be regenerated manually.")


def _write_perf_notes(sim: Simulation, stats: pstats.Stats) -> None:
    """Write the perf_notes.md file to the run directory."""
    import io as _io
    stream = _io.StringIO()
    ps = pstats.Stats(stats, stream=stream)
    ps.sort_stats("cumulative")
    ps.print_stats(10)
    top10 = stream.getvalue()

    total_calls = sim.total_strategic_calls + sim.total_tactical_calls
    import time as _time

    notes = f"""# Performance Notes — {sim.run_id}

## Run Summary
- Seed: {sim.seed}
- Simulated days: {sim.config.run.days}
- Total LLM calls: {total_calls} ({sim.total_strategic_calls} strategic, {sim.total_tactical_calls} tactical)
- Total prompt tokens: {sim.total_prompt_tokens:,}
- Total completion tokens: {sim.total_completion_tokens:,}
- Strategic model: {sim.config.models.strategic.provider}/{sim.config.models.strategic.model}
- Tactical model: {sim.config.models.tactical.provider}/{sim.config.models.tactical.model}

## Top 10 Functions by Cumulative Time

```
{top10}
```

## Analysis

### Where wall-clock time is spent
1. **LLM inference** — the vast majority. Each tactical call blocks on the
   inference server (Ollama or vLLM). With 8 agents × 30 days = 240 main
   tactical calls plus sub-rounds, inference dominates.

2. **Simulation logic** — negligible. State updates, validation, and
   message routing are pure Python with no I/O and run in microseconds.

3. **Logging** — small but non-zero. JSONL writes are synchronous and
   flushed after each entry. For 30 days of 8 agents this is ~hundreds
   of writes, each tiny.

4. **Report generation** — matplotlib + reportlab at end of run. Typically
   a few seconds; negligible relative to inference.

### Speedup opportunities

| Approach | Difficulty | Expected gain |
|---|---|---|
| Parallel agent calls (asyncio or threads) | Medium | 4–8× on tactical tier |
| Switch to vLLM with GPU | Low (config change) | 10–50× inference speedup |
| Batch tactical calls per day | High (API change) | 2–4× on supported backends |
| Reduce sub-round frequency | Low | 10–20% fewer calls |
| Cache strategic prompts | Medium | Negligible (1 call/agent/week) |

**Recommended for Phase 2**: Parallelize tactical calls within a day using
`asyncio` with `httpx.AsyncClient`. All agents receive the same market
snapshot at the start of each day, so their tactical calls are independent
until action-execution time. This alone should yield 4–6× speedup on
multi-agent days.

**For Phase 3 (cluster)**: Switch to vLLM with real GPU. Combined with
parallel calls, a 30-day run should complete in under 10 minutes on a
single A100.
"""

    notes_path = sim.run_dir / "perf_notes.md"
    notes_path.write_text(notes)
    print(f"Performance notes: {notes_path}")


if __name__ == "__main__":
    main()
