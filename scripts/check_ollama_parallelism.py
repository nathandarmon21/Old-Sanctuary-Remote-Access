#!/usr/bin/env python3
"""
Diagnostic script: measure effective Ollama parallelism.

Fires 8 concurrent requests to both tactical (qwen2.5:3b) and strategic
(qwen2.5:7b) models and reports the wall-clock speedup vs sequential.

Acceptance: speedup should be ≥ 3× for 8 concurrent requests. Lower than
that means OLLAMA_NUM_PARALLEL is not set or Ollama needs to be restarted.

Usage:
    python scripts/check_ollama_parallelism.py
"""

from __future__ import annotations

import concurrent.futures
import sys
import time

import httpx


BASE_URL = "http://localhost:11434"
TIMEOUT = 120.0
N = 8  # concurrent requests


def _fire(model: str, tokens: int, call_id: int) -> tuple[int, float, int]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Write exactly 3 sentences about competitive market strategy. "
                        f"Request {call_id}."
                    ),
                }
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": tokens},
        },
        timeout=TIMEOUT,
    )
    elapsed = time.perf_counter() - t0
    tok_count = r.json().get("eval_count", 0)
    return call_id, elapsed, tok_count


def measure(model: str, tokens: int) -> None:
    print(f"\nModel: {model}  ({N} concurrent, {tokens} token limit each)")

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(_fire, model, tokens, i) for i in range(N)]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    wall = time.perf_counter() - wall_start

    results.sort()
    for call_id, elapsed, tok in results:
        print(f"  [{call_id}] {elapsed:5.1f}s  {tok} tokens generated")

    total = sum(t for _, t, _ in results)
    speedup = total / wall
    print(f"  Wall: {wall:.1f}s   Sum: {total:.1f}s   Speedup: {speedup:.1f}×")

    if speedup >= 6.0:
        status = "EXCELLENT — near-full parallelism"
    elif speedup >= 4.0:
        status = "GOOD — solid parallelism"
    elif speedup >= 3.0:
        status = "ACCEPTABLE — partial parallelism"
    else:
        status = "POOR — check OLLAMA_NUM_PARALLEL (run: launchctl getenv OLLAMA_NUM_PARALLEL)"

    print(f"  Status: {status}")
    return speedup


def main() -> None:
    # Verify Ollama is reachable
    try:
        r = httpx.get(f"{BASE_URL}/api/version", timeout=5.0)
        print(f"Ollama version: {r.json().get('version', '?')}")
    except Exception as exc:
        print(f"ERROR: Cannot connect to Ollama at {BASE_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    tactical_speedup = measure("qwen2.5:3b", tokens=80)
    strategic_speedup = measure("qwen2.5:7b", tokens=80)

    print("\n=== Summary ===")
    print(f"  Tactical model (3b): {tactical_speedup:.1f}×")
    print(f"  Strategic model (7b): {strategic_speedup:.1f}×")
    if min(tactical_speedup, strategic_speedup) >= 3.0:
        print("  Overall: PASS — smoke test should complete in ~5-8 minutes")
    else:
        print("  Overall: FAIL — parallelism too low for target iteration speed")
        print("  Fix: launchctl setenv OLLAMA_NUM_PARALLEL 8")
        print("       Then quit and reopen Ollama app from the menu bar")
        sys.exit(1)


if __name__ == "__main__":
    main()
