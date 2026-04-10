"""
Checkpoint save and restore for the Sanctuary simulation.

Checkpoints serialize the full engine state at day boundaries so that
a crashed or interrupted simulation can be resumed from the most recent
checkpoint. Default interval: every 5 days.

Checkpoint files are written to: checkpoints/day_NNN.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_checkpoint(
    checkpoint_dir: Path,
    day: int,
    market_snapshot: dict[str, Any],
    agent_states: dict[str, dict[str, Any]],
    rng_state: dict[str, Any],
    revelation_pending: list[dict[str, Any]],
    counters: dict[str, Any],
) -> Path:
    """
    Save a checkpoint at the given day.

    Args:
        checkpoint_dir: directory for checkpoint files
        day: current simulation day
        market_snapshot: full market state snapshot
        agent_states: per-agent state dicts (history, policy, etc.)
        rng_state: numpy RNG bit_generator state dict
        revelation_pending: list of pending revelation dicts
        counters: engine counters (calls, tokens, parse stats)

    Returns:
        Path to the written checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "day": day,
        "market_snapshot": market_snapshot,
        "agent_states": agent_states,
        "rng_state": _serialize_rng_state(rng_state),
        "revelation_pending": revelation_pending,
        "counters": counters,
    }

    filename = f"day_{day:03d}.json"
    path = checkpoint_dir / filename
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)

    return path


def load_checkpoint(checkpoint_dir: Path, day: int | None = None) -> dict[str, Any]:
    """
    Load a checkpoint.

    Args:
        checkpoint_dir: directory containing checkpoint files
        day: specific day to load, or None to load the latest

    Returns:
        The checkpoint dict.

    Raises:
        FileNotFoundError: if no checkpoint found.
    """
    if day is not None:
        path = checkpoint_dir / f"day_{day:03d}.json"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found for day {day}: {path}")
    else:
        latest_day = find_latest_checkpoint(checkpoint_dir)
        if latest_day is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        path = checkpoint_dir / f"day_{latest_day:03d}.json"

    with open(path) as f:
        return json.load(f)


def find_latest_checkpoint(checkpoint_dir: Path) -> int | None:
    """
    Find the latest checkpoint day number.

    Returns the day number of the most recent checkpoint, or None if
    no checkpoints exist.
    """
    if not checkpoint_dir.exists():
        return None

    days: list[int] = []
    for p in checkpoint_dir.glob("day_*.json"):
        try:
            day_str = p.stem.split("_")[1]
            days.append(int(day_str))
        except (IndexError, ValueError):
            continue

    return max(days) if days else None


def _serialize_rng_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy RNG state to JSON-serializable form."""
    serialized = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            serialized[key] = {"__ndarray__": True, "data": value.tolist(), "dtype": str(value.dtype)}
        elif isinstance(value, dict):
            serialized[key] = _serialize_rng_state(value)
        else:
            serialized[key] = value
    return serialized


def _deserialize_rng_state(state: dict[str, Any]) -> dict[str, Any]:
    """Restore numpy arrays from JSON-serialized RNG state."""
    restored = {}
    for key, value in state.items():
        if isinstance(value, dict) and value.get("__ndarray__"):
            restored[key] = np.array(value["data"], dtype=value["dtype"])
        elif isinstance(value, dict):
            restored[key] = _deserialize_rng_state(value)
        else:
            restored[key] = value
    return restored
