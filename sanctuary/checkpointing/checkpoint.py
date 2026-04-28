"""
Checkpoint save and restore for the Sanctuary simulation.

Checkpoints serialize the full engine state at day boundaries so that
a crashed or interrupted simulation can be resumed from the most recent
checkpoint. Default interval: every 5 days.

Checkpoint files are written to: checkpoints/day_NNN.json
"""

from __future__ import annotations

import json
import os
import tempfile
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
    engine_state: dict[str, Any] | None = None,
    keep: int = 3,
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
        engine_state: optional engine-loop state (prev_day_messages,
            inactivity counters, outcome buffers). Older callers may
            omit this; loaders default missing fields to empty.
        keep: number of most recent checkpoints to keep on disk
            (older files are deleted after a successful save).

    Returns:
        Path to the written checkpoint file.

    The write is atomic: the file is written to a sibling tmp path
    and then renamed into place, so a kill mid-write cannot leave
    a partial JSON file at the canonical path.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "day": day,
        "market_snapshot": market_snapshot,
        "agent_states": agent_states,
        "rng_state": _serialize_rng_state(rng_state),
        "revelation_pending": revelation_pending,
        "counters": counters,
        "engine_state": engine_state or {},
    }

    filename = f"day_{day:03d}.json"
    path = checkpoint_dir / filename

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp", dir=checkpoint_dir,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    if keep is not None and keep > 0:
        prune_old_checkpoints(checkpoint_dir, keep=keep)

    return path


def prune_old_checkpoints(checkpoint_dir: Path, keep: int = 3) -> list[Path]:
    """Delete all but the `keep` most recent checkpoints in `checkpoint_dir`.

    Returns the list of paths that were deleted. Files that don't match
    the day_NNN.json pattern are left alone.
    """
    if not checkpoint_dir.exists():
        return []
    entries: list[tuple[int, Path]] = []
    for p in checkpoint_dir.glob("day_*.json"):
        try:
            day_n = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        entries.append((day_n, p))
    entries.sort(key=lambda x: x[0])
    to_delete = entries[:-keep] if len(entries) > keep else []
    deleted: list[Path] = []
    for _, p in to_delete:
        try:
            p.unlink()
            deleted.append(p)
        except OSError:
            pass
    return deleted


def try_resume(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Load the latest checkpoint if one exists.

    Returns the checkpoint dict, or None if `checkpoint_dir` is missing
    or contains no valid checkpoint file. Used by engine startup to
    detect prior progress without raising.
    """
    if not checkpoint_dir.exists():
        return None
    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    try:
        return load_checkpoint(checkpoint_dir, day=latest)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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
