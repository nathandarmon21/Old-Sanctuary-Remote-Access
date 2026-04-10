"""
Events log writer for the Sanctuary simulation.

The events log (events.jsonl) is the canonical chronological record of
everything that happens in a simulation. Every line is one event with
a common envelope and event-specific payload.

Common envelope: {"timestamp": ISO8601, "day": int, "event_type": str, ...payload}

Event types:
  simulation_start, simulation_end,
  day_start, day_end,
  agent_turn (with full reasoning text),
  message_sent,
  transaction_proposed, transaction_completed,
  quality_revealed,
  factory_completed,
  bankruptcy,
  protocol_hook,
  cot_flag
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventWriter:
    """
    Writes events to events.jsonl, one JSON object per line.

    Flushes after every write so the file is readable mid-simulation
    (safe for `tail -f`).
    """

    def __init__(self, events_path: Path) -> None:
        self._path = events_path
        self._file = open(events_path, "a")

    def write_event(self, event_type: str, day: int, **payload: Any) -> dict[str, Any]:
        """
        Write a single event to the log.

        Returns the event dict (useful for tests and downstream processing).
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "day": day,
            "event_type": event_type,
            **payload,
        }
        self._file.write(json.dumps(event, default=str) + "\n")
        self._file.flush()
        return event

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def read_events(events_path: Path) -> list[dict[str, Any]]:
    """
    Read all events from an events.jsonl file.

    Returns a list of event dicts in chronological order.
    """
    events: list[dict[str, Any]] = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def read_events_by_day(events_path: Path) -> dict[int, list[dict[str, Any]]]:
    """
    Read all events grouped by day number.

    Returns a dict mapping day -> list of events for that day.
    """
    by_day: dict[int, list[dict[str, Any]]] = {}
    for event in read_events(events_path):
        day = event.get("day", 0)
        by_day.setdefault(day, []).append(event)
    return by_day
