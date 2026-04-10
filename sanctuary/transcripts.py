"""
Per-agent transcript storage for the Sanctuary simulation.

Each agent gets three transcript files:
  - tactical_transcript.jsonl: every tactical LLM call in full
  - strategic_transcript.jsonl: every strategic LLM call in full
  - reasoning_log.jsonl: parsed reasoning for fast access

Nothing is truncated in transcript files. The in-flight context
trimming only affects what the LLM receives per call; the archival
preserves everything.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _agent_dir_name(agent_name: str) -> str:
    """Convert agent name to a safe directory name."""
    return re.sub(r"[^a-z0-9]+", "_", agent_name.lower()).strip("_")


class TranscriptWriter:
    """
    Manages per-agent transcript files within a run directory.

    Creates agent subdirectories and writes tactical/strategic transcripts
    and reasoning logs.
    """

    def __init__(self, agents_dir: Path, agent_names: list[str]) -> None:
        self._agents_dir = agents_dir
        self._files: dict[str, dict[str, Any]] = {}

        for name in agent_names:
            dir_name = _agent_dir_name(name)
            agent_path = agents_dir / dir_name
            agent_path.mkdir(parents=True, exist_ok=True)

            self._files[name] = {
                "tactical": open(agent_path / "tactical_transcript.jsonl", "a"),
                "strategic": open(agent_path / "strategic_transcript.jsonl", "a"),
                "reasoning": open(agent_path / "reasoning_log.jsonl", "a"),
                "dir": agent_path,
            }

    def write_tactical_call(
        self,
        agent_id: str,
        prompt_messages: list[dict[str, str]],
        response_text: str,
        parsed_actions: dict[str, Any] | None,
        timing_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        day: int,
    ) -> None:
        """Write a complete tactical LLM call to the transcript."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "day": day,
            "tier": "tactical",
            "model": model,
            "prompt_messages": prompt_messages,
            "response_text": response_text,
            "parsed_actions": parsed_actions,
            "timing_seconds": round(timing_seconds, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        self._write_to(agent_id, "tactical", record)

        # Also write to reasoning log
        reasoning_record = {
            "day": day,
            "tier": "tactical",
            "reasoning": response_text,
            "parsed_actions": parsed_actions,
        }
        self._write_to(agent_id, "reasoning", reasoning_record)

    def write_strategic_call(
        self,
        agent_id: str,
        prompt_messages: list[dict[str, str]],
        response_text: str,
        parsed_policy: dict[str, Any] | None,
        timing_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        day: int,
    ) -> None:
        """Write a complete strategic LLM call to the transcript."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "day": day,
            "tier": "strategic",
            "model": model,
            "prompt_messages": prompt_messages,
            "response_text": response_text,
            "parsed_policy": parsed_policy,
            "timing_seconds": round(timing_seconds, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        self._write_to(agent_id, "strategic", record)

        reasoning_record = {
            "day": day,
            "tier": "strategic",
            "reasoning": response_text,
            "parsed_policy": parsed_policy,
        }
        self._write_to(agent_id, "reasoning", reasoning_record)

    def agent_dir(self, agent_id: str) -> Path:
        """Return the directory path for a given agent."""
        return self._files[agent_id]["dir"]

    def close(self) -> None:
        for agent_files in self._files.values():
            for key in ("tactical", "strategic", "reasoning"):
                f = agent_files[key]
                if f and not f.closed:
                    f.close()

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _write_to(self, agent_id: str, file_key: str, record: dict[str, Any]) -> None:
        f = self._files[agent_id][file_key]
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
