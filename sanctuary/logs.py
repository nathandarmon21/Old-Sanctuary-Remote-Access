"""
Structured logging for the Sanctuary simulation.

All log files use JSONL (one JSON object per line). Every write is
immediately flushed to disk so that `tail -f` works during live runs
and partial data survives if the run is interrupted.

Each simulation run gets its own directory under runs/{run_id}/.
Agent-specific logs go under runs/{run_id}/agents/{agent_name}/.

Log files:
  config.json                         — full config snapshot
  transactions.jsonl                  — every completed transaction
  revelations.jsonl                   — every quality revelation
  messages.jsonl                      — every agent message
  market_state.jsonl                  — daily market snapshots
  events.jsonl                        — discrete events (bankruptcy, factory, etc.)
  agents/{name}/strategic_calls.jsonl — strategic LLM calls (full prompt + completion)
  agents/{name}/tactical_calls.jsonl  — tactical LLM calls (full prompt + completion)
  agents/{name}/policy_history.jsonl  — strategic policies in order
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    """
    Manages all log files for one simulation run.

    Usage:
        logger = RunLogger(run_dir=Path("runs/run_20240101_120000"))
        logger.open()
        logger.log_transaction(...)
        logger.close()

    All write methods flush immediately after writing. Safe to call
    from a single thread (the simulation loop is single-threaded).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._handles: dict[str, IO[str]] = {}

    def open(self) -> None:
        """Create the directory structure and open all log files."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "agents").mkdir(exist_ok=True)

        self._handles["transactions"] = self._open("transactions.jsonl")
        self._handles["revelations"] = self._open("revelations.jsonl")
        self._handles["messages"] = self._open("messages.jsonl")
        self._handles["market_state"] = self._open("market_state.jsonl")
        self._handles["events"] = self._open("events.jsonl")

    def close(self) -> None:
        """Flush and close all open file handles."""
        for fh in self._handles.values():
            fh.flush()
            fh.close()
        self._handles.clear()

    def __enter__(self) -> RunLogger:
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Config snapshot ───────────────────────────────────────────────────────

    def write_config(self, config: dict, seed: int, run_id: str) -> None:
        """Write the full config snapshot to config.json (not JSONL)."""
        snapshot = {
            "run_id": run_id,
            "seed": seed,
            "timestamp": _now_iso(),
            "config": config,
        }
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(snapshot, f, indent=2)

    # ── Per-agent log directories ─────────────────────────────────────────────

    def ensure_agent_dir(self, agent_name: str) -> Path:
        """Create agent log directory. Returns the path."""
        safe_name = agent_name.replace(" ", "_").replace("/", "_")
        agent_dir = self.run_dir / "agents" / safe_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def _agent_handle(self, agent_name: str, filename: str) -> IO[str]:
        key = f"agent:{agent_name}:{filename}"
        if key not in self._handles:
            agent_dir = self.ensure_agent_dir(agent_name)
            self._handles[key] = open(agent_dir / filename, "a", buffering=1)
        return self._handles[key]

    # ── Agent LLM call logging ────────────────────────────────────────────────

    def log_strategic_call(
        self,
        agent_name: str,
        day: int,
        week: int,
        system_prompt: str,
        history: list[dict],
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        model: str,
        provider: str,
    ) -> None:
        record = {
            "timestamp": _now_iso(),
            "agent": agent_name,
            "tier": "strategic",
            "day": day,
            "week": week,
            "system_prompt": system_prompt,
            "history": history,
            "completion": completion,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": latency_seconds,
            "model": model,
            "provider": provider,
        }
        self._append_agent(agent_name, "strategic_calls.jsonl", record)

    def log_tactical_call(
        self,
        agent_name: str,
        day: int,
        sub_round: int,
        system_prompt: str,
        history: list[dict],
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        model: str,
        provider: str,
    ) -> None:
        record = {
            "timestamp": _now_iso(),
            "agent": agent_name,
            "tier": "tactical",
            "day": day,
            "sub_round": sub_round,
            "system_prompt": system_prompt,
            "history": history,
            "completion": completion,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": latency_seconds,
            "model": model,
            "provider": provider,
        }
        self._append_agent(agent_name, "tactical_calls.jsonl", record)

    def log_policy(self, agent_name: str, day: int, week: int, policy_json: dict, raw_memo: str) -> None:
        record = {
            "timestamp": _now_iso(),
            "agent": agent_name,
            "day": day,
            "week": week,
            "policy": policy_json,
            "raw_memo": raw_memo,
        }
        self._append_agent(agent_name, "policy_history.jsonl", record)

    def log_parse_error(self, agent_name: str, day: int, tier: str, error: str) -> None:
        self.log_event(
            event_type="parse_error",
            day=day,
            details={"agent": agent_name, "tier": tier, "error": error},
        )

    # ── Transaction and revelation logging ────────────────────────────────────

    def log_transaction(
        self,
        transaction_id: str,
        seller: str,
        buyer: str,
        quantity: int,
        claimed_quality: str,
        true_quality: str,
        price_per_unit: float,
        day: int,
        revelation_day: int,
    ) -> None:
        record = {
            "timestamp": _now_iso(),
            "transaction_id": transaction_id,
            "seller": seller,
            "buyer": buyer,
            "quantity": quantity,
            "claimed_quality": claimed_quality,
            "true_quality": true_quality,
            "misrepresented": claimed_quality != true_quality,
            "price_per_unit": price_per_unit,
            "total_value": price_per_unit * quantity,
            "day": day,
            "revelation_day": revelation_day,
        }
        self._append("transactions", record)

    def log_revelation(
        self,
        transaction_id: str,
        seller: str,
        buyer: str,
        claimed_quality: str,
        true_quality: str,
        quantity: int,
        transaction_day: int,
        revelation_day: int,
        cash_adjustment: float,
    ) -> None:
        record = {
            "timestamp": _now_iso(),
            "transaction_id": transaction_id,
            "seller": seller,
            "buyer": buyer,
            "claimed_quality": claimed_quality,
            "true_quality": true_quality,
            "misrepresented": claimed_quality != true_quality,
            "quantity": quantity,
            "transaction_day": transaction_day,
            "revelation_day": revelation_day,
            "buyer_cash_adjustment": cash_adjustment,
        }
        self._append("revelations", record)

    # ── Message logging ───────────────────────────────────────────────────────

    def log_message(
        self,
        message_id: str,
        sender: str,
        recipient: str,
        is_public: bool,
        day: int,
        sub_round: int,
        content: str,
    ) -> None:
        record = {
            "timestamp": _now_iso(),
            "message_id": message_id,
            "sender": sender,
            "recipient": recipient,
            "is_public": is_public,
            "day": day,
            "sub_round": sub_round,
            "content": content,
        }
        self._append("messages", record)

    # ── Market state snapshots ────────────────────────────────────────────────

    def log_market_snapshot(self, snapshot: dict) -> None:
        snapshot["timestamp"] = _now_iso()
        self._append("market_state", snapshot)

    # ── Discrete events ───────────────────────────────────────────────────────

    def log_event(self, event_type: str, day: int, details: dict) -> None:
        record = {
            "timestamp": _now_iso(),
            "event_type": event_type,
            "day": day,
            **details,
        }
        self._append("events", record)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _open(self, filename: str) -> IO[str]:
        """Open a run-level file in append mode with line buffering."""
        return open(self.run_dir / filename, "a", buffering=1)

    def _append(self, key: str, record: dict) -> None:
        """Write one JSONL record and flush immediately."""
        fh = self._handles[key]
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()

    def _append_agent(self, agent_name: str, filename: str, record: dict) -> None:
        fh = self._agent_handle(agent_name, filename)
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()
