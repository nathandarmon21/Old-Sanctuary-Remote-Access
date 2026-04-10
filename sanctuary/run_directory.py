"""
Run directory manager for the Sanctuary simulation.

Each simulation produces an immutable directory with a fixed schema:

  runs/<run_id>/
    manifest.json              # run metadata + status
    events.jsonl               # canonical chronological event log
    metrics.json               # computed scalar metrics
    series.csv                 # daily time series
    final_state.json           # end-of-run engine snapshot
    config_used.yaml           # exact YAML that ran
    agents/
      <agent_id>/
        tactical_transcript.jsonl
        strategic_transcript.jsonl
        reasoning_log.jsonl
    checkpoints/

Manifest status transitions: "running" -> "complete" or "crashed".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sanctuary.events import EventWriter
from sanctuary.transcripts import TranscriptWriter


class RunDirectory:
    """
    Manages the complete run directory for one simulation.

    Creates the directory structure on init, provides access to
    EventWriter and TranscriptWriter, and handles manifest updates.
    """

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        seed: int,
        agent_names: list[str],
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.agents_dir = self.run_dir / "agents"
        self.agents_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # Write config_used.yaml
        config_path = self.run_dir / "config_used.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Write initial manifest
        self._manifest = {
            "run_id": self.run_dir.name,
            "seed": seed,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "config_file": str(config_path),
            "days_total": config.get("run", {}).get("days", 30),
            "agent_names": agent_names,
            "metrics": None,
            "error": None,
        }
        self._write_manifest()

        # Open writers
        self.events = EventWriter(self.run_dir / "events.jsonl")
        self.transcripts = TranscriptWriter(self.agents_dir, agent_names)

    def mark_complete(self, metrics: dict[str, Any] | None = None) -> None:
        """Mark the run as successfully completed."""
        self._manifest["status"] = "complete"
        self._manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        if metrics:
            self._manifest["metrics"] = metrics
            self.write_metrics(metrics)
        self._write_manifest()

    def mark_crashed(self, error: str) -> None:
        """Mark the run as crashed with an error description."""
        self._manifest["status"] = "crashed"
        self._manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._manifest["error"] = error
        self._write_manifest()

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        """Write metrics.json."""
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    def write_series(self, series_text: str) -> None:
        """Write series.csv."""
        series_path = self.run_dir / "series.csv"
        with open(series_path, "w") as f:
            f.write(series_text)

    def write_final_state(self, state: dict[str, Any]) -> None:
        """Write final_state.json."""
        state_path = self.run_dir / "final_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def read_manifest(self) -> dict[str, Any]:
        """Read the current manifest."""
        manifest_path = self.run_dir / "manifest.json"
        with open(manifest_path) as f:
            return json.load(f)

    def close(self) -> None:
        """Close all file handles."""
        self.events.close()
        self.transcripts.close()

    def __enter__(self) -> RunDirectory:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _write_manifest(self) -> None:
        manifest_path = self.run_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)
