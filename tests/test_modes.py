"""
Tests for Mode 2 (sanctuary/dev.py) and Mode 3 (sanctuary/replay.py).

Covers: argument parsing, replay data loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanctuary.dev import _parse_args as dev_parse_args
from sanctuary.replay import _parse_args as replay_parse_args, _load_run_data


class TestDevArgs:
    def test_config_required(self):
        with pytest.raises(SystemExit):
            dev_parse_args([])

    def test_default_port(self):
        args = dev_parse_args(["--config", "test.yaml"])
        assert args.port == 8090

    def test_custom_port(self):
        args = dev_parse_args(["--config", "test.yaml", "--port", "9000"])
        assert args.port == 9000


class TestReplayArgs:
    def test_run_required(self):
        with pytest.raises(SystemExit):
            replay_parse_args([])

    def test_default_port(self):
        args = replay_parse_args(["--run", "/tmp/test_run"])
        assert args.port == 8090


class TestReplayDataLoading:
    def test_loads_from_run_directory(self, tmp_path):
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        # Write minimal manifest
        manifest = {"run_id": "test", "status": "complete", "days_total": 3}
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        # Write minimal events
        events = [
            {"timestamp": "2026-01-01T00:00:00Z", "day": 1, "event_type": "day_start"},
            {"timestamp": "2026-01-01T00:00:01Z", "day": 1, "event_type": "day_end"},
            {"timestamp": "2026-01-01T00:00:02Z", "day": 2, "event_type": "day_start"},
            {"timestamp": "2026-01-01T00:00:03Z", "day": 2, "event_type": "day_end"},
        ]
        with open(run_dir / "events.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        data = _load_run_data(run_dir)
        assert data["manifest"]["status"] == "complete"
        assert len(data["events"]) == 4
        assert data["current_state"]["replay_mode"] is True
        assert data["current_state"]["completed"] is True

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_run_data(tmp_path / "nonexistent")

    def test_handles_missing_events(self, tmp_path):
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "test", "status": "crashed"}))
        data = _load_run_data(run_dir)
        assert data["events"] == []
