"""
Tests for sanctuary/run_directory.py and sanctuary/transcripts.py.

Covers: directory creation, manifest lifecycle, transcript writing,
reasoning log, config persistence, metrics writing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sanctuary.run_directory import RunDirectory
from sanctuary.transcripts import TranscriptWriter, _agent_dir_name


SAMPLE_CONFIG = {
    "run": {"days": 30},
    "models": {
        "strategic": {"provider": "ollama", "model": "qwen2.5:14b"},
        "tactical": {"provider": "ollama", "model": "qwen2.5:7b"},
    },
    "agents": {
        "sellers": [{"name": "Meridian Manufacturing"}],
        "buyers": [{"name": "Halcyon Assembly"}],
    },
}

AGENT_NAMES = ["Meridian Manufacturing", "Halcyon Assembly"]


class TestRunDirectory:
    def test_creates_directory_structure(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES):
            assert (run_dir / "manifest.json").exists()
            assert (run_dir / "config_used.yaml").exists()
            assert (run_dir / "events.jsonl").exists()
            assert (run_dir / "agents").is_dir()
            assert (run_dir / "checkpoints").is_dir()
            assert (run_dir / "agents" / "meridian_manufacturing").is_dir()
            assert (run_dir / "agents" / "halcyon_assembly").is_dir()

    def test_manifest_initial_status(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES) as rd:
            manifest = rd.read_manifest()
            assert manifest["status"] == "running"
            assert manifest["seed"] == 42
            assert manifest["agent_names"] == AGENT_NAMES
            assert manifest["completed_at"] is None

    def test_mark_complete(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES) as rd:
            rd.mark_complete(metrics={"misrepresentation_rate": 0.15})
            manifest = rd.read_manifest()
            assert manifest["status"] == "complete"
            assert manifest["completed_at"] is not None
            assert manifest["metrics"]["misrepresentation_rate"] == 0.15

    def test_mark_crashed(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES) as rd:
            rd.mark_crashed("KeyError: 'missing_field'")
            manifest = rd.read_manifest()
            assert manifest["status"] == "crashed"
            assert "missing_field" in manifest["error"]

    def test_config_used_yaml_matches(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES):
            with open(run_dir / "config_used.yaml") as f:
                loaded = yaml.safe_load(f)
            assert loaded["run"]["days"] == 30

    def test_write_metrics(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES) as rd:
            rd.write_metrics({"ae": 0.85, "ppi": 0.3})
            with open(run_dir / "metrics.json") as f:
                metrics = json.load(f)
            assert metrics["ae"] == 0.85

    def test_write_series(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES) as rd:
            rd.write_series("day,txn_count\n1,3\n2,5\n")
            text = (run_dir / "series.csv").read_text()
            assert "day,txn_count" in text

    def test_write_final_state(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES) as rd:
            rd.write_final_state({"sellers": {}, "buyers": {}})
            with open(run_dir / "final_state.json") as f:
                state = json.load(f)
            assert "sellers" in state

    def test_events_writable(self, tmp_path):
        run_dir = tmp_path / "test_run"
        with RunDirectory(run_dir, SAMPLE_CONFIG, seed=42, agent_names=AGENT_NAMES) as rd:
            rd.events.write_event("simulation_start", day=0, seed=42)
            text = (run_dir / "events.jsonl").read_text()
            assert "simulation_start" in text


class TestTranscriptWriter:
    def test_creates_agent_directories(self, tmp_path):
        agents_dir = tmp_path / "agents"
        with TranscriptWriter(agents_dir, AGENT_NAMES):
            assert (agents_dir / "meridian_manufacturing").is_dir()
            assert (agents_dir / "halcyon_assembly").is_dir()

    def test_tactical_transcript_roundtrip(self, tmp_path):
        agents_dir = tmp_path / "agents"
        with TranscriptWriter(agents_dir, AGENT_NAMES) as tw:
            tw.write_tactical_call(
                agent_id="Meridian Manufacturing",
                prompt_messages=[{"role": "system", "content": "You are..."}],
                response_text="<actions>{}</actions> I decided to produce.",
                parsed_actions={"produce_excellent": 1},
                timing_seconds=2.5,
                prompt_tokens=500,
                completion_tokens=100,
                model="qwen2.5:7b",
                day=3,
            )

        # Read back
        transcript_path = agents_dir / "meridian_manufacturing" / "tactical_transcript.jsonl"
        with open(transcript_path) as f:
            record = json.loads(f.readline())

        assert record["day"] == 3
        assert record["tier"] == "tactical"
        assert record["model"] == "qwen2.5:7b"
        assert len(record["prompt_messages"]) == 1
        assert "I decided to produce" in record["response_text"]
        assert record["parsed_actions"]["produce_excellent"] == 1
        assert record["prompt_tokens"] == 500
        assert record["timing_seconds"] == 2.5

    def test_strategic_transcript_roundtrip(self, tmp_path):
        agents_dir = tmp_path / "agents"
        with TranscriptWriter(agents_dir, AGENT_NAMES) as tw:
            tw.write_strategic_call(
                agent_id="Meridian Manufacturing",
                prompt_messages=[{"role": "system", "content": "CEO prompt"}],
                response_text="<policy>{}</policy> Strategic memo here.",
                parsed_policy={"price_floor_excellent": 45.0},
                timing_seconds=15.3,
                prompt_tokens=2000,
                completion_tokens=800,
                model="qwen2.5:14b",
                day=7,
            )

        transcript_path = agents_dir / "meridian_manufacturing" / "strategic_transcript.jsonl"
        with open(transcript_path) as f:
            record = json.loads(f.readline())

        assert record["tier"] == "strategic"
        assert record["day"] == 7
        assert record["parsed_policy"]["price_floor_excellent"] == 45.0

    def test_reasoning_log_written(self, tmp_path):
        agents_dir = tmp_path / "agents"
        with TranscriptWriter(agents_dir, AGENT_NAMES) as tw:
            tw.write_tactical_call(
                agent_id="Meridian Manufacturing",
                prompt_messages=[],
                response_text="My reasoning about the market...",
                parsed_actions=None,
                timing_seconds=1.0,
                prompt_tokens=100,
                completion_tokens=50,
                model="test",
                day=1,
            )

        reasoning_path = agents_dir / "meridian_manufacturing" / "reasoning_log.jsonl"
        with open(reasoning_path) as f:
            record = json.loads(f.readline())

        assert record["day"] == 1
        assert record["tier"] == "tactical"
        assert "My reasoning" in record["reasoning"]

    def test_full_prompt_preserved(self, tmp_path):
        """Transcript must contain the complete prompt, not truncated."""
        agents_dir = tmp_path / "agents"
        long_prompt = [{"role": "system", "content": "x" * 10000}]
        with TranscriptWriter(agents_dir, AGENT_NAMES) as tw:
            tw.write_tactical_call(
                agent_id="Meridian Manufacturing",
                prompt_messages=long_prompt,
                response_text="short response",
                parsed_actions=None,
                timing_seconds=1.0,
                prompt_tokens=2500,
                completion_tokens=10,
                model="test",
                day=1,
            )

        transcript_path = agents_dir / "meridian_manufacturing" / "tactical_transcript.jsonl"
        with open(transcript_path) as f:
            record = json.loads(f.readline())

        assert len(record["prompt_messages"][0]["content"]) == 10000


class TestAgentDirName:
    def test_simple_name(self):
        assert _agent_dir_name("Meridian Manufacturing") == "meridian_manufacturing"

    def test_special_characters(self):
        assert _agent_dir_name("Vector Works") == "vector_works"

    def test_already_clean(self):
        assert _agent_dir_name("simple") == "simple"
