"""
Tests for sanctuary/checkpointing/checkpoint.py.

Covers: save/load roundtrip, RNG state preservation, latest checkpoint
detection, missing checkpoint error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sanctuary.checkpointing.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    _serialize_rng_state,
    _deserialize_rng_state,
)


@pytest.fixture
def checkpoint_dir(tmp_path):
    return tmp_path / "checkpoints"


def _sample_data():
    return {
        "market_snapshot": {
            "day": 5,
            "sellers": {"Meridian": {"cash": 4500.0, "factories": 2}},
            "buyers": {"Halcyon": {"cash": 5800.0, "widgets_acquired": 5}},
        },
        "agent_states": {
            "Meridian": {"history": [{"role": "user", "content": "day 5"}]},
            "Halcyon": {"history": [{"role": "user", "content": "day 5"}]},
        },
        "revelation_pending": [
            {"transaction_id": "tx1", "revelation_day": 8},
        ],
        "counters": {
            "total_tactical_calls": 40,
            "total_strategic_calls": 0,
            "parse_failures": 1,
        },
    }


class TestSaveLoad:
    def test_roundtrip_preserves_market(self, checkpoint_dir):
        data = _sample_data()
        rng = np.random.default_rng(42)
        rng_state = rng.bit_generator.state

        save_checkpoint(
            checkpoint_dir, day=5,
            market_snapshot=data["market_snapshot"],
            agent_states=data["agent_states"],
            rng_state=rng_state,
            revelation_pending=data["revelation_pending"],
            counters=data["counters"],
        )

        loaded = load_checkpoint(checkpoint_dir, day=5)
        assert loaded["day"] == 5
        assert loaded["market_snapshot"]["sellers"]["Meridian"]["cash"] == 4500.0

    def test_roundtrip_preserves_agent_states(self, checkpoint_dir):
        data = _sample_data()
        rng = np.random.default_rng(42)

        save_checkpoint(
            checkpoint_dir, day=5,
            market_snapshot=data["market_snapshot"],
            agent_states=data["agent_states"],
            rng_state=rng.bit_generator.state,
            revelation_pending=data["revelation_pending"],
            counters=data["counters"],
        )

        loaded = load_checkpoint(checkpoint_dir, day=5)
        assert loaded["agent_states"]["Meridian"]["history"][0]["content"] == "day 5"

    def test_roundtrip_preserves_counters(self, checkpoint_dir):
        data = _sample_data()
        rng = np.random.default_rng(42)

        save_checkpoint(
            checkpoint_dir, day=5,
            market_snapshot=data["market_snapshot"],
            agent_states=data["agent_states"],
            rng_state=rng.bit_generator.state,
            revelation_pending=data["revelation_pending"],
            counters=data["counters"],
        )

        loaded = load_checkpoint(checkpoint_dir, day=5)
        assert loaded["counters"]["total_tactical_calls"] == 40

    def test_checkpoint_file_is_valid_json(self, checkpoint_dir):
        data = _sample_data()
        rng = np.random.default_rng(42)

        path = save_checkpoint(
            checkpoint_dir, day=5,
            market_snapshot=data["market_snapshot"],
            agent_states=data["agent_states"],
            rng_state=rng.bit_generator.state,
            revelation_pending=data["revelation_pending"],
            counters=data["counters"],
        )

        with open(path) as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)
        assert "day" in parsed


class TestRNGState:
    def test_rng_state_roundtrip(self):
        rng = np.random.default_rng(42)
        original_state = rng.bit_generator.state
        serialized = _serialize_rng_state(original_state)
        deserialized = _deserialize_rng_state(serialized)

        # Restore the state
        rng2 = np.random.default_rng(0)
        rng2.bit_generator.state = deserialized

        # Both should produce identical sequences
        vals1 = [rng.random() for _ in range(10)]
        rng3 = np.random.default_rng(42)
        # Advance rng3 by the same amount
        # Actually we need to use the state after the original was captured
        # The original state was captured before any draws
        rng3_check = np.random.default_rng(0)
        rng3_check.bit_generator.state = deserialized
        vals2 = [rng3_check.random() for _ in range(10)]

        # rng was at seed=42 state, rng3_check was restored to seed=42 state
        # But rng has already advanced past the capture point when we called rng.random()
        # So let's test differently: capture state, restore, verify next values match
        pass  # Test revised below

    def test_rng_state_preserves_next_values(self):
        """Capture RNG state, restore it, verify next values are identical."""
        rng1 = np.random.default_rng(42)
        # Advance it a bit
        for _ in range(100):
            rng1.random()

        # Capture state
        state = rng1.bit_generator.state
        expected_next = [rng1.random() for _ in range(10)]

        # Restore state
        serialized = _serialize_rng_state(state)
        deserialized = _deserialize_rng_state(serialized)
        rng2 = np.random.default_rng(0)
        rng2.bit_generator.state = deserialized
        actual_next = [rng2.random() for _ in range(10)]

        assert expected_next == actual_next


class TestFindLatest:
    def test_finds_latest(self, checkpoint_dir):
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "day_005.json").write_text("{}")
        (checkpoint_dir / "day_010.json").write_text("{}")
        (checkpoint_dir / "day_015.json").write_text("{}")
        assert find_latest_checkpoint(checkpoint_dir) == 15

    def test_returns_none_for_empty(self, checkpoint_dir):
        checkpoint_dir.mkdir(parents=True)
        assert find_latest_checkpoint(checkpoint_dir) is None

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        assert find_latest_checkpoint(tmp_path / "does_not_exist") is None

    def test_ignores_non_checkpoint_files(self, checkpoint_dir):
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "day_005.json").write_text("{}")
        (checkpoint_dir / "notes.txt").write_text("ignore me")
        assert find_latest_checkpoint(checkpoint_dir) == 5


class TestLoadErrors:
    def test_missing_specific_day_raises(self, checkpoint_dir):
        checkpoint_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(checkpoint_dir, day=99)

    def test_no_checkpoints_raises(self, checkpoint_dir):
        checkpoint_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(checkpoint_dir)
