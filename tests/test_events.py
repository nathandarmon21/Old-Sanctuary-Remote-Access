"""
Tests for sanctuary/events.py.

Covers: event writing, reading, envelope structure, roundtrip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanctuary.events import EventWriter, read_events, read_events_by_day


@pytest.fixture
def events_file(tmp_path):
    return tmp_path / "events.jsonl"


class TestEventWriter:
    def test_writes_valid_json_lines(self, events_file):
        with EventWriter(events_file) as writer:
            writer.write_event("simulation_start", day=0, seed=42, config="test")
            writer.write_event("day_start", day=1)

        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_envelope_has_required_fields(self, events_file):
        with EventWriter(events_file) as writer:
            event = writer.write_event("day_start", day=5)

        assert "timestamp" in event
        assert event["day"] == 5
        assert event["event_type"] == "day_start"

    def test_payload_included(self, events_file):
        with EventWriter(events_file) as writer:
            event = writer.write_event(
                "agent_turn", day=3,
                agent_id="Meridian Manufacturing",
                tier="tactical",
                reasoning="I should produce Excellent widgets.",
                actions={"produce_excellent": 1},
                model="qwen2.5:7b",
                tokens=150,
                latency=2.5,
            )

        assert event["agent_id"] == "Meridian Manufacturing"
        assert event["tier"] == "tactical"
        assert event["reasoning"] == "I should produce Excellent widgets."
        assert event["tokens"] == 150

    def test_agent_turn_reasoning_not_truncated(self, events_file):
        """Agent turn events must include full reasoning text."""
        long_reasoning = "x" * 10000  # very long
        with EventWriter(events_file) as writer:
            writer.write_event(
                "agent_turn", day=1,
                agent_id="test",
                tier="tactical",
                reasoning=long_reasoning,
            )

        events = read_events(events_file)
        assert len(events[0]["reasoning"]) == 10000

    def test_all_event_types_write(self, events_file):
        """Every event type from the spec writes without error."""
        event_types = [
            ("simulation_start", {"seed": 42}),
            ("simulation_end", {"elapsed": 120.0}),
            ("day_start", {}),
            ("day_end", {}),
            ("agent_turn", {"agent_id": "A", "tier": "tactical", "reasoning": "test"}),
            ("message_sent", {"from_agent": "A", "to_agent": "B", "public": False, "body": "hi"}),
            ("transaction_proposed", {"offer_id": "abc", "seller": "A", "buyer": "B"}),
            ("transaction_completed", {"transaction_id": "tx1", "seller": "A", "buyer": "B", "quantity": 1, "claimed_quality": "Excellent", "price_per_unit": 50.0}),
            ("quality_revealed", {"transaction_id": "tx1", "claimed_quality": "Excellent", "true_quality": "Poor"}),
            ("factory_completed", {"agent_id": "A"}),
            ("bankruptcy", {"agent_id": "A", "final_cash": -6000.0}),
            ("protocol_hook", {"protocol": "no_protocol", "hook": "on_day_end", "output": []}),
            ("cot_flag", {"agent_id": "A", "category": "deception_intent", "evidence": "keyword match"}),
        ]

        with EventWriter(events_file) as writer:
            for event_type, payload in event_types:
                writer.write_event(event_type, day=1, **payload)

        events = read_events(events_file)
        assert len(events) == len(event_types)
        for i, (expected_type, _) in enumerate(event_types):
            assert events[i]["event_type"] == expected_type

    def test_events_flushed_immediately(self, events_file):
        """Events should be readable mid-simulation."""
        writer = EventWriter(events_file)
        writer.write_event("day_start", day=1)

        # Read while writer is still open
        events = read_events(events_file)
        assert len(events) == 1
        writer.close()

    def test_chronological_order(self, events_file):
        with EventWriter(events_file) as writer:
            writer.write_event("day_start", day=1)
            writer.write_event("agent_turn", day=1, agent_id="A", tier="tactical", reasoning="r")
            writer.write_event("day_end", day=1)
            writer.write_event("day_start", day=2)

        events = read_events(events_file)
        assert events[0]["event_type"] == "day_start"
        assert events[0]["day"] == 1
        assert events[-1]["event_type"] == "day_start"
        assert events[-1]["day"] == 2


class TestReadEvents:
    def test_roundtrip(self, events_file):
        with EventWriter(events_file) as writer:
            writer.write_event("simulation_start", day=0, seed=42)
            writer.write_event("day_start", day=1)
            writer.write_event("simulation_end", day=30, elapsed=100.0)

        events = read_events(events_file)
        assert len(events) == 3
        assert events[0]["event_type"] == "simulation_start"
        assert events[0]["seed"] == 42
        assert events[2]["elapsed"] == 100.0

    def test_empty_file(self, events_file):
        events_file.write_text("")
        events = read_events(events_file)
        assert events == []


class TestReadEventsByDay:
    def test_groups_by_day(self, events_file):
        with EventWriter(events_file) as writer:
            writer.write_event("day_start", day=1)
            writer.write_event("agent_turn", day=1, agent_id="A", tier="tactical", reasoning="r")
            writer.write_event("day_start", day=2)
            writer.write_event("agent_turn", day=2, agent_id="B", tier="tactical", reasoning="s")
            writer.write_event("agent_turn", day=2, agent_id="C", tier="tactical", reasoning="t")

        by_day = read_events_by_day(events_file)
        assert len(by_day[1]) == 2
        assert len(by_day[2]) == 3
