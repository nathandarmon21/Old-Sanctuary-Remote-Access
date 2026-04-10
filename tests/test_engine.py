"""
Tests for sanctuary/engine.py.

Covers: engine construction, basic run with mock provider, event generation,
strategic/tactical dispatch, protocol hooks, revelation timing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sanctuary.config import SimulationConfig
from sanctuary.engine import SimulationEngine
from sanctuary.events import read_events
from sanctuary.providers.base import ModelProvider, ModelResponse
from sanctuary.protocols.no_protocol import NoProtocol
from sanctuary.run_directory import RunDirectory


# -- Mock provider that returns valid but minimal responses -------------------

class MockProvider(ModelProvider):
    """Returns deterministic canned responses based on system prompt content."""

    def __init__(self):
        super().__init__(model="mock", temperature=0.0, seed=0)

    SELLER_TACTICAL = """
<actions>
{
  "messages": [],
  "offers": [],
  "accept_offers": [],
  "decline_offers": [],
  "produce_excellent": 1,
  "produce_poor": 0,
  "build_factory": false
}
</actions>
Producing one Excellent widget today.
"""

    BUYER_TACTICAL = """
<actions>
{
  "messages": [],
  "buyer_offers": [],
  "accept_offers": [],
  "decline_offers": [],
  "produce_final_goods": 0
}
</actions>
Waiting for offers today.
"""

    SELLER_STRATEGIC = """
<policy>
{
  "price_floor_excellent": 45.0,
  "price_ceiling_excellent": 60.0,
  "quality_stance": "honest",
  "notes": "Focus on volume"
}
</policy>
Week 1 strategic memo: focus on building market share through honest dealing.
"""

    BUYER_STRATEGIC = """
<policy>
{
  "max_price_excellent": 50.0,
  "urgency": "moderate",
  "notes": "Steady acquisition"
}
</policy>
Week 1 strategic memo: acquire widgets at reasonable prices.
"""

    @property
    def provider_name(self) -> str:
        return "mock"

    def complete(self, system_prompt: str, history: list, max_tokens: int = 1024) -> ModelResponse:
        # Detect tier from system prompt content
        if "CEO" in system_prompt or "strategic" in system_prompt.lower():
            if "seller" in system_prompt.lower() or "widget seller" in system_prompt.lower():
                text = self.SELLER_STRATEGIC
            else:
                text = self.BUYER_STRATEGIC
        else:
            if "seller" in system_prompt.lower() or "widget seller" in system_prompt.lower():
                text = self.SELLER_TACTICAL
            else:
                text = self.BUYER_TACTICAL

        return ModelResponse(
            completion=text.strip(),
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_seconds=0.01,
            model="mock",
            provider="mock",
        )


def _make_config(days: int = 3, strategic_days: list[int] | None = None) -> SimulationConfig:
    """Build a minimal config for testing."""
    raw = {
        "run": {
            "days": days,
            "strategic_tier_days": strategic_days or [],
            "max_sub_rounds": 1,
            "max_parallel_llm_calls": 1,
        },
        "models": {
            "strategic": {"provider": "ollama", "model": "mock"},
            "tactical": {"provider": "ollama", "model": "mock"},
        },
        "agents": {
            "sellers": [
                {"name": "Meridian Manufacturing"},
                {"name": "Aldridge Industrial"},
                {"name": "Crestline Components"},
                {"name": "Vector Works"},
            ],
            "buyers": [
                {"name": "Halcyon Assembly"},
                {"name": "Pinnacle Goods"},
                {"name": "Coastal Fabrication"},
                {"name": "Northgate Systems"},
            ],
        },
        "protocol": {"system": "no_protocol"},
    }
    return SimulationConfig.model_validate(raw)


def _run_engine(tmp_path: Path, days: int = 3, strategic_days: list[int] | None = None) -> tuple[SimulationEngine, RunDirectory]:
    """Create and run an engine with mock providers."""
    config = _make_config(days=days, strategic_days=strategic_days)
    from sanctuary.config import config_to_dict
    config_dict = config_to_dict(config)
    agent_names = [s.name for s in config.agents.sellers] + [b.name for b in config.agents.buyers]

    run_dir = tmp_path / "test_run"
    rd = RunDirectory(run_dir, config_dict, seed=42, agent_names=agent_names)

    engine = SimulationEngine(config=config, seed=42, run_directory=rd)

    # Patch providers with mock
    mock = MockProvider()
    engine.strategic_provider = mock
    engine.tactical_provider = mock
    for agent in engine.agents.values():
        agent._strategic_provider = mock
        agent._tactical_provider = mock

    engine.run()
    return engine, rd


class TestEngineBasic:
    def test_runs_to_completion(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=3)
        manifest = rd.read_manifest()
        assert manifest["status"] == "running"  # mark_complete called by caller

    def test_events_log_has_start_and_end(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=3)
        events = read_events(rd.run_dir / "events.jsonl")
        types = [e["event_type"] for e in events]
        assert "simulation_start" in types
        assert "simulation_end" in types

    def test_events_log_has_day_boundaries(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=3)
        events = read_events(rd.run_dir / "events.jsonl")
        day_starts = [e for e in events if e["event_type"] == "day_start"]
        day_ends = [e for e in events if e["event_type"] == "day_end"]
        assert len(day_starts) == 3
        assert len(day_ends) == 3

    def test_tactical_calls_every_day(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=3)
        events = read_events(rd.run_dir / "events.jsonl")
        tactical_turns = [e for e in events if e["event_type"] == "agent_turn" and e.get("tier") == "tactical"]
        # 8 agents x 3 days = 24 tactical turns
        assert len(tactical_turns) == 24

    def test_strategic_calls_on_configured_days(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=7, strategic_days=[7])
        events = read_events(rd.run_dir / "events.jsonl")
        strategic_turns = [e for e in events if e["event_type"] == "agent_turn" and e.get("tier") == "strategic"]
        # 8 agents x 1 strategic day = 8
        assert len(strategic_turns) == 8

    def test_no_strategic_calls_when_not_configured(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=3, strategic_days=[])
        events = read_events(rd.run_dir / "events.jsonl")
        strategic_turns = [e for e in events if e["event_type"] == "agent_turn" and e.get("tier") == "strategic"]
        assert len(strategic_turns) == 0

    def test_day1_strategic_runs(self, tmp_path):
        """Day 1 strategic review fires when day 1 is in strategic_tier_days."""
        engine, rd = _run_engine(tmp_path, days=3, strategic_days=[1])
        events = read_events(rd.run_dir / "events.jsonl")
        strategic_turns = [e for e in events if e["event_type"] == "agent_turn" and e.get("tier") == "strategic"]
        # 8 agents x 1 strategic day (day 1) = 8
        assert len(strategic_turns) == 8
        # All strategic turns should be on day 1
        assert all(e["day"] == 1 for e in strategic_turns)

    def test_day1_strategic_before_tactical(self, tmp_path):
        """On day 1, strategic turns appear before tactical turns in the event log."""
        engine, rd = _run_engine(tmp_path, days=1, strategic_days=[1])
        events = read_events(rd.run_dir / "events.jsonl")
        turns = [e for e in events if e["event_type"] == "agent_turn" and e["day"] == 1]
        strategic_indices = [i for i, e in enumerate(turns) if e.get("tier") == "strategic"]
        tactical_indices = [i for i, e in enumerate(turns) if e.get("tier") == "tactical"]
        assert len(strategic_indices) == 8
        assert len(tactical_indices) == 8
        # Every strategic turn should come before every tactical turn
        assert max(strategic_indices) < min(tactical_indices)

    def test_no_price_update_events(self, tmp_path):
        """Brownian price drift removed -- no price_update events."""
        engine, rd = _run_engine(tmp_path, days=3)
        events = read_events(rd.run_dir / "events.jsonl")
        price_updates = [e for e in events if e["event_type"] == "price_update"]
        assert len(price_updates) == 0

    def test_revelation_at_day_plus_five(self, tmp_path):
        """If a transaction happens on day 1, revelation fires on day 6."""
        # This requires a transaction to happen. With mock provider,
        # sellers produce but don't offer, so no transactions occur.
        # We verify the revelation scheduler is deterministic.
        from sanctuary.revelation import RevelationScheduler
        sched = RevelationScheduler()
        rev_day = sched.schedule("tx1", "S", "B", "E", "P", 1, 1)
        assert rev_day == 6

    def test_asymmetric_seller_cash(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=1)
        sellers = engine.market.sellers
        cash_values = sorted([s.cash for s in sellers.values()], reverse=True)
        # After 1 day of holding costs and production, cash should still
        # reflect the asymmetric starting values approximately
        assert len(set(cash_values)) > 1  # not all the same

    def test_transcripts_written(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=1)
        # Check that at least one agent has a tactical transcript
        agents_dir = rd.run_dir / "agents"
        transcript_files = list(agents_dir.glob("*/tactical_transcript.jsonl"))
        assert len(transcript_files) > 0
        # Check file is non-empty
        with open(transcript_files[0]) as f:
            first_line = f.readline()
        assert len(first_line) > 0
        record = json.loads(first_line)
        assert record["tier"] == "tactical"

    def test_agent_turn_reasoning_not_truncated(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=1)
        events = read_events(rd.run_dir / "events.jsonl")
        turns = [e for e in events if e["event_type"] == "agent_turn"]
        assert len(turns) > 0
        # Check that reasoning is the full response text
        for turn in turns:
            assert len(turn.get("reasoning", "")) > 10

    def test_counters_incremented(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=3)
        assert engine.total_tactical_calls == 24  # 8 agents x 3 days
        assert engine.total_prompt_tokens > 0
        assert engine.total_completion_tokens > 0


class _MessagingMockProvider(MockProvider):
    """MockProvider variant whose sellers send a message each tactical turn."""

    SELLER_TACTICAL = """
<actions>
{
  "messages": [{"to": "Halcyon Assembly", "body": "Let's trade!", "public": false}],
  "offers": [],
  "accept_offers": [],
  "decline_offers": [],
  "produce_excellent": 1,
  "produce_poor": 0,
  "build_factory": false
}
</actions>
Sending a private message to Halcyon Assembly.
"""

    def complete(self, system_prompt: str, history: list, max_tokens: int = 1024) -> ModelResponse:
        # Use "You are the CEO" to identify strategic prompts (more precise
        # than checking for "CEO" anywhere, which also matches tactical
        # prompts that mention "YOUR CEO has set a strategic direction").
        is_strategic = "You are the CEO" in system_prompt
        is_seller = "seller" in system_prompt.lower()

        if is_strategic:
            text = self.SELLER_STRATEGIC if is_seller else self.BUYER_STRATEGIC
        else:
            text = self.SELLER_TACTICAL if is_seller else self.BUYER_TACTICAL

        return ModelResponse(
            completion=text.strip(),
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_seconds=0.01,
            model="mock",
            provider="mock",
        )


class TestEngineMessaging:
    """Verify that the messaging path (router → event log) works end to end."""

    def test_messages_logged_as_events(self, tmp_path):
        """When agents send messages, they appear as message_sent events
        with correct sender/recipient/public/body fields."""
        config = _make_config(days=1)
        from sanctuary.config import config_to_dict
        config_dict = config_to_dict(config)
        agent_names = (
            [s.name for s in config.agents.sellers]
            + [b.name for b in config.agents.buyers]
        )

        run_dir = tmp_path / "test_msg_run"
        rd = RunDirectory(run_dir, config_dict, seed=42, agent_names=agent_names)
        engine = SimulationEngine(config=config, seed=42, run_directory=rd)

        mock = _MessagingMockProvider()
        engine.strategic_provider = mock
        engine.tactical_provider = mock
        for agent in engine.agents.values():
            agent._strategic_provider = mock
            agent._tactical_provider = mock

        engine.run()

        events = read_events(rd.run_dir / "events.jsonl")
        msg_events = [e for e in events if e["event_type"] == "message_sent"]

        # All 4 sellers send a message → at least 4 message_sent events
        assert len(msg_events) >= 4

        # Verify fields are populated correctly (not "?" fallbacks)
        for evt in msg_events:
            assert evt["from_agent"] != "?"
            assert evt["to_agent"] != "?"
            assert evt["body"] != ""
            assert isinstance(evt["public"], bool)

        # Check a specific seller→buyer message exists
        meridian_msgs = [e for e in msg_events if e["from_agent"] == "Meridian Manufacturing"]
        assert len(meridian_msgs) >= 1
        assert meridian_msgs[0]["to_agent"] == "Halcyon Assembly"
        assert meridian_msgs[0]["body"] == "Let's trade!"
        assert meridian_msgs[0]["public"] is False


class TestEngineProtocol:
    def test_protocol_name_in_start_event(self, tmp_path):
        engine, rd = _run_engine(tmp_path, days=1)
        events = read_events(rd.run_dir / "events.jsonl")
        start = [e for e in events if e["event_type"] == "simulation_start"][0]
        assert start["protocol"] == "no_protocol"
