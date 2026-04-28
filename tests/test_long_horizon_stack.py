"""Cross-cutting integration tests for the long-horizon experiment stack.

The individual deliverables (D1-D6) are unit-tested in their own files.
This file verifies they compose: a multi-round day engine with memory
consolidation, checkpoint/resume, and daily metric snapshots all working
together on the same run.
"""

from __future__ import annotations

import json
from pathlib import Path

from sanctuary.config import SimulationConfig, config_to_dict
from sanctuary.engine import SimulationEngine
from sanctuary.events import read_events
from sanctuary.run_directory import RunDirectory

from tests.test_engine import MockProvider


def _make_long_horizon_config(
    days: int,
    n_sellers: int,
    n_buyers: int,
    *,
    multi_round: bool = True,
    checkpoint_interval: int = 2,
) -> SimulationConfig:
    """Build a stack-full config: multi-round on, memory pieces wired,
    checkpoint enabled, daily metrics by default."""
    seller_pool = [
        "Meridian Manufacturing", "Aldridge Industrial",
        "Crestline Components", "Vector Works",
        "Hartwell Industries", "Bellweather Trading",
    ]
    buyer_pool = [
        "Halcyon Assembly", "Pinnacle Goods",
        "Coastal Fabrication", "Northgate Systems",
        "Greendale Manufacturing", "Lakemont Works",
    ]
    raw = {
        "run": {
            "days": days,
            "strategic_tier_days": [1] if days >= 1 else [],
            "max_sub_rounds": 0,
            "max_parallel_llm_calls": 1,
            "multi_round_negotiation": multi_round,
            "max_negotiation_rounds": 5,
            "checkpoint_interval": checkpoint_interval,
        },
        "models": {
            "strategic": {"provider": "ollama", "model": "mock"},
            "tactical": {"provider": "ollama", "model": "mock"},
        },
        "agents": {
            "sellers": [{"name": n} for n in seller_pool[:n_sellers]],
            "buyers": [{"name": n} for n in buyer_pool[:n_buyers]],
        },
        "protocol": {"system": "no_protocol"},
    }
    return SimulationConfig.model_validate(raw)


def _build_engine(tmp_path: Path, config: SimulationConfig, sub: str = "run"):
    config_dict = config_to_dict(config)
    agent_names = (
        [s.name for s in config.agents.sellers]
        + [b.name for b in config.agents.buyers]
    )
    rd = RunDirectory(tmp_path / sub, config_dict, seed=42, agent_names=agent_names)
    engine = SimulationEngine(config=config, seed=42, run_directory=rd)
    mock = MockProvider()
    engine.strategic_provider = mock
    engine.tactical_provider = mock
    for agent in engine.agents.values():
        agent._strategic_provider = mock
        agent._tactical_provider = mock
    return engine, rd


class TestLongHorizonStack:
    """All of D1-D6 wired together on a 3-day 6+6-agent mock run."""

    def test_full_stack_runs_to_completion(self, tmp_path):
        config = _make_long_horizon_config(days=3, n_sellers=6, n_buyers=6)
        engine, rd = _build_engine(tmp_path, config)
        engine.run()

        events = read_events(rd.run_dir / "events.jsonl")
        # Multi-round events present.
        assert any(e["event_type"] == "negotiation_round_start" for e in events)
        # Checkpoint saved at day 2 (interval=2) and day 3 (final).
        ckpt = [e for e in events if e["event_type"] == "checkpoint_saved"]
        saved_days = sorted(e["day"] for e in ckpt)
        assert saved_days == [2, 3]
        # Daily metrics file written.
        assert (rd.run_dir / "daily_metrics.jsonl").exists()
        # Series CSV written.
        assert (rd.run_dir / "series.csv").exists()
        # Simulation completed cleanly.
        assert any(e["event_type"] == "simulation_end" for e in events)

    def test_memory_sections_in_tactical_prompts(self, tmp_path):
        """By day 2, tactical prompts should carry the per-day summary
        and metric ledger sections (D4)."""
        config = _make_long_horizon_config(days=2, n_sellers=6, n_buyers=6)
        engine, _ = _build_engine(tmp_path, config)
        engine.run()

        joined_user = "\n".join(
            m["content"]
            for agent in engine.agents.values()
            for m in agent.tactical_history
            if m["role"] == "user"
        )
        assert "[YESTERDAY'S SUMMARY]" in joined_user
        assert "[YOUR PERFORMANCE LEDGER]" in joined_user

    def test_daily_metrics_has_full_field_set(self, tmp_path):
        """Every row in daily_metrics.jsonl carries the long-horizon
        analysis fields (cumulative misrep, rolling-7, market share)."""
        config = _make_long_horizon_config(days=3, n_sellers=6, n_buyers=6)
        engine, rd = _build_engine(tmp_path, config)
        engine.run()
        text = (rd.run_dir / "daily_metrics.jsonl").read_text().strip()
        rows = [json.loads(line) for line in text.split("\n")]
        assert len(rows) == 3
        for row in rows:
            for key in (
                "day",
                "cumulative_misrep_rate",
                "rolling_7day_misrep_rate",
                "market_share",
                "txn_count",
            ):
                assert key in row, f"missing {key}"


class TestAgentCountFlexibility:
    """Spec lifted the 4+4 hard requirement to at-least-2+2 in the
    foundation commit; verify the engine handles 2+2, 4+4, and 6+6
    without bespoke wiring per count."""

    def test_two_plus_two_runs(self, tmp_path):
        config = _make_long_horizon_config(days=2, n_sellers=2, n_buyers=2,
                                           multi_round=False, checkpoint_interval=10)
        engine, rd = _build_engine(tmp_path, config)
        engine.run()
        events = read_events(rd.run_dir / "events.jsonl")
        assert any(e["event_type"] == "simulation_end" for e in events)

    def test_six_plus_six_runs(self, tmp_path):
        config = _make_long_horizon_config(days=2, n_sellers=6, n_buyers=6,
                                           multi_round=True, checkpoint_interval=10)
        engine, rd = _build_engine(tmp_path, config)
        engine.run()
        events = read_events(rd.run_dir / "events.jsonl")
        assert any(e["event_type"] == "simulation_end" for e in events)
        # Round 1 of multi-round path should list all 12 agents.
        round_starts = [
            e for e in events if e["event_type"] == "negotiation_round_start"
        ]
        assert any(len(e["eligible"]) == 12 for e in round_starts if e["round"] == 1)


class TestDeterminism:
    """Same config + same seed → same final state. This is the
    invariant chained sbatch jobs depend on."""

    def test_two_fresh_runs_identical_market_state(self, tmp_path):
        config = _make_long_horizon_config(days=3, n_sellers=4, n_buyers=4,
                                           multi_round=True, checkpoint_interval=10)
        engine_a, _ = _build_engine(tmp_path, config, sub="A")
        engine_a.run()
        engine_b, _ = _build_engine(tmp_path, config, sub="B")
        engine_b.run()

        for name in engine_a.market.sellers:
            assert (
                engine_a.market.sellers[name].cash
                == engine_b.market.sellers[name].cash
            ), f"non-determinism: seller {name}"
        for name in engine_a.market.buyers:
            assert (
                engine_a.market.buyers[name].cash
                == engine_b.market.buyers[name].cash
            ), f"non-determinism: buyer {name}"
        # Counters too.
        assert engine_a.total_tactical_calls == engine_b.total_tactical_calls
