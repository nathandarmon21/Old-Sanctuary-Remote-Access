"""Tests for checkpoint/resume integration with the engine.

Covers:
- atomic write (the canonical file is only the final, valid JSON)
- prune_old_checkpoints keeps last K
- try_resume returns None for empty/missing dir, dict otherwise
- engine writes checkpoint files at the configured interval
- end-to-end round-trip: save → fresh engine → restore → state matches
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sanctuary.checkpointing.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_old_checkpoints,
    save_checkpoint,
    try_resume,
)
from sanctuary.checkpointing.serialize import (
    deserialize_market,
    serialize_agent,
    serialize_market,
)
from sanctuary.config import SimulationConfig, config_to_dict
from sanctuary.engine import SimulationEngine
from sanctuary.events import read_events
from sanctuary.run_directory import RunDirectory

from tests.test_engine import MockProvider


def _empty_save(checkpoint_dir: Path, day: int, keep: int | None = 3) -> Path:
    rng = np.random.default_rng(0)
    return save_checkpoint(
        checkpoint_dir, day=day,
        market_snapshot={"sellers": {}, "buyers": {}, "fg_prices": {},
                         "current_day": day, "pending_offers": {}, "transactions": []},
        agent_states={}, rng_state=rng.bit_generator.state,
        revelation_pending=[], counters={}, engine_state={},
        keep=keep,
    )


class TestAtomicWriteAndPrune:
    def test_save_produces_only_canonical_file(self, tmp_path):
        cdir = tmp_path / "cp"
        _empty_save(cdir, day=1)
        files = sorted(p.name for p in cdir.iterdir())
        # No .tmp leftovers.
        assert all(not f.startswith(".") for f in files)
        assert files == ["day_001.json"]

    def test_save_overwrites_existing_atomically(self, tmp_path):
        cdir = tmp_path / "cp"
        _empty_save(cdir, day=1)
        # Mutate to a known marker, save again.
        rng = np.random.default_rng(0)
        save_checkpoint(
            cdir, day=1,
            market_snapshot={"current_day": 99, "sellers": {}, "buyers": {},
                             "fg_prices": {}, "pending_offers": {}, "transactions": []},
            agent_states={}, rng_state=rng.bit_generator.state,
            revelation_pending=[], counters={}, engine_state={},
        )
        loaded = load_checkpoint(cdir, day=1)
        assert loaded["market_snapshot"]["current_day"] == 99

    def test_prune_keeps_last_k(self, tmp_path):
        cdir = tmp_path / "cp"
        # Disable inline prune in the setup so all 5 files exist before
        # we call prune_old_checkpoints explicitly.
        for d in (1, 2, 3, 4, 5):
            _empty_save(cdir, day=d, keep=None)
        deleted = prune_old_checkpoints(cdir, keep=3)
        remaining = sorted(p.name for p in cdir.glob("day_*.json"))
        assert remaining == ["day_003.json", "day_004.json", "day_005.json"]
        assert len(deleted) == 2

    def test_save_with_keep_prunes_inline(self, tmp_path):
        cdir = tmp_path / "cp"
        for d in (1, 2, 3, 4, 5):
            _empty_save(cdir, day=d)
        # Prune happened on each save; only last 3 remain.
        remaining = sorted(p.name for p in cdir.glob("day_*.json"))
        assert remaining == ["day_003.json", "day_004.json", "day_005.json"]


class TestTryResume:
    def test_returns_none_for_missing_dir(self, tmp_path):
        assert try_resume(tmp_path / "nope") is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        cdir = tmp_path / "cp"
        cdir.mkdir()
        assert try_resume(cdir) is None

    def test_returns_dict_for_existing(self, tmp_path):
        cdir = tmp_path / "cp"
        _empty_save(cdir, day=7)
        result = try_resume(cdir)
        assert result is not None
        assert result["day"] == 7

    def test_returns_latest_when_multiple(self, tmp_path):
        cdir = tmp_path / "cp"
        for d in (1, 5, 10, 3):
            _empty_save(cdir, day=d)
        result = try_resume(cdir)
        assert result["day"] == 10


# ── Engine integration ───────────────────────────────────────────────────────


def _make_config(days: int, checkpoint_interval: int = 2) -> SimulationConfig:
    raw = {
        "run": {
            "days": days,
            "strategic_tier_days": [],
            "max_sub_rounds": 0,
            "max_parallel_llm_calls": 1,
            "checkpoint_interval": checkpoint_interval,
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


class TestEngineCheckpointing:
    def test_checkpoint_files_written_at_interval(self, tmp_path):
        config = _make_config(days=5, checkpoint_interval=2)
        engine, rd = _build_engine(tmp_path, config)
        engine.run()

        cdir = engine.checkpoint_dir
        assert cdir.exists()
        # day 2, day 4, day 5 (always on final day) — but prune keeps last 3,
        # so all three remain.
        days = sorted(int(p.stem.split("_")[1]) for p in cdir.glob("day_*.json"))
        assert days == [2, 4, 5]

        events = read_events(rd.run_dir / "events.jsonl")
        saved = [e for e in events if e["event_type"] == "checkpoint_saved"]
        assert {e["day"] for e in saved} == {2, 4, 5}

    def test_resume_picks_up_at_next_day(self, tmp_path):
        # Run for 3 days, save checkpoint at day 3.
        config = _make_config(days=3, checkpoint_interval=3)
        engine_a, rd_a = _build_engine(tmp_path, config, sub="phase1")
        engine_a.run()
        cdir_phase1 = engine_a.checkpoint_dir

        # Build a "phase 2" engine in a separate run dir, copy the checkpoint
        # in, configure for 5-day total run, resume.
        config_b = _make_config(days=5, checkpoint_interval=10)
        engine_b, rd_b = _build_engine(tmp_path, config_b, sub="phase2")
        engine_b.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for src in cdir_phase1.glob("day_*.json"):
            dst = engine_b.checkpoint_dir / src.name
            dst.write_bytes(src.read_bytes())

        engine_b.run()

        events = read_events(rd_b.run_dir / "events.jsonl")
        # Resume event present.
        resumed = [e for e in events if e["event_type"] == "checkpoint_resumed"]
        assert len(resumed) == 1
        assert resumed[0]["resume_from_day"] == 4

        # No simulation_start event (we resumed mid-run).
        starts = [e for e in events if e["event_type"] == "simulation_start"]
        assert starts == []

        # Day 4 and 5 ran; day 1-3 did not.
        agent_turns = [
            e for e in events
            if e["event_type"] == "agent_turn" and e.get("tier") == "tactical"
        ]
        days_run = sorted({e["day"] for e in agent_turns})
        assert days_run == [4, 5]

    def test_restore_preserves_market_and_counters(self, tmp_path):
        """Capture state at day 3 and verify a fresh engine restored
        from that snapshot has the same state encoded in the snapshot.

        Note: we compare engine_b against the snapshot dict, not against
        engine_a's live post-run state, because engine_a continues past
        day 3 (here days==3 so it then runs apply_end_of_run_write_offs
        which clears inventory). The snapshot was taken before that
        cleanup, so it's the canonical day-3 state.
        """
        config = _make_config(days=3, checkpoint_interval=3)
        engine_a, rd_a = _build_engine(tmp_path, config, sub="A")
        engine_a.run()

        snapshot = load_checkpoint(engine_a.checkpoint_dir, day=3)

        config_b = _make_config(days=3, checkpoint_interval=3)
        engine_b, rd_b = _build_engine(tmp_path, config_b, sub="B")
        engine_b._restore_from_checkpoint(snapshot)

        snap_sellers = snapshot["market_snapshot"]["sellers"]
        snap_buyers = snapshot["market_snapshot"]["buyers"]
        for name, sd in snap_sellers.items():
            sb = engine_b.market.sellers[name]
            assert sb.cash == sd["cash"]
            assert sb.factories == sd["factories"]
            assert sb.inventory == sd["inventory"]
            assert len(sb.widget_instances) == len(sd["widget_instances"])
        for name, bd in snap_buyers.items():
            bb = engine_b.market.buyers[name]
            assert bb.cash == bd["cash"]
            assert bb.widgets_acquired == bd["widgets_acquired"]

        snap_counters = snapshot["counters"]
        assert engine_b.total_tactical_calls == snap_counters["total_tactical_calls"]
        assert engine_b.total_prompt_tokens == snap_counters["total_prompt_tokens"]

        # Agent histories restored from the snapshot.
        for name, ad in snapshot["agent_states"].items():
            assert len(engine_b.agents[name].history) == len(ad["history"])

        # Day pointer.
        assert engine_b.current_day == 3
        assert engine_b._resumed_from_day == 3

    def test_resume_then_continue_matches_uninterrupted(self, tmp_path):
        """A 5-day uninterrupted run should produce the same final
        market state as a 3-day run + resume + 2-day continuation."""
        # Uninterrupted reference run.
        config_ref = _make_config(days=5, checkpoint_interval=10)
        engine_ref, _ = _build_engine(tmp_path, config_ref, sub="ref")
        engine_ref.run()

        # Interrupted: run 3 days, then resume into a fresh engine and
        # run for 2 more days (total 5).
        config_a = _make_config(days=3, checkpoint_interval=3)
        engine_a, rd_a = _build_engine(tmp_path, config_a, sub="A")
        engine_a.run()

        config_b = _make_config(days=5, checkpoint_interval=10)
        engine_b, rd_b = _build_engine(tmp_path, config_b, sub="B")
        # Wire the phase-1 checkpoint into the phase-2 dir.
        engine_b.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for src in engine_a.checkpoint_dir.glob("day_*.json"):
            (engine_b.checkpoint_dir / src.name).write_bytes(src.read_bytes())
        engine_b.run()

        # Final cash for each seller should match between the two paths.
        # (The MockProvider is deterministic; production decisions don't
        # depend on RNG draws beyond what's restored.)
        for name in engine_ref.market.sellers:
            assert (
                engine_b.market.sellers[name].cash
                == engine_ref.market.sellers[name].cash
            ), f"cash mismatch for {name}"
            assert (
                engine_b.market.sellers[name].inventory
                == engine_ref.market.sellers[name].inventory
            ), f"inventory mismatch for {name}"
        for name in engine_ref.market.buyers:
            assert (
                engine_b.market.buyers[name].cash
                == engine_ref.market.buyers[name].cash
            ), f"buyer cash mismatch for {name}"
