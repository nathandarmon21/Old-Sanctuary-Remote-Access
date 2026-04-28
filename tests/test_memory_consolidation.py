"""Tests for sanctuary/memory.py and engine integration (spec §7).

The memory consolidation layer adds three injection points to the
tactical prompt: yesterday's summary, the performance ledger, and the
strategic-memo digest. Each is computed deterministically from
existing state — no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sanctuary.config import SimulationConfig, config_to_dict
from sanctuary.engine import SimulationEngine
from sanctuary.memory import (
    build_metric_ledger,
    build_per_day_summary,
    digest_recent_memos,
)
from sanctuary.run_directory import RunDirectory

from tests.test_engine import MockProvider


# ── Pure helpers ──────────────────────────────────────────────────────────────


@dataclass
class _StubPolicy:
    """Stand-in for PolicyRecord in helper unit tests — only the fields
    digest_recent_memos reads."""
    week: int
    day: int
    policy_json: dict


class TestDigestRecentMemos:
    def test_empty_returns_empty(self):
        assert digest_recent_memos([]) == ""

    def test_only_latest_returns_empty(self):
        # If there's only one memo, the tactical prompt already shows it
        # in full via current_policy — nothing to digest.
        history = [_StubPolicy(week=1, day=1, policy_json={"price_floor_excellent": 45.0})]
        assert digest_recent_memos(history) == ""

    def test_compresses_prior_memos(self):
        history = [
            _StubPolicy(week=1, day=1, policy_json={
                "price_floor_excellent": 45.0, "notes": "honest start",
            }),
            _StubPolicy(week=2, day=8, policy_json={
                "price_floor_excellent": 47.0, "notes": "tightening",
            }),
            _StubPolicy(week=3, day=15, policy_json={
                "price_floor_excellent": 49.0,
            }),
        ]
        digest = digest_recent_memos(history)
        # Most recent (week 3) is excluded.
        assert "Week 3" not in digest
        assert "Week 1" in digest
        assert "Week 2" in digest
        assert "honest start" in digest
        assert "tightening" in digest

    def test_keeps_only_last_k(self):
        history = [
            _StubPolicy(week=i, day=i*7, policy_json={"notes": f"memo {i}"})
            for i in range(1, 13)
        ]
        digest = digest_recent_memos(history, k=10)
        # Use unique line markers — "Week 1 " has a trailing space to avoid
        # matching "Week 10"/"Week 11".
        assert "Week 1 " not in digest
        assert "Week 2 " not in digest
        # Most recent memo (week 12) is excluded.
        assert "Week 12" not in digest
        # Weeks 3..11 should appear.
        assert "Week 3 " in digest
        assert "Week 11 " in digest

    def test_long_notes_truncated(self):
        long_note = "x" * 200
        history = [
            _StubPolicy(week=1, day=1, policy_json={"notes": long_note}),
            _StubPolicy(week=2, day=8, policy_json={}),
        ]
        digest = digest_recent_memos(history)
        # Only the truncated form appears (with "...") — never the full 200.
        assert "..." in digest
        assert "x" * 200 not in digest


class TestBuildPerDaySummary:
    def test_empty_day_returns_no_activity(self):
        assert "no activity" in build_per_day_summary("Alice", 5, [])

    def test_summarizes_completed_transactions_for_seller(self):
        events = [
            {
                "event_type": "transaction_completed",
                "seller": "Alice", "buyer": "Bob",
                "quantity": 2, "claimed_quality": "Excellent",
                "price_per_unit": 47.5, "transaction_id": "tx1",
                "true_quality": "Excellent", "revelation_day": 10,
            },
        ]
        s = build_per_day_summary("Alice", 5, events)
        assert "Day 5" in s
        assert "sold 2x Excellent" in s
        assert "Bob" in s
        assert "47.50" in s

    def test_summarizes_for_buyer_too(self):
        events = [
            {
                "event_type": "transaction_completed",
                "seller": "Alice", "buyer": "Bob",
                "quantity": 1, "claimed_quality": "Poor",
                "price_per_unit": 30.0, "transaction_id": "tx1",
                "true_quality": "Poor", "revelation_day": 10,
            },
        ]
        s = build_per_day_summary("Bob", 5, events)
        assert "bought 1x Poor" in s
        assert "Alice" in s

    def test_quality_revelation_in_summary(self):
        events = [
            {
                "event_type": "quality_revealed",
                "seller": "Alice", "buyer": "Bob",
                "transaction_id": "tx1", "quantity": 2,
                "claimed_quality": "Excellent", "true_quality": "Poor",
                "misrepresented": True, "adjustment": -20.0,
            },
        ]
        s = build_per_day_summary("Alice", 5, events)
        assert "revelation" in s
        assert "claimed Excellent" in s
        assert "true Poor" in s

    def test_message_count(self):
        events = [
            {"event_type": "message_sent", "from_agent": "Alice",
             "to_agent": "Bob", "public": False, "body": "hi"},
            {"event_type": "message_sent", "from_agent": "Carol",
             "to_agent": "all", "public": True, "body": "ann"},
        ]
        # Bob receives both (private to Bob, public from Carol).
        s = build_per_day_summary("Bob", 5, events)
        assert "received 2" in s


class TestBuildMetricLedger:
    def _stub_seller_state(self):
        @dataclass
        class _S:
            cash: float = 4500.0
            starting_cash: float = 5000.0
        return _S()

    def _stub_tx(self, day, seller, buyer, qty, price, claimed="Excellent",
                 true="Excellent", tx_id="tx1"):
        @dataclass(frozen=True)
        class _Tx:
            day: int
            seller: str
            buyer: str
            quantity: int
            price_per_unit: float
            claimed_quality: str
            true_quality: str
            transaction_id: str
            revelation_day: int = 0
        return _Tx(day, seller, buyer, qty, price, claimed, true, tx_id)

    def test_none_state_returns_empty(self):
        assert build_metric_ledger(
            "Alice", True, None, [], {}, current_day=10,
        ) == ""

    def test_basic_ledger_format(self):
        state = self._stub_seller_state()
        ledger = build_metric_ledger(
            name="Alice", is_seller=True, state=state,
            transactions=[],
            daily_events={},
            current_day=10,
        )
        assert "Days simulated: 10" in ledger
        assert "$4,500.00" in ledger
        assert "$5,000.00" in ledger
        assert "change -$500.00" in ledger
        assert "Offers placed: 0" in ledger

    def test_offers_placed_counted(self):
        state = self._stub_seller_state()
        events = {
            1: [
                {"event_type": "transaction_proposed",
                 "seller": "Alice", "buyer": "Bob"},
                {"event_type": "transaction_proposed",
                 "seller": "Alice", "buyer": "Carol"},
            ],
            2: [
                {"event_type": "transaction_proposed",
                 "seller": "Other", "buyer": "Alice"},  # not Alice's offer
            ],
        }
        ledger = build_metric_ledger(
            name="Alice", is_seller=True, state=state,
            transactions=[], daily_events=events,
            current_day=2,
        )
        assert "Offers placed: 2" in ledger

    def test_weekly_summary_present(self):
        state = self._stub_seller_state()
        txs = [
            self._stub_tx(day=2, seller="Alice", buyer="Bob",
                          qty=1, price=45.0, tx_id="t1"),
            self._stub_tx(day=5, seller="Alice", buyer="Bob",
                          qty=1, price=47.0, tx_id="t2"),
            self._stub_tx(day=10, seller="Alice", buyer="Carol",
                          qty=1, price=50.0, tx_id="t3"),
        ]
        ledger = build_metric_ledger(
            name="Alice", is_seller=True, state=state,
            transactions=txs, daily_events={},
            current_day=10,
        )
        assert "closed: 3" in ledger
        assert "Recent weeks:" in ledger
        assert "Week 1" in ledger  # days 2 & 5 fall in week 1
        assert "Week 2" in ledger  # day 10 is week 2


# ── Engine integration ────────────────────────────────────────────────────────


def _make_config(days: int = 2, multi_round: bool = False) -> SimulationConfig:
    raw = {
        "run": {
            "days": days,
            "strategic_tier_days": [],
            "max_sub_rounds": 0,
            "max_parallel_llm_calls": 1,
            "multi_round_negotiation": multi_round,
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


def _build_engine(tmp_path: Path, config: SimulationConfig):
    config_dict = config_to_dict(config)
    agent_names = (
        [s.name for s in config.agents.sellers]
        + [b.name for b in config.agents.buyers]
    )
    rd = RunDirectory(tmp_path / "run", config_dict, seed=42, agent_names=agent_names)
    engine = SimulationEngine(config=config, seed=42, run_directory=rd)
    mock = MockProvider()
    engine.strategic_provider = mock
    engine.tactical_provider = mock
    for agent in engine.agents.values():
        agent._strategic_provider = mock
        agent._tactical_provider = mock
    return engine, rd


class TestEngineMemoryIntegration:
    def test_per_day_summary_generated(self, tmp_path):
        config = _make_config(days=2)
        engine, _ = _build_engine(tmp_path, config)
        engine.run()
        # Day 1 summary should be in _daily_summaries for every agent.
        for name in engine.agents:
            assert 1 in engine._daily_summaries[name]
            s = engine._daily_summaries[name][1]
            assert s.startswith("Day 1:")

    def test_metric_ledger_appears_in_tactical_history(self, tmp_path):
        """After day 2, day 2's tactical prompts should include the
        ledger and yesterday's summary (computed from day 1)."""
        config = _make_config(days=2)
        engine, _ = _build_engine(tmp_path, config)
        engine.run()

        # Inspect any agent's tactical_history for the ledger marker.
        seller = engine.agents["Meridian Manufacturing"]
        # Last user content for tactical tier should be day 2's prompt.
        user_msgs = [
            m["content"] for m in seller.tactical_history if m["role"] == "user"
        ]
        assert len(user_msgs) >= 2
        day2_prompt = user_msgs[-1]
        assert "[YESTERDAY'S SUMMARY]" in day2_prompt
        assert "[YOUR PERFORMANCE LEDGER]" in day2_prompt

    def test_no_memory_sections_on_day_one(self, tmp_path):
        """On day 1 there's no yesterday yet, and no closed deals — but
        the ledger still has the cash header so it appears. The
        yesterday-summary section should NOT appear on day 1."""
        config = _make_config(days=1)
        engine, _ = _build_engine(tmp_path, config)
        engine.run()

        seller = engine.agents["Meridian Manufacturing"]
        user_msgs = [
            m["content"] for m in seller.tactical_history if m["role"] == "user"
        ]
        day1_prompt = user_msgs[-1]
        assert "[YESTERDAY'S SUMMARY]" not in day1_prompt
        # Ledger always has the cash header.
        assert "[YOUR PERFORMANCE LEDGER]" in day1_prompt

    def test_memory_works_in_multi_round_path(self, tmp_path):
        """Multi-round path also threads the memory inputs through."""
        config = _make_config(days=2, multi_round=True)
        engine, _ = _build_engine(tmp_path, config)
        engine.run()

        seller = engine.agents["Meridian Manufacturing"]
        user_msgs = [
            m["content"] for m in seller.tactical_history if m["role"] == "user"
        ]
        # At least day 2 prompts should include the ledger.
        joined = "\n".join(user_msgs)
        assert "[YOUR PERFORMANCE LEDGER]" in joined
