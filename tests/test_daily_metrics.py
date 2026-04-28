"""Tests for D6: daily metric snapshots.

Covers SeriesTracker's new fields (cumulative misrep, rolling-7, market
share), the JSONL export, and engine integration that writes
daily_metrics.jsonl per run.
"""

from __future__ import annotations

import json
from pathlib import Path

from sanctuary.analytics.series import SeriesTracker
from sanctuary.config import SimulationConfig, config_to_dict
from sanctuary.engine import SimulationEngine
from sanctuary.run_directory import RunDirectory

from tests.test_engine import MockProvider


AGENTS = ["Alice", "Bob", "Carol"]


def _empty_inputs():
    return {
        "agent_cash": {n: 5000.0 for n in AGENTS},
        "agent_inventory": {n: 0 for n in AGENTS},
        "agent_quota": {n: 0 for n in AGENTS},
        "agent_factories": {n: 1 for n in AGENTS},
    }


class TestSeriesTrackerNewFields:
    def test_rolling_7_field_present(self):
        tracker = SeriesTracker(AGENTS)
        tracker.update(day=1, day_events=[], **_empty_inputs())
        rows = tracker.rows()
        assert "rolling_7day_misrep_rate" in rows[0]
        assert "rolling_5day_misrep_rate" in rows[0]  # back-compat
        assert "cumulative_misrep_rate" in rows[0]

    def test_cumulative_misrep_accumulates(self):
        tracker = SeriesTracker(AGENTS)
        # Day 1: 1 honest revelation.
        tracker.update(
            day=1,
            day_events=[{"event_type": "quality_revealed", "misrepresented": False}],
            **_empty_inputs(),
        )
        # Day 2: 1 deceptive revelation.
        tracker.update(
            day=2,
            day_events=[{"event_type": "quality_revealed", "misrepresented": True}],
            **_empty_inputs(),
        )
        rows = tracker.rows()
        assert rows[0]["cumulative_misrep_rate"] == 0.0  # 0/1
        assert rows[1]["cumulative_misrep_rate"] == 0.5  # 1/2

    def test_rolling_7_slides(self):
        tracker = SeriesTracker(AGENTS)
        # Day 1-7: alternating honest/deceptive (4 honest, 3 deceptive).
        for d in range(1, 8):
            misrep = (d % 2 == 0)
            tracker.update(
                day=d,
                day_events=[{"event_type": "quality_revealed", "misrepresented": misrep}],
                **_empty_inputs(),
            )
        # Day 8: 0 events.
        tracker.update(day=8, day_events=[], **_empty_inputs())

        rows = tracker.rows()
        # Day 7: window covers days 1-7 → 3/7 ≈ 0.4286
        assert abs(rows[6]["rolling_7day_misrep_rate"] - 3/7) < 0.01
        # Day 8: window covers days 2-8 → days 2,4,6 deceptive → 3/6 = 0.5
        # (no event on day 8, so 6 entries in window from days 2-7)
        assert abs(rows[7]["rolling_7day_misrep_rate"] - 0.5) < 0.01

    def test_market_share_today(self):
        tracker = SeriesTracker(AGENTS)
        events = [
            {"event_type": "transaction_completed", "seller": "Alice",
             "claimed_quality": "Excellent", "price_per_unit": 50.0},
            {"event_type": "transaction_completed", "seller": "Alice",
             "claimed_quality": "Excellent", "price_per_unit": 48.0},
            {"event_type": "transaction_completed", "seller": "Bob",
             "claimed_quality": "Excellent", "price_per_unit": 52.0},
        ]
        tracker.update(day=1, day_events=events, **_empty_inputs())
        rows = tracker.rows()
        share = rows[0]["market_share"]
        # Tracker rounds to 4 decimal places.
        assert abs(share["Alice"] - 2 / 3) < 1e-3
        assert abs(share["Bob"] - 1 / 3) < 1e-3
        assert "Carol" not in share  # no transactions today

    def test_market_share_empty_when_no_txns(self):
        tracker = SeriesTracker(AGENTS)
        tracker.update(day=1, day_events=[], **_empty_inputs())
        rows = tracker.rows()
        assert rows[0]["market_share"] == {}


class TestSeriesTrackerExports:
    def test_jsonl_round_trips(self):
        tracker = SeriesTracker(AGENTS)
        tracker.update(
            day=1,
            day_events=[{"event_type": "transaction_completed", "seller": "Alice",
                         "claimed_quality": "Excellent", "price_per_unit": 50.0}],
            **_empty_inputs(),
        )
        text = tracker.to_jsonl()
        # One row per line.
        lines = text.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["day"] == 1
        # Nested market_share survives.
        assert parsed["market_share"] == {"Alice": 1.0}

    def test_jsonl_empty_when_no_rows(self):
        tracker = SeriesTracker(AGENTS)
        assert tracker.to_jsonl() == ""

    def test_csv_serializes_market_share_as_json_string(self):
        tracker = SeriesTracker(AGENTS)
        tracker.update(
            day=1,
            day_events=[{"event_type": "transaction_completed", "seller": "Alice",
                         "claimed_quality": "Excellent", "price_per_unit": 50.0}],
            **_empty_inputs(),
        )
        csv_text = tracker.to_csv()
        # Header includes market_share, the value is JSON-encoded.
        assert "market_share" in csv_text.split("\n")[0]
        # Find the data row and pull out the market_share column.
        # Just assert the JSON form is in there.
        assert '{"Alice": 1.0}' in csv_text or "{\"\"Alice\"\":" in csv_text


# ── Engine integration ────────────────────────────────────────────────────────


def _make_config(days: int) -> SimulationConfig:
    raw = {
        "run": {
            "days": days,
            "strategic_tier_days": [],
            "max_sub_rounds": 0,
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


class TestEngineDailyMetrics:
    def test_daily_metrics_jsonl_written(self, tmp_path):
        config = _make_config(days=3)
        engine, rd = _build_engine(tmp_path, config)
        engine.run()

        metrics_path = rd.run_dir / "daily_metrics.jsonl"
        assert metrics_path.exists()
        lines = metrics_path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            row = json.loads(line)
            assert "day" in row
            assert "cumulative_misrep_rate" in row
            assert "rolling_7day_misrep_rate" in row
            assert "market_share" in row

    def test_per_agent_cash_tracked(self, tmp_path):
        config = _make_config(days=2)
        engine, rd = _build_engine(tmp_path, config)
        engine.run()

        rows = engine.series_tracker.rows()
        assert len(rows) == 2
        # Per-agent cash columns present (one per agent, key is sanitized).
        for row in rows:
            for agent in engine.agents:
                safe = agent.lower().replace(" ", "_")
                assert f"cash_{safe}" in row
                assert isinstance(row[f"cash_{safe}"], (int, float))
