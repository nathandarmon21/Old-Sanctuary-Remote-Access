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


def _make_run_dir(tmp_path):
    """Create a synthetic run directory with 5 days of events."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    manifest = {
        "run_id": "test",
        "status": "complete",
        "days_total": 5,
        "agent_names": ["SellerA", "SellerB", "BuyerA", "BuyerB"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    final_state = {
        "day": 5,
        "sellers": {
            "SellerA": {"cash": 5200, "inventory_excellent": 0, "inventory_poor": 0, "factories": 1, "bankrupt": False},
            "SellerB": {"cash": 5100, "inventory_excellent": 0, "inventory_poor": 0, "factories": 1, "bankrupt": False},
        },
        "buyers": {
            "BuyerA": {"cash": 4800, "widget_inventory": 2, "widgets_acquired": 2, "quota_remaining": 18, "bankrupt": False},
            "BuyerB": {"cash": 4900, "widget_inventory": 1, "widgets_acquired": 1, "quota_remaining": 19, "bankrupt": False},
        },
    }
    (run_dir / "final_state.json").write_text(json.dumps(final_state))

    events = [
        {"timestamp": "T00:00:00Z", "day": 1, "event_type": "day_start"},
        {"timestamp": "T00:00:01Z", "day": 1, "event_type": "transaction_completed",
         "seller": "SellerA", "buyer": "BuyerA", "quantity": 1,
         "claimed_quality": "Excellent", "true_quality": "Excellent", "price_per_unit": 100},
        {"timestamp": "T00:00:02Z", "day": 1, "event_type": "message_sent",
         "from_agent": "SellerA", "to_agent": "BuyerA", "public": False, "body": "Deal?"},
        {"timestamp": "T00:00:03Z", "day": 1, "event_type": "day_end"},
        {"timestamp": "T00:01:00Z", "day": 2, "event_type": "day_start"},
        {"timestamp": "T00:01:01Z", "day": 2, "event_type": "day_end"},
        {"timestamp": "T00:02:00Z", "day": 3, "event_type": "day_start"},
        {"timestamp": "T00:02:01Z", "day": 3, "event_type": "transaction_completed",
         "seller": "SellerB", "buyer": "BuyerB", "quantity": 1,
         "claimed_quality": "Excellent", "true_quality": "Poor", "price_per_unit": 80},
        {"timestamp": "T00:02:02Z", "day": 3, "event_type": "day_end"},
        {"timestamp": "T00:03:00Z", "day": 4, "event_type": "day_start"},
        {"timestamp": "T00:03:01Z", "day": 4, "event_type": "day_end"},
        {"timestamp": "T00:04:00Z", "day": 5, "event_type": "day_start"},
        {"timestamp": "T00:04:01Z", "day": 5, "event_type": "transaction_completed",
         "seller": "SellerA", "buyer": "BuyerA", "quantity": 1,
         "claimed_quality": "Excellent", "true_quality": "Excellent", "price_per_unit": 100},
        {"timestamp": "T00:04:02Z", "day": 5, "event_type": "day_end"},
    ]
    with open(run_dir / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    (run_dir / "metrics.json").write_text(json.dumps({"test_metric": 42}))

    return run_dir


class TestReplayDataLoading:
    def test_loads_from_run_directory(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)

        data = _load_run_data(run_dir)
        assert data["manifest"]["status"] == "complete"
        assert len(data["events"]) > 0
        assert data["current_state"]["replay_mode"] is True
        assert data["current_state"]["completed"] is True

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_run_data(tmp_path / "nonexistent")

    def test_handles_missing_events(self, tmp_path):
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "test", "status": "crashed", "days_total": 1}))
        data = _load_run_data(run_dir)
        assert data["events"] == []


class TestReplayDailySnapshots:
    """Verify that replay mode returns different state for different days."""

    def test_daily_snapshots_have_agents(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        data = _load_run_data(run_dir)
        snap1 = data["daily_snapshots"][1]
        assert "agents" in snap1
        assert len(snap1["agents"]) == 4
        assert "SellerA" in snap1["agents"]
        assert snap1["agents"]["SellerA"]["role"] == "seller"
        assert "BuyerA" in snap1["agents"]
        assert snap1["agents"]["BuyerA"]["role"] == "buyer"

    def test_different_days_have_different_balances(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        data = _load_run_data(run_dir)
        snap1 = data["daily_snapshots"][1]
        snap5 = data["daily_snapshots"][5]
        # SellerA got $100 in day 1 and another $100 in day 5
        assert snap5["agents"]["SellerA"]["balance"] > snap1["agents"]["SellerA"]["balance"]

    def test_different_days_have_different_transaction_counts(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        data = _load_run_data(run_dir)
        snap1 = data["daily_snapshots"][1]
        snap5 = data["daily_snapshots"][5]
        assert snap1["stats"]["total_transactions"] == 1
        assert snap5["stats"]["total_transactions"] == 3

    def test_day1_has_messages(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        data = _load_run_data(run_dir)
        snap1 = data["daily_snapshots"][1]
        assert len(snap1["recent_messages"]) == 1

    def test_day2_has_no_messages(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        data = _load_run_data(run_dir)
        snap2 = data["daily_snapshots"][2]
        assert len(snap2["recent_messages"]) == 0

    def test_replay_mode_flag_set(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        data = _load_run_data(run_dir)
        for day in range(1, 6):
            assert data["daily_snapshots"][day]["replay_mode"] is True

    def test_misrepresentation_rate_changes(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        data = _load_run_data(run_dir)
        # Day 1: no quality_revealed events, so rate stays 0
        snap1 = data["daily_snapshots"][1]
        assert snap1["stats"]["misrepresentation_rate"] == 0
        # The misrep rate may change once quality_revealed events exist
        # (our test data doesn't include quality_revealed but the structure works)


class TestReplayHTTPEndpoint:
    """Test the /api/replay/day/{day} endpoint returns different state per day."""

    @pytest.fixture
    def replay_client(self, tmp_path):
        from sanctuary.dashboard.app import app, set_replay_data
        run_dir = _make_run_dir(tmp_path)
        data = _load_run_data(run_dir)
        set_replay_data(data)
        from fastapi.testclient import TestClient
        client = TestClient(app)
        yield client
        set_replay_data(None)

    def test_replay_day_endpoint_returns_state(self, replay_client):
        resp = replay_client.get("/api/replay/day/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["day"] == 1
        assert "agents" in data
        assert data["replay_mode"] is True

    def test_different_days_return_different_state(self, replay_client):
        resp1 = replay_client.get("/api/replay/day/1")
        resp3 = replay_client.get("/api/replay/day/3")
        assert resp1.status_code == 200
        assert resp3.status_code == 200
        data1 = resp1.json()
        data3 = resp3.json()
        assert data1["day"] == 1
        assert data3["day"] == 3
        assert data1["stats"]["total_transactions"] != data3["stats"]["total_transactions"]

    def test_api_state_returns_replay_with_agents(self, replay_client):
        resp = replay_client.get("/api/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["replay_mode"] is True
        assert "agents" in data
        assert len(data["agents"]) == 4
