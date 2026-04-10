"""
Tests for sanctuary/dashboard/app.py.

Covers: API endpoints return valid JSON, protocols endpoint, basic structure.
"""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from sanctuary.dashboard.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestDashboardAPI:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers.get("content-type", "").lower()

    def test_protocols_returns_list(self, client):
        resp = client.get("/api/protocols")
        assert resp.status_code == 200
        data = resp.json()
        assert "protocols" in data
        assert len(data["protocols"]) == 6
        systems = [p["system"] for p in data["protocols"]]
        assert "no_protocol" in systems

    def test_state_without_engine_returns_503(self, client):
        resp = client.get("/api/state")
        assert resp.status_code == 503

    def test_messages_without_engine_returns_empty(self, client):
        resp = client.get("/api/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["messages"] == []

    def test_analytics_without_engine_returns_empty(self, client):
        resp = client.get("/api/analytics")
        assert resp.status_code == 200

    def test_websocket_connects(self, client):
        with client.websocket_connect("/ws") as ws:
            # Should connect without error
            # No init data since no engine is set
            pass
