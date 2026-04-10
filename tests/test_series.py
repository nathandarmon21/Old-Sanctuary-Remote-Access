"""
Tests for sanctuary/analytics/series.py.

Covers: SeriesTracker accumulation, CSV output format.
"""

from __future__ import annotations

import csv
import io

import pytest

from sanctuary.analytics.series import SeriesTracker


AGENTS = ["Meridian Manufacturing", "Halcyon Assembly"]


class TestSeriesTracker:
    def test_one_day_produces_one_row(self):
        tracker = SeriesTracker(AGENTS)
        tracker.update(
            day=1,
            day_events=[],
            agent_cash={"Meridian Manufacturing": 5000.0, "Halcyon Assembly": 6000.0},
            agent_inventory={"Meridian Manufacturing": 8, "Halcyon Assembly": 0},
            agent_quota={"Meridian Manufacturing": 0, "Halcyon Assembly": 20},
            agent_factories={"Meridian Manufacturing": 1, "Halcyon Assembly": 0},
        )
        csv_text = tracker.to_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 1

    def test_csv_has_expected_columns(self):
        tracker = SeriesTracker(AGENTS)
        tracker.update(
            day=1,
            day_events=[],
            agent_cash={"Meridian Manufacturing": 5000.0, "Halcyon Assembly": 6000.0},
            agent_inventory={"Meridian Manufacturing": 8, "Halcyon Assembly": 0},
            agent_quota={"Meridian Manufacturing": 0, "Halcyon Assembly": 20},
            agent_factories={"Meridian Manufacturing": 1, "Halcyon Assembly": 0},
        )
        csv_text = tracker.to_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        row = rows[0]
        assert "day" in row
        assert "txn_count" in row
        assert "avg_price_excellent" in row
        assert "rolling_5day_misrep_rate" in row
        assert "cash_halcyon_assembly" in row
        assert "cash_meridian_manufacturing" in row

    def test_cash_values_tracked(self):
        tracker = SeriesTracker(AGENTS)
        tracker.update(
            day=1,
            day_events=[],
            agent_cash={"Meridian Manufacturing": 4500.0, "Halcyon Assembly": 5800.0},
            agent_inventory={}, agent_quota={}, agent_factories={},
        )
        csv_text = tracker.to_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        row = list(reader)[0]
        assert float(row["cash_meridian_manufacturing"]) == 4500.0
        assert float(row["cash_halcyon_assembly"]) == 5800.0

    def test_transaction_counted(self):
        tracker = SeriesTracker(AGENTS)
        events = [
            {"event_type": "transaction_completed", "claimed_quality": "Excellent", "price_per_unit": 50.0},
            {"event_type": "transaction_completed", "claimed_quality": "Poor", "price_per_unit": 28.0},
        ]
        tracker.update(day=1, day_events=events, agent_cash={}, agent_inventory={}, agent_quota={}, agent_factories={})
        csv_text = tracker.to_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        row = list(reader)[0]
        assert int(row["txn_count"]) == 2

    def test_empty_tracker_produces_empty_csv(self):
        tracker = SeriesTracker(AGENTS)
        assert tracker.to_csv() == ""

    def test_multiple_days(self):
        tracker = SeriesTracker(AGENTS)
        for day in range(1, 4):
            tracker.update(day=day, day_events=[], agent_cash={}, agent_inventory={}, agent_quota={}, agent_factories={})
        csv_text = tracker.to_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 3
