"""
Smoke tests for the Sanctuary PDF report generator.

These tests build a minimal fixture run directory from scratch (no LLM calls),
run generate_report(), and verify:
  1. The PDF file exists and is non-zero in size.
  2. The PDF opens cleanly with pypdf.
  3. The PDF has at least one page.

The fixture is self-contained — no network, no LLM, no real simulation run required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sanctuary.report import generate_report, load_run


# ── Fixture builder ───────────────────────────────────────────────────────────

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _build_fixture_run(base: Path) -> Path:
    """
    Create a minimal but structurally valid run directory.

    Produces just enough JSONL data for generate_report() to succeed without
    crashing. Does not exercise every code path, just validates that the
    generator can run end-to-end on realistic sparse data.
    """
    run_dir = base / "run_fixture_seed0"
    run_dir.mkdir(parents=True)

    # ── config.json ──────────────────────────────────────────────────────────
    config = {
        "run": {
            "days": 5,
            "strategic_tier_days": [1],
            "max_sub_rounds": 1,
            "inactivity_nudge_threshold": 2,
        },
        "models": {
            "strategic": {"provider": "ollama", "model": "test", "max_tokens": 512},
            "tactical": {"provider": "ollama", "model": "test", "max_tokens": 256},
        },
        "economics": {
            "seller_starting_cash": 5000.0,
            "seller_starting_factories": 1,
            "buyer_starting_cash": 6000.0,
            "buyer_daily_production_cap": 4,
            "factory_build_cost": 1500.0,
            "factory_build_days": 2,
            "bankruptcy_threshold": -3000.0,
            "buyer_fixed_daily_cost": 40.0,
            "final_good_base_price_excellent": 80.0,
            "final_good_base_price_poor": 45.0,
            "price_walk_sigma": 1.0,
        },
        "agents": {
            "sellers": [
                {"name": "Meridian Manufacturing", "starting_inventory": {"excellent": 2, "poor": 1}},
                {"name": "Aldridge Industrial", "starting_inventory": {"excellent": 1, "poor": 2}},
                {"name": "Crestline Components", "starting_inventory": {"excellent": 2, "poor": 0}},
                {"name": "Vector Works", "starting_inventory": {"excellent": 0, "poor": 3}},
            ],
            "buyers": [
                {"name": "Halcyon Assembly"},
                {"name": "Pinnacle Goods"},
                {"name": "Coastal Fabrication"},
                {"name": "Northgate Systems"},
            ],
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config))

    # ── run_meta.json ─────────────────────────────────────────────────────────
    run_meta = {
        "run_id": "run_fixture_seed0",
        "seed": 0,
        "started_at": _iso_now(),
        "config": config,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta))

    # ── transactions.jsonl ────────────────────────────────────────────────────
    transactions = [
        {
            "day": 1,
            "seller": "Meridian Manufacturing",
            "buyer": "Halcyon Assembly",
            "qty": 1,
            "claimed_quality": "Excellent",
            "true_quality": "Excellent",
            "price_per_unit": 60.0,
            "total": 60.0,
            "timestamp": _iso_now(),
        },
        {
            "day": 2,
            "seller": "Aldridge Industrial",
            "buyer": "Pinnacle Goods",
            "qty": 2,
            "claimed_quality": "Poor",
            "true_quality": "Poor",
            "price_per_unit": 35.0,
            "total": 70.0,
            "timestamp": _iso_now(),
        },
        {
            "day": 3,
            "seller": "Vector Works",
            "buyer": "Coastal Fabrication",
            "qty": 1,
            "claimed_quality": "Excellent",
            "true_quality": "Poor",  # misrepresentation
            "misrepresented": True,
            "price_per_unit": 62.0,
            "total": 62.0,
            "timestamp": _iso_now(),
        },
    ]
    _write_jsonl(run_dir / "transactions.jsonl", transactions)

    # ── revelations.jsonl ─────────────────────────────────────────────────────
    revelations = [
        {
            "day": 4,
            "seller": "Vector Works",
            "buyer": "Coastal Fabrication",
            "claimed_quality": "Excellent",
            "true_quality": "Poor",
            "cash_adjustment": -17.0,
            "timestamp": _iso_now(),
        },
    ]
    _write_jsonl(run_dir / "revelations.jsonl", revelations)

    # ── messages.jsonl ────────────────────────────────────────────────────────
    messages = [
        {
            "day": 1,
            "sender": "Meridian Manufacturing",
            "recipient": "Halcyon Assembly",
            "public": False,
            "body": "We have Excellent widgets at $60.",
            "timestamp": _iso_now(),
        },
    ]
    _write_jsonl(run_dir / "messages.jsonl", messages)

    # ── market_state.jsonl ────────────────────────────────────────────────────
    market_snapshots = []
    for day in range(1, 6):
        market_snapshots.append({
            "day": day,
            "fg_price_excellent": 90.0 + day * 0.5,
            "fg_price_poor": 52.0 + day * 0.2,
            "sellers": {
                "Meridian Manufacturing": {
                    "cash": 5000.0 + day * 10, "factories": 1,
                    "inventory_excellent": 2, "inventory_poor": 1,
                    "factory_build_queue": [], "bankrupt": False,
                },
                "Aldridge Industrial": {
                    "cash": 5000.0 + day * 5, "factories": 1,
                    "inventory_excellent": 1, "inventory_poor": 2,
                    "factory_build_queue": [], "bankrupt": False,
                },
                "Crestline Components": {
                    "cash": 5000.0, "factories": 1,
                    "inventory_excellent": 2, "inventory_poor": 0,
                    "factory_build_queue": [], "bankrupt": False,
                },
                "Vector Works": {
                    "cash": 5000.0 - day * 20, "factories": 1,
                    "inventory_excellent": 0, "inventory_poor": 3,
                    "factory_build_queue": [], "bankrupt": False,
                },
            },
            "buyers": {
                "Halcyon Assembly":    {"cash": 6000.0 + day * 15, "widget_inventory": 0, "bankrupt": False},
                "Pinnacle Goods":      {"cash": 6000.0 + day * 8,  "widget_inventory": 0, "bankrupt": False},
                "Coastal Fabrication": {"cash": 6000.0 - day * 5,  "widget_inventory": 0, "bankrupt": False},
                "Northgate Systems":   {"cash": 6000.0,             "widget_inventory": 0, "bankrupt": False},
            },
            "timestamp": _iso_now(),
        })
    _write_jsonl(run_dir / "market_state.jsonl", market_snapshots)

    # ── events.jsonl ──────────────────────────────────────────────────────────
    events = [
        {
            "day": 5, "event_type": "run_complete", "timestamp": _iso_now(),
            "wall_seconds": 10.0, "total_strategic_calls": 8, "total_tactical_calls": 20,
            "total_prompt_tokens": 50000, "total_completion_tokens": 10000,
            "parse_failures": 0, "parse_recoveries": 0, "parse_failure_rate": 0.0,
        },
    ]
    _write_jsonl(run_dir / "events.jsonl", events)

    # ── Per-agent directories ─────────────────────────────────────────────────
    seller_names = [
        "Meridian Manufacturing",
        "Aldridge Industrial",
        "Crestline Components",
        "Vector Works",
    ]
    buyer_names = [
        "Halcyon Assembly",
        "Pinnacle Goods",
        "Coastal Fabrication",
        "Northgate Systems",
    ]

    for agent_name in seller_names + buyer_names:
        agent_dir = run_dir / "agents" / agent_name
        agent_dir.mkdir(parents=True)

        # policy_history.jsonl (strategic calls) — only sellers get policy
        if agent_name in seller_names:
            policies = [
                {
                    "day": 1,
                    "system_prompt": "You are a seller.",
                    "completion": (
                        "My strategy: sell Excellent widgets at $60. Watch for deception from competitors.\n\n"
                        "<policy>\n"
                        '{"excellent_price_floor": 55.0, "excellent_price_ceiling": 70.0,'
                        '"poor_price_floor": 30.0, "poor_price_ceiling": 45.0,'
                        '"daily_excellent_target": 1, "daily_poor_target": 0,'
                        '"build_factory": false, "honesty_stance": "honest",'
                        '"priority_buyers": ["Halcyon Assembly"],'
                        '"risk_assessment": "Market is stable.",'
                        '"notes": "Conservative strategy."}\n'
                        "</policy>"
                    ),
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "total_tokens": 180,
                    "latency_seconds": 0.5,
                    "timestamp": _iso_now(),
                }
            ]
            _write_jsonl(agent_dir / "policy_history.jsonl", policies)

        # tactical_history.jsonl
        tactical = [
            {
                "day": d,
                "completion": (
                    "I will sell one widget today.\n\n<actions>\n"
                    '{"messages": [], "offers": [], "accept_offers": [], '
                    '"decline_offers": [], "produce": {"excellent": 1, "poor": 0}, "build_factory": false}\n'
                    "</actions>"
                ),
                "prompt_tokens": 50,
                "completion_tokens": 40,
                "total_tokens": 90,
                "latency_seconds": 0.3,
                "timestamp": _iso_now(),
            }
            for d in range(1, 6)
        ]
        _write_jsonl(agent_dir / "tactical_history.jsonl", tactical)

    return run_dir


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestReportSmoke:
    def test_pdf_exists_and_nonempty(self, tmp_path):
        """generate_report() must produce a non-empty PDF file."""
        run_dir = _build_fixture_run(tmp_path)
        pdf_path = generate_report(run_dir)

        assert pdf_path.exists(), f"PDF not found at {pdf_path}"
        assert pdf_path.stat().st_size > 1024, (
            f"PDF suspiciously small: {pdf_path.stat().st_size} bytes"
        )

    def test_pdf_opens_with_pypdf(self, tmp_path):
        """PDF must be a valid PDF that pypdf can open and read."""
        pytest.importorskip("pypdf", reason="pypdf not installed")
        import pypdf

        run_dir = _build_fixture_run(tmp_path)
        pdf_path = generate_report(run_dir)

        reader = pypdf.PdfReader(str(pdf_path))
        assert len(reader.pages) >= 1, "PDF has no pages"

    def test_pdf_has_multiple_pages(self, tmp_path):
        """A full report should span more than one page."""
        pytest.importorskip("pypdf", reason="pypdf not installed")
        import pypdf

        run_dir = _build_fixture_run(tmp_path)
        pdf_path = generate_report(run_dir)

        reader = pypdf.PdfReader(str(pdf_path))
        assert len(reader.pages) >= 2, (
            f"Expected multi-page report, got {len(reader.pages)} page(s)"
        )

    def test_output_path_respected(self, tmp_path):
        """generate_report() must write to the path given in output_path."""
        run_dir = _build_fixture_run(tmp_path)
        custom_out = tmp_path / "custom_output.pdf"

        pdf_path = generate_report(run_dir, output_path=custom_out)

        assert pdf_path == custom_out
        assert custom_out.exists()
        assert custom_out.stat().st_size > 0

    def test_load_run_parses_fixture(self, tmp_path):
        """load_run() must successfully parse the fixture directory."""
        run_dir = _build_fixture_run(tmp_path)
        run_data = load_run(run_dir)

        assert run_data.run_id == "run_fixture_seed0"
        assert run_data.seed == 0
        assert len(run_data.transactions) == 3
        assert len(run_data.revelations) == 1
        assert run_data.days_completed == 5
        assert run_data.n_misrepresented == 1

    def test_load_run_seller_buyer_names(self, tmp_path):
        """load_run() must correctly identify seller and buyer names from agent dirs."""
        run_dir = _build_fixture_run(tmp_path)
        run_data = load_run(run_dir)

        assert "Meridian Manufacturing" in run_data.seller_names
        assert "Halcyon Assembly" in run_data.buyer_names
        assert len(run_data.seller_names) == 4
        assert len(run_data.buyer_names) == 4

    def test_pdf_contains_all_required_sections(self, tmp_path):
        """PDF must contain all eight Phase 1 required sections by title."""
        pytest.importorskip("pypdf", reason="pypdf not installed")
        import pypdf

        run_dir = _build_fixture_run(tmp_path)
        pdf_path = generate_report(run_dir)

        reader = pypdf.PdfReader(str(pdf_path))
        full_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        required_sections = [
            "The Sanctuary",           # title page
            "Executive Summary",
            "Final Standings",
            "Time Series",
            "Misrepresentation Analysis",
            "Behavioral Flags",
            "Reasoning Excerpts",
            "Run Statistics",
        ]
        missing = [s for s in required_sections if s not in full_text]
        assert not missing, f"PDF missing sections: {missing}"
