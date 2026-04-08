"""
Reproducibility tests for the Sanctuary simulation.

Two runs with the same seed must produce identical transaction logs.
This is tested WITHOUT running actual LLM inference — we mock the
providers to return deterministic canned responses.

The test exercises the full simulation loop (market, revelation,
economics, agent action parsing) to verify that all simulation-side
randomness is correctly seeded.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sanctuary.config import SimulationConfig
from sanctuary.providers.base import ModelProvider, ModelResponse
from sanctuary.simulation import Simulation


# ── Deterministic mock provider ───────────────────────────────────────────────

class DeterministicMockProvider(ModelProvider):
    """
    A mock ModelProvider that returns scripted responses.

    For reproducibility testing we need the LLM to behave identically
    across two runs with the same seed. We achieve this by having the
    mock return a fixed response that always produces valid actions.
    """

    SELLER_TACTICAL_RESPONSE = """
I will produce one Excellent widget today and offer it to the first buyer.

<actions>
{
  "messages": [
    {"to": "Halcyon Assembly", "public": false, "body": "I have Excellent widgets available today at $60."}
  ],
  "offers": [
    {"to": "Halcyon Assembly", "qty": 1, "claimed_quality": "Excellent", "quality_to_send": "Excellent", "price_per_unit": 60.0}
  ],
  "accept_offers": [],
  "decline_offers": [],
  "produce": {"excellent": 1, "poor": 0},
  "build_factory": false
}
</actions>
"""

    BUYER_TACTICAL_RESPONSE = """
I will accept any available offers and produce final goods.

<actions>
{
  "messages": [],
  "offers": [],
  "accept_offers": [],
  "decline_offers": [],
  "produce_final_goods": 2
}
</actions>
"""

    STRATEGIC_RESPONSE = """
My strategy for this week: sell Excellent widgets at $60, maintain honest quality representation.

<policy>
{
  "excellent_price_floor": 55.0,
  "excellent_price_ceiling": 70.0,
  "poor_price_floor": 30.0,
  "poor_price_ceiling": 45.0,
  "daily_excellent_target": 1,
  "daily_poor_target": 0,
  "build_factory": false,
  "honesty_stance": "honest",
  "priority_buyers": ["Halcyon Assembly"],
  "risk_assessment": "Market is stable; watching for price drops from competitors.",
  "notes": "Conservative honest strategy, maximize Excellent widget sales."
}
</policy>
"""

    BUYER_STRATEGIC_RESPONSE = """
My strategy: buy from honest sellers, produce 2 final goods per day.

<policy>
{
  "max_price_excellent": 65.0,
  "max_price_poor": 40.0,
  "daily_production_target": 2,
  "preferred_sellers": ["Meridian Manufacturing"],
  "avoid_sellers": [],
  "risk_assessment": "Watching for misrepresentation; will reduce purchases if revelations show cheating.",
  "notes": "Buy low, produce steadily."
}
</policy>
"""

    def __init__(self, role_hint: str = "seller"):
        super().__init__(model="mock-model", temperature=0.0, seed=42)
        self.role_hint = role_hint
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    def complete(self, system_prompt: str, history: list[dict], max_tokens: int) -> ModelResponse:
        self.call_count += 1

        # Determine response by looking at the system prompt
        is_strategic = "strategic" in system_prompt.lower() or "<policy>" in system_prompt
        is_buyer_prompt = "assembler" in system_prompt.lower() or "final good" in system_prompt.lower()

        if is_strategic:
            if is_buyer_prompt:
                completion = self.BUYER_STRATEGIC_RESPONSE
            else:
                completion = self.STRATEGIC_RESPONSE
        else:
            if is_buyer_prompt:
                completion = self.BUYER_TACTICAL_RESPONSE
            else:
                completion = self.SELLER_TACTICAL_RESPONSE

        return ModelResponse(
            completion=completion,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_seconds=0.01,
            model=self.model,
            provider=self.provider_name,
        )


def _make_test_config() -> SimulationConfig:
    """Build a minimal SimulationConfig for testing (5 days, dev economics)."""
    raw = {
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
    return SimulationConfig.model_validate(raw)


def _run_with_mock(seed: int, tmpdir: Path) -> list[dict]:
    """
    Run a 5-day simulation with mock providers and return the transaction log.
    """
    config = _make_test_config()
    sim = Simulation(config=config, seed=seed)

    # Override run directory to use the temp dir
    from sanctuary.simulation import _runs_dir
    sim.run_dir = tmpdir / sim.run_id

    # Inject mock providers
    mock_provider = DeterministicMockProvider()
    sim.strategic_provider = mock_provider
    sim.tactical_provider = mock_provider
    for agent in sim.agents.values():
        agent._strategic_provider = mock_provider
        agent._tactical_provider = mock_provider

    sim.run()

    # Read transaction log
    tx_file = sim.run_dir / "transactions.jsonl"
    if not tx_file.exists():
        return []
    transactions = []
    with open(tx_file) as f:
        for line in f:
            line = line.strip()
            if line:
                transactions.append(json.loads(line))
    return transactions


class TestReproducibility:
    def test_same_seed_identical_transactions(self, tmp_path):
        """
        Two runs with seed=42 must produce byte-identical transaction logs.
        This is the core reproducibility guarantee of the simulation.
        """
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        dir_a.mkdir()
        dir_b.mkdir()

        txs_a = _run_with_mock(seed=42, tmpdir=dir_a)
        txs_b = _run_with_mock(seed=42, tmpdir=dir_b)

        # Same number of transactions
        assert len(txs_a) == len(txs_b), (
            f"Transaction counts differ: {len(txs_a)} vs {len(txs_b)}"
        )

        # Compare transaction content (exclude timestamp, which is wall-clock)
        def normalize(tx: dict) -> dict:
            return {k: v for k, v in tx.items() if k != "timestamp"}

        for i, (a, b) in enumerate(zip(txs_a, txs_b)):
            assert normalize(a) == normalize(b), (
                f"Transaction {i} differs:\n  run A: {a}\n  run B: {b}"
            )

    def test_different_seeds_different_outcomes(self, tmp_path):
        """
        Two runs with different seeds should (almost certainly) differ.
        Not guaranteed, but with Brownian price walk and revelation scheduling,
        any meaningful run will diverge.
        """
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        dir_a.mkdir()
        dir_b.mkdir()

        # Read market state logs instead of transactions (more data points)
        config = _make_test_config()

        sim_a = Simulation(config=config, seed=42)
        sim_a.run_dir = dir_a / sim_a.run_id
        mock_a = DeterministicMockProvider()
        for agent in sim_a.agents.values():
            agent._strategic_provider = mock_a
            agent._tactical_provider = mock_a
        sim_a.strategic_provider = mock_a
        sim_a.tactical_provider = mock_a
        sim_a.run()

        sim_b = Simulation(config=config, seed=99)
        sim_b.run_dir = dir_b / sim_b.run_id
        mock_b = DeterministicMockProvider()
        for agent in sim_b.agents.values():
            agent._strategic_provider = mock_b
            agent._tactical_provider = mock_b
        sim_b.strategic_provider = mock_b
        sim_b.tactical_provider = mock_b
        sim_b.run()

        # Compare final fg prices from the last market state entry
        def read_last_snapshot(run_dir: Path) -> dict:
            ms_file = list(run_dir.glob("*/market_state.jsonl"))
            if not ms_file:
                return {}
            lines = [l.strip() for l in ms_file[0].read_text().splitlines() if l.strip()]
            return json.loads(lines[-1]) if lines else {}

        snap_a = read_last_snapshot(dir_a)
        snap_b = read_last_snapshot(dir_b)

        if snap_a and snap_b:
            # Prices should differ (seed-dependent random walk)
            assert (
                snap_a.get("fg_price_excellent") != snap_b.get("fg_price_excellent")
                or snap_a.get("fg_price_poor") != snap_b.get("fg_price_poor")
            ), "Different seeds produced identical price walks — RNG may not be seeded correctly"

    def test_revelation_days_reproducible(self, tmp_path):
        """Revelation days in the transaction log are seed-deterministic."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        txs_a = _run_with_mock(seed=7, tmpdir=dir_a)
        txs_b = _run_with_mock(seed=7, tmpdir=dir_b)

        rev_days_a = [t.get("revelation_day") for t in txs_a]
        rev_days_b = [t.get("revelation_day") for t in txs_b]
        assert rev_days_a == rev_days_b, (
            "Revelation day schedules differ between identical-seed runs"
        )
