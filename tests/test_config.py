"""
Tests for sanctuary/config.py.

Verifies that all YAML configs load and validate cleanly,
and that malformed configs raise the expected errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sanctuary.config import SimulationConfig, load_config

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestConfigLoading:
    def test_dev_local_loads(self):
        config = load_config(_CONFIGS_DIR / "dev_local.yaml")
        assert isinstance(config, SimulationConfig)

    def test_production_loads(self):
        config = load_config(_CONFIGS_DIR / "production.yaml")
        assert isinstance(config, SimulationConfig)

    def test_smoke_haiku_loads(self):
        config = load_config(_CONFIGS_DIR / "smoke_haiku.yaml")
        assert isinstance(config, SimulationConfig)

    def test_smoke_3day_loads(self):
        config = load_config(_CONFIGS_DIR / "smoke_3day.yaml")
        assert isinstance(config, SimulationConfig)

    def test_dev_haiku_loads(self):
        config = load_config(_CONFIGS_DIR / "dev_haiku.yaml")
        assert isinstance(config, SimulationConfig)

    def test_dev_available_loads(self):
        config = load_config(_CONFIGS_DIR / "dev_available.yaml")
        assert isinstance(config, SimulationConfig)

    def test_long_horizon_d12_vllm_loads(self):
        """The redesigned 100-day reputation experiment config validates
        and has the post-redesign parameters wired up correctly.

        Headline redesign values (vs. the prior 150-day config):
          - 100 days, defect 0.07, bankruptcy at 0
          - Final-good prices 52/25 (yields V_E=$42, V_P=$15 buyer-side)
          - Tighter starting cash so the $80/day fixed-cost burn bites
        """
        config = load_config(_CONFIGS_DIR / "long_horizon_d12_vllm.yaml")
        assert config.run.days == 100
        assert config.run.multi_round_negotiation is True
        assert config.run.max_negotiation_rounds == 4
        assert config.run.checkpoint_interval == 10
        # Strategic every 7 days, starting at day 1.
        assert config.run.strategic_tier_days[0] == 1
        assert all(
            (b - a) == 7
            for a, b in zip(config.run.strategic_tier_days,
                            config.run.strategic_tier_days[1:])
        )
        # 6+6 agents.
        assert len(config.agents.sellers) == 6
        assert len(config.agents.buyers) == 6
        # Redesigned margin gradient.
        assert config.economics.final_good_base_price_excellent == 52.0
        assert config.economics.final_good_base_price_poor == 25.0
        assert config.economics.production_defect_rate == 0.07
        assert config.economics.bankruptcy_threshold == 0.0
        # Tightened starting cash; top-of-stack down from 7500 to 3500.
        assert config.economics.seller_starting_cash[0] == 3500.0
        assert len(config.economics.seller_starting_cash) == 6
        # vLLM Qwen 2.5 32B AWQ both tiers.
        assert config.models.strategic.model == "Qwen/Qwen2.5-32B-Instruct-AWQ"
        assert config.models.tactical.model == "Qwen/Qwen2.5-32B-Instruct-AWQ"

    def test_dev_local_has_correct_agents(self):
        config = load_config(_CONFIGS_DIR / "dev_local.yaml")
        seller_names = [s.name for s in config.agents.sellers]
        buyer_names = [b.name for b in config.agents.buyers]
        assert "Meridian Manufacturing" in seller_names
        assert "Aldridge Industrial" in seller_names
        assert "Crestline Components" in seller_names
        assert "Vector Works" in seller_names
        assert "Halcyon Assembly" in buyer_names
        assert "Pinnacle Goods" in buyer_names
        assert "Coastal Fabrication" in buyer_names
        assert "Northgate Systems" in buyer_names

    def test_dev_local_uses_ollama(self):
        config = load_config(_CONFIGS_DIR / "dev_local.yaml")
        assert config.models.strategic.provider == "ollama"
        assert config.models.tactical.provider == "ollama"

    def test_production_uses_vllm(self):
        config = load_config(_CONFIGS_DIR / "production.yaml")
        assert config.models.strategic.provider == "vllm"
        assert config.models.tactical.provider == "vllm"

    def test_dev_local_economics(self):
        config = load_config(_CONFIGS_DIR / "dev_local.yaml")
        econ = config.economics
        assert econ.seller_starting_cash == [5000.0, 4500.0, 4000.0, 3500.0]
        assert econ.buyer_starting_cash == 6000.0
        assert econ.bankruptcy_threshold == -5000.0
        assert econ.final_good_base_price_excellent == 55.0
        assert econ.final_good_base_price_poor == 32.0
        assert econ.factory_build_cost == 2000.0
        assert econ.factory_build_days == 3
        assert econ.starting_widgets_per_seller == 8

    def test_strategic_tier_days(self):
        config = load_config(_CONFIGS_DIR / "dev_local.yaml")
        assert config.run.strategic_tier_days == [1, 5, 10, 15, 20, 25, 30]

    def test_protocol_default(self):
        config = load_config(_CONFIGS_DIR / "dev_local.yaml")
        assert config.protocol.system == "no_protocol"

    def test_asymmetric_seller_cash(self):
        config = load_config(_CONFIGS_DIR / "dev_local.yaml")
        cash = config.economics.seller_starting_cash
        assert len(cash) == 4
        assert cash[0] == 5000.0
        assert cash[3] == 3500.0

    def test_uniform_seller_cash_coercion(self):
        """A single float for seller_starting_cash is coerced to a 1-element list.
        The market builder expands it to match the seller count at run time."""
        raw = {
            "models": {
                "strategic": {"provider": "ollama", "model": "test"},
                "tactical": {"provider": "ollama", "model": "test"},
            },
            "economics": {"seller_starting_cash": 5000.0},
            "agents": {
                "sellers": [{"name": f"S{i}"} for i in range(4)],
                "buyers": [{"name": f"B{i}"} for i in range(4)],
            },
        }
        config = SimulationConfig.model_validate(raw)
        # Coerced to a single-entry list; expansion happens at market build time
        assert config.economics.seller_starting_cash == [5000.0]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")

    def test_at_least_two_sellers_required(self):
        raw = {
            "models": {
                "strategic": {"provider": "ollama", "model": "test"},
                "tactical": {"provider": "ollama", "model": "test"},
            },
            "agents": {
                "sellers": [{"name": "S1"}],
                "buyers": [{"name": f"B{i}"} for i in range(4)],
            },
        }
        with pytest.raises(ValidationError, match="2 sellers"):
            SimulationConfig.model_validate(raw)

    def test_unknown_provider_rejected(self):
        raw = {
            "models": {
                "strategic": {"provider": "openai", "model": "gpt-4"},
                "tactical": {"provider": "ollama", "model": "test"},
            },
            "agents": {
                "sellers": [{"name": f"S{i}"} for i in range(4)],
                "buyers": [{"name": f"B{i}"} for i in range(4)],
            },
        }
        with pytest.raises(ValidationError, match="ollama.*vllm|vllm.*ollama|provider"):
            SimulationConfig.model_validate(raw)
