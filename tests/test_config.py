"""
Tests for sanctuary/config.py.

Verifies that both dev_local.yaml and production.yaml load and validate
cleanly, and that malformed configs raise the expected errors.
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
        assert econ.seller_starting_cash == 5_000.0
        assert econ.buyer_starting_cash == 6_000.0
        assert econ.bankruptcy_threshold == -3_000.0
        assert econ.final_good_base_price_excellent == 90.0
        assert econ.final_good_base_price_poor == 52.0

    def test_strategic_tier_days(self):
        config = load_config(_CONFIGS_DIR / "dev_local.yaml")
        assert config.run.strategic_tier_days == [1, 8, 15, 22, 29]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")

    def test_four_sellers_required(self):
        """Config validation rejects fewer or more than 4 sellers."""
        raw = {
            "models": {
                "strategic": {"provider": "ollama", "model": "test"},
                "tactical": {"provider": "ollama", "model": "test"},
            },
            "agents": {
                "sellers": [
                    {"name": "S1"},
                    {"name": "S2"},
                    # only 2 sellers — should fail
                ],
                "buyers": [
                    {"name": "B1"}, {"name": "B2"}, {"name": "B3"}, {"name": "B4"},
                ],
            },
        }
        with pytest.raises(ValidationError, match="4 sellers"):
            SimulationConfig.model_validate(raw)

    def test_unknown_provider_rejected(self):
        raw = {
            "models": {
                "strategic": {"provider": "openai", "model": "gpt-4"},  # unknown
                "tactical": {"provider": "ollama", "model": "test"},
            },
            "agents": {
                "sellers": [{"name": f"S{i}"} for i in range(4)],
                "buyers": [{"name": f"B{i}"} for i in range(4)],
            },
        }
        with pytest.raises(ValidationError, match="ollama.*vllm|vllm.*ollama|provider"):
            SimulationConfig.model_validate(raw)
