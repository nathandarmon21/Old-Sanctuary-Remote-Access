"""
Configuration loading and validation for the Sanctuary simulation.

Config files are YAML. This module loads them, validates them with
Pydantic, and returns typed config objects to the simulation loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Sub-models ────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    provider: str           # "ollama" or "vllm"
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    seed: int | None = None
    base_url: str | None = None   # override default server URL
    timeout: float = 300.0

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, v: str) -> str:
        if v not in ("ollama", "vllm", "anthropic"):
            raise ValueError(f"provider must be 'ollama', 'vllm', or 'anthropic', got {v!r}")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {v}")
        return v


class ModelsConfig(BaseModel):
    strategic: ModelConfig
    tactical: ModelConfig


class RunConfig(BaseModel):
    days: int = 30
    strategic_tier_days: list[int] = Field(default_factory=lambda: [1, 8, 15, 22, 29])
    max_sub_rounds: int = 2
    inactivity_nudge_threshold: int = 2
    # Max concurrent LLM calls per tier.  Set to 1 for fully sequential
    # (safest with Ollama on a single GPU running large models).
    # Set higher when your inference backend actually serves requests in parallel
    # (vLLM, or Ollama with OLLAMA_NUM_PARALLEL>1 and a capable GPU).
    max_parallel_llm_calls: int = 4

    @field_validator("days")
    @classmethod
    def days_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"days must be >= 1, got {v}")
        return v


class EconomicsConfig(BaseModel):
    model_config = {"extra": "ignore"}   # ignore unknown fields (e.g. old buyer_fixed_daily_cost)

    seller_starting_cash: float = 5_000.0
    seller_starting_factories: int = 1
    buyer_starting_cash: float = 6_000.0
    buyer_daily_production_cap: int = 3
    factory_build_cost: float = 1_500.0
    factory_build_days: int = 2
    bankruptcy_threshold: float = -3_000.0
    final_good_base_price_excellent: float = 90.0
    final_good_base_price_poor: float = 52.0
    price_walk_sigma: float = 1.0


class SellerAgentConfig(BaseModel):
    name: str
    starting_inventory: dict[str, int] = Field(
        default_factory=lambda: {"excellent": 0, "poor": 0}
    )


class BuyerAgentConfig(BaseModel):
    name: str


class AgentsConfig(BaseModel):
    sellers: list[SellerAgentConfig]
    buyers: list[BuyerAgentConfig]

    @field_validator("sellers")
    @classmethod
    def four_sellers(cls, v: list) -> list:
        if len(v) != 4:
            raise ValueError(f"Exactly 4 sellers required, got {len(v)}")
        return v

    @field_validator("buyers")
    @classmethod
    def four_buyers(cls, v: list) -> list:
        if len(v) != 4:
            raise ValueError(f"Exactly 4 buyers required, got {len(v)}")
        return v


class SimulationConfig(BaseModel):
    """Top-level validated configuration."""
    run: RunConfig = Field(default_factory=RunConfig)
    models: ModelsConfig
    economics: EconomicsConfig = Field(default_factory=EconomicsConfig)
    agents: AgentsConfig

    model_config = {"arbitrary_types_allowed": True}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> SimulationConfig:
    """
    Load and validate a simulation config from a YAML file.

    Raises:
        FileNotFoundError: if the file does not exist.
        pydantic.ValidationError: if the config is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return SimulationConfig.model_validate(raw)


def config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    """Serialise a SimulationConfig to a plain dict for JSON logging."""
    return config.model_dump()
