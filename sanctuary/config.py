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


# -- Sub-models ----------------------------------------------------------------

class ModelConfig(BaseModel):
    provider: str           # "ollama", "vllm", or "anthropic"
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


class ProtocolConfig(BaseModel):
    system: str = "no_protocol"
    description: str = ""


class RunConfig(BaseModel):
    days: int = 30
    strategic_tier_days: list[int] = Field(default_factory=lambda: [1, 5, 10, 15, 20, 25, 30])
    max_sub_rounds: int = 2
    inactivity_nudge_threshold: int = 2
    max_parallel_llm_calls: int = 2
    checkpoint_interval: int = 5

    @field_validator("days")
    @classmethod
    def days_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"days must be >= 1, got {v}")
        return v


class EconomicsConfig(BaseModel):
    model_config = {"extra": "ignore"}   # ignore unknown fields (e.g. old buyer_fixed_daily_cost)

    seller_starting_cash: list[float] = Field(
        default_factory=lambda: [5_000.0, 4_500.0, 4_000.0, 3_500.0]
    )
    seller_starting_factories: int = 1
    buyer_starting_cash: float = 6_000.0
    buyer_daily_production_cap: int = 5
    factory_build_cost: float = 2_000.0
    factory_build_days: int = 3
    bankruptcy_threshold: float = -5_000.0
    final_good_base_price_excellent: float = 58.0
    final_good_base_price_poor: float = 35.0
    starting_widgets_per_seller: int = 8

    @field_validator("seller_starting_cash", mode="before")
    @classmethod
    def coerce_seller_cash(cls, v: Any) -> list[float]:
        """Accept a single float (uniform) or a list of 4 floats (asymmetric)."""
        if isinstance(v, (int, float)):
            return [float(v)] * 4
        if isinstance(v, list):
            if len(v) != 4:
                raise ValueError(f"seller_starting_cash must have exactly 4 entries, got {len(v)}")
            return [float(x) for x in v]
        raise ValueError(f"seller_starting_cash must be a number or list of 4 numbers, got {type(v)}")


class SellerAgentConfig(BaseModel):
    name: str


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
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)

    model_config = {"arbitrary_types_allowed": True}


# -- Loader --------------------------------------------------------------------

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
