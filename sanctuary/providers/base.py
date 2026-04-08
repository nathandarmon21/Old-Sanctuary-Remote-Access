"""
Abstract base class for LLM inference providers.

To swap providers, change the provider key in the config YAML. The
simulation wires the correct implementation at startup; no other code
needs to change.

Provider contract:
  - Accepts: model name, system prompt, full conversation history
    (list of role-tagged dicts), max_tokens, temperature, seed.
  - Returns: ModelResponse with completion text, token usage, latency.
  - Must NOT trim the conversation history. If the context is too long,
    raise ContextTooLongError rather than silently dropping messages.
    The simulation will halt and alert the researcher.
  - Seed + temperature=0 should produce deterministic output when the
    backend supports it (vLLM does; Ollama does with recent versions).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """Result of a single LLM inference call."""
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model: str
    provider: str


class ProviderError(Exception):
    """Raised when a provider call fails (network error, timeout, etc.)."""


class ContextTooLongError(ProviderError):
    """
    Raised when the conversation history exceeds the model's context window.

    Do NOT catch this silently. Surface it to the researcher immediately.
    Trimming context would invalidate the experiment's full-history guarantee.
    """
    def __init__(self, token_count: int, limit: int, model: str) -> None:
        self.token_count = token_count
        self.limit = limit
        self.model = model
        super().__init__(
            f"Context too long for {model}: {token_count} tokens > {limit} limit. "
            f"Do not trim — alert the researcher and halt the run."
        )


class ModelProvider(ABC):
    """
    Abstract base for inference backends.

    Subclasses implement complete() for their specific backend.
    All other logic (logging, retry, seed derivation) lives in the
    simulation layer, not here.
    """

    def __init__(self, model: str, temperature: float = 0.7, seed: int | None = None) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        max_tokens: int,
    ) -> ModelResponse:
        """
        Run one inference call.

        Args:
            system_prompt: The system role prompt for this call.
            history: Full conversation history as a list of
                     {"role": "user"|"assistant", "content": "..."} dicts.
                     Must be passed in full — do not trim.
            max_tokens: Maximum tokens to generate in the completion.

        Returns:
            ModelResponse with completion text and usage statistics.

        Raises:
            ContextTooLongError: if history exceeds the model's context limit.
            ProviderError: for any other backend failure.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier string for this provider, used in logs."""
        ...

    @staticmethod
    def _elapsed(start: float) -> float:
        return round(time.perf_counter() - start, 4)
