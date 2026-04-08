"""
vLLM provider — inference via the vLLM OpenAI-compatible HTTP API.

vLLM exposes an OpenAI-compatible endpoint. It supports seed-based
determinism and is the production backend for GPU cluster runs (Phase 3+).

Usage in config:
    provider: vllm
    model: Qwen/Qwen2.5-32B-Instruct
    temperature: 0.7

Setup (GPU server):
    pip install vllm
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-32B-Instruct \\
        --port 8000

The base_url can be overridden in the config for non-standard ports or
remote servers.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from sanctuary.providers.base import (
    ContextTooLongError,
    ModelProvider,
    ModelResponse,
    ProviderError,
)


class VLLMProvider(ModelProvider):
    """
    Calls the vLLM /v1/chat/completions endpoint (OpenAI-compatible).

    Seed is passed directly to vLLM, which supports deterministic output
    at temperature=0 with a fixed seed.
    """

    DEFAULT_BASE_URL = "http://localhost:8000"

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(model=model, temperature=temperature, seed=seed)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "vllm"

    def complete(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        max_tokens: int,
    ) -> ModelResponse:
        start = time.perf_counter()

        messages = [{"role": "system", "content": system_prompt}] + history

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot connect to vLLM at {self.base_url}. "
                f"Is the vLLM server running? Error: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"vLLM request timed out after {self.timeout}s for model {self.model}."
            ) from exc

        if response.status_code != 200:
            body = response.text[:500]
            self._check_context_error(body, response.status_code)
            raise ProviderError(
                f"vLLM returned HTTP {response.status_code}: {body}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"vLLM returned non-JSON response: {response.text[:200]}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise ProviderError(f"vLLM returned no choices. Full response: {data}")

        completion = choices[0].get("message", {}).get("content", "")
        if not completion:
            raise ProviderError(f"vLLM returned empty completion content.")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return ModelResponse(
            completion=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=self._elapsed(start),
            model=self.model,
            provider=self.provider_name,
        )

    def _check_context_error(self, error_text: str, status_code: int) -> None:
        """
        Detect context-length errors (typically HTTP 400 from vLLM) and
        raise ContextTooLongError rather than generic ProviderError.
        """
        lower = error_text.lower()
        if status_code == 400 and (
            "maximum context" in lower
            or "context_length_exceeded" in lower
            or "max_position_embeddings" in lower
            or "too long" in lower
        ):
            raise ContextTooLongError(token_count=-1, limit=-1, model=self.model)

    def is_available(self) -> bool:
        """Return True if the vLLM server is reachable."""
        try:
            httpx.get(f"{self.base_url}/v1/models", timeout=5.0)
            return True
        except Exception:
            return False
