"""
Ollama provider — local inference via the Ollama HTTP API.

Ollama runs a local server (default: http://localhost:11434) that hosts
models downloaded with `ollama pull <model>`. It supports seed-based
determinism in recent versions.

Usage in config:
    provider: ollama
    model: qwen2.5:7b
    temperature: 0.7

Setup (macOS):
    brew install ollama
    ollama serve          # start the server
    ollama pull qwen2.5:7b
    ollama pull qwen2.5:3b
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


class OllamaProvider(ModelProvider):
    """
    Calls the Ollama /api/chat endpoint.

    Raises ProviderError on HTTP errors or connection failures.
    Raises ContextTooLongError if Ollama returns a context-length error
    (detected by scanning the error message from the API).
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(model=model, temperature=temperature, seed=seed)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

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
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }
        if self.seed is not None:
            payload["options"]["seed"] = self.seed

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is 'ollama serve' running? Error: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama request timed out after {self.timeout}s for model {self.model}."
            ) from exc

        if response.status_code != 200:
            body = response.text[:500]
            self._check_context_error(body)
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code}: {body}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Ollama returned non-JSON response: {response.text[:200]}") from exc

        completion = data.get("message", {}).get("content", "")
        if not completion:
            raise ProviderError(f"Ollama returned empty completion. Full response: {data}")

        # Ollama reports token counts in the response
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ModelResponse(
            completion=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=self._elapsed(start),
            model=self.model,
            provider=self.provider_name,
        )

    def _check_context_error(self, error_text: str) -> None:
        """
        Detect context-length errors in Ollama error responses and raise
        ContextTooLongError instead of generic ProviderError.
        """
        lower = error_text.lower()
        if "context" in lower and ("length" in lower or "exceed" in lower or "limit" in lower):
            raise ContextTooLongError(
                token_count=-1,   # Ollama doesn't report the count in the error
                limit=-1,
                model=self.model,
            )

    def list_models(self) -> list[str]:
        """Return the names of all locally available Ollama models."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            raise ProviderError(f"Failed to list Ollama models: {exc}") from exc

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return True
        except Exception:
            return False
