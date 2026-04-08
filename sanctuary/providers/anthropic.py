"""
Anthropic provider — Claude models via the Anthropic Messages API.

Reads ANTHROPIC_API_KEY from the environment (or a .env file at the
project root, loaded automatically at import time via python-dotenv).

Usage in config:
    provider: anthropic
    model: claude-haiku-4-5
    temperature: 0.7
    max_tokens: 600

Notes:
  - The Anthropic API does not expose a seed parameter, so
    reproducibility tests should set temperature: 0.0 in the config.
  - System prompts are passed as the top-level "system" parameter,
    NOT as a message with role "system" (API v2 convention).
  - Conversation history is passed as the "messages" list, which must
    alternate strictly between "user" and "assistant" roles.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import anthropic as _anthropic

from sanctuary.providers.base import (
    ContextTooLongError,
    ModelProvider,
    ModelResponse,
    ProviderError,
)

# Load .env at import time so ANTHROPIC_API_KEY is available even when the
# script is not run from a shell that already has it set.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv optional; fall back to environment variable only


class AnthropicProvider(ModelProvider):
    """
    Calls the Anthropic Messages API.

    Raises ProviderError on API errors or connection failures.
    Raises ContextTooLongError if the API returns a context-length error.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,   # stored but not forwarded (API doesn't support it)
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, seed=seed)
        self.timeout = timeout

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env at the project root or export it in your shell."
            )
        self._client = _anthropic.Anthropic(
            api_key=resolved_key,
            timeout=timeout,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def complete(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        max_tokens: int,
    ) -> ModelResponse:
        start = time.perf_counter()

        # Anthropic requires messages to alternate user/assistant.
        # The history from agent.py already has this structure (it's built
        # with alternating user/assistant turns).  Filter to only the keys
        # the API accepts.
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m.get("role") in ("user", "assistant")
        ]

        # Anthropic API requires at least one user message.
        if not messages or messages[-1]["role"] != "user":
            # Should never happen given agent.py's logic, but guard anyway.
            messages.append({"role": "user", "content": "(continue)"})

        try:
            response = self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
        except _anthropic.APIStatusError as exc:
            self._check_context_error(exc)
            raise ProviderError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc
        except _anthropic.APIConnectionError as exc:
            raise ProviderError(
                f"Cannot connect to Anthropic API: {exc}"
            ) from exc
        except _anthropic.APITimeoutError as exc:
            raise ProviderError(
                f"Anthropic API request timed out after {self.timeout}s"
            ) from exc

        # Extract text from the first content block
        completion = ""
        for block in response.content:
            if block.type == "text":
                completion = block.text
                break

        if not completion:
            raise ProviderError(
                f"Anthropic returned empty completion. "
                f"Stop reason: {response.stop_reason}. Model: {self.model}"
            )

        usage = response.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens

        return ModelResponse(
            completion=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=self._elapsed(start),
            model=self.model,
            provider=self.provider_name,
        )

    def _check_context_error(self, exc: _anthropic.APIStatusError) -> None:
        """Re-raise as ContextTooLongError if this is a context-length error."""
        msg = str(exc).lower()
        if exc.status_code == 400 and (
            "context" in msg or "too long" in msg or "max_tokens" in msg
            or "token" in msg and "limit" in msg
        ):
            raise ContextTooLongError(
                token_count=-1,
                limit=-1,
                model=self.model,
            )
