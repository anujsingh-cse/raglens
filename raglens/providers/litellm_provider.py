"""LiteLLM-backed LLM provider — opt-in extra.

LiteLLM (https://github.com/BerriAI/litellm) exposes one client for >50
providers. Users who want a non-NVIDIA-NIM provider (OpenAI, Anthropic,
Gemini, Ollama, vLLM, ...) install ``raglens[litellm]`` and pass a model spec
prefixed with ``litellm:`` (e.g. ``litellm:openai/gpt-4o-mini``).

This module imports LiteLLM lazily so users who stay on the default NVIDIA NIM
provider never pay the dependency cost.
"""

from __future__ import annotations

from raglens.exceptions import ProviderError
from raglens.providers.base import LLM

try:  # Python 3.11+ has typing.assert_never, but we keep broad compat
    from typing import Any
except Exception:  # pragma: no cover
    Any = object  # type: ignore


class LiteLLMProvider(LLM):
    """Routes ``complete`` and ``complete_batch`` through LiteLLM."""

    def __init__(self, model: str = "openai/gpt-4o-mini", *,
                 api_key: str | None = None, base_url: str | None = None,
                 temperature: float = 0.0, **kwargs: Any):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.kwargs = kwargs

    def _client(self):
        try:
            import litellm  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "raglens.litellm_provider requires `litellm`. "
                "Install with: pip install raglens[litellm]"
            ) from e
        return litellm

    def complete(self, prompt: str, system: str | None = None) -> str:
        litellm = self._client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = litellm.completion(
                model=self.model, messages=messages,
                api_key=self.api_key, api_base=self.base_url,
                temperature=self.temperature, **self.kwargs,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            raise ProviderError(f"LiteLLM call failed for {self.model}: {e}") from e

    def complete_batch(self, prompts: list[tuple[str, str | None]]) -> list[str]:
        """Batch via concurrent.futures ThreadPoolExecutor.

        LiteLLM exposes ``acompletion`` for async batching; using threads keeps
        the public API synchronous and provider-agnostic. Counterfactual
        attribution typically fires 5-20 prompts per sample, so this matters.
        """
        from concurrent.futures import ThreadPoolExecutor

        def one(item: tuple[str, str | None]) -> str:
            prompt, system = item
            return self.complete(prompt, system)

        with ThreadPoolExecutor(max_workers=min(8, len(prompts))) as pool:
            return list(pool.map(one, prompts))
