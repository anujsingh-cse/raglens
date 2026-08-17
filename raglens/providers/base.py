"""LLM abstractions: protocol, base class, and provider routing.

The default provider for string model specs is
:class:`raglens.providers.nvidia_nim.NvidiaNimProvider`, which routes through
NVIDIA's OpenAI-compatible NIM endpoint at ``integrate.api.nvidia.com``.

To use a different provider, either pass an :class:`LLM` instance directly,
or prefix the string with ``litellm:`` (e.g. ``"litellm:openai/gpt-4o-mini"``)
to route through the optional LiteLLM extra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from raglens.exceptions import ProviderError


@runtime_checkable
class LLMCallable(Protocol):
    """Anything callable with ``complete(prompt, system=None) -> str``.

    A plain function ``complete(prompt: str, system: str | None = None) -> str``
    satisfies this protocol, so users can drop in their own provider for tests
    or custom routing.
    """

    def complete(self, prompt: str, system: str | None = None) -> str: ...


class LLM:
    """Base interface every provider implements.

    Subclasses must implement :meth:`complete`. They may also implement
    :meth:`complete_batch` for concurrency (counterfactual attribution benefits
    enormously from batching).
    """

    model: str = "base"

    def complete(self, prompt: str, system: str | None = None) -> str:  # pragma: no cover
        raise NotImplementedError

    def complete_batch(self, prompts: list[tuple[str, str | None]]) -> list[str]:
        """Default sequential implementation. Override for real concurrency."""
        return [self.complete(p, s) for (p, s) in prompts]


def make_llm(model: str | LLM | LLMCallable | Callable[..., str], **kwargs) -> LLM:
    """Coerce a user-supplied spec into an :class:`LLM`.

    - :class:`LLM` instances pass through.
    - Strings become a :class:`NvidiaNimProvider` with that model id (default).
      Prefix with ``litellm:`` to route via the optional LiteLLM extra
      (e.g. ``litellm:openai/gpt-4o-mini``).
    - Bare callables matching ``(prompt, system=None) -> str`` are wrapped.
    """
    if isinstance(model, LLM):
        return model
    if isinstance(model, str):
        if model.startswith("litellm:"):
            from raglens.providers.litellm_provider import LiteLLMProvider
            return LiteLLMProvider(model=model[len("litellm:"):], **kwargs)
        from raglens.providers.nvidia_nim import NvidiaNimProvider
        return NvidiaNimProvider(model=model, **kwargs)
    if callable(model):
        wrapper = type("CallableLLM", (LLM,), {})()
        wrapper.model = getattr(model, "__name__", "callable")
        wrapper.complete = lambda prompt, system=None: model(prompt, system)  # type: ignore
        wrapper.complete_batch = lambda prompts: [model(p, s) for p, s, in prompts]  # type: ignore
        return wrapper
    raise ProviderError(f"unsupported LLM spec: {type(model).__name__}")
