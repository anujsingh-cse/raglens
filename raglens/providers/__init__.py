"""LLM providers used by RagLens internals.

Default provider: NVIDIA NIM via :class:`NvidiaNimProvider` (OpenAI-compatible
endpoint at ``integrate.api.nvidia.com``). Other providers can be plugged in
by passing an :class:`LLM` instance directly to :class:`raglens.RagLens`.
"""

from raglens.providers.base import LLM, LLMCallable, make_llm
from raglens.providers.nvidia_nim import NvidiaNimProvider, embed_texts

__all__ = ["LLM", "LLMCallable", "NvidiaNimProvider", "embed_texts", "make_llm"]
