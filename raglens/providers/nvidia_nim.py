"""NVIDIA NIM (NVIDIA Inference Microservices) LLM provider.

NVIDIA NIM exposes an OpenAI-compatible Chat Completions endpoint at
``https://integrate.api.nvidia.com/v1``. This provider routes through the
official ``openai`` SDK pointed at that endpoint, so no LiteLLM dependency is
required. Get a key at https://build.nvidia.com.

Models (https://build.nvidia.com/models):
    - ``meta/llama-3.1-70b-instruct``  (default; universally available judge)
    - ``meta/llama-3.1-8b-instruct``     (cheap path; same family)
    - ``nvidia/llama-3.1-nemotron-70b-instruct``  (strong Nemotron; may need
      separate activation on your account — test with examples/probe_nim.py)
    - ``meta/llama-3.1-405b-instruct``
    - ``mistralai/mistral-large-2-instruct``
    ...and many more. Run ``examples/list_nim_models.py`` to see what your
    account can call.

Example::

    from raglens import RagLens
    lens = RagLens(model="meta/llama-3.1-70b-instruct")
    # NVIDIA_API_KEY is read from the environment automatically.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from raglens.exceptions import ProviderError
from raglens.providers.base import LLM

_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
# `meta/llama-3.1-70b-instruct` is universally available on every NVIDIA NIM
# account we've seen. Nemotron variants are listed by /v1/models but require
# separate activation on the account; using them as a default would 404 for
# many users. Override with `RAGLENS_JUDGE_MODEL` or pass `model=...` directly.
_DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TIMEOUT = 60.0  # NIM can cold-start a model on first call; be generous
_MAX_WORKERS = 8


class NvidiaNimProvider(LLM):
    """LLM provider for NVIDIA NIM accessed via the ``openai`` SDK.

    Args:
        model: NVIDIA NIM model id. Defaults to ``meta/llama-3.1-70b-instruct``
            (universally available on every NIM account; Nemotron-70B is
            listed by /v1/models but 404s on accounts without activation).
        api_key: NVIDIA API key. If omitted, read from the ``NVIDIA_API_KEY``
            environment variable.
        base_url: NIM endpoint. Defaults to the public NIM service.
        temperature: judge temperature. Defaults to 0.0 for determinism.
        max_tokens: cap on generated tokens per response.
        timeout: per-request timeout in seconds. Defaults to 60s; NVIDIA NIM
            can cold-start a model on the first request, which is slow.
        **kwargs: extra kwargs forwarded to ``chat.completions.create``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str = _NIM_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = _DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "NvidiaNimProvider requires an API key. Set NVIDIA_API_KEY in "
                "your environment, or pass api_key=... explicitly. Get one at "
                "https://build.nvidia.com."
            )
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        self._client = None

    def _client_obj(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as e:
                raise ProviderError(
                    "NvidiaNimProvider requires the `openai` package. "
                    "Install with: pip install raglens"
                ) from e
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> str:
        client = self._client_obj()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens if self.max_tokens is not None
                else _DEFAULT_MAX_TOKENS,
                **self.kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(f"NVIDIA NIM call failed for {self.model}: {e}") from e

    def complete_batch(self, prompts: list[tuple[str, str | None]]) -> list[str]:
        """Parallel completion via ThreadPoolExecutor.

        Counterfactual attribution fires 5-20 prompts per sample; doing them
        concurrently is a large wall-time win against NVIDIA NIM.
        """
        def one(item: tuple[str, str | None]) -> str:
            prompt, system = item
            return self.complete(prompt, system)

        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(prompts))) as pool:
            return list(pool.map(one, prompts))


def embed_texts(texts: list[str], *,
                model: str = "nvidia/nv-embedqa-e5-v5",
                input_type: str = "passage",
                api_key: str | None = None,
                base_url: str = _NIM_BASE_URL,
                timeout: float | None = _DEFAULT_TIMEOUT) -> list[list[float]]:
    """Helper for users building a RAG retriever on top of NVIDIA NIM.

    NVIDIA's retrieval embedders are **asymmetric** — the same model uses
    different encoders for stored corpus passages vs. live queries. NIM requires
    you to flag which side of that pair you're embedding via ``input_type``:

      * ``"passage"`` — for documents you store in the vector DB (default).
      * ``"query"``   — for the user query at retrieval time.

    Call this function once with ``input_type="passage"`` to embed your corpus,
    then again with ``input_type="query"`` (and a single-element list) to
    embed the query; cosine-similarity over the two gives you top-k retrieval.
    """
    if not texts:
        return []
    key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise ProviderError(
            "embed_texts requires NVIDIA_API_KEY. Get one at https://build.nvidia.com."
        )
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise ProviderError(
            "embed_texts requires the `openai` package. Install with: pip install raglens"
        ) from e
    client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
    try:
        resp = client.embeddings.create(
            model=model,
            input=texts,
            # `input_type` lives in the request body (NIM extension), not the
            # OpenAI SDK's typed surface — extra_body is the OpenAI SDK's
            # supported escape hatch for vendor extensions like this.
            extra_body={"input_type": input_type},
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        raise ProviderError(f"NVIDIA NIM embed call failed for {model}: {e}") from e
