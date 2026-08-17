# Changelog

All notable changes to RagLens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and uses
date stamps in ISO 8601 (`YYYY-MM-DD`).

## [Unreleased] — 2026-08-17

### Added
- `RagLens.probe(pipeline, dataset)` orchestrator returning a :class:`Report`.
- **Counterfactual chunk attribution** (`CounterfactualAttribution`) —
  leave-one-chunk-out faithfulness, the headline contribution: per-chunk
  causal attribution scores in range ``(-1, 1)``.
- **Judge attribution** (`JudgeAttribution`) — RAGAS-style baseline shipped for
  A/B comparison against the counterfactual approach.
- **Pairwise chunk conflict detection** (`detect_conflicts`).
- **Four-way failure-mode classifier**:
  `ok / retrieval_miss / chunk_dominance / generation_ignore`.
- Built-in metrics — `faithfulness`, `context_relevance`,
  `answer_relevance` — registered via the `@register` decorator pattern.
- `Dataset.from_jsonl` and `from_list` loaders, with tag-based slicing.
- HTML + JSON diagnostic report renderers (stdlib only, no Jinja).
- **NVIDIA NIM as the native default LLM provider** (`NvidiaNimProvider`) —
  the official `openai` SDK pointed at ``integrate.api.nvidia.com/v1``; key
  read from ``NVIDIA_API_KEY``. Also exposes ``embed_texts()`` so the quickstart
  can build a real RAG retriever on the same endpoint.
- Optional LiteLLM-backed provider (opt-in via the ``litellm:`` model prefix
  and ``raglens[litellm]`` extra) for OpenAI / Anthropic / Gemini / Ollama / vLLM.
- Pure-logic test layer runs offline and asserts structural invariants.
- Live integration test layer (``@pytest.mark.integration``) calls NVIDIA NIM,
  auto-skips when ``NVIDIA_API_KEY`` is unset, asserts structural invariants
  only (not specific verdicts).

### Removed
- ``raglens.providers.MockLLM`` and ``NullLLM`` — no mock LLM ships in the
  production package. Tests that previously relied on scripted responses now
  either cover pure-logic code paths without an LLM, or run as live
  integration tests against NVIDIA NIM.

### Internal
- Strict type hints on every public surface; `ruff` lint passes clean.
- Module map and execution flow documented in `ARCHITECTURE.md`.

[Unreleased]: https://github.com/anujsingh-cse/raglens/compare/v0.0.0...HEAD
