# RagLens architecture

RagLens evaluates one run of one RAG pipeline on one dataset, and returns a
**diagnostic** report — not just a score. This document explains how the
pieces fit together so contributors can extend specific surfaces without
touching the rest.

## Module map

```
raglens/
├── __init__.py        Public API exports
├── core.py            RagLens class — runs metrics + attribution on each sample
├── trace.py           Trace / RetrievalStep / GenerationStep data classes
├── dataset.py         Dataset + Sample loaders (JSONL, in-memory)
├── metrics.py         faithfulness / context_relevance / answer_relevance + registry
├── attribution.py     CounterfactualAttribution, JudgeAttribution, failure-mode classifier
├── conflict.py        pairwise chunk conflict detection
├── report.py          Report + SampleReport; HTML/JSON renderers
├── exceptions.py      RagLensError hierarchy
└── providers/
    ├── base.py        LLM abstract base, LLMCallable Protocol, make_llm()
    ├── nvidia_nim.py  **NVIDIA NIM provider (default)** — OpenAI SDK against integrate.api.nvidia.com; embed_texts() helper
    └── litellm_provider.py   LiteLLM-backed provider (opt-in extra, lazy import)
```

## Execution flow

```
Dataset ──► for each Sample ──► pipeline(query)  ──► Trace
                                              │
                ┌─────────────────────────────┴──► metrics.run() ───► float per metric
                │
                └──────► AttributionStrategy.run(trace, llm)         │
                  (Counterfactual | Judge)                            │
                │                                                    │
                ▼                                                    ▼
        ChunkAttribution list + conflicts + failure mode       metric values
                │                                                    │
                └──────────────────► SampleReport ◄───────────────────┘
                                              │
                                              ▼
                                       Report.save_html()
```

The pipeline is **user code**. RagLens calls it with one query and expects
either a `Trace` (for agentic / multi-step pipelines) or a `(answer, chunks)`
tuple where `chunks` may be strings, `RetrievedChunk`, `LangChain Document`, or
`(content, score)` tuples. `_coerce_to_trace()` in `core.py` normalises all of
these to a `Trace`.

## The breakthrough: counterfactual attribution

The novel contribution is, given a `Trace` with N chunks and an LLM judge,
to compute per-chunk **causal** attribution scores via leave-one-chunk-out
removal:

```
attribution(C_k) = faithfulness(answer | all chunks) − faithfulness(answer | all \ {C_k})
```

Intuition:
* `> 0` → the chunk **helped**; removing it lowered faithfulness.
* `< 0` → the chunk **hurt**; removing it raised faithfulness — it was
  misleading the generator.
* `≈ 0` → neutral.

Cost: `(N + 1) * n_claims` judge calls per sample. On a 5-chunk / 3-claim
sample that is 18 calls; fine for regression sets, expensive for live
monitoring. The `JudgeAttribution` strategy (one judge call per claim asking
"which chunk supported this?") is the cheaper RAGAS-style baseline shipped for
comparability studies.

The same judge LLM is reused for conflict detection (`conflict.py`).

## Failure-mode classifier

`classify_failure_mode()` turns the numeric output of attribution into a
diagnosis string consumers can route on:

| Failure mode         | Detected when                                                                 |
|----------------------|-------------------------------------------------------------------------------|
| `ok`                 | `faithfulness ≥ 0.8`                                                          |
| `chunk_dominance`    | Removing some chunk raises faithfulness by ≥ `dominance_threshold`           |
| `generation_ignore`  | `context_relevance ≥ 0.6` but a low baseline; no chunk removal helps         |
| `retrieval_miss`     | `faithfulness < 0.6` and chunks are not relevant                             |

This categorisation is heuristic, not learned; a future version may fit a
classifier on labelled examples. The heuristic contract forms the test surface.

## Extension points

* **New metric** — `@register("name")` decorator in `metrics.py`. Add a
  matching test. The metric registry means no other code changes are needed.
* **New attribution strategy** — subclass `AttributionStrategy` and implement
  `run()`. Register in the `_STRATEGIES` dict in `core.py`.
* **New LLM provider** — implement `LLM.complete()` (optionally override
  `complete_batch` for concurrency). Users pass an instance via `model=`.
* **New RAG framework adapter** — write a function in your own project that
  wraps your pipeline into a `Trace` or `(answer, chunks)` tuple. No first-party
  adapters ship, to keep RagLens framework-agnostic.

## What RagLens intentionally does *not* do

* It does not run a RAG pipeline for production — only to score it offline.
* It does not own embeddings or vector DBs.
* It does not implement retrieval augmentation prompts — those live in the
  user's pipeline.
* It pins the `openai` SDK as its default (against NVIDIA NIM) and exposes
  LiteLLM as an opt-in extra for other providers; users can also pass any
  `LLM` subclass to bypass the SDK entirely.

Keeping these out of scope is what lets RagLens sit cleanly on top of any
existing RAG stack rather than replacing one.
