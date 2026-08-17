# RagLens

> **Score your RAG. Then *diagnose* it.**
> RagLens answers the question nobody else does: *which retrieved chunk helped or hurt this answer?*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NVIDIA NIM](https://img.shields.io/badge/LLM%20provider-NVIDIA%20NIM-76b900.svg)](https://build.nvidia.com)
[![CI](https://github.com/anujsingh-cse/raglens/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/anujsingh-cse/raglens/actions)

`RAGAS` tells you "this answer is 82% faithful."
`DeepEval` tells you "this answer passed."
RagLens tells you: **"chunk `r2_c0` actively misled the model — removing it raises faithfulness from 0.42 to 0.89. Re-rank your retriever."**

It does so with **counterfactual chunk attribution**: leave-one-chunk-out faithfulness, the same idea behind leave-one-out feature importance in classical ML — never before applied to chunk ranking in RAG eval. The judge is a real LLM — by default NVIDIA NIM — no mock LLM, no stub responses, no scripted fixtures.

---

## Why RagLens

Every existing RAG eval framework scores at the *sample* level. They tell you
*whether* something went wrong. They do not tell you *what* went wrong.

RagLens attributes each retrieved chunk's causal contribution to the final
answer, then classifies the failure mode so you know exactly where to fix:

| Framework | Faithfulness | Context Relevance | **Counterfactual attribution** | **Conflict detection** | **Failure-mode classifier** | NVIDIA NIM native | Offline tests |
|---|---|---|---|---|---|---|---|
| RAGAS | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | partial |
| DeepEval | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | partial |
| TruLens | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **RagLens** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓ pure-logic + live NIM** |

---

## The breakthrough, in one picture

```
                          RAG pipeline (yours)
                                   │
                       ┌───────────┴───────────┐
   query ─► retriever ─► chunks ─► generator ─► answer
                       │                         │
                       └─────► Trace ◄───────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                    Counterfactual        Metrics
                    attribution          (faithfulness,
                    (leave-one-out)       relevance,
                          │              answer-rel.)
                          ▼
        attribution(C_k) = F(answer|all) − F(answer|all \ {C_k})
                          │
              ▼                       ▼
         +δ → HELPFUL             −δ → HARMFUL
                          │
                          ▼
                  FailureModeClassifier
                  ──►  ok
                  ──►  retrieval_miss     (retriever didn't find the answer)
                  ──►  chunk_dominance    (a wrong chunk hijacked the answer)
                  ──►  generation_ignore (model ignored its context)
```

This is the missing diagnostic layer between *metric* and *fix*.

---

## Quickstart — real end-to-end RAG, judged by NVIDIA NIM

```bash
pip install raglens
export NVIDIA_API_KEY=nvapi-...   # get one at https://build.nvidia.com
python -m examples.quickstart     # writes quickstart_report.html
```

The quickstart runs a *real* RAG pipeline — embeddings + retrieval + generation
all on NVIDIA NIM — then judges it with RagLens. No mock LLM, no stubbed answer.

For your own RAG:

```python
import os
os.environ.setdefault("NVIDIA_API_KEY", "nvapi-...")

from raglens import RagLens, Dataset

# Your RAG pipeline: takes a query, returns (answer, list_of_chunks).
def my_rag(query: str) -> tuple[str, list[str]]:
    chunks = retriever.invoke(query)
    answer = chain.invoke({"query": query, "context": chunks})
    return answer, chunks

# Default judge: NVIDIA NIM. The string is the NIM model id.
# `meta/llama-3.1-70b-instruct` is universally available on every NIM account.
lens = RagLens(model="meta/llama-3.1-70b-instruct")
dataset = Dataset.from_jsonl("evals.jsonl")

report = lens.probe(my_rag, dataset)
print(report.summary())
print(report.attribution(top_k=3))
report.save_html("report.html")
```

---

## Bring your own judge

RagLens works with any callable matching `(prompt: str, system: str | None) -> str`:

* **NVIDIA NIM** (default) — pass a NIM model id, key read from `NVIDIA_API_KEY`:
  ```python
  RagLens(model="meta/llama-3.1-70b-instruct")        # default; universally available
  RagLens(model="meta/llama-3.1-8b-instruct")         # cheap judge
  RagLens(model="nvidia/llama-3.1-nemotron-nano-8b-v1")  # Nemotron judge (may need activation)
  ```
* **LiteLLM** (covers OpenAI, Anthropic, Gemini, Ollama, vLLM, ...) — opt-in extra:
  ```bash
  pip install raglens[litellm]
  ```
  ```python
  RagLens(model="litellm:openai/gpt-4o-mini")
  RagLens(model="litellm:anthropic/claude-3-5-sonnet-20241022")
  ```
* **Custom provider** — implement `LLM.complete()` and pass an instance:
  ```python
  class MyLocalLLM(LLM):
      def complete(self, prompt, system=None): ...
  RagLens(model=MyLocalLLM())
  ```

LLM calls inside probing are batched (`complete_batch`) so counterfactual
removal runs in parallel by default.

---

## NVIDIA NIM models — what to pick

| Use case | Model id | Why |
|---|---|---|
| **Judge (default)** | `meta/llama-3.1-70b-instruct` | Universally available on every NIM account; solid instruction-following for binary verdicts |
| Cheap judge / fast iteration | `meta/llama-3.1-8b-instruct` | Small + fast; reliable for yes/no verdicts |
| Nemotron (account-activated) | `nvidia/llama-3.1-nemotron-70b-instruct` | NVIDIA's strongest open Nemotron; superior instruction-following — but listed by `/v1/models` yet 404s on accounts without separate activation. Test with `examples/probe_nim.py` first. |
| Strong reasoning | `meta/llama-3.1-405b-instruct` | Best numeric reasoning; on 8B/70B the model may mangle "rate 0-5" |
| Embeddings (default) | `nvidia/nv-embedqa-e5-v5` | Asymmetric retrieval-tuned embedder on NIM; **requires `input_type` (passage/query)** — see `embed_texts()` |
| Embeddings (larger dim) | `nvidia/nv-embed-v1` | 4096-dim embeddings; same `input_type` contract |

Override the judge with `RAGLENS_JUDGE_MODEL`, the generator with
`RAGLENS_GEN_MODEL`, and the embedder with `RAGLENS_EMBED_MODEL`.

---

## Evaluation dataset format

`evals.jsonl` — one JSON object per line:

```jsonl
{"query": "What does RAG stand for?", "expected_answer": "Retrieval-Augmented Generation.", "expected_chunks": ["RAG stands for Retrieval-Augmented Generation."], "tags": {"domain": "ml"}}
{"query": "Which vector DBs are common in RAG?", "expected_answer": "Qdrant, Weaviate, and pgvector.", "tags": {"domain": "ml"}}
```

* `query` — required.
* `expected_answer` — required for faithfulness/answer-relevance; omit for retrieval-only diagnostics.
* `expected_chunks` — optional list of *known-good* substrings or ids, used for retrieval recall scoring (planned 0.2).
* `tags` — free-form dict for slicing the report (e.g. by domain).

---

## Cost model

| Strategy | LLM calls per sample | When to use |
|---|---|---|
| `counterfactual` (default, novel) | `(N + 1) × n_claims + N (relevance) + N²/2 (conflicts)` | Regression sets; production diagnostics |
| `judge` (RAGAS-style baseline) | `n_claims + N²/2 (conflicts)` | Quick scans; comparison-vs-counterfactual studies |

For a 5-chunk / 3-claim sample on `meta/llama-3.1-70b-instruct`:
counterfactual = ~25 calls; judge = ~18 calls. On `nemotron-nano-8b` it's near-free
for prototyping. Both strategies run on real models — outputs are non-deterministic,
which is why tests assert *structural invariants* (scores in [0,1], failure_mode
in allowed set, n_judge_calls > 0), not specific numbers.

---

## Roadmap

Already shipped in this scaffold:

- [x] Counterfactual chunk attribution (the headline contribution)
- [x] Pairwise chunk conflict detection
- [x] Four-way failure-mode classifier
- [x] HTML + JSON reports
- [x] **NVIDIA NIM as the native default LLM provider** (OpenAI-compatible)
- [x] Framework-agnostic: works with LangChain, LlamaIndex, Haystack, raw pipelines
- [x] Pure-logic unit tests run offline; live integration tests against NVIDIA NIM

Coming next:

- [ ] Retrieval recall scoring against `expected_chunks`
- [ ] Streaming-attribution: judge tokens as they arrive (chatbot UX)
- [ ] Agentic RAG: multi-step reasoning attribution across router/self-RAG loops
- [ ] Built-in `langchain`, `llama_index`, `haystack` trace adapters
- [ ] Judge-distillation: calibrate `nemotron-nano-8b` vs `nemotron-70b` to cut cost ~90%
- [ ] Public `raglens-bench` dataset on Hugging Face — versioned, citable
- [ ] arXiv preprint: "Counterfactual chunk attribution for RAG diagnostics"

---

## Repository layout

```
raglens/         the library
├── core.py            RagLens orchestration
├── attribution.py     counterfactual + judge strategies (the breakthrough)
├── conflict.py        pairwise contradiction detection
├── metrics.py         faithfulness, relevance, registry
├── trace.py           execution trace dataclasses
├── dataset.py         eval dataset loader
├── report.py          report + HTML/JSON renderers
└── providers/         LLM abstractions
    ├── base.py        LLM abstract base, LLMCallable Protocol, make_llm()
    ├── nvidia_nim.py  **NVIDIA NIM provider (default)** — OpenAI SDK against integrate.api.nvidia.com
    └── litellm_provider.py   LiteLLM-backed provider (opt-in extra)

tests/           pure-logic + integration (NIM-gated)
examples/        runnable end-to-end RAG demo on NVIDIA NIM
docs/            (planned) deeper docs
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full module map and execution
flow.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Issues labeled `good first issue` are
preferred entry points for new contributors.

Pure-logic tests run offline:

```bash
pip install -e .[dev]
pytest                       # pure-logic only, no API key required
```

Integration tests call NVIDIA NIM and auto-skip when `NVIDIA_API_KEY` is unset:

```bash
export NVIDIA_API_KEY=nvapi-...
pytest -m integration        # live LLM judge
```

---

## License

MIT. Without patent encumbrance.

---

## Citing RagLens

When the arXiv preprint is live, this section will host a BibTeX entry. Until
then, please link to https://github.com/anujsingh-cse/raglens and the
counterfactual-attribution approach described in `ATTRIBUTION.md` (planned).
