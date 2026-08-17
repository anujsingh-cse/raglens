"""End-to-end tests for the RagLens entry point.

Pure-logic tests (HTML/JSON rendering, exception paths, _coerce_to_trace)
run offline with no LLM. Live ``probe()`` runs against NVIDIA NIM and are
marked ``@pytest.mark.integration``.
"""

from __future__ import annotations

import pytest

from raglens import RagLens, Report
from raglens.attribution import AttributionReport, ChunkAttribution
from raglens.core import _coerce_to_trace, _resolve_strategy
from raglens.metrics import get_metric
from raglens.providers import LLM
from raglens.report import SampleReport
from raglens.trace import Trace

# --------------------------------------------------------------------------- #
# Pure-logic: _coerce_to_trace normalization. No LLM.
# --------------------------------------------------------------------------- #

def test_string_chunks_not_iterated_as_characters():
    """A pipeline returning a bare string for chunks must not be split per
    character — `_coerce_to_trace` must wrap it in a list first."""
    trace = _coerce_to_trace("q", ("answer", "single chunk string"))
    assert len(trace.all_chunks) == 1
    assert trace.all_chunks[0].content == "single chunk string"


def test_coerce_accepts_trace_passthrough():
    src = Trace(query="q", final_answer="A.")
    out = _coerce_to_trace("q", src)
    assert out is src
    assert out.query == "q"


def test_coerce_strings_list():
    trace = _coerce_to_trace("q", ("A.", ["chunk-a", "chunk-b"]))
    assert [c.content for c in trace.all_chunks] == ["chunk-a", "chunk-b"]
    assert trace.final_answer == "A."


def test_coerce_rejects_wrong_shape():
    with pytest.raises(TypeError):
        _coerce_to_trace("q", "not a tuple")


def test_coerce_handles_langchain_like_document():
    """A 'Document-like' object exposing .page_content and .metadata is a
    common LangChain/LlamaIndex shape — coerce must not choke on it."""
    class Doc:
        page_content = "the chunk text"
        metadata = {"src": "wiki"}

    trace = _coerce_to_trace("q", ("answer", [Doc()]))
    assert len(trace.all_chunks) == 1
    assert trace.all_chunks[0].content == "the chunk text"
    assert trace.all_chunks[0].metadata == {"src": "wiki"}


# --------------------------------------------------------------------------- #
# Pure-logic: HTML / JSON rendering with a synthetic Report. No LLM.
# --------------------------------------------------------------------------- #

def _synthetic_report() -> Report:
    attr = AttributionReport(
        query="What does RAG stand for?",
        base_faithfulness=0.92,
        chunk_attributions=[
            ChunkAttribution(chunk_id="r0_c0", attribution_score=0.6),
            ChunkAttribution(chunk_id="r0_c1", attribution_score=-0.1),
        ],
        conflicts=[],
        failure_mode="ok",
        n_judge_calls=42,
        strategy="counterfactual",
    )
    return Report(
        samples=[
            SampleReport(
                query="What does RAG stand for?",
                expected_answer="Retrieval-Augmented Generation.",
                tags={"domain": "ml"},
                metrics={
                    "faithfulness": 0.92,
                    "context_relevance": 0.88,
                    "answer_relevance": 0.95,
                },
                attribution=attr,
            )
        ],
        dataset_name="synthetic",
        dataset_version="0",
        model="meta/llama-3.1-70b-instruct",
        lens_name="RagLens",
        strategy="counterfactual",
    )


def test_html_report_renders(tmp_path):
    report = _synthetic_report()
    out = tmp_path / "report.html"
    report.save_html(out)
    text = out.read_text(encoding="utf-8")
    assert "<table" in text
    assert "RagLens report" in text
    assert "What does RAG stand for?" in text
    assert "failure mode" in text.lower()


def test_json_report_serializes():
    import json

    report = _synthetic_report()
    js = report.to_json()
    parsed = json.loads(js)
    assert parsed["dataset"]["name"] == "synthetic"
    assert parsed["samples"][0]["metrics"]["faithfulness"] == 0.92
    assert parsed["samples"][0]["attribution"]["base_faithfulness"] == 0.92


def test_summary_contains_metric_names():
    report = _synthetic_report()
    text = report.summary()
    assert "faithfulness" in text
    assert "context_relevance" in text
    assert "RagLens report" in text


# --------------------------------------------------------------------------- #
# Pure-logic: constructor input validation. No LLM.
# --------------------------------------------------------------------------- #

def test_invalid_strategy_rejected():
    """Strategy validation is pure logic — test directly to avoid instantiating
    a real LLM provider."""
    with pytest.raises(ValueError):
        _resolve_strategy("bogus")


def test_unknown_metric_name_rejected():
    """Metric-name lookup is pure logic — test directly to avoid instantiating
    a real LLM provider."""
    with pytest.raises(KeyError):
        get_metric("definitely_not_a_metric")


def test_passing_an_llm_instance_bypasses_make_llm():
    """A pre-constructed LLM subclass should pass through `make_llm` untouched
    so users can inject custom providers cleanly."""

    class _SentinelLLM(LLM):
        model = "sentinel"

        def complete(self, prompt, system=None):  # noqa: ARG002
            return ""

    sentinel = _SentinelLLM()
    lens = RagLens(model=sentinel)
    assert lens.llm is sentinel


# --------------------------------------------------------------------------- #
# Live integration: full probe() against NVIDIA NIM. Auto-skip if no key.
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_probe_endtoend_against_nvidia_nim(
    nvidia_api_key, nvidia_judge_model, toy_dataset,
):
    """Run RagLens.probe() against a real (tiny) RAG built on NVIDIA NIM."""
    from raglens.providers import NvidiaNimProvider

    judge = NvidiaNimProvider(model=nvidia_judge_model, api_key=nvidia_api_key)
    gen_model = "meta/llama-3.1-8b-instruct"

    corpus = [
        "RAG stands for Retrieval-Augmented Generation. A retriever fetches "
        "relevant documents from a vector store of embedded chunks, and the "
        "generator LLM's prompt is conditioned on those documents so the "
        "answer is grounded in retrievable evidence.",
        "RAG mitigates hallucination by allowing the answer to be traced to a "
        "retrieved chunk. Faithfulness — the fraction of answer claims "
        "supported by the context — is the dominant quality metric.",
        "Vector databases like Qdrant, Weaviate, and pgvector store chunk "
        "embeddings. A cosine-similarity or ANN search over those embeddings "
        "returns the top-k chunks for a given query.",
    ]

    def real_rag(query: str):
        builder = NvidiaNimProvider(model=gen_model, api_key=nvidia_api_key)
        import math

        from raglens.providers import embed_texts

        corpus_embs = embed_texts(corpus, input_type="passage", api_key=nvidia_api_key)
        query_emb = embed_texts([query], input_type="query",
                                api_key=nvidia_api_key)[0]

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        ranked = sorted(range(len(corpus)),
                        key=lambda i: cosine(query_emb, corpus_embs[i]),
                        reverse=True)
        top = [corpus[i] for i in ranked[:2]]

        prompt = (
            "Answer the QUESTION using only the CONTEXT. Use 2-3 sentences.\n\n"
            f"CONTEXT:\n{top[0]}\n\n{top[1]}\n\nQUESTION:\n{query}\n\nAnswer:"
        )
        answer = builder.complete(prompt)
        return answer, top

    lens = RagLens(model=judge, attribution_strategy="counterfactual")
    report = lens.probe(real_rag, toy_dataset)

    assert len(report.samples) == 1
    s = report.samples[0]
    for name in ("faithfulness", "context_relevance", "answer_relevance"):
        v = s.metrics.get(name)
        assert v is not None
        assert 0.0 <= v <= 1.0
    assert s.attribution is not None
    assert -1.0 <= s.attribution.base_faithfulness <= 1.0
    assert s.attribution.failure_mode in {
        "ok", "retrieval_miss", "chunk_dominance", "generation_ignore"}
