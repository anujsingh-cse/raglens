"""Tests for the attribution primitives and failure-mode classifier.

Pure-logic tests (the classifier + conflict parser + dataset I/O) run offline
and need no LLM. End-to-end attribution requires an LLM judge; those tests are
marked ``integration`` and run only when ``NVIDIA_API_KEY`` is set.
"""

from __future__ import annotations

import pytest

from raglens.attribution import (
    CounterfactualAttribution,
    classify_failure_mode,
)
from raglens.conflict import _parse_conflict
from raglens.trace import RetrievalStep, RetrievedChunk, Trace

# --------------------------------------------------------------------------- #
# Failure-mode classifier — pure function, no LLM needed.
# --------------------------------------------------------------------------- #

def test_classifier_ok_when_faithfulness_high():
    fm = classify_failure_mode(
        f_base=0.9, per_removal_scores=[0.9], conflicts=[],
        n_chunks=1, context_relevance=None)
    assert fm == "ok"


def test_classifier_chunk_dominance_when_removal_helps():
    fm = classify_failure_mode(
        f_base=0.4, per_removal_scores=[0.4, 0.4, 0.8], conflicts=[],
        n_chunks=3, context_relevance=0.3)
    assert fm == "chunk_dominance"


def test_classifier_generation_ignore_when_context_relevant_but_ungrounded():
    fm = classify_failure_mode(
        f_base=0.4, per_removal_scores=[0.4, 0.4, 0.4], conflicts=[],
        n_chunks=3, context_relevance=0.7)
    assert fm == "generation_ignore"


def test_classifier_retrieval_miss_when_context_irrelevant():
    fm = classify_failure_mode(
        f_base=0.2, per_removal_scores=[0.2, 0.2, 0.2], conflicts=[],
        n_chunks=3, context_relevance=0.2)
    assert fm == "retrieval_miss"


def test_classifier_fallback_is_generation_ignore_after_thresholds_removed():
    """When faithfulness is mid-range, no recovery, context isn't categorically
    relevant, and faithfulness is above the miss threshold, the residual
    diagnosis is `generation_ignore` (model ignored usable context)."""
    fm = classify_failure_mode(
        f_base=0.65, per_removal_scores=[0.65, 0.65], conflicts=[],
        n_chunks=2, context_relevance=0.3)
    assert fm == "generation_ignore"


# --------------------------------------------------------------------------- #
# Conflict format parsing — pure function, no LLM needed.
# --------------------------------------------------------------------------- #

def test_parse_conflict_format():
    conf, expl = _parse_conflict("CONFIDENCE 0.9 || EXPLANATION a says X, b says not-X")
    assert conf == pytest.approx(0.9)
    assert "says" in expl


def test_parse_conflict_fallback_parses_leading_number():
    conf, _ = _parse_conflict("0.65 chunk a contradicts chunk b")
    assert conf == pytest.approx(0.65, rel=1e-2)


# --------------------------------------------------------------------------- #
# CounterfactualAttribution early-return paths — no LLM calls occur.
# --------------------------------------------------------------------------- #

def test_attribution_empty_chunks_returns_safe_report():
    """No chunks but a non-empty answer ⇒ retrieval_miss, no LLM call."""
    trace = Trace(query="q", final_answer="A.", retrieval=[])
    rep = CounterfactualAttribution().run(trace, _NoopLLM())
    assert rep.chunk_attributions == []
    assert rep.failure_mode == "retrieval_miss"
    assert rep.n_judge_calls == 0


def test_attribution_empty_answer_returns_safe_report():
    """Chunks present but empty answer ⇒ generation_ignore, no LLM call."""
    trace = Trace(
        query="q",
        retrieval=[RetrievalStep(query="q", chunks=[
            RetrievedChunk(content="some context", id="r0_c0")
        ], step_index=0)],
        final_answer="",
    )
    rep = CounterfactualAttribution().run(trace, _NoopLLM())
    assert rep.failure_mode == "generation_ignore"
    assert rep.n_judge_calls == 0


class _NoopLLM:
    """An LLM that explodes if called. Used to *prove* the early-return path
    skips the judge entirely. Not mock data — there is no scripted response to
    assert against; if the LLM is touched, the test fails loudly."""

    model = "noop-never-called"

    def complete(self, prompt, system=None):  # noqa: ARG002
        raise AssertionError("LLM.complete was called on a path expected to early-return")

    def complete_batch(self, prompts):  # noqa: ARG002
        raise AssertionError("LLM.complete_batch was called on a path expected to early-return")


# --------------------------------------------------------------------------- #
# Live integration tests against NVIDIA NIM. Auto-skip if no API key.
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_counterfactual_attribution_against_nvidia_nim(
    nvidia_api_key, nvidia_judge_model,
):
    """Live LLM judge. Asserts only structural invariants — model output is
    non-deterministic, so we check the contract, not specific numbers."""
    from raglens.providers import NvidiaNimProvider

    llm = NvidiaNimProvider(model=nvidia_judge_model, api_key=nvidia_api_key)
    trace = Trace(
        query="What does RAG stand for?",
        retrieval=[RetrievalStep(query="What does RAG stand for?", chunks=[
            RetrievedChunk(
                content=(
                    "RAG stands for Retrieval-Augmented Generation. It retrieves "
                    "relevant documents from a knowledge base and conditions the "
                    "language model on them before generating an answer, so the "
                    "answer is grounded in retrievable evidence rather than the "
                    "model's parametric weights."
                ),
                id="r0_c0",
            ),
            RetrievedChunk(
                content=(
                    "RAG pipelines typically include an embedding model, a vector "
                    "store, a retriever, and a generator LLM. Faithfulness to "
                    "the retrieved context is the dominant quality metric."
                ),
                id="r0_c1",
            ),
        ], step_index=0)],
        final_answer=(
            "RAG stands for Retrieval-Augmented Generation. It grounds the LLM "
            "answer in retrieved documents from a knowledge base."
        ),
    )
    rep = CounterfactualAttribution().run(trace, llm)

    assert 0.0 <= rep.base_faithfulness <= 1.0
    assert len(rep.chunk_attributions) == 2
    for c in rep.chunk_attributions:
        assert -1.0 <= c.attribution_score <= 1.0
    assert rep.failure_mode in {
        "ok", "retrieval_miss", "chunk_dominance", "generation_ignore"}
    assert rep.n_judge_calls > 0
    assert rep.strategy == "counterfactual"
