"""Metrics: the per-sample scoring primitives.

RagLens ships four families. The first three are familiar (Faithfulness,
Context Relevance, Answer Relevance); the fourth, **Attribution**, is what
makes RagLens diagnostic rather than merely judgemental — see
:mod:`raglens.attribution`.

All metrics share the same contract: they receive a :class:`Trace` and an
:class:`LLM`, and return a ``float`` (or ``None`` if not computable).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from raglens.providers import LLM
from raglens.trace import Trace

# --------------------------------------------------------------------------- #
# Claim extraction (shared by metrics that decompose answers into atomic
# verifiable statements — standard RAGAS-style decomposition, kept tiny).
# --------------------------------------------------------------------------- #

_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_claims(answer: str) -> list[str]:
    """Split an answer into atomic claims by sentence punctuation.

    A production version would use an LLM to do semantic decomposition
    (LLMs catch coordinate clauses better than regex). We start with the cheap
    deterministic version so unit tests stay fast and offline; the LLM
    decomposition is wired in :mod:`raglens.attribution` where it matters most.
    """
    claims = [c.strip() for c in _CLAIM_SPLIT_RE.split(answer) if c.strip()]
    return claims or [answer]


# --------------------------------------------------------------------------- #
# Metric registry: name -> callable(trace, llm) -> float | None
# --------------------------------------------------------------------------- #

MetricFn = Callable[[Trace, LLM], "float | None"]
_METRICS: dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    def deco(fn: MetricFn) -> MetricFn:
        _METRICS[name] = fn
        return fn
    return deco


def get_metric(name: str) -> MetricFn:
    if name not in _METRICS:
        raise KeyError(f"unknown metric '{name}'. available: {sorted(_METRICS)}")
    return _METRICS[name]


def available_metrics() -> list[str]:
    return sorted(_METRICS)


# --------------------------------------------------------------------------- #
# Built-in metrics. LLM-judge prompts are intentionally terse to keep cost low;
# the judge's response is parsed with a forgiving matcher (yes/no / 1-5 / 0.0-1.0).
# --------------------------------------------------------------------------- #

_SCORE_RE = re.compile(r"([0-5](?:\.\d+)?)|yes|no|true|false", re.IGNORECASE)


def _parse_judge_score(text: str, scale: float = 5.0) -> float:
    """Parse a judge LLM's response into a float in [0, 1].

    Args:
        scale: divisor for numeric scores. Default 5.0 — prompts request
            "0-5 integer" ratings. For yes/no verdicts the scale is ignored.
    """
    m = _SCORE_RE.search(text.strip())
    if not m:
        return 0.0
    tok = m.group(0).lower()
    if tok in ("yes", "true"):
        return 1.0
    if tok in ("no", "false"):
        return 0.0
    val = float(tok)
    return min(val / scale, 1.0) if scale > 0 else 0.0


@register("faithfulness")
def faithfulness(trace: Trace, llm: LLM) -> float | None:
    """Fraction of atomic claims in the final answer supported by retrieved context."""
    if not trace.final_answer or not trace.all_chunks:
        return None
    context = "\n\n".join(f"[{c.id}] {c.content}" for c in trace.all_chunks)
    claims = extract_claims(trace.final_answer)
    if not claims:
        return None
    supported = 0
    for claim in claims:
        prompt = (
            "You are a strict fact-checker. Decide whether the CLAIM is fully "
            "supported by the CONTEXT. Answer 'yes' or 'no' only.\n\n"
            f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\nVerdict:"
        )
        verdict = llm.complete(prompt)
        if _parse_judge_score(verdict) >= 0.5:
            supported += 1
    return supported / len(claims)


@register("context_relevance")
def context_relevance(trace: Trace, llm: LLM) -> float | None:
    """Mean per-chunk relevance to the query (0-1)."""
    chunks = trace.all_chunks
    if not chunks or not trace.query:
        return None
    scores: list[float] = []
    for chunk in chunks:
        prompt = (
            "Rate how relevant the CONTEXT is for answering the QUERY on a 0-5 "
            "integer scale. Answer with just the number.\n\n"
            f"QUERY:\n{trace.query}\n\nCONTEXT:\n{chunk.content}\n\nRating:"
        )
        scores.append(_parse_judge_score(llm.complete(prompt)))
    return sum(scores) / len(scores)


@register("answer_relevance")
def answer_relevance(trace: Trace, llm: LLM) -> float | None:
    """Does the answer actually address the query? (0-1)."""
    if not trace.final_answer or not trace.query:
        return None
    prompt = (
        "Rate how well the ANSWER addresses the QUERY on a 0-5 integer scale. "
        "Answer with just the number.\n\n"
        f"QUERY:\n{trace.query}\n\nANSWER:\n{trace.final_answer}\n\nRating:"
    )
    return _parse_judge_score(llm.complete(prompt))
