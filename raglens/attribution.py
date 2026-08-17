"""Chunk attribution — RagLens's headline contribution.

The state of the art (RAGAS, DeepEval, TruLens) scores a RAG pipeline at the
*sample* level: "this answer is 0.82 faithful." That tells you *whether*
something is wrong, not *what* is wrong.

RagLens attributes each chunk's *causal* contribution to the final answer's
faithfulness. Concretely:

    For every retrieved chunk C_k we ask:
        "If we re-judge the answer with C_k removed, how does faithfulness
         change?"

      attribution(C_k) = faithfulness(answer | all chunks)
                       - faithfulness(answer | all chunks \\ {C_k})

      > 0  -> C_k was HELPFUL  (removing it made the answer worse)
      < 0  -> C_k was HARMFUL  (removing it made the answer better — the
                                chunk was misleading the generator)
      ~ 0  -> C_k was NEUTRAL

This is counterfactual removal, the same idea behind leave-one-out feature
importance in classical ML. Appling it to RAG chunk ranking is novel.

Two strategies ship:

* :class:`CounterfactualAttribution` — the genuine counterfactual. O(N+1)
  judge calls per sample. Accurate. Use for production eval / regression sets.
* :class:`JudgeAttribution` — single judge call per atomic claim asking
  "which chunk(s) supported this?" RAGAS-ish baseline. Cheaper, less causal.

Both emit the same :class:`AttributionReport`, so users can A/B the cheap vs
accurate path transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from raglens.conflict import Conflict, detect_conflicts
from raglens.metrics import _parse_judge_score, extract_claims
from raglens.providers import LLM
from raglens.trace import Trace


@dataclass(slots=True)
class ChunkAttribution:
    chunk_id: str
    attribution_score: float       # F_base - F_removed, in (-1, 1)
    helped_claims: list[str] = field(default_factory=list)
    harmed_claims: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AttributionReport:
    query: str
    base_faithfulness: float
    chunk_attributions: list[ChunkAttribution]
    conflicts: list[Conflict]
    failure_mode: str            # "ok" | "retrieval_miss" | "chunk_dominance" | "generation_ignore"
    n_judge_calls: int = 0
    strategy: str = "counterfactual"

    def helpful_chunks(self, top_k: int | None = None) -> list[ChunkAttribution]:
        positives = sorted([c for c in self.chunk_attributions if c.attribution_score > 0],
                           key=lambda c: c.attribution_score, reverse=True)
        return positives[:top_k] if top_k else positives

    def harmful_chunks(self, top_k: int | None = None) -> list[ChunkAttribution]:
        negatives = sorted([c for c in self.chunk_attributions if c.attribution_score < 0],
                           key=lambda c: c.attribution_score)
        return negatives[:top_k] if top_k else negatives


# ------------------------------------------------------------------ #
# Strategy interface
# ------------------------------------------------------------------ #

class AttributionStrategy:
    """Subclass and implement :meth:`run`."""

    name = "base"

    def run(self, trace: Trace, llm: LLM) -> AttributionReport:  # pragma: no cover
        raise NotImplementedError


# ------------------------------------------------------------------ #
# FAITHFULNESS primitive (metric-local, kept here so attribution does not
# double-count metric/probe LLM calls)
# ------------------------------------------------------------------ #

def _faithfulness_with(chunks_text: str, claims: list[str], llm: LLM,
                       batch: bool = True) -> tuple[float, int]:
    """Judge every claim against a chunk context. Returns (score, n_calls)."""
    if not claims:
        return 0.0, 0
    prompts = [
        ("You are a strict fact-checker. Decide whether the CLAIM is fully "
         "supported by the CONTEXT. Answer 'yes' or 'no' only.\n\n"
         f"CONTEXT:\n{chunks_text}\n\nCLAIM:\n{claim}\n\nVerdict:", None)
        for claim in claims
    ]
    if batch and len(prompts) > 1:
        responses = llm.complete_batch(prompts)
    else:
        responses = [llm.complete(p, s) for p, s in prompts]
    supported = sum(1 for r in responses if _parse_judge_score(r) >= 0.5)
    return supported / len(claims), len(prompts)


def _chunks_text(chunks) -> str:
    return "\n\n".join(f"[{c.id}] {c.content}" for c in chunks)


# ------------------------------------------------------------------ #
# Counterfactual (cheap enough on top-k, accurate causally) — the breakthrough
# ------------------------------------------------------------------ #

class CounterfactualAttribution(AttributionStrategy):
    """Leave-one-chunk-out faithfulness.

    Cost: ``(N+1) * len(claims)`` judge calls per sample (one baseline, one per
    removal). On a 5-chunk, 3-claim sample that's 18 calls; cheap enough for a
    regression set, not for live monitoring.
    """

    name = "counterfactual"

    def run(self, trace: Trace, llm: LLM) -> AttributionReport:
        trace.index_chunks()
        chunks = list(trace.all_chunks)
        if not chunks or not trace.final_answer:
            fm = "retrieval_miss" if not chunks else "generation_ignore"
            return AttributionReport(query=trace.query, base_faithfulness=0.0,
                                     chunk_attributions=[], conflicts=[],
                                     failure_mode=fm, n_judge_calls=0,
                                     strategy=self.name)

        claims = extract_claims(trace.final_answer)
        n_calls = 0

        f_base, n = _faithfulness_with(_chunks_text(chunks), claims, llm)
        n_calls += n

        attributions: list[ChunkAttribution] = []
        per_removal_scores: list[float] = []
        for k in range(len(chunks)):
            removed = chunks[:k] + chunks[k+1:]
            if not removed:  # only one chunk — removing it drops all context
                f_minus = 0.0
                per_removal_scores.append(f_minus)
                delta = f_base - f_minus
                attributions.append(ChunkAttribution(
                    chunk_id=chunks[k].id or "",
                    attribution_score=round(delta, 4),
                    helped_claims=list(claims) if delta > 0 else [],
                    harmed_claims=[],
                ))
                continue
            f_minus, n = _faithfulness_with(_chunks_text(removed), claims, llm)
            n_calls += n
            delta = f_base - f_minus
            per_removal_scores.append(f_minus)
            helped = list(claims) if delta > 0 else []
            harmed = list(claims) if delta < 0 else []
            attributions.append(ChunkAttribution(
                chunk_id=chunks[k].id or "",
                attribution_score=round(delta, 4),
                helped_claims=helped,
                harmed_claims=harmed,
            ))

        conflicts = detect_conflicts(chunks, llm)
        n_calls += len(chunks) * (len(chunks) - 1) // 2

        ctx_rel = _avg_context_relevance(trace, llm)
        n_calls += len(chunks)  # context_relevance metric makes one call per chunk

        failure_mode = classify_failure_mode(
            f_base=f_base,
            per_removal_scores=per_removal_scores,
            conflicts=conflicts,
            n_chunks=len(chunks),
            context_relevance=ctx_rel,
        )
        return AttributionReport(query=trace.query,
                                 base_faithfulness=round(f_base, 4),
                                 chunk_attributions=attributions,
                                 conflicts=conflicts,
                                 failure_mode=failure_mode,
                                 n_judge_calls=n_calls,
                                 strategy=self.name)


# ------------------------------------------------------------------ #
# Judge attribution (RAGAS-style baseline for comparison)
# ------------------------------------------------------------------ #

class JudgeAttribution(AttributionStrategy):
    """Single judge call per atomic claim — "which chunk supported this?".

    Cost: ``len(claims)`` calls per sample. Faster but not counterfactual: it
    measures *correlation*, not *causation*. Useful as a cheap path or as the
    comparison baseline that justifies the counterfactual approach.
    """

    name = "judge"

    def run(self, trace: Trace, llm: LLM) -> AttributionReport:
        trace.index_chunks()
        chunks = trace.all_chunks
        if not chunks or not trace.final_answer:
            fm = "retrieval_miss" if not chunks else "generation_ignore"
            return AttributionReport(query=trace.query, base_faithfulness=0.0,
                                     chunk_attributions=[], conflicts=[],
                                     failure_mode=fm, n_judge_calls=0,
                                     strategy=self.name)

        claims = extract_claims(trace.final_answer)
        context = _chunks_text(chunks)
        # Single baseline faithfulness pass, to populate base_faithfulness
        f_base, n_calls = _faithfulness_with(context, claims, llm)

        # Per-claim: ask judge for the single chunk id that supports (or "NONE")
        prompts = []
        for claim in claims:
            chunk_lines = "\n".join(f"- {c.id}" for c in chunks)
            prompts.append((
                f"Given the CONTEXT labelled with chunk ids and a CLAIM, "
                f"return the single chunk id that most strongly supports the "
                f"claim, or 'NONE' if unsupported.\n\n"
                f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\n"
                f"AVAILABLE_IDS:\n{chunk_lines}\n\nChunk id (or NONE):",
                None,
            ))
        responses = llm.complete_batch(prompts) if len(prompts) > 1 \
            else [llm.complete(prompts[0][0], prompts[0][1])]
        n_calls += len(responses)

        help_counts: dict[str, int] = {c.id or "": 0 for c in chunks}
        for _claim, resp in zip(claims, responses, strict=False):
            picked = resp.strip().split()[0] if resp.strip() else "NONE"
            if picked in help_counts:
                help_counts[picked] += 1

        attributions = [
            ChunkAttribution(chunk_id=cid, attribution_score=round(cnt / max(1, len(claims)), 4))
            for cid, cnt in help_counts.items()
        ]

        conflicts = detect_conflicts(chunks, llm)
        n_calls += len(chunks) * (len(chunks) - 1) // 2

        return AttributionReport(query=trace.query,
                                 base_faithfulness=round(f_base, 4),
                                 chunk_attributions=attributions,
                                 conflicts=conflicts,
                                 failure_mode="ok",
                                 n_judge_calls=n_calls,
                                 strategy=self.name)


# ------------------------------------------------------------------ #
# Failure-mode classifier — turns numbers into a diagnosis
# ------------------------------------------------------------------ #

def classify_failure_mode(*, f_base: float, per_removal_scores: list[float],
                          conflicts: list[Conflict], n_chunks: int,
                          context_relevance: float | None,
                          dominance_threshold: float = 0.1,
                          miss_threshold: float = 0.6) -> str:
    """Classify *why* a low-faithfulness sample went wrong.

    Returns one of:
      * ``"ok"`` — faithfulness is acceptable.
      * ``"retrieval_miss"`` — chunks don't carry the answer; retriever failed.
      * ``"chunk_dominance"`` — a misleading chunk dominated; re-rank / filter.
      * ``"generation_ignore"`` — chunks are relevant but model ignored them;
        fix the prompt / model.
    """
    if f_base >= 0.8:
        return "ok"

    max_recovery = max(per_removal_scores, default=0.0) - f_base if per_removal_scores else 0.0
    if max_recovery >= dominance_threshold:
        # Removing some chunk *substantially raised* faithfulness -> that
        # chunk was actively misleading the model.
        return "chunk_dominance"

    if context_relevance is not None and context_relevance >= 0.6:
        # Chunks looked relevant to a human judge (or LLM judge) but the model
        # still produced unsupported claims -> the generator dropped them.
        return "generation_ignore"

    if n_chunks and f_base < miss_threshold:
        return "retrieval_miss"

    return "generation_ignore"


def _avg_context_relevance(trace: Trace, llm: LLM) -> float | None:
    """Lightweight inline context-relevance probe used by the classifier."""
    from raglens.metrics import get_metric
    try:
        return get_metric("context_relevance")(trace, llm)
    except Exception:
        return None
