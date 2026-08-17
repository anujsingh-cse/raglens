"""The :class:`RagLens` entry point.

``RagLens`` is a thin orchestrator. The user gives it:

* a ``model`` (provider spec) so judges run,
* a ``pipeline`` callable that turns a query into an answer and its retrieved
  chunks (or a full :class:`Trace` for agentic RAG) — RagLens does not depend
  on any particular RAG framework,
* a :class:`Dataset` of evaluation cases.

It runs built-in metrics plus the chosen attribution strategy on every sample
and returns a :class:`Report` users can ``summary()`` or ``save_html()``.
"""

from __future__ import annotations

from collections.abc import Callable

from raglens.attribution import (
    AttributionReport,
    AttributionStrategy,
    CounterfactualAttribution,
    JudgeAttribution,
)
from raglens.dataset import Dataset
from raglens.metrics import get_metric
from raglens.providers import LLM, LLMCallable, make_llm
from raglens.report import Report, SampleReport
from raglens.trace import GenerationStep, RetrievalStep, RetrievedChunk, Trace

PipelineFn = Callable[[str], "object"]


def _coerce_to_trace(query: str, pipeline_output) -> Trace:
    """Accept either a Trace, or a ``(answer, chunks)`` tuple.

    chunks can be a list of strings, :class:`RetrievedChunk` objects,
    or a list of ``(content, score)`` tuples — common LangChain/LlamaIndex shapes.
    """
    if isinstance(pipeline_output, Trace):
        if not pipeline_output.query:
            pipeline_output.query = query
        return pipeline_output

    if not isinstance(pipeline_output, tuple) or len(pipeline_output) != 2:
        raise TypeError(
            f"pipeline must return a Trace or (answer, chunks); got "
            f"{type(pipeline_output).__name__}")

    answer, raw_chunks = pipeline_output
    if isinstance(raw_chunks, str):
        raw_chunks = [raw_chunks]
    chunks: list[RetrievedChunk] = []
    for i, raw in enumerate(raw_chunks or []):
        if isinstance(raw, RetrievedChunk):
            chunks.append(raw)
        elif isinstance(raw, str):
            chunks.append(RetrievedChunk(content=raw, id=f"r0_c{i}"))
        elif isinstance(raw, tuple) and len(raw) >= 1:
            chunks.append(RetrievedChunk(
                content=str(raw[0]),
                score=float(raw[1]) if len(raw) > 1 and raw[1] is not None else None,
                id=f"r0_c{i}",
            ))
        else:
            # LangChain Document / dict-like
            content = getattr(raw, "page_content", None) or getattr(raw, "content", "") or str(raw)
            meta = getattr(raw, "metadata", {}) or {}
            chunks.append(RetrievedChunk(content=content, metadata=dict(meta), id=f"r0_c{i}"))

    return Trace(
        query=query,
        retrieval=[RetrievalStep(query=query, chunks=chunks, step_index=0)],
        generation=[GenerationStep(prompt="", answer=answer, step_index=0)],
        final_answer=answer,
    )


_STRATEGIES = {
    "counterfactual": CounterfactualAttribution,
    "judge": JudgeAttribution,
}


class RagLens:
    """The orchestrator users construct.

    Args:
        model: provider spec. A string is routed through :class:`NvidiaNimProvider`
            (the default) and used as the NVIDIA NIM model id. Pass an :class:`LLM`
            instance to inject a custom provider, or prefix the string with
            ``litellm:`` (e.g. ``litellm:openai/gpt-4o-mini``) to route via the
            optional LiteLLM extra.
        attribution_strategy: ``"counterfactual"`` (default; expensive, causal)
            or ``"judge"`` (cheap; baseline).
        metrics: iterable of metric names to run. Defaults to faithfulness,
            context_relevance, answer_relevance.

    Example::

        lens = RagLens(model="meta/llama-3.1-70b-instruct")
        report = lens.probe(my_rag, dataset)
        print(report.summary())
    """

    def __init__(
        self,
        model: str | LLM | LLMCallable,
        *,
        attribution_strategy: str | AttributionStrategy = "counterfactual",
        metrics: list[str] | None = None,
        name: str = "RagLens",
        **llm_kwargs,
    ):
        self.llm = make_llm(model, **llm_kwargs)
        self.strategy = _resolve_strategy(attribution_strategy)
        # Store (name, fn) pairs so user-provided names survive intact — the
        # prior `zip(_DEFAULT_METRIC_NAMES, fns)` form silently aligned a
        # user's 2-name list with the first 2 default names.
        self.metric_specs = [
            (n, get_metric(n)) for n in (metrics or [
                "faithfulness", "context_relevance", "answer_relevance"])
        ]
        self.name = name

    def probe(self, pipeline: PipelineFn, dataset: Dataset,
              run_attribution: bool = True, max_samples: int | None = None) -> Report:
        """Run every sample through the pipeline and produce a :class:`Report`."""
        samples = dataset.samples
        if max_samples is not None:
            samples = samples[:max_samples]

        per_sample: list[SampleReport] = []
        for sample in samples:
            pipeline_output = pipeline(sample.query)
            trace = _coerce_to_trace(sample.query, pipeline_output)
            trace.expected_answer = sample.expected_answer

            metric_vals: dict[str, float | None] = {}
            for name, fn in self.metric_specs:
                metric_vals[name] = fn(trace, self.llm)

            attr: AttributionReport | None = None
            if run_attribution:
                attr = self.strategy.run(trace, self.llm)

            per_sample.append(SampleReport(
                query=sample.query,
                expected_answer=sample.expected_answer,
                tags=sample.tags or {},
                metrics=metric_vals,
                attribution=attr,
            ))

        return Report(
            samples=per_sample,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            model=getattr(self.llm, "model", "unknown"),
            lens_name=self.name,
            strategy=self.strategy.name,
        )


def _resolve_strategy(spec: str | AttributionStrategy) -> AttributionStrategy:
    if isinstance(spec, AttributionStrategy):
        return spec
    if spec not in _STRATEGIES:
        raise ValueError(f"unknown attribution strategy '{spec}'. "
                         f"available: {sorted(_STRATEGIES)}")
    return _STRATEGIES[spec]()


__all__ = ["RagLens", "PipelineFn"]
