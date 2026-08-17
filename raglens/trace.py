"""Execution traces: how a RAG pipeline produced an answer for a query.

A :class:`Trace` is the structured unit RagLens reasons about. Adapters for
LangChain / LlamaIndex / Haystack convert their invocation output into a Trace;
users writing raw pipelines can build one directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    """A single retrieved context chunk.

    Attributes:
        content: the chunk text.
        score: optional retrieval score (similarity, relevance). Use ``None``
            if the retriever does not expose one.
        metadata: arbitrary retriever-side metadata (source, doc id, page, ...).
        id: stable identifier used by the attribution algorithm to refer to
            this chunk across counterfactual runs. Auto-assigned if omitted.
    """

    content: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(slots=True)
class RetrievalStep:
    """One retrieval call within a pipeline (agentic RAG may have several)."""

    query: str
    chunks: list[RetrievedChunk]
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationStep:
    """One generation step. Holds the prompt actually sent to the LLM."""

    prompt: str
    answer: str
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trace:
    """A full execution trace of a RAG pipeline for a single query.

    Agentic pipelines (router / corrective / self-RAG) produce multi-step
    traces; plain retriever+reader pipelines produce a single retrieval and
    generation step. RagLens treats both uniformly.
    """

    query: str
    retrieval: list[RetrievalStep] = field(default_factory=list)
    generation: list[GenerationStep] = field(default_factory=list)
    final_answer: str = ""
    expected_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_chunks(self) -> list[RetrievedChunk]:
        """Flatten every retrieved chunk across all retrieval steps."""
        chunks: list[RetrievedChunk] = []
        for step in self.retrieval:
            chunks.extend(step.chunks)
        return chunks

    def index_chunks(self) -> None:
        """Assign stable IDs to every chunk that does not already have one.

        IDs are ``"r{step}_c{pos}"`` so they survive counterfactual removal.
        """
        for r_step in self.retrieval:
            for pos, chunk in enumerate(r_step.chunks):
                if chunk.id is None:
                    chunk.id = f"r{r_step.step_index}_c{pos}"
