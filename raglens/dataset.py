"""Evaluation datasets: the ground-truth cases RagLens runs against."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from raglens.exceptions import DatasetError


@dataclass(slots=True)
class Sample:
    """One evaluation case.

    Args:
        query: the user question.
        expected_answer: reference answer. May be ``None`` for retrieval-only
            diagnostics (no factual grading), but is required for full
            faithfulness / answer-relevance scoring.
        expected_chunks: optional set of *known-good* chunk ids or substrings
            that a perfect retriever should surface. Used to score recall.
        tags: free-form tags for slicing the report (e.g. ``{"domain":"legal"}``).
    """

    query: str
    expected_answer: str | None = None
    expected_chunks: list[str] | None = None
    tags: dict[str, str] | None = None


@dataclass(slots=True)
class Dataset:
    """A collection of :class:`Sample` cases."""

    samples: list[Sample]
    name: str = "untitled"
    version: str = "0"

    def __post_init__(self) -> None:
        if not self.samples:
            raise DatasetError("Dataset must contain at least one Sample.")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None,
                   version: str = "0") -> Dataset:
        """Load samples from a JSONL file. Each line: ``{"query": ..., ...}``."""
        p = Path(path)
        if not p.exists():
            raise DatasetError(f"Dataset file not found: {p}")
        samples: list[Sample] = []
        try:
            for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "query" not in row:
                    raise DatasetError(f"line {lineno+1}: missing 'query' field")
                samples.append(Sample(
                    query=row["query"],
                    expected_answer=row.get("expected_answer") or row.get("reference"),
                    expected_chunks=row.get("expected_chunks"),
                    tags=row.get("tags"),
                ))
        except json.JSONDecodeError as e:
            raise DatasetError(f"invalid JSON in {p}: {e}") from e
        if not samples:
            raise DatasetError(f"no samples parsed from {p}")
        return cls(samples=samples, name=name or p.stem, version=version)

    @classmethod
    def from_list(cls, samples: Iterable[Sample], name: str = "inline",
                  version: str = "0") -> Dataset:
        return cls(samples=list(samples), name=name, version=version)
