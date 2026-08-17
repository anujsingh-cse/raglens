"""Pairwise conflict detection between retrieved chunks.

A "conflict" is two chunks that, taken together, would lead a generator to
contradict itself. Detecting these is one of the cheapest high-signal
diagnostics a RAG team can run, because conflict pairs are exactly the cases
where chunk-re-ranking / chunk-dedup / source-trust policies pay off.
"""

from __future__ import annotations

from dataclasses import dataclass

from raglens.providers import LLM
from raglens.trace import RetrievedChunk


@dataclass(slots=True)
class Conflict:
    chunk_a_id: str
    chunk_b_id: str
    severity: float  # 0..1, judge confidence that the two contradict
    explanation: str = ""


def detect_conflicts(chunks: list[RetrievedChunk], llm: LLM,
                     batch_threshold: int = 4) -> list[Conflict]:
    """Run pairwise contradictions.

    To keep cost reasonable, conflicts are checked pairwise only above a small
    batch threshold; below it (the common case for a 5-chunk retrieval set),
    the whole cross-product is judged in one batched call.
    """
    if len(chunks) < 2:
        return []

    pairs = [(chunks[i], chunks[j])
             for i in range(len(chunks))
             for j in range(i + 1, len(chunks))]

    prompts: list[tuple[str, str | None]] = []
    system = ("You are a contradiction detector for claim pairs. "
             "Respond in this exact format: CONFIDENCE <float 0-1> || "
             "EXPLANATION <one short sentence>")
    for a, b in pairs:
        prompt = (
            "Decide whether CHUNK_A and CHUNK_B contradict each other on a "
            "factual claim relevant to the same query.\n\n"
            f"CHUNK_A:\n{a.content}\n\nCHUNK_B:\n{b.content}\n\n"
            "Format your answer strictly as:\n"
            "CONFIDENCE <0..1> || EXPLANATION <short>"
        )
        prompts.append((prompt, system))

    if len(prompts) >= batch_threshold:
        responses = llm.complete_batch(prompts)
    else:
        responses = [llm.complete(p, s) for p, s in prompts]

    conflicts: list[Conflict] = []
    for (a, b), resp in zip(pairs, responses, strict=True):
        conf, _expl = _parse_conflict(resp)
        if conf >= 0.5:
            conflicts.append(Conflict(
                chunk_a_id=a.id or "",
                chunk_b_id=b.id or "",
                severity=conf,
                explanation=_expl,
            ))
    return conflicts


def _parse_conflict(resp: str) -> tuple[float, str]:
    """Parse `CONFIDENCE 0.7 || EXPLANATION chunk a says X but b says Y`."""
    if "||" in resp:
        left, right = resp.split("||", 1)
        conf = _try_float(left.replace("CONFIDENCE", "").strip())
        expl = right.replace("EXPLANATION", "").strip()
        return conf, expl
    # Fallback: best-effort parse a leading float
    conf = _try_float(resp.strip())
    return conf, resp.strip()[:200]


def _try_float(s: str) -> float:
    import re
    m = re.search(r"([01](?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.0
