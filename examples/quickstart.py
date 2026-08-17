"""RagLens quickstart — real end-to-end RAG judged by NVIDIA NIM.

Run:
    export NVIDIA_API_KEY=nvapi-...   # get one at https://build.nvidia.com
    pip install raglens
    python -m examples.quickstart

What this script demonstrates (no mock LLM, no stub answer):

    1. A tiny knowledge base (3 short paragraphs about RAG itself).
    2. NVIDIA NIM embeds every chunk at ``nvidia/nv-embedqa-e5-v5``.
    3. NVIDIA NIM embeds the query and cosine-similarity retrieves top-2 chunks.
    4. NVIDIA NIM generates an answer from the retrieved chunks
       (``meta/llama-3.1-8b-instruct``).
    5. RagLens wraps the same NVIDIA NIM endpoint as an LLM-judge
       (``meta/llama-3.1-70b-instruct``) and produces a diagnostic
       report: counterfactual chunk attribution + failure-mode classification.
    6. The report is written to ``quickstart_report.html``.
"""

from __future__ import annotations

import math
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads .env from the cwd if present; harmless if missing

from raglens import Dataset, RagLens, Sample  # noqa: E402 — must run after load_dotenv
from raglens.providers import NvidiaNimProvider, embed_texts  # noqa: E402

CORPUS = [
    "RAG stands for Retrieval-Augmented Generation. A retriever fetches "
    "relevant documents from a vector store of embedded chunks, and the "
    "generator LLM's prompt is conditioned on those documents so the answer "
    "is grounded in retrievable evidence rather than the model's parametric "
    "weights.",

    "RAG mitigates hallucination by allowing the answer to be traced to a "
    "retrieved chunk. Faithfulness — the fraction of answer claims supported "
    "by the context — is the dominant quality metric; chunk relevance and "
    "answer relevance are secondary.",

    "Vector databases like Qdrant, Weaviate, and pgvector store chunk "
    "embeddings. A cosine-similarity or approximate-nearest-neighbour search "
    "over those embeddings returns the top-k chunks for a given query, which "
    "are then passed to the generator as in-context examples.",
]

JUDGE_MODEL = os.environ.get("RAGLENS_JUDGE_MODEL",
                             "meta/llama-3.1-70b-instruct")
GEN_MODEL = os.environ.get("RAGLENS_GEN_MODEL",
                           "meta/llama-3.1-8b-instruct")
EMBED_MODEL = os.environ.get("RAGLENS_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_rag(api_key: str):
    """Construct a real RAG pipeline callable: ``(query) -> (answer, chunks)``.

    Embeddings + retrieval + generation all run against NVIDIA NIM. The
    callable returned here is exactly what RagLens.probe() expects.
    """
    generator = NvidiaNimProvider(model=GEN_MODEL, api_key=api_key,
                                  temperature=0.2)

    def rag(query: str) -> tuple[str, list[str]]:
        corpus_embs = embed_texts(CORPUS, model=EMBED_MODEL,
                                  input_type="passage", api_key=api_key)
        query_emb = embed_texts([query], model=EMBED_MODEL,
                                input_type="query", api_key=api_key)[0]

        ranked = sorted(
            range(len(CORPUS)),
            key=lambda i: _cosine(query_emb, corpus_embs[i]),
            reverse=True,
        )
        top = [CORPUS[i] for i in ranked[:2]]

        prompt = (
            "You are a careful assistant. Answer the QUESTION using only the "
            "CONTEXT. Be concise (2-3 sentences) and cite nothing beyond the "
            "context.\n\n"
            f"CONTEXT:\n{top[0]}\n\n{top[1]}\n\n"
            f"QUESTION:\n{query}\n\nAnswer:"
        )
        answer = generator.complete(prompt)
        return answer, top

    return rag


def main() -> int:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print(
            "NVIDIA_API_KEY is not set. Get one at https://build.nvidia.com, "
            "then: export NVIDIA_API_KEY=nvapi-...",
            file=sys.stderr,
        )
        return 1

    dataset = Dataset.from_list([
        Sample(
            query="What does RAG stand for and why does it reduce hallucination?",
            expected_answer=(
                "RAG stands for Retrieval-Augmented Generation. It reduces "
                "hallucination by conditioning the LLM's answer on retrieved "
                "documents, so each claim can be traced to a source chunk."
            ),
            tags={"domain": "ml"},
        ),
        Sample(
            query="Which vector databases are commonly used in RAG pipelines?",
            expected_answer=(
                "Common vector databases for RAG include Qdrant, Weaviate, and "
                "pgvector; they store chunk embeddings and support cosine-"
                "similarity or ANN search."
            ),
            tags={"domain": "ml"},
        ),
    ], name="quickstart_corpus", version="0")

    rag = build_rag(api_key)
    lens = RagLens(model=JUDGE_MODEL, attribution_strategy="counterfactual")

    print(f"Running RagLens against {len(dataset)} samples using NVIDIA NIM...")
    print(f"  judge model : {JUDGE_MODEL}")
    print(f"  gen model   : {GEN_MODEL}")
    print(f"  embed model : {EMBED_MODEL}")
    print()

    report = lens.probe(rag, dataset)
    print(report.summary())
    print(report.attribution(top_k=3))

    out_path = "quickstart_report.html"
    report.save_html(out_path)
    print(f"\nHTML written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
