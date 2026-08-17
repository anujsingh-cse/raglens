"""RagLens — diagnostic evaluation for Retrieval-Augmented Generation pipelines.

RagLens goes beyond faithfulness scoring. It *attributes* every claim in the
generated answer back to the specific retrieved chunks that helped or hurt it,
using counterfactual chunk removal. The output is a diagnosis, not just a score.

Quickstart (default judge is NVIDIA NIM)::

    import os
    os.environ["NVIDIA_API_KEY"] = "nvapi-..."

    from raglens import RagLens, Dataset

    lens = RagLens(model="meta/llama-3.1-70b-instruct")
    dataset = Dataset.from_jsonl("evals.jsonl")

    def my_rag(query: str) -> tuple[str, list[str]]:
        chunks = retriever.invoke(query)
        answer = chain.invoke({"query": query, "context": chunks})
        return answer, chunks

    report = lens.probe(pipeline=my_rag, dataset=dataset)
    print(report.summary())
    print(report.attribution())
    report.save_html("report.html")
"""

from raglens.core import RagLens
from raglens.dataset import Dataset, Sample
from raglens.exceptions import DatasetError, ProviderError, RagLensError
from raglens.report import Report
from raglens.trace import GenerationStep, RetrievalStep, Trace

__version__ = "0.1.0a0"

__all__ = [
    "RagLens",
    "Dataset",
    "Sample",
    "Trace",
    "RetrievalStep",
    "GenerationStep",
    "Report",
    "RagLensError",
    "ProviderError",
    "DatasetError",
    "__version__",
]
