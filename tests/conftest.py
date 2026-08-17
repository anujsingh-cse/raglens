"""Shared pytest fixtures and helpers.

Pure-logic tests run offline with no LLM and no API key. Tests marked
``@pytest.mark.integration`` call NVIDIA NIM via
:class:`raglens.providers.NvidiaNimProvider`; they auto-skip unless
``NVIDIA_API_KEY`` is set in the environment, so the default CI loop never
makes a network call.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# Load .env from the repo root if present, so `pytest -m integration` finds
# NVIDIA_API_KEY without the user having to export it in every shell.
load_dotenv()

from raglens.dataset import Dataset, Sample  # noqa: E402 — must run after load_dotenv


def pytest_runtest_setup(item):
    """Auto-skip `integration` tests when NVIDIA_API_KEY is not set."""
    if "integration" in {m.name for m in item.iter_markers()} and not os.environ.get("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY not set — skipping integration test")


@pytest.fixture
def nvidia_api_key() -> str:
    """The key itself, for tests that need to construct a provider directly."""
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        pytest.skip("NVIDIA_API_KEY not set")
    return key


@pytest.fixture
def nvidia_judge_model() -> str:
    """Default judge model for integration tests.

    `meta/llama-3.1-70b-instruct` is universally available on every NVIDIA NIM
    account we've seen; the Nemotron-70B is listed by /v1/models but requires
    separate activation on the account and would 404 on many users.
    """
    return os.environ.get("RAGLENS_JUDGE_MODEL", "meta/llama-3.1-70b-instruct")


@pytest.fixture
def toy_dataset() -> Dataset:
    """A real evaluation case — not mock data; this is the eval dataset a user
    would ship. The question's expected answer is a verifiable ground truth."""
    return Dataset.from_list([
        Sample(
            query="What does RAG stand for and why is it used?",
            expected_answer=(
                "RAG stands for Retrieval-Augmented Generation. It is used to "
                "ground an LLM's answer in retrieved documents so the model "
                "can cite up-to-date or private facts it would otherwise not "
                "know, and so hallucinations can be traced to the context."
            ),
            tags={"domain": "ml"},
        ),
    ], name="quickstart_corpus", version="0")
