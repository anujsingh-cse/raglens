"""Tests for the metrics module (claim extraction, judge scoring, dataset I/O)."""

import pytest

from raglens import Dataset
from raglens.exceptions import DatasetError
from raglens.metrics import _parse_judge_score, available_metrics, extract_claims, get_metric


def test_extract_claims_splits_on_sentence_punctuation():
    text = "Paris is the capital. France is in Europe. It is large!"
    assert len(extract_claims(text)) == 3


def test_extract_claims_returns_input_if_no_punctuation():
    assert extract_claims("no punctuation here") == ["no punctuation here"]


def test_parse_judge_score_yes():
    assert _parse_judge_score("yes") == 1.0
    assert _parse_judge_score("YES") == 1.0


def test_parse_judge_score_no():
    assert _parse_judge_score("no") == 0.0


def test_parse_judge_score_0_to_5():
    assert _parse_judge_score("0") == 0.0
    assert _parse_judge_score("1") == 0.2   # 1/5, NOT 1.0 — critical edge case
    assert _parse_judge_score("2") == 0.4   # 2/5
    assert _parse_judge_score("3") == 0.6   # 3/5
    assert _parse_judge_score("3.5") == 0.7  # 3.5 / 5
    assert _parse_judge_score("4") == 0.8   # 4/5
    assert _parse_judge_score("5") == 1.0   # 5/5


def test_metrics_registry_has_builtins():
    names = available_metrics()
    for required in ("faithfulness", "context_relevance", "answer_relevance"):
        assert required in names


def test_get_metric_unknown_raises():
    with pytest.raises(KeyError):
        get_metric("definitely_not_a_metric")


def test_dataset_from_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"query": "q1", "expected_answer": "a1"}\n'
        '{"query": "q2"}\n',
        encoding="utf-8")
    ds = Dataset.from_jsonl(p)
    assert len(ds) == 2
    assert ds.samples[0].expected_answer == "a1"
    assert ds.samples[1].query == "q2"


def test_dataset_empty_raises():
    with pytest.raises(DatasetError):
        Dataset(samples=[])


def test_dataset_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError):
        Dataset.from_jsonl(tmp_path / "nope.jsonl")
