import pytest

from ake_backend.evaluation import evaluate


def test_evaluation_excludes_unanswerable_from_retrieval_means():
    rows = [
        {"id": "q1", "answerable": True, "relevant_ids": ["a"], "retrieved_ids": ["x", "a"]},
        {"id": "q2", "answerable": True, "relevant_ids": ["b"], "retrieved_ids": ["b"]},
        {"id": "q3", "answerable": False, "relevant_ids": [], "retrieved_ids": ["x"]},
    ]
    result = evaluate(rows, k=2)
    assert result["answerable_count"] == 2
    assert result["unanswerable_count"] == 1
    assert result["metrics"]["recall_at_k"] == 1
    assert result["metrics"]["mrr_at_k"] == 0.75
    assert result["metrics"]["precision_at_k"] == 0.5


def test_invalid_ground_truth_is_rejected():
    with pytest.raises(ValueError):
        evaluate([{"id": "q1", "answerable": True, "relevant_ids": [], "retrieved_ids": []}], k=2)
    with pytest.raises(ValueError):
        evaluate([], k=0)
