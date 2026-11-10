import math

import pytest

from ake_backend.rerank import rank_results


class FakeScorer:
    def score(self, pairs):
        return [0.2, 0.9]


def test_reranking_sorts_and_keeps_original_results():
    results = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
    ordered = rank_results("question", results, FakeScorer())
    assert [x["id"] for x in ordered] == ["b", "a"]
    assert ordered[0]["rerank_score"] == 0.9
    assert "rerank_score" not in results[0]


def test_invalid_scores_rejected():
    class Bad:
        def score(self, pairs):
            return [math.nan]

    with pytest.raises(ValueError):
        rank_results("q", [{"id": "a", "text": "x"}], Bad())
