"""Cross-encoder reranking of retrieved query-document pairs."""

import math


class CrossEncoderScorer:
    def __init__(self, model_name: str, revision: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, revision=revision, device="cpu")

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(x) for x in self.model.predict(pairs, batch_size=16, show_progress_bar=False)]


def rank_results(query: str, results: list[dict], scorer) -> list[dict]:
    if not results:
        return []
    scores = scorer.score([(query, result["text"]) for result in results])
    if len(scores) != len(results) or not all(math.isfinite(score) for score in scores):
        raise ValueError("Reranker returned invalid scores")
    ranked = [{**item, "rerank_score": float(score)} for item, score in zip(results, scores)]
    return sorted(ranked, key=lambda item: (-item["rerank_score"], item["id"]))
