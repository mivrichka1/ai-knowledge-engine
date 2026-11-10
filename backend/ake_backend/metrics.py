"""Per-query retrieval metrics. Duplicate IDs occupy ranks but score only once."""

import math
from collections.abc import Mapping, Sequence


def _validate(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def _hits(ranked: Sequence[str], relevant: set[str], k: int):
    _validate(k)
    seen = set()  # type: set[str]
    for item in ranked[:k]:
        yield item in relevant and item not in seen
        seen.add(item)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Precision with fixed K denominator, including when fewer than K are returned."""
    return sum(_hits(ranked, relevant, k)) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    hits = sum(_hits(ranked, relevant, k))
    return hits / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, hit in enumerate(_hits(ranked, relevant, k), 1):
        if hit:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(_hits(ranked, relevant, k)))


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    _validate(k)
    if any(not math.isfinite(value) or value < 0 for value in relevance.values()):
        raise ValueError("relevance grades must be finite and non-negative")
    seen = set()  # type: set[str]
    dcg = 0.0
    for rank, item in enumerate(ranked[:k], 1):
        if item not in seen:
            dcg += (2 ** relevance.get(item, 0.0) - 1) / math.log2(rank + 1)
            seen.add(item)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0
