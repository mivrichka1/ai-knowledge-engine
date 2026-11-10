"""Rank fusion independent of database and search implementation."""

from collections.abc import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]], k: int = 60, top_k: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked chunk IDs; count an ID once per list, using its first rank."""
    if k <= 0 or (top_k is not None and top_k <= 0):
        raise ValueError("k and top_k must be positive")
    scores = {}  # type: dict[str, float]
    for ranking in rankings:
        seen = set()  # type: set[str]
        for rank, item in enumerate(ranking, 1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    result = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return result if top_k is None else result[:top_k]
