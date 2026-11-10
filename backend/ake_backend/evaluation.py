"""Reproducible offline aggregation of manually labelled retrieval results."""

import argparse
import json
from statistics import mean

from .metrics import hit_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def evaluate(rows: list[dict], k: int = 10) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    seen = set()
    per_query = []
    unanswered = 0
    for row in rows:
        identifier = row["id"]
        if not identifier or identifier in seen:
            raise ValueError("Each query must have a unique nonempty ID")
        seen.add(identifier)
        relevant = set(row["relevant_ids"])
        if not isinstance(row["answerable"], bool):
            raise ValueError("answerable must be boolean")
        if not row["answerable"]:
            if relevant:
                raise ValueError("Unanswerable query cannot have relevant IDs")
            unanswered += 1
            continue
        if not relevant:
            raise ValueError("Answerable query needs relevance labels")
        ranked = row["retrieved_ids"]
        per_query.append(
            {
                "id": identifier,
                "precision_at_k": precision_at_k(ranked, relevant, k),
                "recall_at_k": recall_at_k(ranked, relevant, k),
                "mrr_at_k": reciprocal_rank(ranked, relevant, k),
                "hit_at_k": hit_at_k(ranked, relevant, k),
                "ndcg_at_k": ndcg_at_k(ranked, {x: 1.0 for x in relevant}, k),
            }
        )
    names = ("precision_at_k", "recall_at_k", "mrr_at_k", "hit_at_k", "ndcg_at_k")
    metrics = {name: mean(x[name] for x in per_query) if per_query else None for name in names}
    return {
        "k": k,
        "answerable_count": len(per_query),
        "unanswerable_count": unanswered,
        "metrics": metrics,
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a JSON file of labelled retrieval results"
    )
    parser.add_argument("input")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as f:
        rows = json.load(f)
    print(json.dumps(evaluate(rows, args.k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
