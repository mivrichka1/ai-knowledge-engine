"""Run the partially labelled seed set against an already uploaded document.

Set AKE_TOKEN, AKE_BASE_ID and AKE_DOCUMENT_ID. The token is never printed.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from ake_backend.evaluation import evaluate


def run_experiment(client, config, base, document, strategies):
    status_response = client.get(f"/documents/{document}")
    status_response.raise_for_status()
    initial = status_response.json()
    if initial["status"] != "ready":
        raise ValueError("The selected document must be fully indexed")
    if not isinstance(initial.get("index_version"), int):
        raise ValueError("Document index version is unavailable")
    settings_response = client.get("/settings")
    settings_response.raise_for_status()
    settings = settings_response.json()
    chunks = []
    while True:
        response = client.get(
            f"/documents/{document}/chunks", params={"offset": len(chunks), "limit": 500}
        )
        response.raise_for_status()
        page = response.json()
        chunks.extend(page)
        if len(page) < 500:
            break
    results = {}
    for strategy in strategies:
        rows = []
        latencies = []
        for query in config["queries"]:
            relevant = [
                c["id"]
                for c in chunks
                if c["page"] == query["page"] and query["anchor"].casefold() in c["text"].casefold()
            ]
            if not relevant:
                raise ValueError(
                    f"No chunk matches anchor for {query['id']}; do not evaluate unverified labels"
                )
            response = client.post(
                "/search",
                json={
                    "knowledge_base_id": base,
                    "document_ids": [document],
                    "query": query["question"],
                    "strategy": strategy,
                    "limit": 5,
                },
            )
            response.raise_for_status()
            value = response.json()
            rows.append(
                {
                    "id": query["id"],
                    "answerable": True,
                    "relevant_ids": relevant,
                    "retrieved_ids": [x["id"] for x in value["results"]],
                }
            )
            latencies.append(value["latency_ms"])
        results[strategy] = {"evaluation": evaluate(rows, 5), "latency_ms": latencies}
    final_response = client.get(f"/documents/{document}")
    final_response.raise_for_status()
    final = final_response.json()
    if final["status"] != "ready" or final.get("index_version") != initial.get("index_version"):
        raise ValueError("Document index changed during the experiment; run again")
    return {
        "dataset": config["name"],
        "declared_source_version": config["source_version"],
        "note": f"{len(config['queries'])} queries; partial anchor labels; source version is declared, not independently verified; not a reliable quality comparison.",
        "document_index_version": initial.get("index_version"),
        "document_index_metadata": initial.get("index_metadata"),
        "search_settings": {
            key: settings.get(key)
            for key in (
                "embedding_model",
                "embedding_revision",
                "reranker_model",
                "reranker_revision",
            )
        },
        "results": results,
    }


def main():
    dataset_path = Path(
        os.environ.get("AKE_DATASET", str(Path(__file__).with_name("attention_seed.json")))
    )
    config = json.loads(dataset_path.read_text())
    strategies = ["keyword", "vector", "hybrid"] + (
        ["hybrid_rerank"] if os.environ.get("AKE_INCLUDE_RERANKER") == "1" else []
    )
    with httpx.Client(
        base_url=os.environ.get("AKE_API", "http://127.0.0.1:8000/api/v1"),
        headers={"Authorization": f"Bearer {os.environ['AKE_TOKEN']}"},
        timeout=180,
    ) as client:
        report = run_experiment(
            client, config, os.environ["AKE_BASE_ID"], os.environ["AKE_DOCUMENT_ID"], strategies
        )
    report["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
