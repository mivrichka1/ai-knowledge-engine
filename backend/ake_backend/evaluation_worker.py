"""Run durable evaluation jobs through the same authenticated retrieval API."""

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import jwt
from sqlalchemy import or_, select

from .db import Document, DocumentChunk, EvaluationDataset, EvaluationRun
from .evaluation import evaluate


async def process_evaluation(maker, settings, transport=None) -> bool:
    now = datetime.now(UTC)
    lease = str(uuid4())
    async with maker() as session:
        async with session.begin():
            run = await session.scalar(
                select(EvaluationRun)
                .where(
                    or_(
                        (EvaluationRun.status == "pending")
                        & or_(
                            EvaluationRun.lease_until.is_(None), EvaluationRun.lease_until <= now
                        ),
                        (EvaluationRun.status == "processing") & (EvaluationRun.lease_until < now),
                    )
                )
                .order_by(EvaluationRun.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if run is None:
                return False
            if run.attempts >= 3:
                run.status, run.error = "failed", "Evaluation attempts exhausted"
                return True
            dataset = await session.get(EvaluationDataset, run.dataset_id)
            if dataset is None:
                run.status, run.error = "failed", "Dataset not found"
                return True
            relevant = {id_ for query in dataset.queries for id_ in query["relevant_ids"]}
            matching = (
                set(
                    (
                        await session.scalars(
                            select(DocumentChunk.id)
                            .join(Document, DocumentChunk.document_id == Document.id)
                            .where(
                                Document.knowledge_base_id == dataset.knowledge_base_id,
                                DocumentChunk.id.in_(relevant),
                            )
                        )
                    ).all()
                )
                if relevant
                else set()
            )
            if matching != relevant:
                run.status, run.error = (
                    "failed",
                    "Dataset labels refer to missing or changed chunks",
                )
                return True
            run.status, run.attempts, run.lease_token = "processing", run.attempts + 1, lease
            run.lease_until = now + timedelta(hours=2)
            run_id, user_id, base_id = run.id, run.user_id, dataset.knowledge_base_id
            strategies, top_k, queries = list(run.strategies), run.top_k, list(dataset.queries)
            dataset_version = dataset.version
            documents = [
                {
                    "id": doc.id,
                    "sha256": doc.sha256,
                    "status": doc.status,
                    "index_version": doc.index_version,
                    "index_metadata": doc.index_metadata,
                }
                for doc in (
                    await session.scalars(
                        select(Document)
                        .where(Document.knowledge_base_id == dataset.knowledge_base_id)
                        .order_by(Document.id)
                    )
                ).all()
            ]
            if any(document["status"] not in {"ready", "failed"} for document in documents):
                run.status = "failed"
                run.error = (
                    "Document indexing is incomplete; retry the experiment when indexing finishes"
                )
                run.lease_token, run.lease_until = None, None
                return True
    try:
        access = jwt.encode(
            {
                "sub": user_id,
                "iat": now,
                "exp": now + timedelta(hours=2),
                "type": "evaluation",
                "run_id": run_id,
                "lease": lease,
            },
            settings.jwt_secret,
            algorithm="HS256",
        )
        outputs = {}
        async with httpx.AsyncClient(
            base_url=settings.internal_api_url,
            headers={"Authorization": f"Bearer {access}"},
            timeout=180,
            transport=transport,
        ) as client:
            for strategy in strategies:
                rows, latencies = [], []
                for query in queries:
                    async with maker() as session:
                        current = await session.get(EvaluationRun, run_id)
                        if (
                            current is None
                            or current.status != "processing"
                            or current.lease_token != lease
                        ):
                            return True
                    response = await client.post(
                        "/search",
                        json={
                            "knowledge_base_id": base_id,
                            "query": query["question"],
                            "strategy": strategy,
                            "limit": top_k,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    rows.append(
                        {
                            "id": query["id"],
                            "answerable": query["answerable"],
                            "relevant_ids": query["relevant_ids"],
                            "retrieved_ids": [hit["id"] for hit in result["results"]],
                        }
                    )
                    latencies.append(float(result["latency_ms"]))
                outputs[strategy] = {
                    **evaluate(rows, top_k),
                    "latencies_ms": latencies,
                    "query_results": rows,
                }
        payload = {
            "dataset_version": dataset_version,
            "documents": documents,
            "strategies": outputs,
            "configuration": {
                "top_k": top_k,
                "embedding_model": settings.embedding_model,
                "embedding_revision": settings.embedding_revision,
                "reranker_model": settings.reranker_model,
                "reranker_revision": settings.reranker_revision,
                "chunking": "See each document's index_metadata; null means unknown",
                "rrf_k": 60,
                "retrieval_version": "websearch-any-fallback-rrf-v1",
                "index_version": "0007-hnsw-cosine",
            },
        }
        payload["config_sha256"] = hashlib.sha256(
            json.dumps(payload["configuration"], sort_keys=True).encode()
        ).hexdigest()
        async with maker() as session:
            async with session.begin():
                run = await session.scalar(
                    select(EvaluationRun).where(EvaluationRun.id == run_id).with_for_update()
                )
                if run and run.lease_token == lease:
                    current = [
                        {
                            "id": doc.id,
                            "sha256": doc.sha256,
                            "status": doc.status,
                            "index_version": doc.index_version,
                            "index_metadata": doc.index_metadata,
                        }
                        for doc in (
                            await session.scalars(
                                select(Document)
                                .where(Document.knowledge_base_id == base_id)
                                .order_by(Document.id)
                            )
                        ).all()
                    ]
                    if current != documents:
                        raise ValueError("Document index changed during evaluation")
                    run.result, run.status, run.error = payload, "complete", None
                    run.lease_token, run.lease_until = None, None
    except Exception as exc:
        logging.getLogger("ake.evaluation").error(
            json.dumps(
                {"event": "evaluation_failed", "run_id": run_id, "error_type": type(exc).__name__}
            )
        )
        async with maker() as session:
            async with session.begin():
                run = await session.scalar(
                    select(EvaluationRun).where(EvaluationRun.id == run_id).with_for_update()
                )
                if run and run.lease_token == lease:
                    run.status = "failed" if run.attempts >= 3 else "pending"
                    run.error = "Evaluation failed; see worker logs"
                    run.lease_token = None
                    run.lease_until = (
                        None
                        if run.status == "failed"
                        else datetime.now(UTC) + timedelta(seconds=2**run.attempts)
                    )
    return True
