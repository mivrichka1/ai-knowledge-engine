"""Opt-in integration test; DATABASE_URL must point at a disposable database."""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_pgvector_extension_and_cosine_distance():
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            assert result.scalar_one()
            value = await connection.scalar(text("SELECT '[1,0]'::vector <=> '[0,1]'::vector"))
            assert value == pytest.approx(1.0)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_api_registration_on_migrated_postgres():
    from uuid import uuid4

    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import delete, select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from ake_backend.config import Settings
    from ake_backend.db import User, get_session
    from ake_backend.main import create_app

    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    engine = create_async_engine(url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(Settings(jwt_secret="postgres-test-secret-at-least-32-characters"))

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    email = f"{uuid4().hex}@example.com"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            result = await client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": "correct horse battery staple"},
            )
            assert result.status_code == 201, result.text
            async with maker() as session:
                user = await session.scalar(select(User).where(User.email == email))
                assert user is not None
                assert user.password_hash != "correct horse battery staple"
    finally:
        async with maker() as session:
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
        await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_upload_and_worker(tmp_path):
    from uuid import uuid4

    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from ake_backend.config import Settings
    from ake_backend.db import User, get_session
    from ake_backend.main import create_app
    from ake_backend.worker import process_one

    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    engine = create_async_engine(url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(
        Settings(
            jwt_secret="postgres-test-secret-at-least-32-characters",
            document_storage_path=str(tmp_path),
        )
    )

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    email = f"{uuid4().hex}@example.com"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": "correct horse battery staple"},
            )
            login = await client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": "correct horse battery staple"},
            )
            headers = {"Authorization": "Bearer " + login.json()["access_token"]}
            base = (
                await client.post(
                    "/api/v1/knowledge-bases", json={"name": "Example"}, headers=headers
                )
            ).json()
            upload = await client.post(
                f"/api/v1/knowledge-bases/{base['id']}/documents",
                files={"file": ("file.txt", b"Actual postgres pipeline", "text/plain")},
                headers=headers,
            )
            assert upload.status_code == 202, upload.text

            class FakeProvider:
                model_name = "test-model"
                revision = "test-revision"
                dimension = 384

                def encode(self, texts):
                    return [[1.0] + [0.0] * 383 for _ in texts]

            if os.environ.get("TEST_REAL_MODEL"):
                from ake_backend.embeddings import SentenceTransformerProvider

                app.state.embedding_provider = SentenceTransformerProvider(
                    app.state.settings.embedding_model, 384, app.state.settings.embedding_revision
                )
            else:
                app.state.embedding_provider = FakeProvider()
            import asyncio

            claims = await asyncio.gather(
                process_one(maker, str(tmp_path), app.state.embedding_provider),
                process_one(maker, str(tmp_path), app.state.embedding_provider),
            )
            assert sorted(claims) == [False, True]
            chunks = await client.get(
                "/api/v1/documents/" + upload.json()["id"] + "/chunks", headers=headers
            )
            assert chunks.json()[0]["text"] == "Actual postgres pipeline"
            found = await client.post(
                "/api/v1/search",
                json={"knowledge_base_id": base["id"], "query": "postgres"},
                headers=headers,
            )
            assert found.status_code == 200
            filtered = await client.post(
                "/api/v1/search",
                json={
                    "knowledge_base_id": base["id"],
                    "query": "postgres",
                    "strategy": "keyword",
                    "document_ids": ["00000000-0000-0000-0000-000000000000"],
                },
                headers=headers,
            )
            assert filtered.status_code == 200 and filtered.json()["results"] == []
            for strategy in ["vector", "hybrid"]:
                filtered = await client.post(
                    "/api/v1/search",
                    json={
                        "knowledge_base_id": base["id"],
                        "query": "postgres",
                        "strategy": strategy,
                        "document_ids": ["00000000-0000-0000-0000-000000000000"],
                    },
                    headers=headers,
                )
                assert filtered.status_code == 200 and filtered.json()["results"] == []
            fallback = await client.post(
                "/api/v1/search",
                json={
                    "knowledge_base_id": base["id"],
                    "query": "postgres missingword",
                    "strategy": "keyword",
                },
                headers=headers,
            )
            assert fallback.status_code == 200 and fallback.json()["results"]
            assert fallback.json()["query_mode"] == "any_terms"
            assert found.json()["latency_ms"] >= 0
            assert found.json()["results"][0]["text"] == "Actual postgres pipeline"
            semantic = await client.post(
                "/api/v1/search",
                json={"knowledge_base_id": base["id"], "query": "meaning", "strategy": "vector"},
                headers=headers,
            )
            assert semantic.status_code == 200, semantic.text
            assert semantic.json()["results"][0]["text"] == "Actual postgres pipeline"
            hybrid = await client.post(
                "/api/v1/search",
                json={"knowledge_base_id": base["id"], "query": "postgres", "strategy": "hybrid"},
                headers=headers,
            )
            assert hybrid.status_code == 200, hybrid.text
            assert hybrid.json()["results"][0]["text"] == "Actual postgres pipeline"

            class FakeReranker:
                def score(self, pairs):
                    return [0.7] * len(pairs)

            if os.environ.get("TEST_REAL_MODEL"):
                from ake_backend.rerank import CrossEncoderScorer

                app.state.reranker = CrossEncoderScorer(
                    app.state.settings.reranker_model, app.state.settings.reranker_revision
                )
            else:
                app.state.reranker = FakeReranker()
            reranked = await client.post(
                "/api/v1/search",
                json={
                    "knowledge_base_id": base["id"],
                    "query": "postgres",
                    "strategy": "hybrid_rerank",
                },
                headers=headers,
            )
            assert reranked.status_code == 200, reranked.text
            assert isinstance(reranked.json()["results"][0]["rerank_score"], float)
            compared = await client.post(
                "/api/v1/search/compare",
                json={
                    "knowledge_base_id": base["id"],
                    "query": "postgres",
                    "strategies": ["keyword", "vector"],
                },
                headers=headers,
            )
            assert compared.status_code == 200, compared.text
            assert [x["strategy"] for x in compared.json()["comparisons"]] == ["keyword", "vector"]

            class FakeLLM:
                async def complete(self, question, context):
                    import re

                    source = re.search(r"\[([a-f0-9-]+)\]", context).group(1)
                    return "The document describes a pipeline [" + source + "]."

                async def extract(self, context):
                    import json
                    import re

                    source = re.search(r"\[([a-f0-9-]+)\]", context).group(1)
                    return json.dumps({"title": {"value": "Example", "source_ids": [source]}})

            app.state.llm_provider = FakeLLM()
            answer = await client.post(
                "/api/v1/ask",
                json={
                    "knowledge_base_id": base["id"],
                    "question": "postgres",
                    "allow_provider_transfer": True,
                },
                headers=headers,
            )
            assert answer.status_code == 200, answer.text
            assert answer.json()["sources"][0]["id"] in answer.json()["answer"]
            assert answer.json()["sufficient_context"] is True
            assert answer.json()["citations"] == [answer.json()["sources"][0]["id"]]
            assert answer.json()["processing_time_ms"] >= 0

            class AbstainingLLM:
                async def complete(self, question, context):
                    return "INSUFFICIENT_EVIDENCE"

            app.state.llm_provider = AbstainingLLM()
            abstention = await client.post(
                "/api/v1/ask",
                json={
                    "knowledge_base_id": base["id"],
                    "question": "unknown",
                    "allow_provider_transfer": True,
                },
                headers=headers,
            )
            assert (
                abstention.status_code == 200 and abstention.json()["sufficient_context"] is False
            )
            from ake_backend.llm import LLMError

            class FailingLLM:
                async def complete(self, question, context):
                    raise LLMError("provider failed")

            app.state.llm_provider = FailingLLM()
            failure = await client.post(
                "/api/v1/ask",
                json={
                    "knowledge_base_id": base["id"],
                    "question": "unknown",
                    "allow_provider_transfer": True,
                },
                headers=headers,
            )
            assert failure.status_code == 502
            app.state.llm_provider = FakeLLM()
            conv = await client.post(
                "/api/v1/chat/conversations",
                json={"knowledge_base_id": base["id"], "title": "Research"},
                headers=headers,
            )
            assert conv.status_code == 201, conv.text
            message = await client.post(
                "/api/v1/chat/conversations/" + conv.json()["id"] + "/messages",
                json={"question": "postgres", "allow_provider_transfer": True},
                headers=headers,
            )
            assert message.status_code == 200, message.text

            class HistoryLLM(FakeLLM):
                async def complete(self, question, context):
                    assert "postgres" in question and "What about it?" in question
                    return await super().complete(question, context)

            app.state.llm_provider = HistoryLLM()
            second = await client.post(
                "/api/v1/chat/conversations/" + conv.json()["id"] + "/messages",
                json={"question": "What about it?", "allow_provider_transfer": True},
                headers=headers,
            )
            assert second.status_code == 200, second.text
            retrieval_stats = await client.get("/api/v1/analytics/retrieval", headers=headers)
            assert retrieval_stats.status_code == 200
            assert any(
                item["strategy"] == "keyword" and item["count"] >= 1
                for item in retrieval_stats.json()["strategies"]
            )
            latency_stats = await client.get("/api/v1/analytics/latency", headers=headers)
            assert latency_stats.status_code == 200
            overview = await client.get("/api/v1/analytics/overview", headers=headers)
            assert overview.status_code == 200
            assert overview.json()["documents"] == 1 and overview.json()["questions"] == 2
            history = await client.get(
                "/api/v1/chat/conversations/" + conv.json()["id"], headers=headers
            )
            assert len(history.json()["messages"]) == 4
            extraction = await client.post(
                "/api/v1/documents/" + upload.json()["id"] + "/extract",
                json={"allow_provider_transfer": True},
                headers=headers,
            )
            assert extraction.status_code == 200, extraction.text
            assert extraction.json()["data"]["title"]["value"] == "Example"
            original_chunk = chunks.json()[0]["id"]
            reindex = await client.post(
                "/api/v1/documents/" + upload.json()["id"] + "/reindex", headers=headers
            )
            assert reindex.status_code == 202
            pending_search = await client.post(
                "/api/v1/search",
                json={"knowledge_base_id": base["id"], "query": "postgres", "strategy": "keyword"},
                headers=headers,
            )
            assert pending_search.status_code == 200 and pending_search.json()["results"] == []
            assert await process_one(maker, str(tmp_path), app.state.embedding_provider)
            again = await client.get(
                "/api/v1/documents/" + upload.json()["id"] + "/chunks", headers=headers
            )
            assert again.json()[0]["id"] == original_chunk
            deleted = await client.delete(
                "/api/v1/documents/" + upload.json()["id"], headers=headers
            )
            assert deleted.status_code == 204, deleted.text
            assert (
                await client.get("/api/v1/documents/" + upload.json()["id"], headers=headers)
            ).status_code == 404
            after_delete = await client.get(
                "/api/v1/chat/conversations/" + conv.json()["id"], headers=headers
            )
            assert [m["role"] for m in after_delete.json()["messages"]] == ["user", "user"]
    finally:
        async with maker() as session:
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_during_indexing_does_not_resurrect_document(tmp_path):
    import asyncio
    import threading
    from uuid import uuid4

    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from ake_backend.config import Settings
    from ake_backend.db import User, get_session
    from ake_backend.main import create_app
    from ake_backend.worker import process_one

    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    engine = create_async_engine(url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(
        Settings(
            jwt_secret="postgres-test-secret-at-least-32-characters",
            document_storage_path=str(tmp_path),
        )
    )

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    started, release = threading.Event(), threading.Event()

    class BlockingProvider:
        model_name = "test"
        revision = "test"
        dimension = 384

        def encode(self, texts):
            started.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test timed out")
            return [[1.0] + [0.0] * 383 for _ in texts]

    email = f"{uuid4().hex}@example.com"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": "correct horse battery staple"},
            )
            login = await client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": "correct horse battery staple"},
            )
            headers = {"Authorization": "Bearer " + login.json()["access_token"]}
            base = (
                await client.post("/api/v1/knowledge-bases", json={"name": "Race"}, headers=headers)
            ).json()["id"]
            upload = await client.post(
                f"/api/v1/knowledge-bases/{base}/documents",
                files={"file": ("race.txt", b"concurrency", "text/plain")},
                headers=headers,
            )
            doc = upload.json()["id"]
            task = asyncio.create_task(process_one(maker, str(tmp_path), BlockingProvider()))
            assert await asyncio.to_thread(started.wait, 5)
            assert (
                await client.delete("/api/v1/documents/" + doc, headers=headers)
            ).status_code == 204
            release.set()
            assert await task
            assert (
                await client.get("/api/v1/documents/" + doc, headers=headers)
            ).status_code == 404
    finally:
        release.set()
        async with maker() as session:
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_upload_idempotency_on_postgres(tmp_path):
    import asyncio
    from uuid import uuid4

    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from ake_backend.config import Settings
    from ake_backend.db import User, get_session
    from ake_backend.main import create_app

    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    engine = create_async_engine(url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(
        Settings(
            jwt_secret="postgres-test-secret-at-least-32-characters",
            document_storage_path=str(tmp_path),
        )
    )

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    email = f"{uuid4().hex}@example.com"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            credentials = {"email": email, "password": "correct horse battery staple"}
            await client.post("/api/v1/auth/register", json=credentials)
            login = await client.post("/api/v1/auth/login", json=credentials)
            headers = {
                "Authorization": "Bearer " + login.json()["access_token"],
                "Idempotency-Key": "same-operation",
            }
            base = (
                await client.post(
                    "/api/v1/knowledge-bases", json={"name": "Concurrency"}, headers=headers
                )
            ).json()["id"]

            async def upload():
                return await client.post(
                    f"/api/v1/knowledge-bases/{base}/documents",
                    files={"file": ("same.txt", b"same content", "text/plain")},
                    headers=headers,
                )

            first, second = await asyncio.gather(upload(), upload())
            assert first.status_code == 202 and second.status_code == 202
            assert first.json()["id"] == second.json()["id"]
            assert (
                len(
                    (
                        await client.get(
                            f"/api/v1/knowledge-bases/{base}/documents", headers=headers
                        )
                    ).json()
                )
                == 1
            )
            assert len(list(tmp_path.iterdir())) == 1
    finally:
        async with maker() as session:
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
        await engine.dispose()
