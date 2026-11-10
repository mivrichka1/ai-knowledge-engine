import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ake_backend.config import Settings
from ake_backend.db import Base, get_session
from ake_backend.main import create_app


@pytest.mark.asyncio
async def test_dataset_validation_and_ownership(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(Settings(jwt_secret="test-only-secret-at-least-32-characters"))

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:

        async def login(email):
            await client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": "correct horse battery staple"},
            )
            result = await client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": "correct horse battery staple"},
            )
            return {"Authorization": "Bearer " + result.json()["access_token"]}

        owner = await login("eval-owner@example.com")
        other = await login("eval-other@example.com")
        base = (
            await client.post("/api/v1/knowledge-bases", json={"name": "Eval"}, headers=owner)
        ).json()["id"]
        dataset = {
            "knowledge_base_id": base,
            "name": "Manual set",
            "queries": [
                {
                    "id": "q1",
                    "question": "Unknown?",
                    "answerable": False,
                    "relevant_ids": [],
                    "category": "unanswerable",
                }
            ],
        }
        created = await client.post("/api/v1/evaluation/datasets", json=dataset, headers=owner)
        assert created.status_code == 201, created.text
        assert len((await client.get("/api/v1/evaluation/datasets", headers=owner)).json()) == 1
        assert (await client.get("/api/v1/evaluation/datasets", headers=other)).json() == []
        assert (
            await client.post(
                "/api/v1/evaluation/runs",
                json={"dataset_id": created.json()["id"], "strategies": ["keyword"], "top_k": 5},
                headers=other,
            )
        ).status_code == 404
        run = await client.post(
            "/api/v1/evaluation/runs",
            json={"dataset_id": created.json()["id"], "strategies": ["keyword"], "top_k": 5},
            headers=owner,
        )
        assert run.status_code == 202
        assert (
            await client.get("/api/v1/evaluation/runs/" + run.json()["id"], headers=other)
        ).status_code == 404
        assert (await client.get("/api/v1/evaluation/runs", headers=other)).json() == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_evaluation_worker_persists_real_result_payload(tmp_path):
    import httpx

    from ake_backend.db import Document, DocumentChunk
    from ake_backend.evaluation_worker import process_evaluation

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/eval.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(jwt_secret="test-only-secret-at-least-32-characters")
    app = create_app(settings)

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/api/v1/auth/register",
            json={"email": "eval@example.com", "password": "correct horse battery staple"},
        )
        login = await client.post(
            "/api/v1/auth/login",
            json={"email": "eval@example.com", "password": "correct horse battery staple"},
        )
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}
        base = (
            await client.post("/api/v1/knowledge-bases", json={"name": "Eval"}, headers=headers)
        ).json()["id"]
        async with maker() as session:
            doc = Document(
                knowledge_base_id=base,
                filename="a.txt",
                storage_key="key",
                format="txt",
                size=4,
                sha256="a" * 64,
                status="ready",
            )
            session.add(doc)
            await session.flush()
            chunk = DocumentChunk(
                document_id=doc.id,
                position=0,
                page=None,
                text="text",
                token_count=1,
                start_offset=0,
                end_offset=4,
            )
            session.add(chunk)
            await session.commit()
            chunk_id = chunk.id
        dataset = {
            "knowledge_base_id": base,
            "name": "Reviewed",
            "queries": [
                {
                    "id": "q1",
                    "question": "Find text",
                    "answerable": True,
                    "relevant_ids": [chunk_id],
                    "category": "exact",
                }
            ],
        }
        created = await client.post("/api/v1/evaluation/datasets", json=dataset, headers=headers)
        run = await client.post(
            "/api/v1/evaluation/runs",
            json={"dataset_id": created.json()["id"], "strategies": ["keyword"], "top_k": 5},
            headers=headers,
        )

        def search_handler(request):
            return httpx.Response(
                200, json={"strategy": "keyword", "latency_ms": 12.0, "results": [{"id": chunk_id}]}
            )

        assert await process_evaluation(
            maker, settings, transport=httpx.MockTransport(search_handler)
        )
        result = await client.get(
            "/api/v1/evaluation/runs/" + run.json()["id"] + "/results", headers=headers
        )
        assert result.status_code == 200
        assert result.json()["strategies"]["keyword"]["metrics"]["recall_at_k"] == 1.0
        assert result.json()["documents"][0]["index_metadata"] is None
        assert "index_metadata" in result.json()["configuration"]["chunking"]
        second = await client.post(
            "/api/v1/evaluation/runs",
            json={"dataset_id": created.json()["id"], "strategies": ["keyword"], "top_k": 5},
            headers=headers,
        )

        async def changed_index(request):
            async with maker() as session:
                item = await session.get(Document, doc.id)
                item.index_version += 1
                await session.commit()
            return httpx.Response(
                200, json={"strategy": "keyword", "latency_ms": 12.0, "results": [{"id": chunk_id}]}
            )

        assert await process_evaluation(
            maker, settings, transport=httpx.MockTransport(changed_index)
        )
        invalidated = await client.get(
            "/api/v1/evaluation/runs/" + second.json()["id"], headers=headers
        )
        assert invalidated.json()["status"] == "pending" and invalidated.json()["result"] is None
    await engine.dispose()


@pytest.mark.asyncio
async def test_evaluation_failure_retries_and_does_not_publish_partial_results(tmp_path):
    import httpx

    from ake_backend.db import EvaluationDataset, EvaluationRun, KnowledgeBase, User
    from ake_backend.evaluation_worker import process_evaluation

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/retry.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(jwt_secret="test-only-secret-at-least-32-characters")
    async with maker() as session:
        user = User(email="retry@example.com", password_hash="test")
        session.add(user)
        await session.flush()
        base = KnowledgeBase(owner_id=user.id, name="Test")
        session.add(base)
        await session.flush()
        dataset = EvaluationDataset(
            knowledge_base_id=base.id,
            name="Test",
            version="v1",
            queries=[{"id": "q", "question": "question", "answerable": False, "relevant_ids": []}],
        )
        session.add(dataset)
        await session.flush()
        run = EvaluationRun(
            dataset_id=dataset.id,
            user_id=user.id,
            strategies=["keyword"],
            top_k=5,
            status="pending",
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    assert await process_evaluation(maker, settings, transport=transport)
    async with maker() as session:
        run = await session.get(EvaluationRun, run_id)
        assert run.status == "pending"
        assert run.attempts == 1
        assert run.result is None
        assert run.lease_until is not None
    assert not await process_evaluation(maker, settings, transport=transport)
    await engine.dispose()


@pytest.mark.asyncio
async def test_evaluation_waits_for_document_indexing(tmp_path):
    from ake_backend.db import Document, EvaluationDataset, EvaluationRun, KnowledgeBase, User
    from ake_backend.evaluation_worker import process_evaluation

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/pending.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(jwt_secret="test-only-secret-at-least-32-characters")
    async with maker() as session:
        user = User(email="pending@example.com", password_hash="test")
        session.add(user)
        await session.flush()
        base = KnowledgeBase(owner_id=user.id, name="Test")
        session.add(base)
        await session.flush()
        session.add(
            Document(
                knowledge_base_id=base.id,
                filename="a.txt",
                storage_key="key",
                format="txt",
                size=1,
                sha256="a" * 64,
                status="processing",
            )
        )
        dataset = EvaluationDataset(
            knowledge_base_id=base.id,
            name="Test",
            version="v1",
            queries=[{"id": "q", "question": "question", "answerable": False, "relevant_ids": []}],
        )
        session.add(dataset)
        await session.flush()
        run = EvaluationRun(
            dataset_id=dataset.id,
            user_id=user.id,
            strategies=["keyword"],
            top_k=5,
            status="pending",
        )
        session.add(run)
        await session.commit()
        run_id = run.id
    assert await process_evaluation(maker, settings)
    async with maker() as session:
        run = await session.get(EvaluationRun, run_id)
        assert run.status == "failed"
        assert "index" in run.error.lower()
        assert run.result is None
    await engine.dispose()


@pytest.mark.asyncio
async def test_owner_can_cancel_evaluation_but_other_user_cannot(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/cancel.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(Settings(jwt_secret="test-only-secret-at-least-32-characters"))

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:

        async def auth(email):
            credentials = {"email": email, "password": "correct horse battery staple"}
            await client.post("/api/v1/auth/register", json=credentials)
            login = await client.post("/api/v1/auth/login", json=credentials)
            return {"Authorization": "Bearer " + login.json()["access_token"]}

        owner = await auth("owner-cancel@example.com")
        other = await auth("other-cancel@example.com")
        base = (
            await client.post("/api/v1/knowledge-bases", json={"name": "Test"}, headers=owner)
        ).json()["id"]
        data = {
            "knowledge_base_id": base,
            "name": "Test",
            "queries": [
                {
                    "id": "q",
                    "question": "Question?",
                    "answerable": False,
                    "relevant_ids": [],
                    "category": "unknown",
                }
            ],
        }
        dataset = (
            await client.post("/api/v1/evaluation/datasets", json=data, headers=owner)
        ).json()["id"]
        run = (
            await client.post(
                "/api/v1/evaluation/runs",
                json={"dataset_id": dataset, "strategies": ["keyword"], "top_k": 5},
                headers=owner,
            )
        ).json()["id"]
        assert (
            await client.post(f"/api/v1/evaluation/runs/{run}/cancel", headers=other)
        ).status_code == 404
        assert (
            await client.post(f"/api/v1/evaluation/runs/{run}/cancel", headers=owner)
        ).status_code == 200
        assert (await client.get(f"/api/v1/evaluation/runs/{run}", headers=owner)).json()[
            "status"
        ] == "cancelled"
        assert (
            await client.post(f"/api/v1/evaluation/runs/{run}/cancel", headers=owner)
        ).status_code == 200
    await engine.dispose()


@pytest.mark.asyncio
async def test_cancelled_evaluation_stops_before_next_query(tmp_path):
    import httpx

    from ake_backend.db import EvaluationDataset, EvaluationRun, KnowledgeBase, User
    from ake_backend.evaluation_worker import process_evaluation

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/stop.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(jwt_secret="test-only-secret-at-least-32-characters")
    async with maker() as session:
        user = User(email="stop@example.com", password_hash="test")
        session.add(user)
        await session.flush()
        base = KnowledgeBase(owner_id=user.id, name="Test")
        session.add(base)
        await session.flush()
        queries = [
            {"id": str(i), "question": "Question?", "answerable": False, "relevant_ids": []}
            for i in range(2)
        ]
        dataset = EvaluationDataset(
            knowledge_base_id=base.id, name="Test", version="v1", queries=queries
        )
        session.add(dataset)
        await session.flush()
        run = EvaluationRun(
            dataset_id=dataset.id,
            user_id=user.id,
            strategies=["keyword"],
            top_k=5,
            status="pending",
        )
        session.add(run)
        await session.commit()
        run_id = run.id
    calls = []

    async def handler(request):
        calls.append(request)
        async with maker() as session:
            item = await session.get(EvaluationRun, run_id)
            item.status, item.lease_token = "cancelled", None
            await session.commit()
        return httpx.Response(200, json={"strategy": "keyword", "latency_ms": 1, "results": []})

    assert await process_evaluation(maker, settings, transport=httpx.MockTransport(handler))
    assert len(calls) == 1
    async with maker() as session:
        item = await session.get(EvaluationRun, run_id)
        assert item.status == "cancelled" and item.result is None
    await engine.dispose()


@pytest.mark.asyncio
async def test_evaluation_token_limited_to_active_search(tmp_path):
    from datetime import UTC, datetime, timedelta

    import jwt

    from ake_backend.db import EvaluationDataset, EvaluationRun, KnowledgeBase, User

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/service.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(jwt_secret="test-only-secret-at-least-32-characters")
    app = create_app(settings)

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    now = datetime.now(UTC)
    async with maker() as session:
        user = User(email="service@example.com", password_hash="test")
        session.add(user)
        await session.flush()
        base = KnowledgeBase(owner_id=user.id, name="Test")
        session.add(base)
        await session.flush()
        dataset = EvaluationDataset(
            knowledge_base_id=base.id, name="Test", version="v1", queries=[]
        )
        session.add(dataset)
        await session.flush()
        run = EvaluationRun(
            dataset_id=dataset.id,
            user_id=user.id,
            strategies=["keyword"],
            top_k=5,
            status="processing",
            lease_token="lease-test",
            lease_until=now + timedelta(minutes=5),
        )
        session.add(run)
        await session.commit()
        run_id, user_id, base_id = run.id, user.id, base.id
        another = KnowledgeBase(owner_id=user.id, name="Other")
        session.add(another)
        await session.commit()
        other_base_id = another.id
    token = jwt.encode(
        {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "type": "evaluation",
            "run_id": run_id,
            "lease": "lease-test",
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    headers = {"Authorization": "Bearer " + token}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/v1/auth/me", headers=headers)).status_code == 401
        wrong_base = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": other_base_id, "query": "test"},
            headers=headers,
        )
        assert wrong_base.status_code == 403
        search = await client.post(
            "/api/v1/search", json={"knowledge_base_id": base_id, "query": "test"}, headers=headers
        )
        assert search.status_code == 503  # Auth passed; SQLite does not support PostgreSQL search.
        async with maker() as session:
            run = await session.get(EvaluationRun, run_id)
            run.status, run.lease_token = "cancelled", None
            await session.commit()
        assert (
            await client.post(
                "/api/v1/search",
                json={"knowledge_base_id": base_id, "query": "test"},
                headers=headers,
            )
        ).status_code == 401
    await engine.dispose()
