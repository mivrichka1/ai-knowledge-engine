import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ake_backend.config import Settings
from ake_backend.db import Base, get_session
from ake_backend.main import create_app


@pytest_asyncio.fixture
async def client(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(
        Settings(
            jwt_secret="test-only-secret-with-at-least-32-characters",
            document_storage_path=str(tmp_path / "storage"),
        )
    )

    async def override():
        async with maker() as session:
            yield session

    app.state.test_maker = maker
    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await engine.dispose()


async def register(client, email):
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "correct horse battery staple"},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.asyncio
async def test_auth_rotation_and_logout(client):
    await register(client, "one@example.com")
    bad = await client.post(
        "/api/v1/auth/login",
        json={"email": "one@example.com", "password": "bad password"},
    )
    assert bad.status_code == 401
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "one@example.com", "password": "correct horse battery staple"},
    )
    assert login.status_code == 200
    old = login.json()["refresh_token"]
    refresh = await client.post("/api/v1/auth/refresh", json={"refresh_token": old})
    assert refresh.status_code == 200
    assert (
        await client.post("/api/v1/auth/refresh", json={"refresh_token": old})
    ).status_code == 401
    old_access = login.json()["access_token"]
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {old_access}"})
    ).status_code == 401
    token = refresh.json()["access_token"]
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    ).status_code == 200
    assert (
        await client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": refresh.json()["refresh_token"]},
        )
    ).status_code == 204
    assert (
        await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh.json()["refresh_token"]},
        )
    ).status_code == 401
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    ).status_code == 401


@pytest.mark.asyncio
async def test_owner_isolation_and_crud(client):
    await register(client, "one@example.com")
    await register(client, "two@example.com")

    async def headers(email):
        r = await client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "correct horse battery staple"},
        )
        return {"Authorization": "Bearer " + r.json()["access_token"]}

    a, b = await headers("one@example.com"), await headers("two@example.com")
    created = await client.post("/api/v1/knowledge-bases", json={"name": "Research"}, headers=a)
    assert created.status_code == 201
    base_id = created.json()["id"]
    assert (await client.get(f"/api/v1/knowledge-bases/{base_id}", headers=b)).status_code == 404
    assert (
        await client.patch(f"/api/v1/knowledge-bases/{base_id}", json={"name": "Stolen"}, headers=b)
    ).status_code == 404
    assert (await client.delete(f"/api/v1/knowledge-bases/{base_id}", headers=b)).status_code == 404
    assert (await client.get("/api/v1/knowledge-bases", headers=b)).json() == []
    assert (
        await client.patch(f"/api/v1/knowledge-bases/{base_id}", json={"name": "New"}, headers=a)
    ).json()["name"] == "New"
    assert (await client.delete(f"/api/v1/knowledge-bases/{base_id}", headers=a)).status_code == 204
    assert (await client.get("/api/v1/knowledge-bases")).status_code == 401


@pytest.mark.asyncio
async def test_document_upload_and_cross_user_access(client):
    await register(client, "owner@example.com")
    await register(client, "other@example.com")

    async def token(email):
        r = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": "correct horse battery staple"}
        )
        return {"Authorization": "Bearer " + r.json()["access_token"]}

    owner, other = await token("owner@example.com"), await token("other@example.com")
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Docs"}, headers=owner)
    ).json()
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base['id']}/documents",
        files={"file": ("test.txt", b"Hello document", "text/plain")},
        headers=owner,
    )
    assert upload.status_code == 202, upload.text
    doc = upload.json()
    assert doc["status"] == "pending"
    listed = await client.get(f"/api/v1/knowledge-bases/{base['id']}/documents", headers=owner)
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == doc["id"]
    assert (
        await client.get(f"/api/v1/knowledge-bases/{base['id']}/documents", headers=other)
    ).status_code == 404
    assert (await client.get("/api/v1/documents/" + doc["id"], headers=other)).status_code == 404
    assert (
        await client.get("/api/v1/documents/" + doc["id"] + "/file", headers=other)
    ).status_code == 404
    downloaded = await client.get("/api/v1/documents/" + doc["id"] + "/file", headers=owner)
    assert downloaded.status_code == 200 and downloaded.content == b"Hello document"
    assert (
        await client.get("/api/v1/documents/" + doc["id"] + "/chunks", headers=other)
    ).status_code == 404
    assert (await client.get("/api/v1/documents/" + doc["id"], headers=owner)).status_code == 200
    assert (
        await client.post(
            f"/api/v1/knowledge-bases/{base['id']}/documents",
            files={"file": ("../evil.txt", b"x", "text/plain")},
            headers=owner,
        )
    ).status_code == 422


@pytest.mark.asyncio
async def test_worker_parses_queued_document(client):
    from ake_backend.worker import process_one

    await register(client, "worker@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "worker@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Worker"}, headers=headers)
    ).json()
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base['id']}/documents",
        files={"file": ("paper.txt", b"First paragraph of the document.", "text/plain")},
        headers=headers,
    )
    result = upload.json()
    assert await process_one(
        client._transport.app.state.test_maker,
        client._transport.app.state.settings.document_storage_path,
    )
    doc = await client.get("/api/v1/documents/" + result["id"], headers=headers)
    assert doc.json()["status"] == "parsed"
    assert doc.json()["index_metadata"]["chunk_strategy"] == "fixed_whitespace"
    chunks = await client.get("/api/v1/documents/" + result["id"] + "/chunks", headers=headers)
    assert chunks.status_code == 200
    assert chunks.json()[0]["text"] == "First paragraph of the document."
    chunk_id = chunks.json()[0]["id"]
    await register(client, "intruder@example.com")
    intruder_login = await client.post(
        "/api/v1/auth/login",
        json={"email": "intruder@example.com", "password": "correct horse battery staple"},
    )
    intruder_headers = {"Authorization": "Bearer " + intruder_login.json()["access_token"]}
    assert (
        await client.get("/api/v1/chunks/" + chunk_id, headers=intruder_headers)
    ).status_code == 404
    assert (await client.get("/api/v1/chunks/" + chunk_id, headers=headers)).status_code == 200


@pytest.mark.asyncio
async def test_worker_marks_invalid_pdf_failed(client):
    from ake_backend.worker import process_one

    await register(client, "broken@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "broken@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Broken"}, headers=headers)
    ).json()
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base['id']}/documents",
        files={"file": ("bad.pdf", b"%PDF-invalid", "application/pdf")},
        headers=headers,
    )
    assert upload.status_code == 202
    assert await process_one(
        client._transport.app.state.test_maker,
        client._transport.app.state.settings.document_storage_path,
    )
    doc = await client.get("/api/v1/documents/" + upload.json()["id"], headers=headers)
    assert doc.json()["status"] == "failed"
    assert doc.json()["error"]


@pytest.mark.asyncio
async def test_worker_exhausted_lease_fails_instead_of_sticking(client):
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select

    from ake_backend.db import ProcessingJob
    from ake_backend.worker import process_one

    await register(client, "crash@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "crash@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Crash"}, headers=headers)
    ).json()
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base['id']}/documents",
        files={"file": ("file.txt", b"text", "text/plain")},
        headers=headers,
    )
    maker = client._transport.app.state.test_maker
    async with maker() as session:
        job = await session.scalar(
            select(ProcessingJob).where(ProcessingJob.id == upload.json()["job_id"])
        )
        job.status, job.attempts = "processing", 3
        job.lease_until = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()
    assert await process_one(maker, client._transport.app.state.settings.document_storage_path)
    doc = await client.get("/api/v1/documents/" + upload.json()["id"], headers=headers)
    assert doc.json()["status"] == "failed"


@pytest.mark.asyncio
async def test_auth_endpoints_rate_limit(client):
    for _ in range(20):
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "missing@example.com", "password": "valid-password"},
        )
        assert response.status_code == 401
    assert (
        await client.post(
            "/api/v1/auth/login",
            json={"email": "missing@example.com", "password": "valid-password"},
        )
    ).status_code == 429


@pytest.mark.asyncio
async def test_rename_does_not_erase_description(client):
    await register(client, "rename@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "rename@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    created = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": "Before", "description": "Keep this"},
        headers=headers,
    )
    result = await client.patch(
        "/api/v1/knowledge-bases/" + created.json()["id"], json={"name": "After"}, headers=headers
    )
    assert result.json()["description"] == "Keep this"


@pytest.mark.asyncio
async def test_reindex_replaces_chunks_without_duplicates(client):
    from ake_backend.worker import process_one

    await register(client, "reindex@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "reindex@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "R"}, headers=headers)
    ).json()["id"]
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base}/documents",
        files={"file": ("text.txt", b"one two three", "text/plain")},
        headers=headers,
    )
    doc = upload.json()["id"]
    maker = client._transport.app.state.test_maker
    path = client._transport.app.state.settings.document_storage_path
    assert await process_one(maker, path)
    original = (await client.get(f"/api/v1/documents/{doc}/chunks", headers=headers)).json()
    assert (
        await client.post(f"/api/v1/documents/{doc}/reindex", headers=headers)
    ).status_code == 202
    assert (
        await client.post(f"/api/v1/documents/{doc}/reindex", headers=headers)
    ).status_code == 409
    assert await process_one(maker, path)
    chunks = (await client.get(f"/api/v1/documents/{doc}/chunks", headers=headers)).json()
    assert len(chunks) == 1 and chunks[0]["text"] == "one two three"
    assert chunks[0]["id"] == original[0]["id"]


@pytest.mark.asyncio
async def test_document_status_visible_to_owner_only(client):
    await register(client, "status@example.com")
    await register(client, "status-other@example.com")

    async def auth(email):
        response = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": "correct horse battery staple"}
        )
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    owner, other = await auth("status@example.com"), await auth("status-other@example.com")
    base = (await client.post("/api/v1/knowledge-bases", json={"name": "S"}, headers=owner)).json()[
        "id"
    ]
    result = await client.post(
        f"/api/v1/knowledge-bases/{base}/documents",
        files={"file": ("test.txt", b"hello", "text/plain")},
        headers=owner,
    )
    doc = result.json()["id"]
    status = await client.get(f"/api/v1/documents/{doc}/status", headers=owner)
    assert status.status_code == 200 and status.json()["job_id"] == result.json()["job_id"]
    assert (await client.get(f"/api/v1/documents/{doc}/status", headers=other)).status_code == 404


@pytest.mark.asyncio
async def test_expired_or_tampered_access_token_is_rejected(client):
    from datetime import UTC, datetime, timedelta

    import jwt

    await register(client, "expired@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "expired@example.com", "password": "correct horse battery staple"},
    )
    token = login.json()["access_token"]
    claims = jwt.decode(token, options={"verify_signature": False})
    claims["exp"] = datetime.now(UTC) - timedelta(seconds=1)
    expired = jwt.encode(claims, client._transport.app.state.settings.jwt_secret, algorithm="HS256")
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer " + expired})
    ).status_code == 401
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer " + token + "x"})
    ).status_code == 401


@pytest.mark.asyncio
async def test_worker_rejects_modified_file(client):
    from pathlib import Path

    from sqlalchemy import select

    from ake_backend.db import Document
    from ake_backend.worker import process_one

    await register(client, "tamper@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "tamper@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Integrity"}, headers=headers)
    ).json()["id"]
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base}/documents",
        files={"file": ("file.txt", b"original", "text/plain")},
        headers=headers,
    )
    maker = client._transport.app.state.test_maker
    async with maker() as session:
        doc = await session.scalar(select(Document).where(Document.id == upload.json()["id"]))
        Path(
            client._transport.app.state.settings.document_storage_path, doc.storage_key
        ).write_bytes(b"modified")
    await process_one(maker, client._transport.app.state.settings.document_storage_path)
    result = await client.get("/api/v1/documents/" + upload.json()["id"], headers=headers)
    assert result.json()["status"] == "failed"


@pytest.mark.asyncio
async def test_search_analytics_is_owner_scoped(client):
    from sqlalchemy import select

    from ake_backend.db import SearchEvent, User

    await register(client, "analytics-a@example.com")
    await register(client, "analytics-b@example.com")
    maker = client._transport.app.state.test_maker
    async with maker() as session:
        owner = await session.scalar(select(User).where(User.email == "analytics-a@example.com"))
        session.add(
            SearchEvent(user_id=owner.id, strategy="keyword", latency_ms=5.0, result_count=1)
        )
        await session.commit()
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "analytics-b@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    result = await client.get("/api/v1/analytics/retrieval", headers=headers)
    assert result.json()["strategies"] == []


@pytest.mark.asyncio
async def test_chunk_pagination_and_validation(client):
    from ake_backend.db import Document, DocumentChunk

    await register(client, "pagination@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "pagination@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Pages"}, headers=headers)
    ).json()["id"]
    maker = client._transport.app.state.test_maker
    async with maker() as session:
        doc = Document(
            knowledge_base_id=base,
            filename="a.txt",
            storage_key="testkey",
            format="txt",
            size=1,
            sha256="a" * 64,
            status="ready",
        )
        session.add(doc)
        await session.flush()
        session.add_all(
            [
                DocumentChunk(
                    document_id=doc.id,
                    position=i,
                    page=None,
                    text=f"chunk {i}",
                    token_count=2,
                    start_offset=0,
                    end_offset=7,
                )
                for i in range(4)
            ]
        )
        await session.commit()
        doc_id = doc.id
    page = await client.get(f"/api/v1/documents/{doc_id}/chunks?offset=1&limit=2", headers=headers)
    assert page.status_code == 200 and [c["position"] for c in page.json()] == [1, 2]
    assert (
        await client.get(f"/api/v1/documents/{doc_id}/chunks?limit=0", headers=headers)
    ).status_code == 422


@pytest.mark.asyncio
async def test_cancel_document_job_prevents_processing(client):
    from ake_backend.worker import process_one

    await register(client, "cancel-doc@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "cancel-doc@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Cancel"}, headers=headers)
    ).json()["id"]
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base}/documents",
        files={"file": ("a.txt", b"content", "text/plain")},
        headers=headers,
    )
    document_id = upload.json()["id"]
    await register(client, "cancel-intruder@example.com")
    intruder = await client.post(
        "/api/v1/auth/login",
        json={"email": "cancel-intruder@example.com", "password": "correct horse battery staple"},
    )
    intruder_headers = {"Authorization": "Bearer " + intruder.json()["access_token"]}
    assert (
        await client.post(f"/api/v1/documents/{document_id}/cancel", headers=intruder_headers)
    ).status_code == 404
    result = await client.post(f"/api/v1/documents/{document_id}/cancel", headers=headers)
    assert result.status_code == 200
    assert result.json()["status"] == "cancelled"
    assert not await process_one(
        client._transport.app.state.test_maker,
        client._transport.app.state.settings.document_storage_path,
    )
    assert (await client.get(f"/api/v1/documents/{document_id}", headers=headers)).json()[
        "status"
    ] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_during_document_processing_does_not_commit_chunks(client):
    import asyncio
    import threading

    from ake_backend.worker import process_one

    await register(client, "cancel-active@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "cancel-active@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Cancel"}, headers=headers)
    ).json()["id"]
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base}/documents",
        files={"file": ("a.txt", b"content", "text/plain")},
        headers=headers,
    )
    document_id = upload.json()["id"]
    started, release = threading.Event(), threading.Event()

    class Provider:
        model_name = "test"
        revision = "v1"

        def encode(self, texts):
            started.set()
            if not release.wait(5):
                raise RuntimeError("test timeout")
            return [[1.0] + [0.0] * 383 for _ in texts]

    task = asyncio.create_task(
        process_one(
            client._transport.app.state.test_maker,
            client._transport.app.state.settings.document_storage_path,
            Provider(),
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        result = await client.post(f"/api/v1/documents/{document_id}/cancel", headers=headers)
        assert result.status_code == 200
    finally:
        release.set()
    assert await task
    assert (
        await client.get(f"/api/v1/documents/{document_id}/chunks", headers=headers)
    ).json() == []
    assert (await client.get(f"/api/v1/documents/{document_id}", headers=headers)).json()[
        "status"
    ] == "cancelled"


@pytest.mark.asyncio
async def test_unconfigured_llm_returns_clear_error_before_retrieval(client):
    await register(client, "no-provider@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "no-provider@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Unconfigured"}, headers=headers)
    ).json()["id"]
    response = await client.post(
        "/api/v1/ask",
        json={"knowledge_base_id": base, "question": "Question?", "allow_provider_transfer": True},
        headers=headers,
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "LLM provider is not configured"


@pytest.mark.asyncio
async def test_worker_rejects_chunk_size_beyond_model_context(client):
    from ake_backend.worker import process_one

    await register(client, "chunk-size@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "chunk-size@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Config"}, headers=headers)
    ).json()["id"]
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base}/documents",
        files={"file": ("a.txt", b"some words", "text/plain")},
        headers=headers,
    )

    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

    class Model:
        max_seq_length = 128

    class Provider:
        tokenizer = Tokenizer()
        model = Model()

    await process_one(
        client._transport.app.state.test_maker,
        client._transport.app.state.settings.document_storage_path,
        Provider(),
        chunk_size=127,
        chunk_overlap=16,
    )
    result = await client.get("/api/v1/documents/" + upload.json()["id"], headers=headers)
    assert result.json()["status"] == "failed"
    assert "model context" in result.json()["error"]


@pytest.mark.asyncio
async def test_settings_expose_safe_configuration_without_secrets(client):
    await register(client, "settings@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "settings@example.com", "password": "correct horse battery staple"},
    )
    response = await client.get(
        "/api/v1/settings", headers={"Authorization": "Bearer " + login.json()["access_token"]}
    )
    assert response.status_code == 200
    assert response.json()["chunk_size"] == 120 and response.json()["chunk_overlap"] == 16
    assert "jwt_secret" not in response.text and "llm_api_key" not in response.text
    assert (await client.get("/api/v1/settings")).status_code == 401


@pytest.mark.asyncio
async def test_upload_idempotency_returns_same_job_and_rejects_changed_file(client):
    await register(client, "idempotency@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "idempotency@example.com", "password": "correct horse battery staple"},
    )
    headers = {
        "Authorization": "Bearer " + login.json()["access_token"],
        "Idempotency-Key": "unique-upload-key-1",
    }
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Idempotency"}, headers=headers)
    ).json()["id"]

    async def upload(data):
        return await client.post(
            f"/api/v1/knowledge-bases/{base}/documents",
            files={"file": ("test.txt", data, "text/plain")},
            headers=headers,
        )

    first = await upload(b"first document")
    again = await upload(b"first document")
    assert first.status_code == 202 and again.status_code == 202
    assert first.json()["id"] == again.json()["id"]
    assert first.json()["job_id"] == again.json()["job_id"]
    assert (await upload(b"different document")).status_code == 409
    assert (
        len((await client.get(f"/api/v1/knowledge-bases/{base}/documents", headers=headers)).json())
        == 1
    )


@pytest.mark.asyncio
async def test_indexed_pdf_reports_page_count(client):
    import pymupdf

    from ake_backend.worker import process_one

    await register(client, "page-count@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "page-count@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Pages"}, headers=headers)
    ).json()["id"]
    pdf = pymupdf.open()
    for text in ["Page one", "Page two"]:
        pdf.new_page().insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base}/documents",
        files={"file": ("pages.pdf", data, "application/pdf")},
        headers=headers,
    )
    await process_one(
        client._transport.app.state.test_maker,
        client._transport.app.state.settings.document_storage_path,
    )
    document = await client.get("/api/v1/documents/" + upload.json()["id"], headers=headers)
    assert document.json()["index_metadata"]["page_count"] == 2
    assert document.json()["index_metadata"]["parser_version"] == "pdf-columns-v1"


@pytest.mark.asyncio
async def test_upload_rejects_invalid_content_and_oversize(client):
    await register(client, "invalid-upload@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "invalid-upload@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Uploads"}, headers=headers)
    ).json()["id"]
    endpoint = f"/api/v1/knowledge-bases/{base}/documents"
    for filename, contents, expected in [
        ("x.exe", b"content", 415),
        ("x.pdf", b"not a pdf", 415),
        ("x.docx", b"not zip", 415),
        ("x.txt", b"\xff", 415),
        ("x.txt", b"bad\x00text", 415),
        ("x.txt", b"", 422),
        ("..\\evil.txt", b"x", 422),
    ]:
        response = await client.post(
            endpoint,
            files={"file": (filename, contents, "application/octet-stream")},
            headers=headers,
        )
        assert response.status_code == expected, (filename, response.text)
    client._transport.app.state.settings.max_upload_size = 3
    response = await client.post(
        endpoint, files={"file": ("x.txt", b"1234", "text/plain")}, headers=headers
    )
    assert response.status_code == 413
    assert (await client.get(endpoint, headers=headers)).json() == []


@pytest.mark.asyncio
async def test_stream_requires_consent_and_configured_provider(client):
    await register(client, "stream-user@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "stream-user@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Stream"}, headers=headers)
    ).json()["id"]
    conv = (
        await client.post(
            "/api/v1/chat/conversations",
            json={"knowledge_base_id": base, "title": "Stream"},
            headers=headers,
        )
    ).json()["id"]
    path = f"/api/v1/chat/conversations/{conv}/stream"
    assert (
        await client.post(
            path, json={"question": "Test", "allow_provider_transfer": False}, headers=headers
        )
    ).status_code == 400
    response = await client.post(
        path, json={"question": "Test", "allow_provider_transfer": True}, headers=headers
    )
    assert (
        response.status_code == 503
        and response.json()["detail"] == "LLM provider is not configured"
    )
    assert (
        await client.post(path, json={"question": "Test", "allow_provider_transfer": True})
    ).status_code == 401


@pytest.mark.asyncio
async def test_stream_api_returns_provisional_deltas_then_saves_only_validated_answer(
    client, monkeypatch
):
    import ake_backend.rag as rag
    from ake_backend.db import Document

    await register(client, "stream-valid@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "stream-valid@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Stream"}, headers=headers)
    ).json()["id"]
    async with client._transport.app.state.test_maker() as session:
        document = Document(
            knowledge_base_id=base,
            filename="source.txt",
            storage_key="stream-test-key",
            format="txt",
            size=1,
            sha256="a" * 64,
            status="ready",
        )
        session.add(document)
        await session.commit()
        document_id = document.id

    async def fake_retrieve(document_ids, retrieve, limit):
        return [
            {
                "id": "source-1",
                "document_id": document_id,
                "page": 1,
                "text": "Evidence",
                "score": 1.0,
            }
        ]

    monkeypatch.setattr(rag, "retrieve_for_documents", fake_retrieve)

    class Provider:
        async def stream(self, question, context):
            yield "Verified "
            yield "[source-1]."

    client._transport.app.state.llm_provider = Provider()
    conv = (
        await client.post(
            "/api/v1/chat/conversations",
            json={"knowledge_base_id": base, "title": "Stream"},
            headers=headers,
        )
    ).json()["id"]
    response = await client.post(
        f"/api/v1/chat/conversations/{conv}/stream",
        json={"question": "Question", "allow_provider_transfer": True},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: delta") == 2 and "event: final" in response.text
    history = (await client.get(f"/api/v1/chat/conversations/{conv}", headers=headers)).json()[
        "messages"
    ]
    assert len(history) == 2 and history[1]["text"] == "Verified [source-1]."
    assert history[1]["generation_metadata"]["prompt_version"] == "rag-v2"
    assert history[1]["generation_metadata"]["status"] == "answered"

    class InvalidProvider:
        async def stream(self, question, context):
            yield "Unverified [unknown]."

    client._transport.app.state.llm_provider = InvalidProvider()
    response = await client.post(
        f"/api/v1/chat/conversations/{conv}/stream",
        json={"question": "Another", "allow_provider_transfer": True},
        headers=headers,
    )
    assert "event: error" in response.text and "event: final" not in response.text
    assert (
        len(
            (await client.get(f"/api/v1/chat/conversations/{conv}", headers=headers)).json()[
                "messages"
            ]
        )
        == 2
    )


@pytest.mark.asyncio
async def test_worker_stops_when_chunk_count_limit_exceeded(client):
    import re

    from ake_backend.worker import process_one

    await register(client, "chunk-limit@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "chunk-limit@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Limit"}, headers=headers)
    ).json()["id"]
    upload = await client.post(
        f"/api/v1/knowledge-bases/{base}/documents",
        files={"file": ("a.txt", b"one two three four five", "text/plain")},
        headers=headers,
    )

    class Tokenizer:
        def __call__(self, text, **kwargs):
            spans = [match.span() for match in re.finditer(r"\S+", text)]
            return {"offset_mapping": spans, "input_ids": list(range(len(spans)))}

    class Provider:
        tokenizer = Tokenizer()

    await process_one(
        client._transport.app.state.test_maker,
        client._transport.app.state.settings.document_storage_path,
        Provider(),
        chunk_size=1,
        chunk_overlap=0,
        max_chunks=3,
    )
    doc = await client.get("/api/v1/documents/" + upload.json()["id"], headers=headers)
    assert doc.json()["status"] == "failed" and "chunk limit" in doc.json()["error"].lower()


@pytest.mark.asyncio
async def test_upload_logs_request_and_job_correlation(client, caplog):
    import logging

    await register(client, "correlation@example.com")
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "correlation@example.com", "password": "correct horse battery staple"},
    )
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = (
        await client.post("/api/v1/knowledge-bases", json={"name": "Correlation"}, headers=headers)
    ).json()["id"]
    with caplog.at_level(logging.INFO, logger="ake.index"):
        response = await client.post(
            f"/api/v1/knowledge-bases/{base}/documents",
            files={"file": ("private.txt", b"private example content", "text/plain")},
            headers=headers,
        )
    assert response.status_code == 202
    assert response.json()["job_id"] in caplog.text
    assert response.headers["x-request-id"] in caplog.text
    assert "private example content" not in caplog.text
    assert "private.txt" not in caplog.text
