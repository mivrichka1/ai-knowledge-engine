import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ake_backend.config import Settings
from ake_backend.db import Base, Document, get_session
from ake_backend.main import create_app


@pytest.mark.asyncio
async def test_comparison_requires_citations_from_each_selected_document(tmp_path, monkeypatch):
    import ake_backend.rag as rag

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/compare.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(Settings(jwt_secret="test-only-secret-at-least-32-characters"))

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        credentials = {"email": "compare@example.com", "password": "correct horse battery staple"}
        await client.post("/api/v1/auth/register", json=credentials)
        headers = {
            "Authorization": "Bearer "
            + (await client.post("/api/v1/auth/login", json=credentials)).json()["access_token"]
        }
        base = (
            await client.post(
                "/api/v1/knowledge-bases", json={"name": "Comparison"}, headers=headers
            )
        ).json()["id"]
        async with maker() as session:
            docs = [
                Document(
                    knowledge_base_id=base,
                    filename=f"{i}.txt",
                    storage_key=f"key{i}",
                    format="txt",
                    size=1,
                    sha256="a" * 64,
                    status="ready",
                )
                for i in range(2)
            ]
            session.add_all(docs)
            await session.commit()
            ids = [d.id for d in docs]
        sources = [
            {
                "id": "source-a",
                "document_id": ids[0],
                "page": 1,
                "text": "First source",
                "score": 1.0,
            },
            {
                "id": "source-b",
                "document_id": ids[1],
                "page": 1,
                "text": "Second source",
                "score": 0.8,
            },
        ]

        async def retrieve(document_ids, retrieve, limit):
            assert document_ids == ids
            return sources

        monkeypatch.setattr(rag, "retrieve_for_documents", retrieve)

        class Provider:
            async def complete(self, question, context):
                return "First claim [source-a], second claim [source-b]."

        app.state.llm_provider = Provider()
        body = {
            "knowledge_base_id": base,
            "question": "Compare the documents",
            "document_ids": ids,
            "allow_provider_transfer": True,
        }
        response = await client.post("/api/v1/ask", json=body, headers=headers)
        assert response.status_code == 200 and len(response.json()["sources"]) == 2

        class MissingCitation:
            async def complete(self, question, context):
                return "Only one document is mentioned [source-a]."

        app.state.llm_provider = MissingCitation()
        assert (await client.post("/api/v1/ask", json=body, headers=headers)).status_code == 502

        async def missing_document(document_ids, retrieve, limit):
            return sources[:1]

        monkeypatch.setattr(rag, "retrieve_for_documents", missing_document)
        response = await client.post("/api/v1/ask", json=body, headers=headers)
        assert response.status_code == 200 and response.json()["status"] == "insufficient_evidence"
    await engine.dispose()
