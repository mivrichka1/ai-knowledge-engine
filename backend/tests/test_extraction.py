import pytest
from pydantic import ValidationError

from ake_backend.extraction import ExtractionResult, validate_evidence


def test_schema_and_evidence_validation():
    result = ExtractionResult.model_validate(
        {
            "title": {"value": "BERT", "source_ids": ["a"]},
            "publication_year": {"value": 2019, "source_ids": ["a"]},
        }
    )
    validate_evidence(result, {"a"})
    with pytest.raises(ValueError):
        validate_evidence(result, {"b"})
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(
            {"publication_year": {"value": "not a year", "source_ids": ["a"]}}
        )


def test_nonempty_value_must_have_evidence():
    result = ExtractionResult.model_validate({"title": {"value": "Title", "source_ids": []}})
    with pytest.raises(ValueError):
        validate_evidence(result, {"a"})


def test_extraction_context_selects_beginning_and_relevant_sections():
    from ake_backend.extraction import select_extraction_sources

    chunks = [
        {"id": str(i), "text": text, "page": i + 1}
        for i, text in enumerate(
            [
                "Title and authors",
                "Introduction",
                "Background",
                "Unrelated discussion",
                "Methods and experimental setup",
                "Results and key findings",
                "Limitations and future work",
                "References",
            ]
        )
    ]
    chosen = select_extraction_sources(chunks, limit=5)
    assert [x["id"] for x in chosen] == ["0", "1", "4", "5", "6"]
    assert select_extraction_sources([], limit=5) == []
    with pytest.raises(ValueError):
        select_extraction_sources(chunks, limit=0)


@pytest.mark.asyncio
async def test_extraction_rejects_unknown_citations_and_invalid_json(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from ake_backend.config import Settings
    from ake_backend.db import Base, Document, DocumentChunk, get_session
    from ake_backend.main import create_app

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/extract.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(Settings(jwt_secret="test-only-secret-at-least-32-characters"))

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        credentials = {
            "email": "extract-owner@example.com",
            "password": "correct horse battery staple",
        }
        await client.post("/api/v1/auth/register", json=credentials)
        login = await client.post("/api/v1/auth/login", json=credentials)
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}
        base = (
            await client.post("/api/v1/knowledge-bases", json={"name": "Paper"}, headers=headers)
        ).json()["id"]
        async with maker() as session:
            document = Document(
                knowledge_base_id=base,
                filename="paper.txt",
                storage_key="extract-key",
                format="txt",
                size=4,
                sha256="a" * 64,
                status="ready",
            )
            session.add(document)
            await session.flush()
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    position=0,
                    page=None,
                    text="actual source",
                    token_count=2,
                    start_offset=0,
                    end_offset=13,
                )
            )
            await session.commit()
            document_id = document.id

        class Fake:
            async def extract(self, context):
                return '{"title":{"value":"Invented","source_ids":["unknown"]}}'

        app.state.llm_provider = Fake()
        response = await client.post(
            f"/api/v1/documents/{document_id}/extract",
            json={"allow_provider_transfer": True},
            headers=headers,
        )
        assert response.status_code == 502

        class BadJson:
            async def extract(self, context):
                return "not valid JSON"

        app.state.llm_provider = BadJson()
        response = await client.post(
            f"/api/v1/documents/{document_id}/extract",
            json={"allow_provider_transfer": True},
            headers=headers,
        )
        assert response.status_code == 502
        assert (
            await client.post(
                f"/api/v1/documents/{document_id}/extract",
                json={"allow_provider_transfer": False},
                headers=headers,
            )
        ).status_code == 400
    await engine.dispose()
