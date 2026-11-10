import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ake_backend.config import Settings
from ake_backend.db import get_session
from ake_backend.main import create_app


@pytest.mark.asyncio
async def test_readiness_and_request_id():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    maker = async_sessionmaker(engine)
    app = create_app(Settings(jwt_secret="test-only-secret-at-least-32-characters"))

    async def db():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.headers["x-request-id"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_request_log_does_not_include_credentials(tmp_path, caplog):
    import logging

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/privacy.db")
    from ake_backend.db import Base

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(Settings(jwt_secret="test-only-secret-at-least-32-characters"))

    async def db():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = db
    with caplog.at_level(logging.INFO, logger="ake.requests"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/register",
                json={"email": "private@example.com", "password": "private-password-example"},
            )
            assert response.status_code == 201
    assert "private-password-example" not in caplog.text
    assert "private@example.com" not in caplog.text
    assert "request_complete" in caplog.text
    await engine.dispose()
