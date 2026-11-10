import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ake_backend.config import Settings
from ake_backend.db import Base, Conversation, KnowledgeBase, Message, User, get_session
from ake_backend.main import create_app


@pytest.mark.asyncio
async def test_feedback_is_owner_scoped_and_updatable(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/feedback.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(Settings(jwt_secret="test-only-secret-at-least-32-characters"))

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:

        async def auth(email):
            creds = {"email": email, "password": "correct horse battery staple"}
            await client.post("/api/v1/auth/register", json=creds)
            return {
                "Authorization": "Bearer "
                + (await client.post("/api/v1/auth/login", json=creds)).json()["access_token"]
            }

        owner = await auth("feedback-owner@example.com")
        other = await auth("feedback-other@example.com")
        async with maker() as session:
            user = await session.scalar(
                select(User).where(User.email == "feedback-owner@example.com")
            )
            base = KnowledgeBase(owner_id=user.id, name="Base")
            session.add(base)
            await session.flush()
            conversation = Conversation(knowledge_base_id=base.id, title="Test")
            session.add(conversation)
            await session.flush()
            message = Message(
                conversation_id=conversation.id, role="assistant", text="Answer", sources=[]
            )
            session.add(message)
            await session.commit()
            mid = message.id
        assert (
            await client.put(
                f"/api/v1/chat/messages/{mid}/feedback", json={"rating": 1}, headers=other
            )
        ).status_code == 404
        first = await client.put(
            f"/api/v1/chat/messages/{mid}/feedback",
            json={"rating": 1, "comment": "Helpful"},
            headers=owner,
        )
        assert first.status_code == 200 and first.json()["rating"] == 1
        second = await client.put(
            f"/api/v1/chat/messages/{mid}/feedback", json={"rating": -1}, headers=owner
        )
        assert second.json()["rating"] == -1
        assert (
            await client.put(
                f"/api/v1/chat/messages/{mid}/feedback", json={"rating": 0}, headers=owner
            )
        ).status_code == 422
        assert (
            await client.put(
                f"/api/v1/chat/messages/{mid}/feedback", json={"rating": True}, headers=owner
            )
        ).status_code == 422
        assert (
            await client.get(f"/api/v1/chat/conversations/{conversation.id}", headers=owner)
        ).json()["messages"][0]["feedback"]["rating"] == -1
        overview = (await client.get("/api/v1/analytics/overview", headers=owner)).json()
        assert overview["feedback"] == {"helpful": 0, "unhelpful": 1}
        other_overview = (await client.get("/api/v1/analytics/overview", headers=other)).json()
        assert other_overview["feedback"] == {"helpful": 0, "unhelpful": 0}
    await engine.dispose()


@pytest.mark.asyncio
async def test_feedback_rejects_user_messages(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/user_message.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app(Settings(jwt_secret="test-only-secret-at-least-32-characters"))

    async def override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        creds = {"email": "feedback-user@example.com", "password": "correct horse battery staple"}
        await client.post("/api/v1/auth/register", json=creds)
        headers = {
            "Authorization": "Bearer "
            + (await client.post("/api/v1/auth/login", json=creds)).json()["access_token"]
        }
        async with maker() as session:
            user = await session.scalar(select(User).where(User.email == creds["email"]))
            base = KnowledgeBase(owner_id=user.id, name="Base")
            session.add(base)
            await session.flush()
            conversation = Conversation(knowledge_base_id=base.id, title="Test")
            session.add(conversation)
            await session.flush()
            message = Message(
                conversation_id=conversation.id, role="user", text="Question", sources=[]
            )
            session.add(message)
            await session.commit()
            mid = message.id
        response = await client.put(
            f"/api/v1/chat/messages/{mid}/feedback", json={"rating": 1}, headers=headers
        )
        assert response.status_code == 404
    await engine.dispose()
