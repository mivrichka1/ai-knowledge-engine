import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import Settings
from .db import (
    Conversation,
    Document,
    DocumentChunk,
    EvaluationDataset,
    EvaluationRun,
    Feedback,
    KnowledgeBase,
    Message,
    ProcessingJob,
    RefreshSession,
    SearchEvent,
    User,
    get_session,
)
from .llm import RAG_PROMPT_VERSION

Session = Annotated[AsyncSession, Depends(get_session)]
bearer = HTTPBearer(auto_error=False)
password_hasher = PasswordHash.recommended()
dummy_password_hash = password_hasher.hash(secrets.token_urlsafe(32))


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: EmailStr


class Tokens(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshIn(BaseModel):
    refresh_token: str = Field(min_length=40, max_length=200)


class BaseIn(BaseModel):
    name: str = Field(min_length=1, max_length=200, pattern=r".*\S.*")
    description: str | None = Field(default=None, max_length=4000)


class SearchIn(BaseModel):
    knowledge_base_id: str
    query: str = Field(min_length=1, max_length=1000, pattern=r".*\S.*")
    limit: int = Field(default=10, ge=1, le=50)
    strategy: str = Field(default="keyword", pattern="^(keyword|vector|hybrid|hybrid_rerank)$")
    document_ids: list[str] = Field(default_factory=list, max_length=50)


class CompareIn(BaseModel):
    knowledge_base_id: str
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    query: str = Field(min_length=1, max_length=1000, pattern=r".*\S.*")
    limit: int = Field(default=10, ge=1, le=50)
    strategies: list[Literal["keyword", "vector", "hybrid", "hybrid_rerank"]] = Field(
        min_length=1, max_length=4
    )


class AskIn(BaseModel):
    knowledge_base_id: str
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    question: str = Field(min_length=1, max_length=2000, pattern=r".*\S.*")
    allow_provider_transfer: bool = False
    history: str = Field(default="", max_length=4000)


class EvaluationQuery(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=1000)
    answerable: bool
    relevant_ids: list[str] = Field(default_factory=list, max_length=100)
    category: str = Field(min_length=1, max_length=100)


class DatasetIn(BaseModel):
    knowledge_base_id: str
    name: str = Field(min_length=1, max_length=200)
    queries: list[EvaluationQuery] = Field(min_length=1, max_length=200)


class EvaluationRunIn(BaseModel):
    dataset_id: str
    strategies: list[Literal["keyword", "vector", "hybrid", "hybrid_rerank"]] = Field(
        min_length=1, max_length=4
    )
    top_k: int = Field(default=5, ge=1, le=50)


class SourceOut(BaseModel):
    id: str
    document_id: str
    page: int | None
    text: str
    score: float


class AnswerOut(BaseModel):
    prompt_version: str = RAG_PROMPT_VERSION
    answer: str
    status: Literal["answered", "insufficient_evidence"]
    sufficient_context: bool
    sources: list[SourceOut]
    citations: list[str]
    limitations: list[str]
    model_name: str | None
    processing_time_ms: float


class ExtractIn(BaseModel):
    allow_provider_transfer: bool = False


class ConversationIn(BaseModel):
    knowledge_base_id: str
    title: str = Field(default="New conversation", min_length=1, max_length=200)


class FeedbackIn(BaseModel):
    rating: Literal[-1, 1]

    @field_validator("rating", mode="before")
    @classmethod
    def rating_is_integer(cls, value):
        if type(value) is not int:
            raise ValueError("rating must be an integer")
        return value

    comment: str | None = Field(default=None, max_length=2000)


class MessageIn(BaseModel):
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    question: str = Field(min_length=1, max_length=2000, pattern=r".*\S.*")
    allow_provider_transfer: bool = False


class BasePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200, pattern=r".*\S.*")
    description: str | None = Field(default=None, max_length=4000)


class BaseOut(BaseIn):
    model_config = ConfigDict(from_attributes=True)
    id: str


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def unauthorized():
    return HTTPException(
        status_code=401,
        detail="Invalid credentials or session",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def current_user(
    request: Request,
    db: Session,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> User:
    if not credentials:
        raise unauthorized()
    try:
        claims = jwt.decode(
            credentials.credentials,
            request.app.state.settings.jwt_secret,
            algorithms=["HS256"],
            options={"require": ["sub", "exp", "iat", "type"]},
        )
        token_type = claims["type"]
        if token_type == "access":
            session_id = claims.get("sid")
            if not isinstance(session_id, str):
                raise unauthorized()
            active_session = await db.scalar(
                select(RefreshSession.id).where(
                    RefreshSession.id == session_id,
                    RefreshSession.user_id == claims["sub"],
                    RefreshSession.revoked.is_(False),
                    RefreshSession.expires_at > datetime.now(UTC),
                )
            )
            if active_session is None:
                raise unauthorized()
        elif token_type == "evaluation":
            if request.method != "POST" or request.url.path != "/api/v1/search":
                raise unauthorized()
            if not all(
                isinstance(claims.get(key), str) and claims[key] for key in ("run_id", "lease")
            ):
                raise unauthorized()
            active_run = await db.scalar(
                select(EvaluationDataset.knowledge_base_id)
                .join(EvaluationRun, EvaluationRun.dataset_id == EvaluationDataset.id)
                .where(
                    EvaluationRun.id == claims.get("run_id"),
                    EvaluationRun.user_id == claims["sub"],
                    EvaluationRun.status == "processing",
                    EvaluationRun.lease_token == claims.get("lease"),
                    EvaluationRun.lease_until > datetime.now(UTC),
                )
            )
            if active_run is None:
                raise unauthorized()
            request.state.evaluation_base_id = active_run
        else:
            raise unauthorized()
        user = await db.get(User, claims["sub"])
    except (jwt.PyJWTError, ValueError, TypeError):
        raise unauthorized()
    if user is None or not user.is_active:
        raise unauthorized()
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def issue_tokens(user: User, request: Request, db: AsyncSession) -> Tokens:
    settings = request.app.state.settings
    now = datetime.now(UTC)
    session_id = str(uuid4())
    access = jwt.encode(
        {
            "sub": user.id,
            "sid": session_id,
            "iat": now,
            "exp": now + timedelta(minutes=settings.access_minutes),
            "type": "access",
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    refresh = secrets.token_urlsafe(48)
    db.add(
        RefreshSession(
            id=session_id,
            user_id=user.id,
            token_hash=digest(refresh),
            expires_at=now + timedelta(days=settings.refresh_days),
        )
    )
    return Tokens(access_token=access, refresh_token=refresh)


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        app.state.sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        yield
        await engine.dispose()

    app = FastAPI(title="AI Knowledge Engine", lifespan=lifespan)
    logging.getLogger("ake").setLevel(getattr(logging, settings.log_level))
    app.state.settings = settings
    app.state.embedding_lock = asyncio.Lock()
    app.state.reranker_lock = asyncio.Lock()
    app.state.llm_semaphore = asyncio.Semaphore(2)
    from .rate_limit import RateLimiter

    app.state.auth_limiter = RateLimiter(limit=20, window_seconds=60)

    def limit_auth(request: Request):
        key = request.client.host if request.client else "unknown"
        try:
            request.app.state.auth_limiter.check(key)
        except ValueError:
            raise HTTPException(
                429, "Too many authentication attempts", headers={"Retry-After": "60"}
            )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logging.getLogger("ake.requests").error(
                json.dumps(
                    {"event": "request_failed", "request_id": request_id, "method": request.method}
                )
            )
            raise
        response.headers["X-Request-ID"] = request_id
        logging.getLogger("ake.requests").info(
            json.dumps(
                {
                    "event": "request_complete",
                    "request_id": request_id,
                    "method": request.method,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        )
        return response

    @app.get("/api/v1/health/ready")
    async def ready(db: Session):
        try:
            await db.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(503, "Database unavailable")
        return {"status": "ready"}

    @app.get("/api/v1/health/live")
    async def live():
        return {"status": "ok"}

    @app.post("/api/v1/auth/register", response_model=UserOut, status_code=201)
    async def register(data: Credentials, request: Request, db: Session):
        limit_auth(request)
        hashed = await asyncio.to_thread(password_hasher.hash, data.password)
        user = User(email=data.email.lower(), password_hash=hashed)
        db.add(user)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            raise HTTPException(409, "Email already registered")
        await db.refresh(user)
        return user

    @app.post("/api/v1/auth/login", response_model=Tokens)
    async def login(data: Credentials, request: Request, db: Session):
        limit_auth(request)
        user = await db.scalar(select(User).where(User.email == data.email.lower()))
        valid_password = await asyncio.to_thread(
            password_hasher.verify,
            data.password,
            user.password_hash if user else dummy_password_hash,
        )
        if not user or not user.is_active or not valid_password:
            raise unauthorized()
        tokens = issue_tokens(user, request, db)
        await db.commit()
        return tokens

    @app.post("/api/v1/auth/refresh", response_model=Tokens)
    async def refresh(data: RefreshIn, request: Request, db: Session):
        limit_auth(request)
        now = datetime.now(UTC)
        result = await db.execute(
            update(RefreshSession)
            .where(
                RefreshSession.token_hash == digest(data.refresh_token),
                RefreshSession.revoked.is_(False),
                RefreshSession.expires_at > now,
            )
            .values(revoked=True)
            .returning(RefreshSession.user_id)
        )
        user_id = result.scalar_one_or_none()
        user = await db.get(User, user_id) if user_id else None
        if user is None or not user.is_active:
            await db.rollback()
            raise unauthorized()
        tokens = issue_tokens(user, request, db)
        await db.commit()
        return tokens

    @app.post("/api/v1/auth/logout", status_code=204)
    async def logout(data: RefreshIn, db: Session):
        await db.execute(
            update(RefreshSession)
            .where(RefreshSession.token_hash == digest(data.refresh_token))
            .values(revoked=True)
        )
        await db.commit()
        return Response(status_code=204)

    @app.get("/api/v1/auth/me", response_model=UserOut)
    async def me(user: CurrentUser):
        return user

    async def owned(base_id: str, user: User, db: AsyncSession):
        item = await db.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.id == base_id, KnowledgeBase.owner_id == user.id
            )
        )
        if item is None:
            raise HTTPException(404, "Knowledge base not found")
        return item

    @app.get("/api/v1/knowledge-bases", response_model=list[BaseOut])
    async def list_bases(user: CurrentUser, db: Session, offset: int = 0, limit: int = 50):
        if offset < 0 or not 1 <= limit <= 100:
            raise HTTPException(422, "Invalid pagination")
        return (
            await db.scalars(
                select(KnowledgeBase)
                .where(KnowledgeBase.owner_id == user.id)
                .order_by(KnowledgeBase.created_at, KnowledgeBase.id)
                .offset(offset)
                .limit(limit)
            )
        ).all()

    @app.post("/api/v1/knowledge-bases", response_model=BaseOut, status_code=201)
    async def create_base(data: BaseIn, user: CurrentUser, db: Session):
        item = KnowledgeBase(owner_id=user.id, **data.model_dump())
        db.add(item)
        await db.commit()
        await db.refresh(item)
        return item

    @app.get("/api/v1/knowledge-bases/{base_id}", response_model=BaseOut)
    async def get_base(base_id: str, user: CurrentUser, db: Session):
        return await owned(base_id, user, db)

    @app.patch("/api/v1/knowledge-bases/{base_id}", response_model=BaseOut)
    async def update_base(base_id: str, data: BasePatch, user: CurrentUser, db: Session):
        item = await owned(base_id, user, db)
        changes = data.model_dump(exclude_unset=True)
        if "name" in changes and changes["name"] is None:
            raise HTTPException(422, "Name cannot be null")
        for key, value in changes.items():
            setattr(item, key, value)
        await db.commit()
        return item

    @app.delete("/api/v1/knowledge-bases/{base_id}", status_code=204)
    async def delete_base(base_id: str, user: CurrentUser, db: Session):
        item = await owned(base_id, user, db)
        if await db.scalar(
            select(Document.id).where(Document.knowledge_base_id == base_id).limit(1)
        ):
            raise HTTPException(409, "Cannot delete a knowledge base containing documents yet")
        await db.delete(item)
        await db.commit()
        return Response(status_code=204)

    @app.get("/api/v1/knowledge-bases/{base_id}/documents")
    async def list_documents(
        base_id: str, user: CurrentUser, db: Session, limit: int = 50, offset: int = 0
    ):
        await owned(base_id, user, db)
        if offset < 0 or not 1 <= limit <= 100:
            raise HTTPException(422, "Invalid pagination")
        items = (
            await db.scalars(
                select(Document)
                .where(Document.knowledge_base_id == base_id)
                .order_by(Document.created_at, Document.id)
                .offset(offset)
                .limit(limit)
            )
        ).all()
        return [
            {"id": item.id, "filename": item.filename, "status": item.status, "error": item.error}
            for item in items
        ]

    @app.post("/api/v1/knowledge-bases/{base_id}/documents", status_code=202)
    async def upload_document(
        base_id: str,
        file: UploadFile,
        user: CurrentUser,
        db: Session,
        request: Request,
        idempotency_key: Annotated[str | None, Header(min_length=1, max_length=128)] = None,
    ):
        await owned(base_id, user, db)
        name = file.filename or ""
        if (
            not name
            or len(name) > 255
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
            or any(ord(c) < 32 for c in name)
        ):
            raise HTTPException(422, "Invalid filename")
        kind = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if kind not in {"txt", "md", "pdf", "docx"}:
            raise HTTPException(415, "Unsupported format")
        data = await file.read(request_limit := settings.max_upload_size + 1)
        await file.close()
        if len(data) >= request_limit:
            raise HTTPException(413, "File too large")
        if not data:
            raise HTTPException(422, "Empty file")
        if kind == "pdf" and not data.startswith(b"%PDF-"):
            raise HTTPException(415, "Invalid PDF signature")
        if kind == "docx" and not data.startswith(b"PK\x03\x04"):
            raise HTTPException(415, "Invalid DOCX signature")
        if kind in {"txt", "md"}:
            try:
                decoded = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise HTTPException(415, "Text must be UTF-8")
            if "\x00" in decoded:
                raise HTTPException(415, "Invalid text content")
        file_hash = hashlib.sha256(data).hexdigest()

        async def existing_upload():
            if not idempotency_key:
                return None
            item = await db.scalar(
                select(Document).where(
                    Document.knowledge_base_id == base_id,
                    Document.idempotency_key == idempotency_key,
                )
            )
            if item is None:
                return None
            if item.status == "deleting":
                raise HTTPException(409, "Original document is being deleted")
            if item.sha256 != file_hash or item.filename != name:
                raise HTTPException(409, "Idempotency key was already used for another upload")
            job = await db.scalar(
                select(ProcessingJob)
                .where(ProcessingJob.document_id == item.id)
                .order_by(ProcessingJob.created_at, ProcessingJob.id)
                .limit(1)
            )
            if job is None:
                raise HTTPException(409, "Original upload job is unavailable")
            logging.getLogger("ake.index").info(
                json.dumps(
                    {
                        "event": "index_request_reused",
                        "request_id": request.state.request_id,
                        "document_id": item.id,
                        "job_id": job.id,
                    }
                )
            )
            return {"id": item.id, "job_id": job.id, "status": item.status}

        previous = await existing_upload()
        if previous is not None:
            return previous
        key = str(uuid4())
        root = Path(settings.document_storage_path)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = root / key
        try:
            with path.open("xb") as out:
                out.write(data)
            item = Document(
                knowledge_base_id=base_id,
                filename=name,
                storage_key=key,
                format=kind,
                size=len(data),
                sha256=file_hash,
                idempotency_key=idempotency_key,
                status="pending",
            )
            db.add(item)
            await db.flush()
            job = ProcessingJob(document_id=item.id, status="pending")
            db.add(job)
            await db.commit()
        except IntegrityError:
            await db.rollback()
            path.unlink(missing_ok=True)
            previous = await existing_upload()
            if previous is not None:
                return previous
            raise HTTPException(409, "Upload conflicts with an existing document")
        except Exception:
            await db.rollback()
            path.unlink(missing_ok=True)
            raise
        logging.getLogger("ake.index").info(
            json.dumps(
                {
                    "event": "index_queued",
                    "request_id": request.state.request_id,
                    "document_id": item.id,
                    "job_id": job.id,
                }
            )
        )
        return {"id": item.id, "job_id": job.id, "status": item.status}

    @app.get("/api/v1/documents/{document_id}")
    async def get_document(document_id: str, user: CurrentUser, db: Session):
        item = await db.scalar(
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(Document.id == document_id, KnowledgeBase.owner_id == user.id)
        )
        if item is None:
            raise HTTPException(404, "Document not found")
        return {
            "id": item.id,
            "filename": item.filename,
            "status": item.status,
            "error": item.error,
            "index_version": item.index_version,
            "index_metadata": item.index_metadata,
        }

    @app.get("/api/v1/documents/{document_id}/chunks")
    async def get_chunks(
        document_id: str, user: CurrentUser, db: Session, offset: int = 0, limit: int = 100
    ):
        if offset < 0 or not 1 <= limit <= 500:
            raise HTTPException(422, "Invalid pagination")
        item = await db.scalar(
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(Document.id == document_id, KnowledgeBase.owner_id == user.id)
        )
        if item is None:
            raise HTTPException(404, "Document not found")
        chunks = (
            await db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == item.id)
                .order_by(DocumentChunk.position, DocumentChunk.id)
                .offset(offset)
                .limit(limit)
            )
        ).all()
        return [
            {"id": c.id, "position": c.position, "page": c.page, "text": c.text} for c in chunks
        ]

    async def search(data: SearchIn, request: Request, user: CurrentUser, db: Session):
        started = time.perf_counter()

        def search_response(strategy, results):
            return {
                "strategy": strategy,
                "results": results,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }

        await owned(data.knowledge_base_id, user, db)
        if db.bind is None or db.bind.dialect.name != "postgresql":
            raise HTTPException(503, "Keyword search requires PostgreSQL")
        if data.strategy == "hybrid_rerank":
            from .rerank import CrossEncoderScorer, rank_results

            candidates = await search(
                data.model_copy(update={"strategy": "hybrid", "limit": 50}), request, user, db
            )
            if not candidates["results"]:
                return search_response("hybrid_rerank", [])
            try:
                async with request.app.state.reranker_lock:
                    scorer = getattr(request.app.state, "reranker", None)
                    if scorer is None:
                        scorer = await asyncio.to_thread(
                            CrossEncoderScorer, settings.reranker_model, settings.reranker_revision
                        )
                        request.app.state.reranker = scorer
                    ranked = await asyncio.to_thread(
                        rank_results, data.query, candidates["results"], scorer
                    )
            except Exception:
                raise HTTPException(503, "Reranking model is unavailable")
            return search_response("hybrid_rerank", ranked[: data.limit])
        if data.strategy == "hybrid":
            from .retrieval import reciprocal_rank_fusion

            keyword = await search(
                data.model_copy(update={"strategy": "keyword", "limit": 50}), request, user, db
            )
            vector = await search(
                data.model_copy(update={"strategy": "vector", "limit": 50}), request, user, db
            )
            results = keyword["results"] + vector["results"]
            by_id = {item["id"]: item for item in results}
            fused = reciprocal_rank_fusion(
                [
                    [item["id"] for item in keyword["results"]],
                    [item["id"] for item in vector["results"]],
                ],
                top_k=data.limit,
            )
            return search_response(
                "hybrid", [{**by_id[id_], "score": score} for id_, score in fused]
            )
        if data.strategy == "vector":
            from .embeddings import (
                EmbeddingInputTooLong,
                SentenceTransformerProvider,
                validate_vectors,
            )

            try:
                async with request.app.state.embedding_lock:
                    provider = getattr(request.app.state, "embedding_provider", None)
                    if provider is None:
                        provider = await asyncio.to_thread(
                            SentenceTransformerProvider,
                            settings.embedding_model,
                            384,
                            settings.embedding_revision,
                        )
                        request.app.state.embedding_provider = provider
                    assert provider is not None
                    vector = (await asyncio.to_thread(provider.encode, [data.query]))[0]
                    validate_vectors([vector], 1, 384)
            except EmbeddingInputTooLong:
                raise HTTPException(422, "Query exceeds the embedding model context; shorten it")
            except Exception:
                raise HTTPException(503, "Embedding generation failed")
            query_vector = "[" + ",".join(str(float(value)) for value in vector) + "]"
            stmt = text("""
                SELECT c.id, c.document_id, c.page, c.text,
                       1 - (c.embedding <=> CAST(:vector AS vector)) AS score
                FROM document_chunks c JOIN documents d ON d.id = c.document_id
                JOIN knowledge_bases kb ON kb.id = d.knowledge_base_id
                WHERE kb.id = :base_id AND kb.owner_id = :owner_id AND (NOT :filter_enabled OR d.id = ANY(CAST(:document_ids AS varchar[]))) AND d.status = 'ready'
                  AND c.embedding IS NOT NULL AND c.embedding_model = :model AND c.embedding_revision = :revision
                ORDER BY c.embedding <=> CAST(:vector AS vector), c.id LIMIT :limit
            """)
            rows = await db.execute(
                stmt,
                {
                    "vector": query_vector,
                    "base_id": data.knowledge_base_id,
                    "owner_id": user.id,
                    "filter_enabled": bool(data.document_ids),
                    "document_ids": data.document_ids,
                    "model": provider.model_name,
                    "revision": provider.revision,
                    "limit": data.limit,
                },
            )
            return search_response("vector", [dict(row) for row in rows.mappings()])
        query = text("""
            SELECT c.id, c.document_id, c.page, c.text,
                   ts_rank_cd(to_tsvector('simple', c.text), websearch_to_tsquery('simple', :query)) AS score
            FROM document_chunks c JOIN documents d ON d.id = c.document_id
            JOIN knowledge_bases kb ON kb.id = d.knowledge_base_id
            WHERE kb.id = :base_id AND kb.owner_id = :owner_id AND d.status IN ('ready', 'parsed') AND (NOT :filter_enabled OR d.id = ANY(CAST(:document_ids AS varchar[])))
              AND to_tsvector('simple', c.text) @@ websearch_to_tsquery('simple', :query)
            ORDER BY score DESC, c.id ASC LIMIT :limit
        """)
        result = await db.execute(
            query,
            {
                "query": data.query,
                "base_id": data.knowledge_base_id,
                "owner_id": user.id,
                "filter_enabled": bool(data.document_ids),
                "document_ids": data.document_ids,
                "limit": data.limit,
            },
        )
        hits = [dict(row) for row in result.mappings()]
        mode = "websearch"
        if not hits and not re.search(r'[-"]|\bOR\b', data.query):
            fallback = text("""
                SELECT c.id, c.document_id, c.page, c.text,
                    ts_rank_cd(to_tsvector('simple', c.text), CAST(replace(CAST(plainto_tsquery('simple', :query) AS text), ' & ', ' | ') AS tsquery)) AS score
                FROM document_chunks c JOIN documents d ON d.id = c.document_id
                JOIN knowledge_bases kb ON kb.id = d.knowledge_base_id
                WHERE kb.id = :base_id AND kb.owner_id = :owner_id AND d.status IN ('ready', 'parsed') AND (NOT :filter_enabled OR d.id = ANY(CAST(:document_ids AS varchar[])))
                  AND to_tsvector('simple', c.text) @@ CAST(replace(CAST(plainto_tsquery('simple', :query) AS text), ' & ', ' | ') AS tsquery)
                ORDER BY score DESC, c.id ASC LIMIT :limit
            """)
            fallback_rows = await db.execute(
                fallback,
                {
                    "query": data.query,
                    "base_id": data.knowledge_base_id,
                    "owner_id": user.id,
                    "filter_enabled": bool(data.document_ids),
                    "document_ids": data.document_ids,
                    "limit": data.limit,
                },
            )
            hits = [dict(row) for row in fallback_rows.mappings()]
            mode = "any_terms"
        response = search_response("keyword", hits)
        response["query_mode"] = mode
        return response

    async def record_search(result: dict, user: User, db: AsyncSession):
        db.add(
            SearchEvent(
                user_id=user.id,
                strategy=result["strategy"],
                latency_ms=result["latency_ms"],
                result_count=len(result["results"]),
            )
        )
        await db.commit()

    @app.post("/api/v1/search")
    async def search_endpoint(data: SearchIn, request: Request, user: CurrentUser, db: Session):
        evaluation_base_id = getattr(request.state, "evaluation_base_id", None)
        if evaluation_base_id is not None and evaluation_base_id != data.knowledge_base_id:
            raise HTTPException(403, "Evaluation token is restricted to its dataset")
        result = await search(data, request, user, db)
        await record_search(result, user, db)
        return result

    @app.post("/api/v1/search/compare")
    async def compare_search(data: CompareIn, request: Request, user: CurrentUser, db: Session):
        if len(set(data.strategies)) != len(data.strategies):
            raise HTTPException(422, "Duplicate search strategies")
        comparisons = []
        for strategy in data.strategies:
            comparisons.append(
                await search(
                    SearchIn(
                        knowledge_base_id=data.knowledge_base_id,
                        query=data.query,
                        limit=data.limit,
                        strategy=strategy,
                        document_ids=data.document_ids,
                    ),
                    request,
                    user,
                    db,
                )
            )
        for comparison in comparisons:
            await record_search(comparison, user, db)
        return {"comparisons": comparisons}

    async def prepare_rag(data: AskIn, request: Request, user: User, db: AsyncSession):
        from .llm import OpenAICompatibleProvider
        from .rag import build_context, retrieve_for_documents

        await owned(data.knowledge_base_id, user, db)
        if not data.allow_provider_transfer:
            raise HTTPException(
                400,
                "Confirm that document excerpts and conversation history may be sent to the configured provider",
            )
        provider = getattr(request.app.state, "llm_provider", None)
        if provider is None:
            if not settings.llm_model:
                raise HTTPException(503, "LLM provider is not configured")
            provider = OpenAICompatibleProvider(
                settings.llm_base_url, settings.llm_api_key, settings.llm_model
            )
        if len(set(data.document_ids)) != len(data.document_ids) or len(data.document_ids) > 4:
            raise HTTPException(422, "Choose up to four distinct documents")
        query = ((data.history[-400:] + " " if data.history else "") + data.question)[-1000:]

        async def retrieve_document(document_id):
            result = await search(
                SearchIn(
                    knowledge_base_id=data.knowledge_base_id,
                    query=query,
                    strategy="hybrid",
                    limit=8,
                    document_ids=[document_id] if document_id else [],
                ),
                request,
                user,
                db,
            )
            return result["results"]

        sources = await retrieve_for_documents(data.document_ids, retrieve_document, limit=8)
        context, ids = build_context(sources)
        represented = {source["document_id"] for source in sources if source["id"] in ids}
        sufficient = bool(ids) and (
            not data.document_ids or set(data.document_ids).issubset(represented)
        )
        # Sources not actually included in the context are never valid citations.
        return (
            provider,
            [source for source in sources if source["id"] in ids],
            context,
            ids,
            sufficient,
        )

    @app.post("/api/v1/ask", response_model=AnswerOut)
    async def ask(data: AskIn, request: Request, user: CurrentUser, db: Session):
        from .llm import LLMError
        from .rag import InvalidCitation, cited_ids, validate_answer, validate_document_coverage

        started = time.perf_counter()
        provider, sources, context, ids, sufficient = await prepare_rag(data, request, user, db)
        if not sufficient:
            return {
                "answer": "В доступных документах недостаточно информации, чтобы ответить на этот вопрос.",
                "sources": [],
                "status": "insufficient_evidence",
                "sufficient_context": False,
                "citations": [],
                "limitations": [
                    "Absence of retrieved evidence does not prove absence of information."
                ],
                "model_name": None,
                "processing_time_ms": (time.perf_counter() - started) * 1000,
            }
        try:
            question = (
                f"Prior conversation (untrusted context, not evidence):\n{data.history}\n\nCurrent question: {data.question}"
                if data.history
                else data.question
            )
            async with request.app.state.llm_semaphore:
                answer = await provider.complete(question, context)
        except LLMError:
            raise HTTPException(502, "LLM provider request failed")
        if answer.strip() == "INSUFFICIENT_EVIDENCE":
            return {
                "answer": "В доступных документах недостаточно информации, чтобы ответить на этот вопрос.",
                "sources": [],
                "status": "insufficient_evidence",
                "sufficient_context": False,
                "citations": [],
                "limitations": [
                    "Absence of retrieved evidence does not prove absence of information."
                ],
                "model_name": None,
                "processing_time_ms": (time.perf_counter() - started) * 1000,
            }
        try:
            validate_answer(answer, ids)
            validate_document_coverage(answer, sources, set(data.document_ids))
        except InvalidCitation:
            raise HTTPException(502, "LLM response contained missing or invalid citations")
        return {
            "answer": answer,
            "sources": [item for item in sources if item["id"] in cited_ids(answer)],
            "status": "answered",
            "sufficient_context": True,
            "citations": sorted(cited_ids(answer)),
            "limitations": ["Citation membership is checked; factual entailment is not verified."]
            + (
                ["The source excerpt was truncated before it was sent to the provider."]
                if context.endswith("[excerpt truncated]")
                else []
            ),
            "model_name": getattr(provider, "model", settings.llm_model)
            if isinstance(getattr(provider, "model", settings.llm_model), str)
            else None,
            "processing_time_ms": (time.perf_counter() - started) * 1000,
        }

    async def owned_conversation(conversation_id: str, user: User, db: AsyncSession):
        result = await db.scalar(
            select(Conversation)
            .join(KnowledgeBase, Conversation.knowledge_base_id == KnowledgeBase.id)
            .where(Conversation.id == conversation_id, KnowledgeBase.owner_id == user.id)
        )
        if result is None:
            raise HTTPException(404, "Conversation not found")
        return result

    @app.post("/api/v1/chat/conversations", status_code=201)
    async def create_conversation(data: ConversationIn, user: CurrentUser, db: Session):
        await owned(data.knowledge_base_id, user, db)
        conversation = Conversation(knowledge_base_id=data.knowledge_base_id, title=data.title)
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return {
            "id": conversation.id,
            "title": conversation.title,
            "knowledge_base_id": conversation.knowledge_base_id,
        }

    @app.get("/api/v1/chat/conversations")
    async def list_conversations(user: CurrentUser, db: Session):
        rows = (
            await db.scalars(
                select(Conversation)
                .join(KnowledgeBase, Conversation.knowledge_base_id == KnowledgeBase.id)
                .where(KnowledgeBase.owner_id == user.id)
                .order_by(Conversation.created_at.desc())
                .limit(100)
            )
        ).all()
        return [
            {"id": row.id, "title": row.title, "knowledge_base_id": row.knowledge_base_id}
            for row in rows
        ]

    @app.get("/api/v1/chat/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str, user: CurrentUser, db: Session):
        conversation = await owned_conversation(conversation_id, user, db)
        messages = (
            await db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at, Message.id)
                .limit(500)
            )
        ).all()
        feedback_rows = (
            await db.scalars(
                select(Feedback).where(Feedback.message_id.in_([m.id for m in messages]))
            )
        ).all()
        feedback_by_message = {
            f.message_id: {"rating": f.rating, "comment": f.comment} for f in feedback_rows
        }
        return {
            "id": conversation.id,
            "title": conversation.title,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "text": m.text,
                    "sources": m.sources,
                    "generation_metadata": m.generation_metadata,
                    "feedback": feedback_by_message.get(m.id),
                }
                for m in messages
            ],
        }

    @app.put("/api/v1/chat/messages/{message_id}/feedback")
    async def set_feedback(message_id: str, data: FeedbackIn, user: CurrentUser, db: Session):
        message = await db.scalar(
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .join(KnowledgeBase, Conversation.knowledge_base_id == KnowledgeBase.id)
            .where(
                Message.id == message_id,
                Message.role == "assistant",
                KnowledgeBase.owner_id == user.id,
            )
            .with_for_update(of=Message)
        )
        if message is None:
            raise HTTPException(404, "Message not found")
        feedback = await db.get(Feedback, message_id)
        if feedback is None:
            feedback = Feedback(message_id=message_id, rating=data.rating, comment=data.comment)
            db.add(feedback)
        else:
            feedback.rating, feedback.comment = data.rating, data.comment
        await db.commit()
        return {"message_id": message_id, "rating": feedback.rating, "comment": feedback.comment}

    @app.post("/api/v1/chat/conversations/{conversation_id}/messages")
    async def send_message(
        conversation_id: str, data: MessageIn, request: Request, user: CurrentUser, db: Session
    ):
        conversation = await owned_conversation(conversation_id, user, db)
        previous = (
            await db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(6)
            )
        ).all()
        history = "\n".join(
            f"{message.role}: {message.text[:600]}" for message in reversed(previous)
        )[-4000:]
        result = await ask(
            AskIn(
                knowledge_base_id=conversation.knowledge_base_id,
                question=data.question,
                allow_provider_transfer=data.allow_provider_transfer,
                history=history,
                document_ids=data.document_ids,
            ),
            request,
            user,
            db,
        )
        for doc_id in sorted({source["document_id"] for source in result["sources"]}):
            current = await db.scalar(
                select(Document)
                .where(
                    Document.id == doc_id,
                    Document.knowledge_base_id == conversation.knowledge_base_id,
                    Document.status == "ready",
                )
                .with_for_update()
            )
            if current is None:
                raise HTTPException(409, "Source document changed while generating the answer")
        db.add_all(
            [
                Message(
                    conversation_id=conversation.id, role="user", text=data.question, sources=[]
                ),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    text=result["answer"],
                    sources=result["sources"],
                    generation_metadata={
                        "prompt_version": RAG_PROMPT_VERSION,
                        "model_name": result.get("model_name"),
                        "status": result["status"],
                        "processing_time_ms": result.get("processing_time_ms"),
                    },
                ),
            ]
        )
        await db.commit()
        return result

    @app.post("/api/v1/chat/conversations/{conversation_id}/stream")
    async def stream_message(
        conversation_id: str, data: MessageIn, request: Request, user: CurrentUser, db: Session
    ):
        from .rag_stream import stream_answer

        started = time.perf_counter()
        conversation = await owned_conversation(conversation_id, user, db)
        previous = (
            await db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(6)
            )
        ).all()
        history = "\n".join(
            f"{message.role}: {message.text[:600]}" for message in reversed(previous)
        )[-4000:]
        payload = AskIn(
            knowledge_base_id=conversation.knowledge_base_id,
            question=data.question,
            allow_provider_transfer=data.allow_provider_transfer,
            history=history,
            document_ids=data.document_ids,
        )
        provider, sources, context, _, sufficient = await prepare_rag(payload, request, user, db)
        if sufficient and not callable(getattr(provider, "stream", None)):
            raise HTTPException(503, "Configured LLM provider does not support streaming")
        question = (
            f"Prior conversation (untrusted context, not evidence):\n{history}\n\nCurrent question: {data.question}"
            if history
            else data.question
        )

        def frame(event):
            return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

        async def emit():
            async def events():
                if not sufficient:
                    yield {
                        "event": "final",
                        "data": {
                            "answer": "В доступных документах недостаточно информации, чтобы ответить на этот вопрос.",
                            "status": "insufficient_evidence",
                            "sources": [],
                        },
                    }
                    return
                try:
                    async with asyncio.timeout(120):
                        async with request.app.state.llm_semaphore:
                            async for event in stream_answer(
                                provider, question, context, sources, set(data.document_ids)
                            ):
                                yield event
                except TimeoutError:
                    yield {
                        "event": "error",
                        "data": {"message": "Generation timed out; discard provisional text."},
                    }

            async for event in events():
                if event["event"] == "final":
                    try:
                        result = event["data"]
                        result["prompt_version"] = RAG_PROMPT_VERSION
                        result["processing_time_ms"] = (time.perf_counter() - started) * 1000
                        result["model_name"] = (
                            getattr(provider, "model", settings.llm_model)
                            if isinstance(getattr(provider, "model", settings.llm_model), str)
                            else None
                        )
                        for doc_id in sorted(
                            {source["document_id"] for source in result["sources"]}
                        ):
                            current = await db.scalar(
                                select(Document)
                                .where(
                                    Document.id == doc_id,
                                    Document.knowledge_base_id == conversation.knowledge_base_id,
                                    Document.status == "ready",
                                )
                                .with_for_update()
                            )
                            if current is None:
                                raise ValueError("Source document changed")
                        db.add_all(
                            [
                                Message(
                                    conversation_id=conversation.id,
                                    role="user",
                                    text=data.question,
                                    sources=[],
                                ),
                                Message(
                                    conversation_id=conversation.id,
                                    role="assistant",
                                    text=result["answer"],
                                    sources=result["sources"],
                                    generation_metadata={
                                        "prompt_version": result["prompt_version"],
                                        "model_name": result["model_name"],
                                        "status": result["status"],
                                        "processing_time_ms": result["processing_time_ms"],
                                    },
                                ),
                            ]
                        )
                        await db.commit()
                    except Exception:
                        await db.rollback()
                        yield frame(
                            {
                                "event": "error",
                                "data": {
                                    "message": "Answer could not be saved; discard provisional text."
                                },
                            }
                        )
                        return
                yield frame(event)

        return StreamingResponse(
            emit(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/v1/documents/{document_id}/extract")
    async def extract_document(
        document_id: str, data: ExtractIn, request: Request, user: CurrentUser, db: Session
    ):
        from pydantic import ValidationError

        from .extraction import ExtractionResult, select_extraction_sources, validate_evidence
        from .llm import LLMError, OpenAICompatibleProvider
        from .rag import build_context

        item = await db.scalar(
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(Document.id == document_id, KnowledgeBase.owner_id == user.id)
        )
        if item is None:
            raise HTTPException(404, "Document not found")
        if not data.allow_provider_transfer:
            raise HTTPException(
                400,
                "Confirm that document excerpts and conversation history may be sent to the configured provider",
            )
        chunks = (
            await db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.position)
                .limit(500)
            )
        ).all()
        sources = select_extraction_sources(
            [{"id": c.id, "text": c.text, "page": c.page} for c in chunks]
        )
        context, ids = build_context(sources)
        if not ids:
            raise HTTPException(422, "No extractable document text")
        provider = getattr(request.app.state, "llm_provider", None)
        if provider is None:
            if not settings.llm_model:
                raise HTTPException(503, "LLM provider is not configured")
            provider = OpenAICompatibleProvider(
                settings.llm_base_url, settings.llm_api_key, settings.llm_model
            )
        try:
            async with request.app.state.llm_semaphore:
                raw = await provider.extract(context)
            result = ExtractionResult.model_validate_json(raw)
            validate_evidence(result, ids)
        except (LLMError, ValidationError, ValueError):
            raise HTTPException(502, "Provider returned invalid extraction or evidence")
        return {
            "data": result.model_dump(),
            "sources": [source for source in sources if source["id"] in ids],
            "note": "Sections selected heuristically from up to 500 chunks; source membership checked, factual support requires review.",
        }

    @app.get("/api/v1/chunks/{chunk_id}")
    async def get_chunk(chunk_id: str, user: CurrentUser, db: Session):
        chunk = await db.scalar(
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(DocumentChunk.id == chunk_id, KnowledgeBase.owner_id == user.id)
        )
        if chunk is None:
            raise HTTPException(404, "Chunk not found")
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "position": chunk.position,
            "page": chunk.page,
            "text": chunk.text,
        }

    @app.delete("/api/v1/documents/{document_id}", status_code=204)
    async def delete_document(document_id: str, user: CurrentUser, db: Session):
        item = await db.scalar(
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(Document.id == document_id, KnowledgeBase.owner_id == user.id)
            .with_for_update(of=Document)
        )
        if item is None:
            raise HTTPException(404, "Document not found")
        item.status = "deleting"
        key = item.storage_key
        await db.commit()
        try:
            if str(UUID(key)) != key:
                raise ValueError("Invalid storage key")
            await asyncio.to_thread(
                (Path(settings.document_storage_path) / key).unlink, missing_ok=True
            )
        except (OSError, ValueError):
            raise HTTPException(503, "File deletion failed; retry deletion")
        messages = (
            await db.scalars(
                select(Message)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(
                    Conversation.knowledge_base_id == item.knowledge_base_id,
                    Message.role == "assistant",
                )
            )
        ).all()
        for message in messages:
            if any(source.get("document_id") == document_id for source in message.sources):
                await db.delete(message)
        await db.delete(item)
        await db.commit()
        return Response(status_code=204)

    @app.post("/api/v1/documents/{document_id}/cancel")
    async def cancel_document_job(document_id: str, user: CurrentUser, db: Session):
        document = await db.scalar(
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(Document.id == document_id, KnowledgeBase.owner_id == user.id)
        )
        if document is None:
            raise HTTPException(404, "Document not found")
        if document.status == "cancelled":
            return {"id": document_id, "status": "cancelled"}
        result = await db.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.document_id == document_id,
                ProcessingJob.status.in_(["pending", "processing"]),
            )
            .values(status="cancelled", lease_token=None, lease_until=None)
            .returning(ProcessingJob.id)
        )
        if result.first() is None:
            await db.rollback()
            raise HTTPException(409, "No active indexing job")
        await db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status="cancelled", error="Processing cancelled")
        )
        await db.commit()
        return {"id": document_id, "status": "cancelled"}

    @app.post("/api/v1/documents/{document_id}/reindex", status_code=202)
    async def reindex_document(document_id: str, user: CurrentUser, db: Session):
        item = await db.scalar(
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(Document.id == document_id, KnowledgeBase.owner_id == user.id)
            .with_for_update(of=Document)
        )
        if item is None:
            raise HTTPException(404, "Document not found")
        if item.status in {"pending", "processing", "deleting"}:
            raise HTTPException(409, "Document already has an active operation")
        item.status, item.error = "pending", None
        job = ProcessingJob(document_id=item.id, status="pending")
        db.add(job)
        await db.commit()
        return {"document_id": item.id, "job_id": job.id, "status": item.status}

    @app.get("/api/v1/documents/{document_id}/status")
    async def document_status(document_id: str, user: CurrentUser, db: Session):
        item = await db.scalar(
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(Document.id == document_id, KnowledgeBase.owner_id == user.id)
        )
        if item is None:
            raise HTTPException(404, "Document not found")
        job = await db.scalar(
            select(ProcessingJob)
            .where(ProcessingJob.document_id == item.id)
            .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
            .limit(1)
        )
        return {
            "document_id": item.id,
            "status": item.status,
            "error": item.error,
            "job_id": job.id if job else None,
            "attempts": job.attempts if job else 0,
        }

    @app.get("/api/v1/documents/{document_id}/file")
    async def document_file(document_id: str, user: CurrentUser, db: Session):
        item = await db.scalar(
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(
                Document.id == document_id,
                KnowledgeBase.owner_id == user.id,
                Document.status != "deleting",
            )
        )
        if item is None:
            raise HTTPException(404, "Document not found")
        try:
            if str(UUID(item.storage_key)) != item.storage_key:
                raise ValueError("Invalid key")
            path = Path(settings.document_storage_path) / item.storage_key
            if not path.is_file():
                raise FileNotFoundError
        except (ValueError, OSError):
            raise HTTPException(404, "Document file not found")
        return FileResponse(
            path,
            filename=item.filename,
            media_type="application/pdf" if item.format == "pdf" else "application/octet-stream",
            content_disposition_type="inline" if item.format == "pdf" else "attachment",
            headers={"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"},
        )

    @app.get("/api/v1/settings")
    async def get_settings(user: CurrentUser):
        return {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "reranker_model": settings.reranker_model,
            "reranker_revision": settings.reranker_revision,
            "llm_model": settings.llm_model or None,
            "llm_configured": bool(settings.llm_model),
            "max_upload_size": settings.max_upload_size,
            "max_chunks_per_document": settings.max_chunks_per_document,
            "note": "Configuration is controlled by environment variables. Reindex existing documents after changing chunk or embedding settings.",
        }

    @app.get("/api/v1/analytics/overview")
    async def analytics_overview(user: CurrentUser, db: Session):
        bases = await db.scalar(
            select(func.count(KnowledgeBase.id)).where(KnowledgeBase.owner_id == user.id)
        )
        docs = await db.scalar(
            select(func.count(Document.id))
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(KnowledgeBase.owner_id == user.id)
        )
        questions = await db.scalar(
            select(func.count(Message.id))
            .join(Conversation, Message.conversation_id == Conversation.id)
            .join(KnowledgeBase, Conversation.knowledge_base_id == KnowledgeBase.id)
            .where(KnowledgeBase.owner_id == user.id, Message.role == "user")
        )
        grouped = await db.execute(
            select(Document.status, func.count(Document.id))
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(KnowledgeBase.owner_id == user.id)
            .group_by(Document.status)
        )
        ratings = await db.execute(
            select(Feedback.rating, func.count(Feedback.message_id))
            .join(Message, Feedback.message_id == Message.id)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .join(KnowledgeBase, Conversation.knowledge_base_id == KnowledgeBase.id)
            .where(KnowledgeBase.owner_id == user.id)
            .group_by(Feedback.rating)
        )
        rating_counts = dict(ratings.all())
        return {
            "feedback": {"helpful": rating_counts.get(1, 0), "unhelpful": rating_counts.get(-1, 0)},
            "knowledge_bases": bases,
            "documents": docs,
            "questions": questions,
            "processing_status": {state: count for state, count in grouped},
        }

    @app.get("/api/v1/analytics/retrieval")
    async def analytics_retrieval(user: CurrentUser, db: Session):
        rows = await db.execute(
            select(
                SearchEvent.strategy, func.count(SearchEvent.id), func.sum(SearchEvent.result_count)
            )
            .where(SearchEvent.user_id == user.id)
            .group_by(SearchEvent.strategy)
        )
        return {
            "strategies": [
                {"strategy": strategy, "count": count, "total_results": total}
                for strategy, count, total in rows
            ]
        }

    @app.get("/api/v1/analytics/latency")
    async def analytics_latency(user: CurrentUser, db: Session):
        rows = await db.execute(
            select(
                SearchEvent.strategy, func.count(SearchEvent.id), func.avg(SearchEvent.latency_ms)
            )
            .where(SearchEvent.user_id == user.id)
            .group_by(SearchEvent.strategy)
        )
        return {
            "strategies": [
                {"strategy": strategy, "count": count, "mean_latency_ms": average}
                for strategy, count, average in rows
            ]
        }

    @app.post("/api/v1/evaluation/datasets", status_code=201)
    async def create_dataset(data: DatasetIn, user: CurrentUser, db: Session):
        await owned(data.knowledge_base_id, user, db)
        if len({q.id for q in data.queries}) != len(data.queries):
            raise HTTPException(422, "Query IDs must be unique")
        if any(q.answerable != bool(q.relevant_ids) for q in data.queries):
            raise HTTPException(
                422,
                "Answerable queries require relevance labels; unanswerable queries must have none",
            )
        ids = {id_ for q in data.queries for id_ in q.relevant_ids}
        if ids:
            matching = set(
                (
                    await db.scalars(
                        select(DocumentChunk.id)
                        .join(Document, DocumentChunk.document_id == Document.id)
                        .where(
                            Document.knowledge_base_id == data.knowledge_base_id,
                            DocumentChunk.id.in_(ids),
                        )
                    )
                ).all()
            )
            if matching != ids:
                raise HTTPException(
                    422, "A relevance label is missing or belongs to another knowledge base"
                )
        queries = [q.model_dump() for q in data.queries]
        version = hashlib.sha256(
            json.dumps(queries, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        dataset = EvaluationDataset(
            knowledge_base_id=data.knowledge_base_id,
            name=data.name,
            version=version,
            queries=queries,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        return {
            "id": dataset.id,
            "name": dataset.name,
            "version": dataset.version,
            "query_count": len(queries),
        }

    @app.get("/api/v1/evaluation/datasets")
    async def evaluation_datasets(user: CurrentUser, db: Session):
        rows = (
            await db.scalars(
                select(EvaluationDataset)
                .join(KnowledgeBase, EvaluationDataset.knowledge_base_id == KnowledgeBase.id)
                .where(KnowledgeBase.owner_id == user.id)
                .order_by(EvaluationDataset.created_at.desc())
                .limit(100)
            )
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "query_count": len(row.queries),
                "knowledge_base_id": row.knowledge_base_id,
            }
            for row in rows
        ]

    @app.post("/api/v1/evaluation/runs", status_code=202)
    async def create_evaluation_run(data: EvaluationRunIn, user: CurrentUser, db: Session):
        dataset = await db.scalar(
            select(EvaluationDataset)
            .join(KnowledgeBase, EvaluationDataset.knowledge_base_id == KnowledgeBase.id)
            .where(EvaluationDataset.id == data.dataset_id, KnowledgeBase.owner_id == user.id)
        )
        if dataset is None:
            raise HTTPException(404, "Evaluation dataset not found")
        if len(set(data.strategies)) != len(data.strategies):
            raise HTTPException(422, "Duplicate strategies")
        run = EvaluationRun(
            dataset_id=dataset.id,
            user_id=user.id,
            strategies=data.strategies,
            top_k=data.top_k,
            status="pending",
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        return {"id": run.id, "status": run.status}

    @app.get("/api/v1/evaluation/runs")
    async def evaluation_runs(user: CurrentUser, db: Session):
        rows = (
            await db.scalars(
                select(EvaluationRun)
                .where(EvaluationRun.user_id == user.id)
                .order_by(EvaluationRun.created_at.desc())
                .limit(100)
            )
        ).all()
        return [
            {
                "id": row.id,
                "dataset_id": row.dataset_id,
                "status": row.status,
                "strategies": row.strategies,
                "error": row.error,
            }
            for row in rows
        ]

    @app.get("/api/v1/evaluation/runs/{run_id}")
    async def evaluation_run(run_id: str, user: CurrentUser, db: Session):
        run = await db.scalar(
            select(EvaluationRun).where(
                EvaluationRun.id == run_id, EvaluationRun.user_id == user.id
            )
        )
        if run is None:
            raise HTTPException(404, "Evaluation run not found")
        return {
            "id": run.id,
            "dataset_id": run.dataset_id,
            "status": run.status,
            "strategies": run.strategies,
            "top_k": run.top_k,
            "error": run.error,
            "result": run.result,
        }

    @app.post("/api/v1/evaluation/runs/{run_id}/cancel")
    async def cancel_evaluation(run_id: str, user: CurrentUser, db: Session):
        run = await db.scalar(
            select(EvaluationRun)
            .where(EvaluationRun.id == run_id, EvaluationRun.user_id == user.id)
            .with_for_update()
        )
        if run is None:
            raise HTTPException(404, "Evaluation run not found")
        if run.status == "cancelled":
            return {"id": run.id, "status": run.status}
        if run.status not in {"pending", "processing"}:
            raise HTTPException(409, "Only pending or running experiments can be cancelled")
        run.status, run.lease_token, run.lease_until = "cancelled", None, None
        await db.commit()
        return {"id": run.id, "status": run.status}

    @app.get("/api/v1/evaluation/runs/{run_id}/results")
    async def evaluation_results(run_id: str, user: CurrentUser, db: Session):
        run = await evaluation_run(run_id, user, db)
        if run["status"] != "complete":
            raise HTTPException(409, "Evaluation has not completed")
        return run["result"]

    @app.get("/api/v1/evaluation/runs/{run_id}/export")
    async def evaluation_export(run_id: str, user: CurrentUser, db: Session):
        return await evaluation_results(run_id, user, db)

    return app
