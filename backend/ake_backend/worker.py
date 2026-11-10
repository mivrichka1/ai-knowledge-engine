"""PostgreSQL-backed parsing worker; parsed does not mean embedded/indexed."""

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .config import Settings
from .db import Document, DocumentChunk, ProcessingJob
from .embeddings import EmbeddingInputTooLong, SentenceTransformerProvider, validate_vectors
from .parsers import PDF_PARSER_VERSION, ParseError, parse_document
from .text import ChunkLimitError, TextNormalizer, fixed_chunks, tokenizer_chunks


async def process_one(
    maker,
    storage_path: str,
    provider=None,
    chunk_size: int = 120,
    chunk_overlap: int = 16,
    max_chunks: int = 5000,
) -> bool:
    now = datetime.now(UTC)
    started = time.perf_counter()
    token = str(uuid4())
    async with maker() as session:
        async with session.begin():
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    or_(
                        (ProcessingJob.status == "pending")
                        & or_(
                            ProcessingJob.lease_until.is_(None), ProcessingJob.lease_until <= now
                        ),
                        (ProcessingJob.status == "processing") & (ProcessingJob.lease_until < now),
                    ),
                )
                .order_by(ProcessingJob.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if not job:
                return False
            document = await session.get(Document, job.document_id, with_for_update=True)
            if document is None or document.status == "deleting":
                job.status = "cancelled"
                return True
            if job.attempts >= 3:
                job.status = "failed"
                if document:
                    document.status, document.error = "failed", "Processing attempts exhausted"
                return True
            job.status = "processing"
            job.attempts += 1
            job.lease_token = token
            job.lease_until = now + timedelta(minutes=5)
            job_id, document_id = job.id, job.document_id
            document.status = "processing"
            storage_key, kind, expected_hash, expected_size = (
                document.storage_key,
                document.format,
                document.sha256,
                document.size,
            )
    try:
        # Storage keys are generated UUIDs, never names supplied by users.
        if str(UUID(storage_key)) != storage_key:
            raise ParseError("Invalid storage key")
        path = Path(storage_path) / storage_key
        if path.stat().st_size != expected_size:
            raise ParseError("Stored file integrity check failed")
        data = await asyncio.to_thread(path.read_bytes)
        if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_hash:
            raise ParseError("Stored file integrity check failed")
        pages = await asyncio.to_thread(parse_document, data, kind)
        tokenizer = getattr(provider, "tokenizer", None)
        if tokenizer is not None and getattr(provider, "model", None) is not None:
            special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
            if chunk_size + special_tokens > provider.model.max_seq_length:
                raise ParseError("Configured chunk size exceeds the embedding model context")
        rows: list[DocumentChunk] = []
        for page in pages:
            text = TextNormalizer().normalize(page.text)
            if not text:
                continue
            if len(rows) >= max_chunks:
                raise ParseError("Document chunk limit exceeded")
            tokenizer = getattr(provider, "tokenizer", None)
            try:
                chunks = (
                    tokenizer_chunks(
                        text,
                        tokenizer,
                        size=chunk_size,
                        overlap=chunk_overlap,
                        max_chunks=max_chunks - len(rows),
                    )
                    if tokenizer
                    else fixed_chunks(text)
                )
            except ChunkLimitError as exc:
                raise ParseError("Document chunk limit exceeded") from exc
            for chunk in chunks:
                if len(rows) >= max_chunks:
                    raise ParseError("Document chunk limit exceeded")
                rows.append(
                    DocumentChunk(
                        id=str(
                            uuid5(
                                UUID(document_id),
                                f"{page.page}:{chunk.start}:{chunk.end}:{hashlib.sha256(chunk.text.encode()).hexdigest()}",
                            )
                        ),
                        document_id=document_id,
                        position=len(rows),
                        page=page.page,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        start_offset=chunk.start,
                        end_offset=chunk.end,
                    )
                )
        if not rows:
            raise ParseError("No extractable text")
        if provider is not None:
            vectors = await asyncio.to_thread(provider.encode, [row.text for row in rows])
            validate_vectors(vectors, len(rows), 384)
            for row, vector in zip(rows, vectors):
                row.embedding, row.embedding_model, row.embedding_revision = (
                    vector,
                    provider.model_name,
                    provider.revision,
                )
        async with maker() as session:
            async with session.begin():
                job = await session.scalar(
                    select(ProcessingJob).where(ProcessingJob.id == job_id).with_for_update()
                )
                if not job or job.lease_token != token or job.status != "processing":
                    return True
                document = await session.get(Document, document_id, with_for_update=True)
                if document is None or document.status == "deleting":
                    job.status = "cancelled"
                    return True
                await session.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
                )
                session.add_all(rows)
                document = await session.get(Document, document_id)
                if document is not None:
                    document.status, document.error = (
                        ("ready" if provider is not None else "parsed"),
                        None,
                    )
                    document.index_version += 1
                    uses_model_tokens = (
                        provider is not None and getattr(provider, "tokenizer", None) is not None
                    )
                    document.index_metadata = {
                        "page_count": len(pages) if kind == "pdf" else None,
                        "parser_version": PDF_PARSER_VERSION if kind == "pdf" else None,
                        "chunk_strategy": "model_tokenizer"
                        if uses_model_tokens
                        else "fixed_whitespace",
                        "chunk_size": chunk_size if uses_model_tokens else 512,
                        "chunk_overlap": chunk_overlap if uses_model_tokens else 64,
                        "embedding_model": provider.model_name if provider is not None else None,
                        "embedding_revision": provider.revision if provider is not None else None,
                    }
                job.status, job.lease_token, job.lease_until = "complete", None, None
        logging.getLogger("ake.worker").info(
            json.dumps(
                {
                    "event": "index_complete",
                    "document_id": document_id,
                    "job_id": job_id,
                    "chunk_count": len(rows),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        )
    except Exception as exc:
        logging.getLogger("ake.worker").error(
            json.dumps(
                {
                    "event": "index_failed",
                    "document_id": document_id,
                    "job_id": job_id,
                    "error_type": type(exc).__name__,
                }
            )
        )
        async with maker() as session:
            async with session.begin():
                job = await session.scalar(
                    select(ProcessingJob).where(ProcessingJob.id == job_id).with_for_update()
                )
                if job and job.lease_token == token:
                    document = await session.get(Document, document_id)
                    # Malformed files are not transient; other errors get up to three attempts.
                    terminal = (
                        isinstance(exc, (ParseError, EmbeddingInputTooLong)) or job.attempts >= 3
                    )
                    job.status = "failed" if terminal else "pending"
                    job.lease_token = None
                    job.lease_until = (
                        None if terminal else datetime.now(UTC) + timedelta(seconds=2**job.attempts)
                    )
                    if document:
                        document.status = "failed" if terminal else "pending"
                        document.error = (
                            str(exc)
                            if isinstance(exc, (ParseError, EmbeddingInputTooLong))
                            else "Processing failed; see worker logs"
                        )
    return True


async def main():
    settings = Settings()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("ake").setLevel(getattr(logging, settings.log_level))
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    provider = await asyncio.to_thread(
        SentenceTransformerProvider, settings.embedding_model, 384, settings.embedding_revision
    )
    try:
        while True:
            if not await process_one(
                maker,
                settings.document_storage_path,
                provider,
                settings.chunk_size,
                settings.chunk_overlap,
                settings.max_chunks_per_document,
            ):
                from .evaluation_worker import process_evaluation

                if not await process_evaluation(maker, settings):
                    await asyncio.sleep(2)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
