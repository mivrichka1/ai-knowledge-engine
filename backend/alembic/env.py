import asyncio
import os

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from ake_backend.db import Base
from alembic import context

config = context.config
url = os.environ.get("DATABASE_URL")
if url:
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
target_metadata = Base.metadata


def offline():
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def include_object(obj, name, type_, reflected, compare_to):
    # This PostgreSQL expression index is managed explicitly by migration 0004.
    return not (
        type_ == "index" and name in {"ix_document_chunks_fts", "ix_document_chunks_embedding_hnsw"}
    )


def migrate(connection):
    context.configure(
        connection=connection, target_metadata=target_metadata, include_object=include_object
    )
    with context.begin_transaction():
        context.run_migrations()


async def online():
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with engine.connect() as connection:
        await connection.run_sync(migrate)
    await engine.dispose()


if context.is_offline_mode():
    offline()
else:
    asyncio.run(online())
