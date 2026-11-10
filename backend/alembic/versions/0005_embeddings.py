"""Add model-versioned vector storage for document chunks."""

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    vector_type = Vector(384) if op.get_bind().dialect.name == "postgresql" else sa.Text()
    op.add_column("document_chunks", sa.Column("embedding", vector_type, nullable=True))
    op.add_column("document_chunks", sa.Column("embedding_model", sa.String(255), nullable=True))
    op.add_column("document_chunks", sa.Column("embedding_revision", sa.String(64), nullable=True))


def downgrade():
    op.drop_column("document_chunks", "embedding_revision")
    op.drop_column("document_chunks", "embedding_model")
    op.drop_column("document_chunks", "embedding")
