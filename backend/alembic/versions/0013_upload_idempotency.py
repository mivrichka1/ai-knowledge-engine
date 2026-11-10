"""Make optional upload idempotency keys unique per knowledge base."""

import sqlalchemy as sa

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("documents") as batch:
        batch.add_column(sa.Column("idempotency_key", sa.String(128), nullable=True))
        batch.create_unique_constraint(
            "uq_documents_upload_key", ["knowledge_base_id", "idempotency_key"]
        )


def downgrade():
    with op.batch_alter_table("documents") as batch:
        batch.drop_constraint("uq_documents_upload_key", type_="unique")
        batch.drop_column("idempotency_key")
