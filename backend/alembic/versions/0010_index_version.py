"""Track atomic document index replacements for experiment provenance."""

import sqlalchemy as sa

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "documents", sa.Column("index_version", sa.Integer(), nullable=False, server_default="0")
    )
    if op.get_bind().dialect.name != "sqlite":
        op.alter_column("documents", "index_version", server_default=None)


def downgrade():
    op.drop_column("documents", "index_version")
