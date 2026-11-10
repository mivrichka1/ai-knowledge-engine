"""Record actual indexing configuration for each document."""

import sqlalchemy as sa

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("documents", sa.Column("index_metadata", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("documents", "index_metadata")
