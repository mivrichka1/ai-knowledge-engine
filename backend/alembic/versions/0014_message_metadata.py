"""Record generation provenance for newly saved assistant messages."""

import sqlalchemy as sa

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("messages", sa.Column("generation_metadata", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("messages", "generation_metadata")
