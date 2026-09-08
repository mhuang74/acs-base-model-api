"""Add chat_snapshots.model — persists the model id used for each Continue.

Nullable so existing rows backfill cleanly; the workbench falls back to
displaying just the timestamp/tokens when the column is NULL.

Revision ID: 0009_add_chat_snapshot_model
Revises: 0008_model_warm_window
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009_add_chat_snapshot_model"
down_revision: Union[str, None] = "0008_model_warm_window"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_snapshots",
        sa.Column("model", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_snapshots", "model")
