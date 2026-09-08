"""Add chat_sessions.model — persists the model id chosen for each workbench session.

Nullable so existing rows backfill cleanly; the UI falls back to
``settings.default_model_id`` when the column is NULL.

Revision ID: 0006_add_chat_session_model
Revises: 0005_add_user_lifecycle
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_add_chat_session_model"
down_revision: Union[str, None] = "0005_add_user_lifecycle"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("model", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "model")
