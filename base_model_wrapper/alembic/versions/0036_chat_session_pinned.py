"""Sidebar chat pinning: ``chat_sessions.pinned_at`` (ACS-257).

Pinned chats sort above the recency-ordered rest of the workbench sidebar.
Nullable timestamp; NULL = not pinned. Additive + reversible.

Revision ID: 0036_chat_session_pinned
Revises: 0035_harvest_jobs
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0036_chat_session_pinned"
down_revision: Union[str, None] = "0035_harvest_jobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("pinned_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "pinned_at")
