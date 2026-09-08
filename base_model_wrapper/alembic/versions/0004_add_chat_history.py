"""Add chat history: saved sessions + per-Continue snapshots.

Two new tables backing the workbench:

- ``chat_sessions`` — one row per saved workbench prompt. Stores the current
  rolling ``prompt_text`` plus the last-used generation params so a refresh
  picks up where the user left off.
- ``chat_snapshots`` — one row per ``Continue`` generation. ``prompt_before``
  is what was sent upstream; ``completion_text`` is the (possibly partial)
  reply. Reverting a session = copy ``prompt_before`` back to the parent
  session's ``prompt_text``.

This is the first time the wrapper persists prompt/completion text — all
prior tables store only token counts. See ``tutorial.html`` for the
user-facing note.

Revision ID: 0004_add_chat_history
Revises: 0003_add_limits_backend
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_add_chat_history"
down_revision: Union[str, None] = "0003_add_limits_backend"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False, server_default="Untitled"),
        sa.Column("prompt_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_max_tokens", sa.Integer(), nullable=False, server_default="200"),
        sa.Column("last_temperature", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Sidebar query: list this user's non-archived sessions, newest first.
    op.create_index(
        "ix_chat_sessions_user_updated",
        "chat_sessions",
        ["user_id", sa.text("updated_at DESC")],
        postgresql_where=sa.text("archived_at IS NULL"),
    )

    op.create_table(
        "chat_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("prompt_before", sa.Text(), nullable=False),
        sa.Column("completion_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("n_completion", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_tokens", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("cancelled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index(
        "ix_chat_snapshots_session_ts",
        "chat_snapshots",
        ["session_id", sa.text("ts DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_snapshots_session_ts", table_name="chat_snapshots")
    op.drop_table("chat_snapshots")
    op.drop_index("ix_chat_sessions_user_updated", table_name="chat_sessions")
    op.drop_table("chat_sessions")
