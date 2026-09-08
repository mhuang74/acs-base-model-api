"""Add chat_generations — durable record of an in-flight workbench Continue.

Decouples the generation lifecycle from the browser HTTP connection so a
closed tab does not abort the upstream work and a reopened tab can replay
the accumulated tokens and tail live. The streaming task flushes
``completion_text`` periodically; a wrapper restart marks any rows still
``running`` as ``failed`` on boot.

Revision ID: 0010_add_chat_generations
Revises: 0009_add_chat_snapshot_model
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_add_chat_generations"
down_revision: Union[str, None] = "0009_add_chat_snapshot_model"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_generations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("prompt_before", sa.Text(), nullable=False),
        sa.Column(
            "completion_text", sa.Text(), nullable=False, server_default=""
        ),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("max_tokens", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("n_prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("n_completion_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    # "Is there a running generation for this session?" lookup on every page
    # load and every POST /generations (concurrency check).
    op.create_index(
        "ix_chat_generations_session_status",
        "chat_generations",
        ["session_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chat_generations_session_status", table_name="chat_generations"
    )
    op.drop_table("chat_generations")
