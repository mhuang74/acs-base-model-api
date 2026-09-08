"""Add chat_sessions.api_key_id — persists the API key chosen for each workbench chat.

Nullable FK to api_keys with ON DELETE SET NULL: existing rows backfill cleanly
and the route falls back to the user's newest active key when NULL. Deleting a
key reverts any chats pointing at it to that default rather than orphaning them.

Revision ID: 0013_add_chat_session_api_key
Revises: 0012_add_email_logs
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0015_add_chat_session_api_key"
down_revision: Union[str, None] = "0014_add_signup_invites"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_chat_sessions_api_key_id",
        "chat_sessions",
        "api_keys",
        ["api_key_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_chat_sessions_api_key_id", "chat_sessions", type_="foreignkey"
    )
    op.drop_column("chat_sessions", "api_key_id")
