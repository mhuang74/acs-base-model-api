"""chat_sessions.last_temperature default 0.7 -> 1.0 (ACS-145).

Base models are faithfully sampled at temperature 1.0; 0.7 was a chat-tuned
default that biases the next-token distribution. New chats should start at the
neutral 1.0. Existing rows keep their stored value — this only changes the
column default applied to freshly-inserted chat sessions.

Revision ID: 0022_chat_temperature_default
Revises: 0021_admin_delete_user_cascade
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0022_chat_temperature_default"
down_revision: Union[str, None] = "0021_admin_delete_user_cascade"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "chat_sessions",
        "last_temperature",
        existing_type=sa.Float(),
        existing_nullable=False,
        server_default="1.0",
    )


def downgrade() -> None:
    op.alter_column(
        "chat_sessions",
        "last_temperature",
        existing_type=sa.Float(),
        existing_nullable=False,
        server_default="0.7",
    )
