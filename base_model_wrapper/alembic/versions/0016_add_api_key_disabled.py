"""Add api_keys.disabled_at — reversible "pause" distinct from revoke.

A key is usable iff ``revoked_at IS NULL AND disabled_at IS NULL``. ``revoked_at``
remains a permanent soft-delete (the trash control); ``disabled_at`` is a
reversible pause the owner can toggle on/off. Nullable with no default so every
existing key backfills as not-paused.

Revision ID: 0016_add_api_key_disabled
Revises: 0015_add_chat_session_api_key
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0016_add_api_key_disabled"
down_revision: Union[str, None] = "0015_add_chat_session_api_key"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "disabled_at")
