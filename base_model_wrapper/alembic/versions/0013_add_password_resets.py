"""Add password_resets — single-use, hashed, short-lived password-reset tokens.

Backs the self-service forgot-password flow. We persist only the SHA-256 hash
of each token (never plaintext), mirroring ``api_keys.key_hash``: lookup hashes
the presented token and matches ``token_hash`` (unique-indexed). ``used_at``
enforces single-use via a guarded UPDATE; ``expires_at`` enforces a short
window. ON DELETE CASCADE on the user FK drops a user's reset rows with the
account.

Revision ID: 0013_add_password_resets
Revises: 0012_add_email_logs
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_add_password_resets"
down_revision: Union[str, None] = "0012_add_email_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "password_resets",
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
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Lookup is by hashing the presented token and matching token_hash, so it
    # must be unique + indexed. user_id index speeds the "invalidate prior
    # outstanding tokens for this user" sweep on a fresh request.
    op.create_index(
        "ix_password_resets_token_hash", "password_resets", ["token_hash"], unique=True
    )
    op.create_index("ix_password_resets_user_id", "password_resets", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_password_resets_user_id", table_name="password_resets")
    op.drop_index("ix_password_resets_token_hash", table_name="password_resets")
    op.drop_table("password_resets")
