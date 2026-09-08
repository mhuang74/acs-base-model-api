"""Add user lifecycle columns: status, approval timestamps, one-shot pending key.

Two related concerns:

- **Signup/approval queue**: ``status`` ('pending' | 'approved' | 'rejected')
  with audit columns ``approved_at``, ``approved_by_user_id``, ``rejected_at``.
  The column defaults to 'approved' so existing rows backfill without changing
  behaviour; new ``POST /signup`` rows opt in to 'pending' explicitly.

- **One-shot post-approval key delivery**: when an admin approves a signup we
  generate one API key, encrypt the plaintext (itsdangerous + session_secret)
  and stash it on ``pending_key_plaintext`` + ``pending_key_id``. The first
  ``/dashboard`` render after approval decrypts, shows the key once, and clears
  both columns in the same transaction. Approval email never carries the key.

Revision ID: 0005_add_user_lifecycle
Revises: 0004_add_chat_history
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_add_user_lifecycle"
down_revision: Union[str, None] = "0004_add_chat_history"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "status", sa.Text(), nullable=False, server_default="approved"
        ),
    )
    op.add_column(
        "users",
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column(
            "approved_by_user_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "users_approved_by_user_id_fkey",
        "users",
        "users",
        ["approved_by_user_id"],
        ["id"],
        ondelete="SET NULL",
        use_alter=True,
    )
    op.add_column(
        "users",
        sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("pending_key_plaintext", sa.Text(), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column(
            "pending_key_id", postgresql.UUID(as_uuid=True), nullable=True
        ),
    )
    op.create_foreign_key(
        "users_pending_key_id_fkey",
        "users",
        "api_keys",
        ["pending_key_id"],
        ["id"],
        ondelete="SET NULL",
        use_alter=True,
    )


def downgrade() -> None:
    op.drop_constraint("users_pending_key_id_fkey", "users", type_="foreignkey")
    op.drop_column("users", "pending_key_id")
    op.drop_column("users", "pending_key_plaintext")
    op.drop_column("users", "rejected_at")
    op.drop_constraint(
        "users_approved_by_user_id_fkey", "users", type_="foreignkey"
    )
    op.drop_column("users", "approved_by_user_id")
    op.drop_column("users", "approved_at")
    op.drop_column("users", "status")
