"""Add signup_invites table for single-use invite tokens.

Admins create tokens (link-only or email-targeted) from /admin/users; each
token is single-use and expiring. The plaintext token is stored directly so
admins can re-copy the magic link from the admin table at any time.

Revision ID: 0013_add_signup_invites
Revises: 0012_add_email_logs
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014_add_signup_invites"
down_revision: Union[str, None] = "0013_add_password_resets"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "signup_invites",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Plaintext token — stored directly so admins can re-copy the link.
        # Unique index for fast lookup on the acceptance path.
        sa.Column("token", sa.Text(), nullable=False, unique=True),
        # NULL → link-only; non-NULL → intended recipient email (informational
        # only — does not restrict which email the invitee signs up with).
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column(
            "created_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "accepted_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Fast lookup by plaintext token on the acceptance path.
    op.create_index("ix_signup_invites_token", "signup_invites", ["token"], unique=True)
    # Admin list queries: newest-first, filter by email.
    op.create_index("ix_signup_invites_created_at", "signup_invites", ["created_at"])
    op.create_index("ix_signup_invites_email", "signup_invites", ["email"])


def downgrade() -> None:
    op.drop_index("ix_signup_invites_email", table_name="signup_invites")
    op.drop_index("ix_signup_invites_created_at", table_name="signup_invites")
    op.drop_index("ix_signup_invites_token", table_name="signup_invites")
    op.drop_table("signup_invites")
