"""Make signup invites multi-use with a configurable cap + redemption log.

Adds ``signup_invites.max_uses`` (NULL = unlimited; a positive integer caps the
number of accounts the link can create) and a new ``signup_invite_redemptions``
table — one row per account created from a link.

Existing single-use semantics are preserved on read: the legacy
``accepted_at`` / ``accepted_by_user_id`` columns are kept and continue to be
stamped on the *first* redemption (so old rows and old queries still make
sense). Already-accepted invites are backfilled into a redemption row so the
admin "created accounts" count is correct for historical links. Existing rows
get ``max_uses = 1`` (they were single-use); the column default for brand-new
rows is left to the application.

Revision ID: 0016_invite_multi_use
Revises: 0015_add_chat_session_api_key
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0017_invite_multi_use"
down_revision: Union[str, None] = "0016_add_api_key_disabled"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Per-invite usage cap. NULL = unlimited. Existing rows were single-use, so
    # backfill them to 1 before relying on the column.
    op.add_column(
        "signup_invites",
        sa.Column("max_uses", sa.Integer(), nullable=True),
    )
    op.execute("UPDATE signup_invites SET max_uses = 1 WHERE max_uses IS NULL")

    # Redemption log: one row per account created from a link.
    op.create_table(
        "signup_invite_redemptions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "invite_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("signup_invites.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # SET NULL so deleting a user doesn't erase the redemption count.
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "redeemed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # Count + batch-resolve redemptions per invite without a full scan.
    op.create_index(
        "ix_signup_invite_redemptions_invite_id",
        "signup_invite_redemptions",
        ["invite_id"],
    )

    # Backfill: every already-accepted invite becomes one redemption row so the
    # admin count/emails are accurate for historical links.
    op.execute(
        """
        INSERT INTO signup_invite_redemptions (invite_id, user_id, redeemed_at)
        SELECT id, accepted_by_user_id, accepted_at
        FROM signup_invites
        WHERE accepted_at IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_index(
        "ix_signup_invite_redemptions_invite_id",
        table_name="signup_invite_redemptions",
    )
    op.drop_table("signup_invite_redemptions")
    op.drop_column("signup_invites", "max_uses")
