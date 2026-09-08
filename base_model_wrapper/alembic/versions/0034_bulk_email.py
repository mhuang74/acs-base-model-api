"""Admin bulk email: batches + items, opt-outs, update subscribers (ACS-228).

Four additive tables:

- ``bulk_email_batches`` / ``bulk_email_items`` — one CSV-uploaded campaign
  and its per-recipient, fully-addressed emails (draft → preview/edit →
  send-now or scheduled). Items link to ``email_logs`` once a send is
  attempted, so the existing /admin/emails audit covers bulk mail too.
- ``email_optouts`` — addresses that clicked the one-click unsubscribe link
  every bulk email carries; suppressed from all future bulk sends.
- ``update_subscribers`` — addresses from the public landing-page
  "subscribe to updates" field (consent timestamp = created_at).

Additive + reversible; no existing rows touched.

Revision ID: 0034_bulk_email
Revises: 0033_signup_source
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0034_bulk_email"
down_revision: Union[str, None] = "0033_signup_source"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "bulk_email_batches",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_by_user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="draft"),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            index=True,
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'scheduled', 'sending', 'sent', 'canceled')",
            name="ck_bulk_email_batches_status",
        ),
    )
    op.create_table(
        "bulk_email_items",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "batch_id",
            UUID(as_uuid=True),
            sa.ForeignKey("bulk_email_batches.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("from_email", sa.Text(), nullable=False),
        sa.Column("to_email", sa.Text(), nullable=False),
        sa.Column("cc", sa.Text(), nullable=True),
        sa.Column("bcc", sa.Text(), nullable=True),
        sa.Column("reply_to", sa.Text(), nullable=True),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("body_html", sa.Text(), nullable=False),
        sa.Column("body_text", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "email_log_id",
            UUID(as_uuid=True),
            sa.ForeignKey("email_logs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'suppressed', 'sent', 'failed', 'skipped')",
            name="ck_bulk_email_items_status",
        ),
    )
    op.create_table(
        "email_optouts",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("source", sa.Text(), nullable=False, server_default="link"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_table(
        "update_subscribers",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("source", sa.Text(), nullable=False, server_default="landing"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("update_subscribers")
    op.drop_table("email_optouts")
    op.drop_table("bulk_email_items")
    op.drop_table("bulk_email_batches")
