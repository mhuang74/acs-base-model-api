"""Add email_logs + email_events — full transactional-email audit + webhooks.

Every send attempt (approve/reject/…) lands one row in ``email_logs``,
including suppressed sends (email disabled / no API key) and failures, with the
full rendered HTML + text body retained. Resend delivery webhooks land one row
each in ``email_events``, keeping the entire verified payload as JSONB, joined
back to the send by ``resend_message_id`` (and FK when matched).

CHECK on ``send_status`` pins it to sent/skipped/failed so a bad insert fails
loudly. ``email_events.email_log_id`` is nullable (a webhook may arrive for a
message id we have no log row for) with ON DELETE CASCADE so dropping a log row
takes its events with it.

Revision ID: 0012_add_email_logs
Revises: 0011_add_feedback
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_add_email_logs"
down_revision: Union[str, None] = "0011_add_feedback"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "email_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("to_email", sa.Text(), nullable=False),
        sa.Column("from_email", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("body_html", sa.Text(), nullable=False),
        sa.Column("body_text", sa.Text(), nullable=False),
        sa.Column("send_status", sa.Text(), nullable=False),
        sa.Column("skip_reason", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("resend_message_id", sa.Text(), nullable=True),
        sa.Column("last_event", sa.Text(), nullable=True),
        sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "send_status IN ('sent', 'skipped', 'failed')",
            name="ck_email_logs_send_status",
        ),
    )
    # List view sorts newest-first; webhook lookup + admin per-user view join by
    # resend_message_id / user_id.
    op.create_index("ix_email_logs_created_at", "email_logs", ["created_at"])
    op.create_index(
        "ix_email_logs_resend_message_id", "email_logs", ["resend_message_id"]
    )
    op.create_index("ix_email_logs_user_id", "email_logs", ["user_id"])

    op.create_table(
        "email_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "email_log_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("email_logs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("resend_message_id", sa.Text(), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # Webhook matching joins by resend_message_id; the detail timeline loads a
    # log row's events by email_log_id; received_at orders the timeline.
    op.create_index(
        "ix_email_events_resend_message_id", "email_events", ["resend_message_id"]
    )
    op.create_index(
        "ix_email_events_email_log_id", "email_events", ["email_log_id"]
    )
    op.create_index("ix_email_events_received_at", "email_events", ["received_at"])


def downgrade() -> None:
    op.drop_index("ix_email_events_received_at", table_name="email_events")
    op.drop_index("ix_email_events_email_log_id", table_name="email_events")
    op.drop_index(
        "ix_email_events_resend_message_id", table_name="email_events"
    )
    op.drop_table("email_events")
    op.drop_index("ix_email_logs_user_id", table_name="email_logs")
    op.drop_index(
        "ix_email_logs_resend_message_id", table_name="email_logs"
    )
    op.drop_index("ix_email_logs_created_at", table_name="email_logs")
    op.drop_table("email_logs")
