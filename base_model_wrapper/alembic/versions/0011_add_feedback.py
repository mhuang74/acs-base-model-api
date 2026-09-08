"""Add feedback + feedback_screenshots — in-app user feedback with triage.

Gives logged-in users a low-friction floating button + modal to send feedback
(used heavily during conferences) and admins a ``/admin/feedback`` triage page.
Screenshots are stored inline as ``bytea`` because this app has no object
storage and runs single-replica; the route caps each image at ~2 MB and at
most 5 per submission.

``feedback.user_id`` is nullable (anonymous submissions, or the submitter's
account later deleted via SET NULL). CHECK constraints pin ``category`` to
bug/feature/general and ``status`` to open/resolved so a bad insert fails
loudly rather than silently storing garbage.

Revision ID: 0011_add_feedback
Revises: 0010_add_chat_generations
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_add_feedback"
down_revision: Union[str, None] = "0010_add_chat_generations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback",
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
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "is_anonymous",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("page_path", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "resolved_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("admin_notes", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "category IN ('bug', 'feature', 'general')",
            name="ck_feedback_category",
        ),
        sa.CheckConstraint(
            "status IN ('open', 'resolved')",
            name="ck_feedback_status",
        ),
    )
    # Triage list filters on status/category and sorts newest-first; the
    # admin-side join back to the submitting user uses user_id.
    op.create_index("ix_feedback_status", "feedback", ["status"])
    op.create_index("ix_feedback_created_at", "feedback", ["created_at"])
    op.create_index("ix_feedback_user_id", "feedback", ["user_id"])

    op.create_table(
        "feedback_screenshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "feedback_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("feedback.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("image_bytes", postgresql.BYTEA(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # Loading a feedback row's thumbnails fetches its screenshots by feedback_id.
    op.create_index(
        "ix_feedback_screenshots_feedback_id",
        "feedback_screenshots",
        ["feedback_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_feedback_screenshots_feedback_id", table_name="feedback_screenshots"
    )
    op.drop_table("feedback_screenshots")
    op.drop_index("ix_feedback_user_id", table_name="feedback")
    op.drop_index("ix_feedback_created_at", table_name="feedback")
    op.drop_index("ix_feedback_status", table_name="feedback")
    op.drop_table("feedback")
