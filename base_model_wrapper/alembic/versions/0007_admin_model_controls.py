"""Add probe_schedule + probe_results tables for the editable capacity-probe cron.

The wrapper's APScheduler reads ``probe_schedule.cron_expression`` at startup
and on /admin edits. Seeds the one row with the previous Modal-side cron
expression so the schedule keeps firing on the same cadence after the
migration off ``modal.Cron``.

Revision ID: 0007_admin_model_controls
Revises: 0006_add_chat_session_model
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_admin_model_controls"
down_revision: Union[str, None] = "0006_add_chat_session_model"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "probe_schedule",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "cron_expression",
            sa.Text(),
            nullable=False,
            server_default="0 7,10,14,16,19,23 * * *",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    # Seed the single row so the scheduler has something to read on first boot.
    op.execute(
        "INSERT INTO probe_schedule (id, cron_expression) "
        "VALUES (1, '0 7,10,14,16,19,23 * * *')"
    )

    op.create_table(
        "probe_results",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "fired_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("ok", sa.Boolean(), nullable=False),
        sa.Column("elapsed_s", sa.Float(), nullable=True),
        sa.Column("gpu_type", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_probe_results_fired_at", "probe_results", ["fired_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_probe_results_fired_at", table_name="probe_results")
    op.drop_table("probe_results")
    op.drop_table("probe_schedule")
