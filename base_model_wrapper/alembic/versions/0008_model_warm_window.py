"""Add model_warm_window table for editable per-model warm/cool crons.

The wrapper's APScheduler reads enabled rows at startup (and on /admin
edits) and installs two CronTriggers per row — one for warm (sets
``min_containers=1``) and one for cool (sets it back to 0). Seeds the
Trinity row with the previous Modal-side ``modal.Cron`` expressions so
the schedule keeps firing on the same cadence after the migration off
``modal_app.py``.

Revision ID: 0008_model_warm_window
Revises: 0007_admin_model_controls
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_model_warm_window"
down_revision: Union[str, None] = "0007_admin_model_controls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_warm_window",
        sa.Column("model_id", sa.Text(), primary_key=True),
        sa.Column("warm_cron", sa.Text(), nullable=False),
        sa.Column("cool_cron", sa.Text(), nullable=False),
        sa.Column(
            "timezone",
            sa.Text(),
            nullable=False,
            server_default="UTC",
        ),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
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
    # Seed Trinity with the previous Modal-side cron expressions so the
    # schedule keeps firing on the same cadence after the migration.
    op.execute(
        "INSERT INTO model_warm_window "
        "(model_id, warm_cron, cool_cron, timezone, enabled) "
        "VALUES ('trinity-base', '15 8 * * 3-6,0', '0 17 * * 3-6,0', "
        "'America/Los_Angeles', TRUE)"
    )


def downgrade() -> None:
    op.drop_table("model_warm_window")
