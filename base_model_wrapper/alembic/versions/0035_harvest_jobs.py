"""Self-serve bulk activation harvest: jobs table + per-key quota (ACS-245).

``POST /v1/harvest`` lets an API key spawn the offline bulk-harvest Modal
function (``acs-<model>-harvest``) itself and poll ``GET /v1/harvest/<id>``
for the manifest/shard URLs, instead of an operator ``modal run``.

Additive + reversible:
  * ``harvest_jobs`` — one row per spawned job. Doubles as the monthly quota
    tally (rows started this month) and the per-key running-job concurrency
    count. ``params``/``result`` hold request/response METADATA only (prompt
    count, layers, shard URLs, timings) — never prompt text or tensors.
  * ``api_keys.monthly_harvest_budget`` — BIGINT NOT NULL DEFAULT 0
    (0 = unlimited, mirroring ``monthly_activation_budget`` from 0031).

Revision ID: 0035_harvest_jobs
Revises: 0034_bulk_email
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0035_harvest_jobs"
down_revision: Union[str, None] = "0034_bulk_email"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "harvest_jobs",
        # UUID4 as text — an opaque API token clients present back, not a join
        # target needing the native UUID type.
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "key_id",
            UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("modal_call_id", sa.Text(), nullable=True),
        # 'pending' = row inserted inside the quota-gate transaction, Modal
        # spawn not yet confirmed (flips to 'running' or 'failed' right after).
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("params", JSONB(), nullable=False),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # Terminal timestamp; anchors the presigned-URL freshness window.
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed')",
            name="ck_harvest_jobs_status",
        ),
    )
    op.add_column(
        "api_keys",
        sa.Column(
            "monthly_harvest_budget",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "monthly_harvest_budget")
    op.drop_table("harvest_jobs")
