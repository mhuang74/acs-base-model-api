"""Periodic per-model GPU cost samples (cost monitor).

Revision ID: 0020_gpu_cost_sample
Revises: 0019_api_request_event_fields
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0020_gpu_cost_sample"
down_revision: Union[str, None] = "0019_api_request_event_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gpu_cost_sample",
        sa.Column(
            "id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("gpu_type", sa.Text(), nullable=False),
        sa.Column("gpu_count", sa.Integer(), nullable=False),
        sa.Column("running_containers", sa.Integer(), nullable=False),
        sa.Column("hourly_usd_per_container", sa.Float(), nullable=False),
        sa.Column("period_seconds", sa.Integer(), nullable=False),
        sa.Column("est_usd", sa.Float(), nullable=False),
    )
    op.create_index("ix_gpu_cost_sample_ts", "gpu_cost_sample", ["ts"])


def downgrade() -> None:
    op.drop_index("ix_gpu_cost_sample_ts", table_name="gpu_cost_sample")
    op.drop_table("gpu_cost_sample")
