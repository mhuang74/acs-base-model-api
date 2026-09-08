"""Record Modal placement and GPU telemetry for probe results.

Revision ID: 0018_probe_result_gpu_telemetry
Revises: 0017_invite_multi_use
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0018_probe_result_gpu_telemetry"
down_revision: Union[str, None] = "0017_invite_multi_use"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("probe_results", sa.Column("cloud", sa.Text(), nullable=True))
    op.add_column("probe_results", sa.Column("region", sa.Text(), nullable=True))
    op.add_column("probe_results", sa.Column("gpu_count", sa.Integer(), nullable=True))
    op.add_column(
        "probe_results", sa.Column("gpu_memory_total_mb", sa.Float(), nullable=True)
    )
    op.add_column(
        "probe_results",
        sa.Column("gpu_memory_total_std_mb", sa.Float(), nullable=True),
    )
    op.add_column(
        "probe_results", sa.Column("gpu_memory_used_mb", sa.Float(), nullable=True)
    )
    op.add_column(
        "probe_results",
        sa.Column("gpu_memory_used_std_mb", sa.Float(), nullable=True),
    )
    op.add_column(
        "probe_results", sa.Column("gpu_utilization_pct", sa.Float(), nullable=True)
    )
    op.add_column(
        "probe_results",
        sa.Column("gpu_utilization_std_pct", sa.Float(), nullable=True),
    )
    op.add_column(
        "probe_results", sa.Column("gpu_temperature_c", sa.Float(), nullable=True)
    )
    op.add_column(
        "probe_results",
        sa.Column("gpu_temperature_std_c", sa.Float(), nullable=True),
    )
    op.add_column("probe_results", sa.Column("gpu_power_w", sa.Float(), nullable=True))
    op.add_column(
        "probe_results", sa.Column("gpu_power_std_w", sa.Float(), nullable=True)
    )
    op.add_column(
        "probe_results", sa.Column("driver_version", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("probe_results", "driver_version")
    op.drop_column("probe_results", "gpu_power_std_w")
    op.drop_column("probe_results", "gpu_power_w")
    op.drop_column("probe_results", "gpu_temperature_std_c")
    op.drop_column("probe_results", "gpu_temperature_c")
    op.drop_column("probe_results", "gpu_utilization_std_pct")
    op.drop_column("probe_results", "gpu_utilization_pct")
    op.drop_column("probe_results", "gpu_memory_used_std_mb")
    op.drop_column("probe_results", "gpu_memory_used_mb")
    op.drop_column("probe_results", "gpu_memory_total_std_mb")
    op.drop_column("probe_results", "gpu_memory_total_mb")
    op.drop_column("probe_results", "gpu_count")
    op.drop_column("probe_results", "region")
    op.drop_column("probe_results", "cloud")
