"""Server-side compare-snapshot barrier: compare_run_id + per-lane config (ACS-186).

ACS-180 (#158) added ``compare_snapshots`` but the snapshot was assembled and
POSTed by the *browser* after Run-all settled — closing the tab mid-run left no
snapshot. ACS-186 moves the write server-side: give each Compare "Run all" a
shared ``compare_run_id`` (UUID) stamped on its N lane ``chat_generations`` rows,
so the last lane to reach a terminal state can detect "whole batch done" and
assemble the ``CompareSnapshot`` from the persisted lane data.

Additive only, three changes:

1. ``chat_generations.compare_run_id`` (nullable UUID, indexed) — the batch id.
   NULL for single-pane Continue + loom generations.
2. ``chat_generations.compare_config`` (nullable JSONB) — the clamped per-lane
   vLLM request body + lane index, so the barrier can rebuild the snapshot lane
   faithfully (top_p/top_k/min_p/penalties/seed/stop aren't on the base columns).
3. ``compare_snapshots.compare_run_id`` (nullable UUID, UNIQUE) — the dedup
   point. Postgres allows multiple NULLs, so legacy #158 rows don't collide; the
   UNIQUE constraint guarantees exactly one snapshot per batch even when the
   server barrier and a racing client POST both fire.

Downgrade drops all three (index, unique constraint, columns) — fully reversible.

Revision ID: 0029_compare_run_barrier
Revises: 0028_add_compare_snapshots
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0029_compare_run_barrier"
down_revision: Union[str, None] = "0028_add_compare_snapshots"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_generations",
        sa.Column("compare_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "chat_generations",
        sa.Column("compare_config", postgresql.JSONB(), nullable=True),
    )
    op.create_index(
        "ix_chat_generations_compare_run_id",
        "chat_generations",
        ["compare_run_id"],
    )
    op.add_column(
        "compare_snapshots",
        sa.Column("compare_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_unique_constraint(
        "uq_compare_snapshots_compare_run_id",
        "compare_snapshots",
        ["compare_run_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_compare_snapshots_compare_run_id",
        "compare_snapshots",
        type_="unique",
    )
    op.drop_column("compare_snapshots", "compare_run_id")
    op.drop_index(
        "ix_chat_generations_compare_run_id",
        table_name="chat_generations",
    )
    op.drop_column("chat_generations", "compare_config")
    op.drop_column("chat_generations", "compare_run_id")
