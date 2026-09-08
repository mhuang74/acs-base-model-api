"""Add loom_nodes — workbench tree/branching exploration nodes (ACS-148).

A *loom* hangs off a ``chat_sessions`` row (the owner-scoped container the
workbench already persists), so ownership + the sidebar reuse the existing chat
plumbing. Each ``loom_nodes`` row is one continuation; ``parent_id`` is a
self-FK to the node it branched from (NULL for a root, whose ``text`` is the
seed prompt). ``logprobs`` (JSONB) carries the per-token heatmap payload and
``seed`` pins the sampler seed for reproducibility. Additive + backward
compatible: no existing table is touched.

Revision ID: 0025_add_loom_nodes
Revises: 0024_backfill_trinity_cost_label
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0025_add_loom_nodes"
down_revision: Union[str, None] = "0024_backfill_trinity_cost_label"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "loom_nodes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("loom_nodes.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("text", sa.Text(), nullable=False, server_default=""),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("seed", sa.BigInteger(), nullable=True),
        sa.Column("logprobs", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # "Load the whole tree for this session" on every loom page open.
    op.create_index(
        "ix_loom_nodes_session_id",
        "loom_nodes",
        ["session_id"],
    )
    # "Fetch the children of this node" when expanding a branch.
    op.create_index(
        "ix_loom_nodes_parent_id",
        "loom_nodes",
        ["parent_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_loom_nodes_parent_id", table_name="loom_nodes")
    op.drop_index("ix_loom_nodes_session_id", table_name="loom_nodes")
    op.drop_table("loom_nodes")
