"""Add compare_snapshots — saved snapshots for Workbench Compare mode (ACS-180).

Single-pane Continue already snapshots each generation into ``chat_snapshots``
(one row per Continue, single model). Compare mode had no history at all. We add
a dedicated ``compare_snapshots`` table storing ONE row per Compare *Run all*,
with the per-lane detail (model + sampling params + final completion) in a JSONB
``lanes`` column. This deliberately snapshots whatever lanes existed at Run-all
time as a unit, so add/remove lanes between runs stays unrestricted.

Additive only: new table + CASCADE FK to ``chat_sessions`` + an index on
``session_id``. Downgrade drops the table (and its index) — fully reversible.

Revision ID: 0028_add_compare_snapshots
Revises: 0027_looms_first_class
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0028_add_compare_snapshots"
down_revision: Union[str, None] = "0027_looms_first_class"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "compare_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("prompt", sa.Text(), nullable=False, server_default=""),
        sa.Column("lanes", postgresql.JSONB(), nullable=False),
        sa.Column("n_lanes", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "ix_compare_snapshots_session_id",
        "compare_snapshots",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_compare_snapshots_session_id", table_name="compare_snapshots")
    op.drop_table("compare_snapshots")
