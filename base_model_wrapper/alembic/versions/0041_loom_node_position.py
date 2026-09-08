"""Add loom_nodes.position — a stable sibling ordinal (ACS-340).

Siblings generated in one batch share a ``created_at`` to the microsecond, so
ordering by time alone is unstable and export→import silently reorders a node's
children. ``position`` is a 0-based ordinal within each parent, assigned at
creation (root / sibling / split-child / generated branch) and used to sort
siblings by ``(position, created_at)``.

The column is added ``NOT NULL DEFAULT 0``. Critically we then **backfill**
existing rows: without it every pre-existing loom would have every sibling at
position 0 and keep reordering exactly as before. The backfill numbers each
parent's children 0..N-1 by their current ``(created_at, id)`` order — i.e. the
order the UI showed before this migration — so nothing visibly moves.

Not unique per ``(parent_id, position)`` on purpose: two concurrent generations
under one parent can read the same ``max+1`` and collide; that rare case
degrades to ``created_at`` order rather than losing a branch to an insert error.

Revision ID: 0041_loom_node_position
Revises: 0040_discord_guild_joined
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0041_loom_node_position"
down_revision: str | None = "0040_discord_guild_joined"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Backfill exported as a module constant so the migration test can execute the
# exact statement against simulated pre-existing rows (the alembic-on-empty-DB
# harness never runs the backfill over real data otherwise). NULL parent_ids
# (roots) partition together in Postgres, which is what we want — roots are
# numbered as one sibling group per loom.
BACKFILL_POSITION_SQL = """
WITH ordered AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY loom_id, parent_id
               ORDER BY created_at, id
           ) - 1 AS pos
    FROM loom_nodes
)
UPDATE loom_nodes
SET position = ordered.pos
FROM ordered
WHERE loom_nodes.id = ordered.id
"""


def upgrade() -> None:
    op.add_column(
        "loom_nodes",
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
    )
    op.execute(BACKFILL_POSITION_SQL)


def downgrade() -> None:
    op.drop_column("loom_nodes", "position")
