"""Make looms first-class saved objects — ACS-148 follow-up.

A *loom* used to be 1:1 with a ``chat_sessions`` row (``loom_nodes.session_id``
FK). On review, a loom should be its own saved object — like a workbench
chat — with its own list, its own ``/loom/<id>`` URL, created/named/listed/
deleted on its own. So we:

  1. create the ``looms`` table (id, user_id, title, timestamps, model,
     api_key_id — the last two for model/key parity with chat sessions);
  2. **backfill**: for each distinct ``session_id`` that currently owns loom
     nodes, create one ``looms`` row **reusing the session's id as the loom id**
     (so the old ``/workbench/{session_id}/loom`` URL redirects deterministically
     to ``/loom/{session_id}`` — idempotent, no duplicate looms), owned by that
     session's user, titled after the session (fallback "Untitled loom");
  3. add ``loom_nodes.loom_id`` (FK → looms, CASCADE), repoint every node to its
     backfilled loom, then drop the old ``session_id`` column + its index.

Downgrade is fully reversible **because backfilled loom ids equal the origin
session id**: re-add ``session_id``, copy ``loom_id`` back into it (they match
for every migrated node — new looms created after this migration have no
originating session and are dropped along with their nodes on downgrade), then
drop ``loom_id`` and the ``looms`` table.

Revision ID: 0027_looms_first_class
Revises: 0026_raise_default_token_budget
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0027_looms_first_class"
down_revision: Union[str, None] = "0026_raise_default_token_budget"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. New first-class looms table.
    op.create_table(
        "looms",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "title", sa.Text(), nullable=False, server_default="Untitled loom"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_looms_user_id", "looms", ["user_id"])

    # 2. Backfill one loom per distinct session that owns loom nodes, reusing the
    #    session id as the loom id (deterministic old-URL redirect). Copy the
    #    session's title/model/api_key_id/created_at so the loom looks continuous.
    op.execute(
        """
        INSERT INTO looms (id, user_id, title, created_at, updated_at, model, api_key_id)
        SELECT cs.id,
               cs.user_id,
               COALESCE(NULLIF(cs.title, ''), 'Untitled loom'),
               cs.created_at,
               cs.updated_at,
               cs.model,
               cs.api_key_id
        FROM chat_sessions cs
        WHERE cs.id IN (SELECT DISTINCT session_id FROM loom_nodes)
        """
    )

    # 3. Add loom_id, repoint every node (loom_id == its old session_id, which is
    #    now the backfilled loom's id), then make it NOT NULL + FK + index and
    #    drop the old session_id column.
    op.add_column(
        "loom_nodes",
        sa.Column("loom_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.execute("UPDATE loom_nodes SET loom_id = session_id")
    op.alter_column("loom_nodes", "loom_id", nullable=False)
    op.create_foreign_key(
        "fk_loom_nodes_loom_id",
        "loom_nodes",
        "looms",
        ["loom_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_loom_nodes_loom_id", "loom_nodes", ["loom_id"])

    op.drop_index("ix_loom_nodes_session_id", table_name="loom_nodes")
    op.drop_constraint("loom_nodes_session_id_fkey", "loom_nodes", type_="foreignkey")
    op.drop_column("loom_nodes", "session_id")


def downgrade() -> None:
    # Re-add session_id and repoint nodes back. Migrated looms have id == origin
    # session id, so loom_id round-trips exactly. Looms created after the upgrade
    # (no originating session) can't be represented in the old schema — drop them
    # and their nodes so we don't leave dangling FKs.
    op.execute("DELETE FROM loom_nodes WHERE loom_id NOT IN (SELECT id FROM chat_sessions)")

    op.add_column(
        "loom_nodes",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.execute("UPDATE loom_nodes SET session_id = loom_id")
    op.alter_column("loom_nodes", "session_id", nullable=False)
    op.create_foreign_key(
        "loom_nodes_session_id_fkey",
        "loom_nodes",
        "chat_sessions",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_loom_nodes_session_id", "loom_nodes", ["session_id"])

    op.drop_index("ix_loom_nodes_loom_id", table_name="loom_nodes")
    op.drop_constraint("fk_loom_nodes_loom_id", "loom_nodes", type_="foreignkey")
    op.drop_column("loom_nodes", "loom_id")

    op.drop_index("ix_looms_user_id", table_name="looms")
    op.drop_table("looms")
