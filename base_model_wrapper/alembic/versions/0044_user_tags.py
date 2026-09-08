"""Generic user tags + auto-tag-from-invite (ACS-371).

``user_tags`` gives accounts admin-assigned labels — the hiring cohort, the
HAAISS tutorial batch, the internal team — so "which accounts belong to this
group" stops needing a new column per group. ``signup_invites.tag`` carries a
label onto every account created from that link, so a cohort tags itself at
claim time instead of relying on somebody labelling people afterwards.

Shape mirrors ``signup_invite_redemptions`` (migration 0014): own UUID PK,
CASCADE on the owning user, an index on the FK we look up by. Chosen over a
``users.tags`` ARRAY/JSONB column — see the ``UserTag`` docstring for why.

Additive + backward compatible: a new table plus one nullable column; existing
rows are untouched and every query without a tag filter behaves as before.

Revision ID: 0044_user_tags
Revises: 0043_user_suspend_state
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0044_user_tags"
down_revision: str | None = "0043_user_suspend_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_tags",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # SET NULL, not CASCADE: deleting the admin who applied a tag must not
        # delete the tag — it describes the tagged user, not the tagger.
        sa.Column(
            "created_by_user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        # Named explicitly because ``add_tag`` targets it by name in its
        # ON CONFLICT clause — an auto-generated name would break that.
        sa.UniqueConstraint("user_id", "tag", name="uq_user_tags_user_id_tag"),
    )
    # "This user's tags" — every roster render.
    op.create_index("ix_user_tags_user_id", "user_tags", ["user_id"])
    # "Everyone with this tag" — the roster filter and the future Metabase
    # internal-team exclusion.
    op.create_index("ix_user_tags_tag", "user_tags", ["tag"])

    op.add_column("signup_invites", sa.Column("tag", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("signup_invites", "tag")
    op.drop_index("ix_user_tags_tag", table_name="user_tags")
    op.drop_index("ix_user_tags_user_id", table_name="user_tags")
    op.drop_table("user_tags")
