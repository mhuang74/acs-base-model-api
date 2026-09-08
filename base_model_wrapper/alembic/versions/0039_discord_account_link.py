"""Add users.discord_user_id / discord_username / discord_connected_at (ACS-269).

Filled by the OAuth2 ``identify guilds.join`` "Connect Discord" flow — the
exact Discord account id, so engagement and offboarding can join platform and
Discord identities. ``discord_user_id`` is UNIQUE: one Discord account can't
claim two platform accounts. All nullable; existing rows stay NULL. Additive.

Revision ID: 0039_discord_account_link
Revises: 0038_signup_profile_link
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0039_discord_account_link"
down_revision: Union[str, None] = "0038_signup_profile_link"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("discord_user_id", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("discord_username", sa.Text(), nullable=True))
    op.add_column(
        "users", sa.Column("discord_connected_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_unique_constraint("uq_users_discord_user_id", "users", ["discord_user_id"])


def downgrade() -> None:
    op.drop_constraint("uq_users_discord_user_id", "users", type_="unique")
    op.drop_column("users", "discord_connected_at")
    op.drop_column("users", "discord_username")
    op.drop_column("users", "discord_user_id")
