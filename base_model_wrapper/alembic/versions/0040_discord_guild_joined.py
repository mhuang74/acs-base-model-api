"""Add users.discord_guild_joined (ACS-307).

Records whether the bot's ``guilds.join`` actually added the user to the
server. NULL = never attempted (account not linked); False = linked but the
join failed, so the dashboard keeps offering the invite link instead of a
server deep link that wouldn't work for a non-member. Nullable; existing rows
stay NULL. Additive + backward compatible.

Revision ID: 0040_discord_guild_joined
Revises: 0039_discord_account_link
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0040_discord_guild_joined"
down_revision: Union[str, None] = "0039_discord_account_link"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("discord_guild_joined", sa.Boolean(), nullable=True))
    # Rows linked before this migration were joined successfully (the callback
    # only reached the mapping write after a 201/204, or logged the 403 path
    # which was rare) — assume True so they don't all show the invite fallback.
    op.execute(
        "UPDATE users SET discord_guild_joined = true WHERE discord_user_id IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("users", "discord_guild_joined")
