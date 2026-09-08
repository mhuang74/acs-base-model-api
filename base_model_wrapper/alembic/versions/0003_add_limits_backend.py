"""Add limits backend (item 9): daily budgets, user-aggregate, I/O split.

Adds four nullable columns and one new table. All new columns default NULL so
existing rows behave identically to before — backward compat is non-negotiable.

- ``users.monthly_token_budget_total`` — per-user aggregate ceiling across all
  the user's keys for the current month. NULL = unlimited.
- ``api_keys.daily_token_budget`` — per-key daily ceiling. NULL = unlimited.
- ``api_keys.monthly_input_token_budget`` / ``monthly_output_token_budget`` —
  split the existing monthly budget into per-direction limits (matches the
  Anthropic shape). NULL on either side = no per-direction limit; the
  pre-existing ``monthly_token_budget`` total still applies.
- ``usage_daily`` — mirrors ``usage_monthly`` (PK is ``(key_id, period_start)``
  where period_start is a calendar date in UTC).

Revision ID: 0003_add_limits_backend
Revises: 0002_add_web_auth
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_add_limits_backend"
down_revision: Union[str, None] = "0002_add_web_auth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("monthly_token_budget_total", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "api_keys",
        sa.Column("daily_token_budget", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "api_keys",
        sa.Column("monthly_input_token_budget", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "api_keys",
        sa.Column("monthly_output_token_budget", sa.BigInteger(), nullable=True),
    )

    op.create_table(
        "usage_daily",
        sa.Column(
            "key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id"),
            primary_key=True,
        ),
        sa.Column("period_start", sa.Date(), primary_key=True),
        sa.Column("tokens_prompt", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "tokens_completion", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column("request_count", sa.BigInteger(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("usage_daily")
    op.drop_column("api_keys", "monthly_output_token_budget")
    op.drop_column("api_keys", "monthly_input_token_budget")
    op.drop_column("api_keys", "daily_token_budget")
    op.drop_column("users", "monthly_token_budget_total")
