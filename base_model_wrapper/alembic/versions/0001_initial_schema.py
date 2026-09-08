"""Initial schema — users, api_keys, api_requests, usage_monthly.

Mirrors the schema in wrapper-implementation-plan.md verbatim. `pgcrypto`
extension enabled explicitly even though Postgres ≥13 has `gen_random_uuid()`
built-in — safety net for older instances.

Revision ID: 0001_initial_schema
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True),
                  primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("name", sa.Text()),
        sa.Column("org", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text("now()")),
        sa.Column("notes", sa.Text()),
    )

    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True),
                  primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key_hash", sa.LargeBinary(), nullable=False, unique=True),
        sa.Column("key_prefix", sa.Text(), nullable=False),
        sa.Column("name", sa.Text()),
        sa.Column("monthly_token_budget", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text("now()")),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("scopes", postgresql.ARRAY(sa.Text()),
                  nullable=False, server_default="{completions}"),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])

    op.create_table(
        "api_requests",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text("now()")),
        sa.Column("key_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("api_keys.id"), nullable=False),
        sa.Column("ip", postgresql.INET()),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("model", sa.Text()),
        sa.Column("n_prompt", sa.Integer()),
        sa.Column("n_completion", sa.Integer()),
        sa.Column("status", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("error_kind", sa.Text()),
    )
    op.create_index("ix_api_requests_key_id_ts", "api_requests", ["key_id", sa.text("ts DESC")])
    op.create_index("ix_api_requests_ts", "api_requests", [sa.text("ts DESC")])

    op.create_table(
        "usage_monthly",
        sa.Column("key_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("api_keys.id"), primary_key=True),
        sa.Column("period_start", sa.Date(), primary_key=True),
        sa.Column("tokens_prompt", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("tokens_completion", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("request_count", sa.BigInteger(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("usage_monthly")
    op.drop_index("ix_api_requests_ts", table_name="api_requests")
    op.drop_index("ix_api_requests_key_id_ts", table_name="api_requests")
    op.drop_table("api_requests")
    op.drop_index("ix_api_keys_user_id", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_table("users")
