"""Per-key activation quota + activation request telemetry (ACS-199).

Exposing activation harvesting/steering through the wrapper adds a per-key
monthly cap on activation requests (each wakes an expensive activation GPU
engine) and per-request telemetry so activation cost is attributable and
visible in the usage dashboards.

Additive + reversible:
  * ``api_keys.monthly_activation_budget`` — BIGINT NOT NULL DEFAULT 0
    (0 = unlimited, mirroring the legacy ``monthly_token_budget`` convention).
  * ``api_requests.activation`` — BOOLEAN NOT NULL DEFAULT false (whether the
    request used the activation engine); this column also backs the monthly
    quota count.
  * ``api_requests.activation_layers`` — INTEGER NULL (# residual-stream layers
    captured; the heavy payload/GPU-cost dimension).
  * ``api_requests.activation_steering_vectors`` — INTEGER NULL (# steering
    vectors applied).

All request-metadata only — no tensors / prompt / completion content, so the
privacy guarantee is untouched.

Revision ID: 0031_activation_quota_telemetry
Revises: 0030_chat_snapshot_logprobs
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0031_activation_quota_telemetry"
down_revision: Union[str, None] = "0030_chat_snapshot_logprobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column(
            "monthly_activation_budget",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "api_requests",
        sa.Column(
            "activation",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "api_requests",
        sa.Column("activation_layers", sa.Integer(), nullable=True),
    )
    op.add_column(
        "api_requests",
        sa.Column("activation_steering_vectors", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("api_requests", "activation_steering_vectors")
    op.drop_column("api_requests", "activation_layers")
    op.drop_column("api_requests", "activation")
    op.drop_column("api_keys", "monthly_activation_budget")
