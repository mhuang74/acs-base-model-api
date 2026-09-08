"""Beta event-logging fields on api_requests.

Adds per-request telemetry columns used by the beta usage-pattern dashboards:
- ``upstream_latency_ms`` — persist the Modal/vLLM round-trip time (already
  emitted to stdout, previously not stored).
- ``ttft_ms`` — time-to-first-token for streaming requests.
- ``stream`` / ``cold_boot`` — interactive-vs-batch and cold-start frequency.
- ``workload_type`` — client-declared ``X-Acs-Workload`` (batch|interactive).
- ``req_max_tokens`` / ``temperature`` / ``top_p`` and the ``*_set`` adoption
  flags — sampling-parameter distributions/adoption.

All columns are pure request metadata (no prompt/completion/logprobs content),
additive, and nullable or server-defaulted, so existing rows backfill cleanly.
Time-range dashboard queries are already served by ``ix_api_requests_ts`` /
``ix_api_requests_key_id_ts`` from migration 0001, so no new index is needed.

Revision ID: 0019_api_request_event_fields
Revises: 0018_probe_result_gpu_telemetry
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0019_api_request_event_fields"
down_revision: Union[str, None] = "0018_probe_result_gpu_telemetry"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("api_requests", sa.Column("upstream_latency_ms", sa.Integer(), nullable=True))
    op.add_column("api_requests", sa.Column("ttft_ms", sa.Integer(), nullable=True))
    op.add_column(
        "api_requests",
        sa.Column("stream", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "api_requests",
        sa.Column("cold_boot", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("api_requests", sa.Column("workload_type", sa.Text(), nullable=True))
    op.add_column("api_requests", sa.Column("req_max_tokens", sa.Integer(), nullable=True))
    op.add_column("api_requests", sa.Column("temperature", sa.Float(), nullable=True))
    op.add_column("api_requests", sa.Column("top_p", sa.Float(), nullable=True))
    op.add_column(
        "api_requests",
        sa.Column("logprobs_set", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "api_requests",
        sa.Column("prompt_logprobs_set", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "api_requests",
        sa.Column("seed_set", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "api_requests",
        sa.Column("echo_set", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("api_requests", "echo_set")
    op.drop_column("api_requests", "seed_set")
    op.drop_column("api_requests", "prompt_logprobs_set")
    op.drop_column("api_requests", "logprobs_set")
    op.drop_column("api_requests", "top_p")
    op.drop_column("api_requests", "temperature")
    op.drop_column("api_requests", "req_max_tokens")
    op.drop_column("api_requests", "workload_type")
    op.drop_column("api_requests", "cold_boot")
    op.drop_column("api_requests", "stream")
    op.drop_column("api_requests", "ttft_ms")
    op.drop_column("api_requests", "upstream_latency_ms")
