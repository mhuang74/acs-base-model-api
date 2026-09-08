"""Record full Sampling settings on every Run (chat_snapshots) — issue #12.

Single-pane ``Continue`` Runs previously persisted only max_tokens +
temperature (base columns) plus the logprobs payload; top_p / top_k / min_p /
the three penalties / seed / stop / the logprobs toggle-count lived only in
uncontrolled form inputs, so a page reload reset them and exports described
nothing reproducible. Compare lanes already carried their config via
``chat_generations.compare_config`` (ACS-186) — that path is unchanged.

Additive-only, one column each on ``chat_snapshots``:

- ``top_p``, ``top_k``, ``min_p``, ``presence_penalty``,
  ``frequency_penalty``, ``repetition_penalty``, ``seed``, ``stop``,
  ``logprobs_count``, ``model_echo`` — nullable; NULL = unset.
- ``recorded_incomplete`` — NOT NULL, default false. True marks pre-migration
  rows (and v1-imported Runs), whose extended settings were never captured.

The ONLY data write here is the completeness-marker backfill:
``UPDATE chat_snapshots SET recorded_incomplete = true``. Every row that
existed before this migration gets the marker, because its extended settings
were never captured — leaving them at the server default (false) would read
as "fully recorded" while their settings columns are NULL, exactly the lie
the spec forbids. No sampling value is fabricated: the settings columns stay
NULL, render defaults-with-placeholder in the UI, restore as defaults, and
travel in v2 exports with the marker set.

Revision ID: 0046_run_sampling_settings
Revises: 0045_internal_tag_backfill
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0046_run_sampling_settings"
down_revision: Union[str, None] = "0045_internal_tag_backfill"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chat_snapshots", sa.Column("top_p", sa.Float(), nullable=True))
    op.add_column("chat_snapshots", sa.Column("top_k", sa.Integer(), nullable=True))
    op.add_column("chat_snapshots", sa.Column("min_p", sa.Float(), nullable=True))
    op.add_column(
        "chat_snapshots", sa.Column("presence_penalty", sa.Float(), nullable=True)
    )
    op.add_column(
        "chat_snapshots", sa.Column("frequency_penalty", sa.Float(), nullable=True)
    )
    op.add_column(
        "chat_snapshots", sa.Column("repetition_penalty", sa.Float(), nullable=True)
    )
    op.add_column(
        "chat_snapshots", sa.Column("seed", sa.BigInteger(), nullable=True)
    )
    op.add_column("chat_snapshots", sa.Column("stop", sa.Text(), nullable=True))
    op.add_column(
        "chat_snapshots", sa.Column("logprobs_count", sa.Integer(), nullable=True)
    )
    op.add_column(
        "chat_snapshots",
        sa.Column(
            "recorded_incomplete",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column("chat_snapshots", sa.Column("model_echo", sa.Text(), nullable=True))
    # The ONLY data write in this migration: mark every row that exists here as
    # recorded_incomplete. These are pre-migration Runs whose extended settings
    # were never captured — leaving them at the server default (false) would
    # read as "fully recorded" while their settings columns are NULL, exactly
    # the lie the spec forbids. This is the completeness marker doing its job,
    # NOT a settings backfill: no sampling value is fabricated (they stay NULL,
    # restore as defaults, and travel in exports with the marker set).
    op.execute("UPDATE chat_snapshots SET recorded_incomplete = true")


def downgrade() -> None:
    op.drop_column("chat_snapshots", "model_echo")
    op.drop_column("chat_snapshots", "recorded_incomplete")
    op.drop_column("chat_snapshots", "logprobs_count")
    op.drop_column("chat_snapshots", "stop")
    op.drop_column("chat_snapshots", "seed")
    op.drop_column("chat_snapshots", "repetition_penalty")
    op.drop_column("chat_snapshots", "frequency_penalty")
    op.drop_column("chat_snapshots", "presence_penalty")
    op.drop_column("chat_snapshots", "min_p")
    op.drop_column("chat_snapshots", "top_k")
    op.drop_column("chat_snapshots", "top_p")