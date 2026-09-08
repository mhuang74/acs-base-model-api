"""Persist per-token logprobs on chat_snapshots so a saved single-pane snapshot
can be re-coloured with the logprobs heatmap on expand/revert (ACS-189).

``ChatSnapshot`` already stores ``completion_text`` but no per-token logprobs, so
a past snapshot couldn't drive the heatmap. Add one nullable JSONB column holding
the same normalised ``[{token, logprob, top:[{token, logprob}]}]`` shape the loom
stores (``loom_nodes.logprobs``) and the shared heatmap primitive consumes.

Additive + reversible: NULL for legacy rows and for runs where logprobs were off,
so the UI falls back to plain text. Populated server-side from the streamed chunks
(see workbench_generations.py).

Revision ID: 0030_chat_snapshot_logprobs
Revises: 0029_compare_run_barrier
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0030_chat_snapshot_logprobs"
down_revision: Union[str, None] = "0029_compare_run_barrier"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_snapshots",
        sa.Column("logprobs", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_snapshots", "logprobs")
