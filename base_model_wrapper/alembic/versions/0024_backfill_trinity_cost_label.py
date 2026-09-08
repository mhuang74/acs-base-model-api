"""Backfill gpu_cost_sample.model_id trinity-base -> trinity-truebase (ACS-151).

The ACS-147 rename flipped the registry id trinity-base -> trinity-truebase but
left the append-only ``gpu_cost_sample`` rows untouched (migration 0023 was
deliberately scoped to the operational tables ``model_warm_window`` /
``chat_sessions``). That splits Trinity into two series on the cost dashboard
and — more importantly — means the cost-spike alert and any raw query see two
ids for one Modal app (``acs-trinity-base``, unchanged). The Modal app,
gpu_type/count, and rates are identical across the historical rows, so this is a
pure label backfill: fold the old id into the new one.

Forward-only. Downgrade is a deliberate no-op: once folded, a 'trinity-truebase'
row can't be distinguished from one that was always 'trinity-truebase' (written
after the 2026-06-26 rename), so a blind reverse UPDATE would mislabel the
genuinely-post-rename rows. The per-tile CASE normalization in
``docs/runbooks/metabase-dashboard-queries.sql`` becomes a redundant no-op after
this (kept as defence against a future rename).

Revision ID: 0024_backfill_trinity_cost_label
Revises: 0023_rename_trinity_truebase
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0024_backfill_trinity_cost_label"
down_revision: Union[str, None] = "0023_rename_trinity_truebase"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE gpu_cost_sample SET model_id = 'trinity-truebase' "
        "WHERE model_id = 'trinity-base'"
    )


def downgrade() -> None:
    # Forward-only label fold — see module docstring. Reversing would mislabel
    # rows that were always 'trinity-truebase'. Intentional no-op.
    pass
