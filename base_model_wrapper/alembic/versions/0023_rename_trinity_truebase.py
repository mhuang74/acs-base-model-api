"""Rename model id trinity-base -> trinity-truebase (ACS-147).

The public model id ``trinity-base`` is misleading — it serves
``arcee-ai/Trinity-Large-TrueBase`` (the *TrueBase* variant), and there is a
separate regular Trinity-Large-Base. Hard rename to ``trinity-truebase`` (no
alias) per the 2026-06-26 decision.

This migration carries the rename across the operational rows keyed by the model
id so they keep matching the renamed registry entry:

* ``model_warm_window.model_id`` — the always-on / scheduled-warm cron row
  (seeded in 0008). If left as ``trinity-base`` the scheduler can't resolve the
  app name from the (renamed) registry and the warm/cool jobs break.
* ``chat_sessions.model`` — a tester's saved per-chat model selection. If left
  as ``trinity-base`` the workbench picker would have no matching option and a
  submit would 404.

Append-only log tables (``api_requests.model``, ``gpu_cost_sample.model_id``)
are intentionally NOT rewritten — they are historical records; dashboards that
span the rename can coalesce the two ids.

Revision ID: 0023_rename_trinity_truebase
Revises: 0022_chat_temperature_default
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0023_rename_trinity_truebase"
down_revision: Union[str, None] = "0022_chat_temperature_default"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE model_warm_window SET model_id = 'trinity-truebase' "
        "WHERE model_id = 'trinity-base'"
    )
    op.execute(
        "UPDATE chat_sessions SET model = 'trinity-truebase' WHERE model = 'trinity-base'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE model_warm_window SET model_id = 'trinity-base' "
        "WHERE model_id = 'trinity-truebase'"
    )
    op.execute(
        "UPDATE chat_sessions SET model = 'trinity-base' WHERE model = 'trinity-truebase'"
    )
