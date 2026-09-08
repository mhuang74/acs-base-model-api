"""Raise existing default monthly token budgets 500k -> 5M (ops bump).

New users/keys get the new default from settings (``default_monthly_token_budget_total``
and ``default_per_key_budget``, both bumped 500_000 -> 5_000_000). This one-shot
migration lifts the EXISTING beta users + their auto-created keys that are still at
the old 500k default up to 5M, so current users see the same higher limit without
per-user admin edits.

Scoped to rows at *exactly* the old default (500_000): a manually-set budget,
unlimited aggregate (``NULL``), or an unbounded key (``0``) is left untouched.

Forward-only: downgrade is a deliberate no-op — a 5M row can't be distinguished
from one a later admin set to 5M on purpose.

Revision ID: 0026_raise_default_token_budget
Revises: 0025_add_loom_nodes
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0026_raise_default_token_budget"
down_revision: Union[str, None] = "0025_add_loom_nodes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD = 500_000
_NEW = 5_000_000


def upgrade() -> None:
    op.execute(
        f"UPDATE users SET monthly_token_budget_total = {_NEW} "
        f"WHERE monthly_token_budget_total = {_OLD}"
    )
    op.execute(
        f"UPDATE api_keys SET monthly_token_budget = {_NEW} "
        f"WHERE monthly_token_budget = {_OLD}"
    )


def downgrade() -> None:
    # Forward-only ops bump — a 5M row can't be told apart from a deliberately
    # set 5M one. Intentional no-op.
    pass
