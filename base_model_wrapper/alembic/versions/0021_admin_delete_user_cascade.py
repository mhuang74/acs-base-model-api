"""Cascade api_requests / usage_monthly / usage_daily on api_keys delete.

Enables admin "delete user" by removing the FK-violation blocker: deleting a
user already cascades to ``api_keys`` (via ``api_keys.user_id`` ON DELETE
CASCADE), but the three child tables ``api_requests``, ``usage_monthly``, and
``usage_daily`` all FK back to ``api_keys.id`` with no ondelete clause (default
NO ACTION). Any user with request history therefore can't be deleted today.

This migration drops and re-adds those three FKs with ``ON DELETE CASCADE`` so
deleting an api_key (and transitively, a user) sweeps the child rows away.
Downgrade restores NO ACTION.

Revision ID: 0021_admin_delete_user_cascade
Revises: 0020_gpu_cost_sample
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0021_admin_delete_user_cascade"
down_revision: Union[str, None] = "0020_gpu_cost_sample"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (constraint_name, table_name) for the three key_id → api_keys.id FKs.
_FKS = [
    ("api_requests_key_id_fkey", "api_requests"),
    ("usage_monthly_key_id_fkey", "usage_monthly"),
    ("usage_daily_key_id_fkey", "usage_daily"),
]


def _recreate(*, ondelete: str | None) -> None:
    for name, table in _FKS:
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(
            name,
            table,
            "api_keys",
            ["key_id"],
            ["id"],
            ondelete=ondelete,
        )


def upgrade() -> None:
    _recreate(ondelete="CASCADE")


def downgrade() -> None:
    _recreate(ondelete=None)
