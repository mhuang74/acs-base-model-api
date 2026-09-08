"""Add the suspend audit stamps: users.suspended_at / suspended_by_user_id (ACS-353).

A first-class, *reversible* 'suspended' account status — for a cohort whose
access was always time-boxed (hiring candidates, event tutorial accounts), or a
long-inactive account we want to park before deleting. Until now the only
admin options were reject (terminal, sweeps the user's API keys irreversibly)
and delete (destroys the ``api_requests`` history the Metabase C1/C2 tiles
read).

No change to ``users.status`` itself: it is plain ``Text`` with no CHECK
(migration 0005), and every gate in the app is deny-by-default
(``status != 'approved'``), so the new value needs no schema change to be
enforced. Only the audit stamps are new — mirroring the ``approved_at`` /
``approved_by_user_id`` pair so "who suspended this account, and when" is
answerable. ``suspended_by_user_id`` is a self-FK, hence ``use_alter=True``
(same treatment as ``approved_by_user_id``; the users<->api_keys FK cycle
already forces it).

Additive + backward compatible: both columns nullable, existing rows stay NULL.

Revision ID: 0043_user_suspend_state
Revises: 0042_harvest_cancelled_status
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0043_user_suspend_state"
down_revision: str | None = "0042_harvest_cancelled_status"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK = "fk_users_suspended_by_user_id"


def upgrade() -> None:
    op.add_column("users", sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("suspended_by_user_id", UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        _FK, "users", "users", ["suspended_by_user_id"], ["id"], ondelete="SET NULL"
    )


def downgrade() -> None:
    # ``status`` is deliberately left alone — suspended rows STAY suspended.
    #
    # The tempting move is to fold them back to 'approved' so no row is left in
    # a status the older build has no UI for. That fails OPEN: because suspend
    # leaves ``api_keys`` untouched by design, re-approving instantly restores
    # full web *and* API access — with working keys — to every account an admin
    # deliberately cut off, silently and with the audit stamps dropped in the
    # next two statements. Rolling a migration would hand a whole suspended
    # cohort its access back.
    #
    # Failing closed costs much less than that. Every gate on the older build is
    # deny-by-default (``status != 'approved'``), so a suspended row keeps being
    # refused at login and at the API; it just reports the more generic "not
    # currently active" / ``account_not_approved`` instead of the suspend-
    # specific copy, and its pill renders unstyled. Recovery is still one click:
    # Approve sets 'approved' on the old build too. The dated "(suspended): …"
    # Notes entry also survives, so the reason isn't lost.
    op.drop_constraint(_FK, "users", type_="foreignkey")
    op.drop_column("users", "suspended_by_user_id")
    op.drop_column("users", "suspended_at")
