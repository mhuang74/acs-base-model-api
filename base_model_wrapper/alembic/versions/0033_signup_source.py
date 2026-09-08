"""Add users.signup_source — per-channel signup attribution (ACS-210).

The /signup links we post to communities carry a ``?src=<channel>`` tag
(e.g. ``?src=constellation``); the form persists it so applications record
which channel they came from. Nullable: organic signups, untagged links, and
all pre-existing rows stay NULL. Additive + backward compatible.

Revision ID: 0033_signup_source
Revises: 0032_signup_use_case_agree
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0033_signup_source"
down_revision: Union[str, None] = "0032_signup_use_case_agree"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("signup_source", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "signup_source")
