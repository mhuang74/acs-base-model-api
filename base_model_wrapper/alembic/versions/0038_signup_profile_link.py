"""Add users.signup_profile_link (ACS-303).

Optional link to the applicant's own profile page (research-org page, Google
Scholar, LinkedIn, or a pseudonymous X/LessWrong profile) — pins the account
to an online identity for review and the ACS-155 identity work. Nullable;
existing rows stay NULL. Additive + backward compatible.

Revision ID: 0038_signup_profile_link
Revises: 0037_signup_split_questions
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0038_signup_profile_link"
down_revision: Union[str, None] = "0037_signup_split_questions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("signup_profile_link", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "signup_profile_link")
