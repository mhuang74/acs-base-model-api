"""Add users.signup_outcome / signup_prior_work / signup_referral (ACS-302).

The single "how do you want to use this" question splits into four; the
existing ``signup_use_case`` column is reused for the required "Planned
usage" answer, and these three nullable columns carry the optional ones
(ideal results, prior-work links, how-did-you-hear). Pre-redesign rows and
invite-onboarded accounts stay NULL. Additive + backward compatible.

Revision ID: 0037_signup_split_questions
Revises: 0036_chat_session_pinned
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0037_signup_split_questions"
down_revision: Union[str, None] = "0036_chat_session_pinned"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("signup_outcome", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("signup_prior_work", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("signup_referral", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "signup_referral")
    op.drop_column("users", "signup_prior_work")
    op.drop_column("users", "signup_outcome")
