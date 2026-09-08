"""Add users.signup_use_case + users.agreed_terms_at (ACS-24 / ACS-170).

The public /signup application flow now collects a short free-text "who are you
/ how will you use this" (``signup_use_case``) so admins have something to
evaluate at approval time, and records when the applicant ticked the mandatory
usage-rules agreement (``agreed_terms_at``). Both are nullable: existing rows,
invite-accept users, and admin-created accounts never went through this gate, so
they backfill as NULL. Additive + backward compatible — no existing column is
touched.

Revision ID: 0032_signup_use_case_agree
Revises: 0031_activation_quota_telemetry
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0032_signup_use_case_agree"
down_revision: Union[str, None] = "0031_activation_quota_telemetry"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("signup_use_case", sa.Text(), nullable=True))
    op.add_column(
        "users",
        sa.Column("agreed_terms_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "agreed_terms_at")
    op.drop_column("users", "signup_use_case")
