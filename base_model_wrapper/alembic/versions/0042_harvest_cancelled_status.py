"""Allow a 'cancelled' harvest-job status (ACS-344).

``DELETE /v1/harvest/{job_id}`` lets a key cancel its own in-flight harvest
job — freeing the one-job-per-key concurrency slot so the owner can resubmit
without waiting out a stuck job's whole wall clock. Cancellation records a new
terminal status, ``cancelled``, which the original ``ck_harvest_jobs_status``
CHECK (migration 0035) did not permit. Widen the constraint to include it.

Additive + reversible: the downgrade re-narrows the constraint, first mapping
any ``cancelled`` rows to ``failed`` so the tighter CHECK can be re-applied.

Revision ID: 0042_harvest_cancelled_status
Revises: 0041_loom_node_position
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0042_harvest_cancelled_status"
down_revision: str | None = "0041_loom_node_position"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CK = "ck_harvest_jobs_status"


def upgrade() -> None:
    op.drop_constraint(_CK, "harvest_jobs", type_="check")
    op.create_check_constraint(
        _CK,
        "harvest_jobs",
        "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
    )


def downgrade() -> None:
    # Fold cancelled rows into failed so the narrower CHECK can be re-applied.
    op.execute("UPDATE harvest_jobs SET status = 'failed' WHERE status = 'cancelled'")
    op.drop_constraint(_CK, "harvest_jobs", type_="check")
    op.create_check_constraint(
        _CK,
        "harvest_jobs",
        "status IN ('pending', 'running', 'done', 'failed')",
    )
