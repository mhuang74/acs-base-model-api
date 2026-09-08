"""Backfill the 'internal' user tag from the @acsresearch.org heuristic (ACS-374).

Internal-team accounts were identified by a duplicated email-domain match: five
sites in ``docs/runbooks/metabase-dashboard-queries.sql`` plus a Jinja
``endswith`` in the admin roster. The SQL file's own header admits the rule is
insufficient — team members who test from a personal address are handled by a
commented ``NOT IN ('personal-test@…')`` list that, by policy, is edited in the
hosted Metabase UI and kept out of git. So the real membership list lived in
three commented-out SQL fragments and one operator's memory.

The ``internal`` tag (ACS-371) replaces both. This revision seeds it from the
domain rule so nothing changes on day one; from then on membership is edited in
``/admin/users`` like any other tag, and the personal-address case stops needing
an uncommittable escape hatch.

Data-only and idempotent (``ON CONFLICT DO NOTHING`` against
``uq_user_tags_user_id_tag``). The downgrade removes only the rows this
revision's rule would have created — a hand-applied ``internal`` tag on an
address outside the domain is left alone, because re-running the upgrade could
not recreate it.

Revision ID: 0045_internal_tag_backfill
Revises: 0044_user_tags
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0045_internal_tag_backfill"
down_revision: str | None = "0044_user_tags"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        INSERT INTO user_tags (user_id, tag)
        SELECT id, 'internal' FROM users WHERE email LIKE '%@acsresearch.org'
        ON CONFLICT ON CONSTRAINT uq_user_tags_user_id_tag DO NOTHING
        """
    )


def downgrade() -> None:
    # Remove only rows this revision created, identified by
    # ``created_by_user_id IS NULL`` — ``services.user_tags.add_tag`` always
    # stamps the acting admin, so NULL means "backfilled, not curated".
    #
    # Both asymmetries matter, and the domain predicate alone only handles one:
    #   - a tag added by hand to an out-of-domain address must survive (the
    #     upgrade could not recreate it, so deleting it loses information);
    #   - a tag an admin RE-added by hand to an in-domain address must survive
    #     too, so the operator's intent isn't silently reverted.
    #
    # What this does NOT fix, because nothing here can: the upgrade re-seeds
    # from the domain rule unconditionally, so a downgrade/upgrade cycle
    # re-adds `internal` to an in-domain account an admin deliberately
    # UN-tagged — there is no tombstone recording that removal. Verified, not
    # assumed. If you roll this revision back and forward, re-check the tag on
    # any @acsresearch.org account you had excluded.
    op.execute(
        """
        DELETE FROM user_tags
        WHERE tag = 'internal'
          AND created_by_user_id IS NULL
          AND user_id IN (SELECT id FROM users WHERE email LIKE '%@acsresearch.org')
        """
    )
