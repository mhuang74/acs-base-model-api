"""Admin-assigned labels on user accounts (ACS-371).

Tags answer "which accounts belong to this group" — the hiring cohort, the
HAAISS tutorial batch, the internal team — without a new column per group. See
:class:`wrapper.models.UserTag` for why this is a join table rather than an
array or JSONB column.

The vocabulary is open (free text, no migration to add a value, filter options
derived from the rows that exist), so :func:`normalize_tag` is the only thing
keeping it from fragmenting into ``Hiring-2026-08`` / ``hiring 2026 08`` /
``hiring_2026_08``. Call it at every write site.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import UserTag

#: Max stored length. Long enough for `hiring-2026-08` / `haaiss-summer-school`,
#: short enough to stay readable as a chip in the roster table.
MAX_TAG_LEN = 32

_INVALID = re.compile(r"[^a-z0-9-]+")
_DASHES = re.compile(r"-{2,}")


def normalize_tag(raw: str | None) -> str | None:
    """Canonicalise a tag, or return None if nothing usable is left.

    lowercase → non-``[a-z0-9-]`` runs collapse to a single ``-`` → dashes
    deduped → trimmed of leading/trailing dashes → truncated to
    :data:`MAX_TAG_LEN` (then re-trimmed, so a cut never leaves a trailing dash).

    Modelled on ``_normalize_signup_source`` in ``routes/web.py``: normalise at
    the boundary so the stored value is the only form anything downstream sees.
    """
    if not raw:
        return None
    s = _INVALID.sub("-", raw.strip().lower())
    s = _DASHES.sub("-", s).strip("-")
    if not s:
        return None
    return s[:MAX_TAG_LEN].strip("-") or None


async def load_tags_for_users(
    session: AsyncSession, user_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[str]]:
    """Map user id -> sorted tags, in one query.

    Returns ``{}`` for an empty input rather than issuing a ``WHERE id IN ()``.
    Users with no tags are simply absent — callers use ``.get(uid, [])``, the
    same shape as ``customer360.load_roster_engagement``.
    """
    if not user_ids:
        return {}
    rows = (
        await session.execute(
            select(UserTag.user_id, UserTag.tag)
            .where(UserTag.user_id.in_(user_ids))
            .order_by(UserTag.tag)
        )
    ).all()
    out: dict[uuid.UUID, list[str]] = {}
    for user_id, tag in rows:
        out.setdefault(user_id, []).append(tag)
    return out


async def all_tags(session: AsyncSession) -> list[tuple[str, int]]:
    """Every tag in use with its account count, most-used first.

    Drives the roster filter's options — an open vocabulary has no canonical
    list, so the rows that exist *are* the list (the ``EmailLog.kind`` pattern).
    """
    rows = (
        await session.execute(
            select(UserTag.tag, func.count())
            .group_by(UserTag.tag)
            .order_by(func.count().desc(), UserTag.tag)
        )
    ).all()
    return [(tag, count) for tag, count in rows]


async def add_tag(
    session: AsyncSession,
    user_ids: list[uuid.UUID],
    tag: str,
    *,
    created_by_user_id: uuid.UUID | None = None,
) -> int:
    """Tag every listed user; returns how many rows were actually inserted.

    ``ON CONFLICT DO NOTHING`` against ``uq_user_tags_user_id_tag`` makes this
    idempotent, so re-tagging an already-tagged selection is a no-op rather than
    an IntegrityError — which matters for the bulk path, where a partial overlap
    between the selection and the existing tag is the normal case.

    ``tag`` must already be normalized; callers own that so an invalid value is
    reported to the admin instead of silently coerced.
    """
    if not user_ids:
        return 0
    stmt = (
        pg_insert(UserTag)
        .values(
            [
                {"user_id": uid, "tag": tag, "created_by_user_id": created_by_user_id}
                for uid in user_ids
            ]
        )
        .on_conflict_do_nothing(constraint="uq_user_tags_user_id_tag")
    )
    return (await session.execute(stmt)).rowcount or 0


async def remove_tag(session: AsyncSession, user_ids: list[uuid.UUID], tag: str) -> int:
    """Remove one tag from every listed user; returns rows deleted."""
    if not user_ids:
        return 0
    result = await session.execute(
        delete(UserTag).where(UserTag.user_id.in_(user_ids), UserTag.tag == tag)
    )
    return result.rowcount or 0
