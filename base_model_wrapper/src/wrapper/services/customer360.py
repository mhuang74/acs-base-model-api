"""Per-user engagement aggregates for the admin CRM views (ACS-300).

The engagement-bucket definition (active ≤7d · cooling 8–14d · churned >14d ·
never_activated) is ported from, and must stay in sync with, the Metabase C1
query "Per-user summary — customer360" in
``docs/runbooks/metabase-dashboard-queries.sql`` — that file's C1 comment
points back here. Unlike the Metabase C tiles, these queries do NOT exclude
internal team accounts — the roster shows every account and lets its
``internal`` tag speak for itself.

That exclusion moved from an ``@acsresearch.org`` domain match to the
``internal`` user tag in ACS-374, because the domain rule could not express a
team member testing from a personal address. If you change the rule, change it
in the SQL file too — the two are a documented two-way contract.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Shared CASE expression — the single source of the bucket thresholds in app code.
_BUCKET_CASE = """
    CASE
      WHEN count(ar.id) = 0                          THEN 'never_activated'
      WHEN max(ar.ts) >= now() - interval '7 days'   THEN 'active'
      WHEN max(ar.ts) >= now() - interval '14 days'  THEN 'cooling'
      ELSE 'churned'
    END
"""

# 'trinity-base' was renamed to 'trinity-truebase' mid-beta; normalize so the
# favorite-model mode() doesn't split one model across two names (ACS-147).
_MODEL_NORM = (
    "CASE WHEN ar.model = 'trinity-base' THEN 'trinity-truebase' ELSE ar.model END"
)


@dataclass(frozen=True)
class RosterEngagement:
    engagement: str
    days_since_last: int | None
    tokens_30d: int
    active_weeks: int


@dataclass(frozen=True)
class UserEngagement:
    engagement: str
    days_since_last: int | None
    tokens_30d: int
    active_weeks: int
    requests: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    favorite_model: str | None
    first_request_at: dt.datetime | None
    last_request_at: dt.datetime | None
    access_granted_at: dt.datetime | None
    days_to_first_request: float | None


_ROSTER_SQL = text(f"""
    SELECT u.id AS user_id,
           {_BUCKET_CASE} AS engagement,
           date_part('day', now() - max(ar.ts))::int AS days_since_last,
           coalesce(sum(coalesce(ar.n_prompt, 0) + coalesce(ar.n_completion, 0))
                    FILTER (WHERE ar.ts >= now() - interval '30 days'), 0)::bigint
                                                     AS tokens_30d,
           count(DISTINCT date_trunc('week', ar.ts))::int AS active_weeks
    FROM users u
    LEFT JOIN api_keys     ak ON ak.user_id = u.id
    LEFT JOIN api_requests ar ON ar.key_id  = ak.id
    GROUP BY u.id
""")


async def load_roster_engagement(
    session: AsyncSession,
) -> dict[uuid.UUID, RosterEngagement]:
    """One engagement row per user (all statuses), for the /admin/users roster."""
    rows = (await session.execute(_ROSTER_SQL)).mappings().all()
    return {
        r["user_id"]: RosterEngagement(
            engagement=r["engagement"],
            days_since_last=r["days_since_last"],
            tokens_30d=int(r["tokens_30d"] or 0),
            active_weeks=int(r["active_weeks"] or 0),
        )
        for r in rows
    }


_USER_SQL = text(f"""
    SELECT {_BUCKET_CASE} AS engagement,
           date_part('day', now() - max(ar.ts))::int AS days_since_last,
           coalesce(sum(coalesce(ar.n_prompt, 0) + coalesce(ar.n_completion, 0))
                    FILTER (WHERE ar.ts >= now() - interval '30 days'), 0)::bigint
                                                     AS tokens_30d,
           count(DISTINCT date_trunc('week', ar.ts))::int AS active_weeks,
           count(ar.id)                              AS requests,
           coalesce(sum(coalesce(ar.n_prompt, 0) + coalesce(ar.n_completion, 0)), 0)::bigint
                                                     AS total_tokens,
           coalesce(sum(coalesce(ar.n_prompt, 0)), 0)::bigint     AS input_tokens,
           coalesce(sum(coalesce(ar.n_completion, 0)), 0)::bigint AS output_tokens,
           mode() WITHIN GROUP (ORDER BY {_MODEL_NORM})           AS favorite_model,
           min(ar.ts)                                AS first_request_at,
           max(ar.ts)                                AS last_request_at,
           coalesce(inv.invited_at, u.approved_at)   AS access_granted_at,
           round((extract(epoch FROM (min(ar.ts) - coalesce(inv.invited_at, u.approved_at)))
                  / 86400.0)::numeric, 2)            AS days_to_first_request
    FROM users u
    LEFT JOIN LATERAL (
        SELECT min(si.created_at) AS invited_at
        FROM signup_invite_redemptions r
        LEFT JOIN signup_invites si ON si.id = r.invite_id
        WHERE r.user_id = u.id
    ) inv ON true
    LEFT JOIN api_keys     ak ON ak.user_id = u.id
    LEFT JOIN api_requests ar ON ar.key_id  = ak.id
    WHERE u.id = :user_id
    GROUP BY u.id, inv.invited_at, u.approved_at
""")


async def load_user_engagement(
    session: AsyncSession, user_id: uuid.UUID
) -> UserEngagement | None:
    """Engagement + lifetime usage for one user (the /admin/users/{id} page)."""
    r = (
        (await session.execute(_USER_SQL, {"user_id": str(user_id)})).mappings().first()
    )
    if r is None:
        return None
    return UserEngagement(
        engagement=r["engagement"],
        days_since_last=r["days_since_last"],
        tokens_30d=int(r["tokens_30d"] or 0),
        active_weeks=int(r["active_weeks"] or 0),
        requests=int(r["requests"] or 0),
        total_tokens=int(r["total_tokens"] or 0),
        input_tokens=int(r["input_tokens"] or 0),
        output_tokens=int(r["output_tokens"] or 0),
        favorite_model=r["favorite_model"],
        first_request_at=r["first_request_at"],
        last_request_at=r["last_request_at"],
        access_granted_at=r["access_granted_at"],
        days_to_first_request=(
            float(r["days_to_first_request"])
            if r["days_to_first_request"] is not None
            else None
        ),
    )
