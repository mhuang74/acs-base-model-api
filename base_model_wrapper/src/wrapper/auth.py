"""Bearer-key authentication + multi-dimensional budget enforcement.

The hot path:
  1. Parse `Authorization: Bearer <key>` from the request.
  2. Hash, look up the `api_keys` row (rejecting if missing, revoked, paused,
     or wrong scope).
  3. Read this month's / today's usage rows + the user's aggregate monthly
     spend (sum across every key they own, including revoked).
  4. Caller (the proxy) is responsible for clamping `max_tokens` against
     ``effective_remaining()`` and for incrementing usage in BOTH
     ``usage_monthly`` AND ``usage_daily`` after upstream returns.

Limit dimensions (item 9):
  - ``monthly_token_budget`` per key (0 = unlimited, kept for back-compat).
  - ``monthly_input_token_budget`` / ``monthly_output_token_budget`` per key
    (NULL = no per-direction cap).
  - ``daily_token_budget`` per key (NULL = unlimited).
  - ``users.monthly_token_budget_total`` per user, across all their keys
    (NULL = unlimited).
  Any non-NULL dimension whose remaining headroom hits zero blocks the call.

Last-used-at is updated at most once per ``last_used_throttle_s`` per key —
avoids a write storm on every request.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass, field
from typing import NoReturn

from fastapi import HTTPException, Request, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from . import keys as keymod
from .error_kinds import ErrorKind
from .logging import log_auth_failure
from .models import ApiKey, UsageDaily, UsageMonthly, User


def _raise_auth_error(
    request: Request,
    *,
    status_code: int,
    error_kind: str,
    message: str,
    code: str,
    key_prefix: str | None = None,
    log_ip: bool = False,
) -> NoReturn:
    """Emit a structured auth-failure log line, then raise ``HTTPException``.

    Centralises every 401/403 so each rejection leaves exactly one structured
    trace (request_id, endpoint, error_kind, key_prefix, ip) for triage —
    auth failures are raised before the normal per-request logging path runs,
    so without this they're invisible. No full token/secret/hash is ever
    logged; ``key_prefix`` is the public ``acs-bm-<prefix>`` segment only.
    """
    request_id = getattr(request.state, "request_id", None) or "unknown"
    # Read the path straight from the scope rather than via ``request.url.path``:
    # building ``request.url`` requires a fully-populated scope (scheme/server/
    # path), and auth failures can be raised very early — before the full scope
    # is assembled — so a partial scope would otherwise raise ``KeyError: 'path'``
    # while rendering the rejection itself. ``.get`` keeps the error path robust.
    endpoint = request.scope.get("path", "unknown")
    ip = request.client.host if (request.client and log_ip) else None
    log_auth_failure(
        request_id=request_id,
        endpoint=endpoint,
        error_kind=error_kind,
        status=status_code,
        key_prefix=key_prefix,
        ip=ip,
    )
    raise HTTPException(
        status_code,
        detail={
            "error": {"message": message, "type": "invalid_request_error", "code": code}
        },
    )


@dataclass
class AuthedCaller:
    """Snapshot of an authenticated caller + their current budget state.

    ``monthly_token_budget`` keeps the legacy convention: ``0`` means
    unlimited. Every other budget field is ``int | None`` where ``None``
    means "no limit in this dimension" — they were added by item 9 and
    existing keys/users have NULL in the DB, so they default to None here.
    """

    key_id: uuid.UUID
    key_prefix: str
    user_email: str
    monthly_token_budget: int  # 0 = unlimited (legacy convention)
    tokens_used_this_month: int
    # --- item 9 additions; all default-None so existing call sites compile. -
    user_id: uuid.UUID | None = None
    daily_token_budget: int | None = None
    tokens_used_today: int = 0
    monthly_token_budget_total: int | None = None
    user_tokens_used_this_month: int = 0
    monthly_input_token_budget: int | None = None
    monthly_output_token_budget: int | None = None
    input_tokens_used_this_month: int = 0
    output_tokens_used_this_month: int = 0
    # Per-key monthly activation-request cap (ACS-199). 0 = unlimited (legacy
    # convention). Enforced separately from token budgets — the count is a tally
    # of activation requests, checked pre-flight in the completions route.
    monthly_activation_budget: int = 0
    # Per-key monthly cap on self-serve bulk harvest JOBS (ACS-245). 0 =
    # unlimited (legacy convention). Counted against harvest_jobs rows started
    # this month, checked pre-flight in POST /v1/harvest.
    monthly_harvest_budget: int = 0
    # Filled later by the limit-clamp code if it wants to surface which
    # dimension blocked a call; not used by the hot path.
    _last_blocking_dim: str | None = field(default=None, repr=False)

    # ----- legacy accessors (kept so existing tests / call sites work) ------

    @property
    def remaining_budget(self) -> int | None:
        """Per-key monthly total remaining. Legacy alias for the per-key
        dimension — kept so older callers + tests still work. New code should
        prefer ``effective_remaining()`` which considers every dimension.
        """
        if self.monthly_token_budget == 0:
            return None
        return max(0, self.monthly_token_budget - self.tokens_used_this_month)

    # ----- new multi-dimensional helpers ------------------------------------

    def _per_key_monthly_remaining(self) -> int | None:
        """``monthly_token_budget == 0`` is the legacy "unlimited" sentinel."""
        if self.monthly_token_budget == 0:
            return None
        return max(0, self.monthly_token_budget - self.tokens_used_this_month)

    def _user_aggregate_remaining(self) -> int | None:
        if self.monthly_token_budget_total is None:
            return None
        return max(
            0, self.monthly_token_budget_total - self.user_tokens_used_this_month
        )

    def _daily_remaining(self) -> int | None:
        if self.daily_token_budget is None:
            return None
        return max(0, self.daily_token_budget - self.tokens_used_today)

    def _input_remaining(self) -> int | None:
        if self.monthly_input_token_budget is None:
            return None
        return max(
            0, self.monthly_input_token_budget - self.input_tokens_used_this_month
        )

    def _output_remaining(self) -> int | None:
        if self.monthly_output_token_budget is None:
            return None
        return max(
            0, self.monthly_output_token_budget - self.output_tokens_used_this_month
        )

    def effective_remaining(self) -> dict[str, int | None]:
        """Return remaining headroom across every budget dimension.

        Keys: ``key_monthly``, ``user_monthly``, ``daily``, ``input``,
        ``output``. A value of ``None`` for any dimension means that
        dimension is uncapped for this caller; an integer is the remaining
        token count, clamped at 0.

        Callers turn this into a single ``max_tokens`` clamp by taking the
        minimum of every non-None value (with ``input`` handled specially —
        prompt tokens count against it, not output).
        """
        return {
            "key_monthly": self._per_key_monthly_remaining(),
            "user_monthly": self._user_aggregate_remaining(),
            "daily": self._daily_remaining(),
            "input": self._input_remaining(),
            "output": self._output_remaining(),
        }

    def is_unlimited(self) -> bool:
        """True iff every dimension is uncapped — i.e. backward-compat path
        where the only configured limit was ``monthly_token_budget == 0``.
        ``run_completion_nonstream`` uses this to skip the tokenizer call
        entirely (matches the pre-item-9 behaviour and the existing test
        ``test_run_completion_nonstream_unlimited_budget_skips_clamp``).
        """
        return all(v is None for v in self.effective_remaining().values())


# In-process cache of last `last_used_at` write timestamp per key. Bounded by
# the number of distinct keys we hand out — fine to keep in process memory.
_LAST_USED_WRITE: dict[uuid.UUID, float] = {}


def _current_period_start(now: dt.datetime | None = None) -> dt.date:
    n = now or dt.datetime.now(tz=dt.UTC)
    return dt.date(n.year, n.month, 1)


def _current_day_start(now: dt.datetime | None = None) -> dt.date:
    """UTC calendar day used as the ``usage_daily.period_start`` value."""
    n = now or dt.datetime.now(tz=dt.UTC)
    return dt.date(n.year, n.month, n.day)


def _extract_bearer(request: Request, *, log_ip: bool = False) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        _raise_auth_error(
            request,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_kind="missing_authorization",
            message="Missing or malformed Authorization header. Expected 'Bearer <key>'.",
            code="missing_authorization",
            key_prefix=None,
            log_ip=log_ip,
        )
    return auth.split(" ", 1)[1].strip()


async def _load_usage_state(
    session: AsyncSession,
    *,
    key_id: uuid.UUID,
    user_id: uuid.UUID,
) -> dict[str, int]:
    """Fetch the per-key + per-user usage figures used by ``AuthedCaller``.

    One round trip per dimension (3 statements). Cheap because each one is
    a PK lookup or a small aggregate — no full scans.
    """
    period_start = _current_period_start()
    day_start = _current_day_start()

    monthly_row = (
        await session.execute(
            select(UsageMonthly).where(
                UsageMonthly.key_id == key_id,
                UsageMonthly.period_start == period_start,
            )
        )
    ).scalar_one_or_none()
    daily_row = (
        await session.execute(
            select(UsageDaily).where(
                UsageDaily.key_id == key_id,
                UsageDaily.period_start == day_start,
            )
        )
    ).scalar_one_or_none()
    # Aggregate across every key the user owns this month (revoked keys still
    # count — usage history persists once written).
    user_agg = (
        await session.execute(
            select(
                func.coalesce(func.sum(UsageMonthly.tokens_prompt), 0),
                func.coalesce(func.sum(UsageMonthly.tokens_completion), 0),
            )
            .join(ApiKey, UsageMonthly.key_id == ApiKey.id)
            .where(ApiKey.user_id == user_id)
            .where(UsageMonthly.period_start == period_start)
        )
    ).one()
    user_prompt, user_completion = int(user_agg[0] or 0), int(user_agg[1] or 0)

    return {
        "tokens_used_this_month": (
            (monthly_row.tokens_prompt + monthly_row.tokens_completion)
            if monthly_row else 0
        ),
        "input_tokens_used_this_month": monthly_row.tokens_prompt if monthly_row else 0,
        "output_tokens_used_this_month": (
            monthly_row.tokens_completion if monthly_row else 0
        ),
        "tokens_used_today": (
            (daily_row.tokens_prompt + daily_row.tokens_completion)
            if daily_row else 0
        ),
        "user_tokens_used_this_month": user_prompt + user_completion,
    }


async def authenticate(
    request: Request,
    session: AsyncSession,
    last_used_throttle_s: int = 300,
    required_scope: str = "completions",
    log_ip: bool = False,
) -> AuthedCaller:
    plaintext = _extract_bearer(request, log_ip=log_ip)
    key_hash = keymod.hash_key(plaintext)
    # Public prefix of the *presented* token, for log decoration only. None if
    # the caller sent a malformed key. Never the secret/hash — the full key is
    # still hashed above for the DB lookup; we never trust the prefix alone.
    presented_prefix = keymod.parse_prefix(plaintext)

    row = (
        await session.execute(
            select(ApiKey, User).join(User, ApiKey.user_id == User.id).where(
                ApiKey.key_hash == key_hash
            )
        )
    ).one_or_none()

    if row is None:
        _raise_auth_error(
            request,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_kind="invalid_api_key",
            message="Invalid API key.",
            code="invalid_api_key",
            key_prefix=presented_prefix,
            log_ip=log_ip,
        )

    api_key, user = row
    if api_key.revoked_at is not None:
        _raise_auth_error(
            request,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_kind="key_revoked",
            message="This API key has been revoked.",
            code="key_revoked",
            key_prefix=api_key.key_prefix,
            log_ip=log_ip,
        )
    # Reversible pause — rejected the same way as revoke, but the user can
    # resume it from the dashboard. Enforced server-side (not just hidden in UI).
    if api_key.disabled_at is not None:
        _raise_auth_error(
            request,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_kind="key_disabled",
            message="This API key is paused. Resume it from your dashboard to use it again.",
            code="key_disabled",
            key_prefix=api_key.key_prefix,
            log_ip=log_ip,
        )
    # The key's owner must be an approved account (ACS-212). Rejecting or
    # otherwise un-approving a user previously only blocked their web login;
    # their existing keys kept working. There is no auth cache — every request
    # hits this query — so a status flip cuts API access immediately.
    if user.status != "approved":
        # Suspension is reversible and usually scheduled (a time-boxed cohort),
        # so it gets its own kind + copy pointing at the way back — "not
        # approved" would read as a verdict on them (ACS-353).
        suspended = user.status == "suspended"
        kind = ErrorKind.account_suspended if suspended else ErrorKind.account_not_approved
        _raise_auth_error(
            request,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_kind=kind,
            message=(
                "This account's access is suspended. "
                "Email infra@acsresearch.org if you'd like it restored."
                if suspended
                else "This account is not approved for API access."
            ),
            code=kind,
            key_prefix=api_key.key_prefix,
            log_ip=log_ip,
        )
    if required_scope not in (api_key.scopes or []):
        _raise_auth_error(
            request,
            status_code=status.HTTP_403_FORBIDDEN,
            error_kind="scope_denied",
            message=f"Key lacks required scope: {required_scope}.",
            code="scope_denied",
            key_prefix=api_key.key_prefix,
            log_ip=log_ip,
        )

    # Throttled last-used-at update.
    now_ts = time.monotonic()
    prev = _LAST_USED_WRITE.get(api_key.id, 0.0)
    if now_ts - prev > last_used_throttle_s:
        _LAST_USED_WRITE[api_key.id] = now_ts
        await session.execute(
            update(ApiKey).where(ApiKey.id == api_key.id).values(last_used_at=dt.datetime.now(tz=dt.UTC))
        )

    usage_state = await _load_usage_state(
        session, key_id=api_key.id, user_id=api_key.user_id
    )

    return AuthedCaller(
        key_id=api_key.id,
        key_prefix=api_key.key_prefix,
        user_email=user.email,
        monthly_token_budget=api_key.monthly_token_budget,
        tokens_used_this_month=usage_state["tokens_used_this_month"],
        user_id=api_key.user_id,
        daily_token_budget=api_key.daily_token_budget,
        tokens_used_today=usage_state["tokens_used_today"],
        monthly_token_budget_total=user.monthly_token_budget_total,
        user_tokens_used_this_month=usage_state["user_tokens_used_this_month"],
        monthly_input_token_budget=api_key.monthly_input_token_budget,
        monthly_output_token_budget=api_key.monthly_output_token_budget,
        input_tokens_used_this_month=usage_state["input_tokens_used_this_month"],
        output_tokens_used_this_month=usage_state["output_tokens_used_this_month"],
        monthly_activation_budget=api_key.monthly_activation_budget,
        monthly_harvest_budget=api_key.monthly_harvest_budget,
    )


async def primary_authed_caller(
    session: AsyncSession, user_id: uuid.UUID
) -> AuthedCaller | None:
    """Build an AuthedCaller for the user's most recently-created active API key.

    Used by web routes (e.g. `/chat`) that need to charge usage against the
    logged-in user but don't have a bearer header — the user authenticated
    via cookie session, and we attribute API calls to one of their owned keys.

    Returns None if the user has no active (non-revoked) keys. Required scope
    is "completions" — matches what ``authenticate()`` enforces.
    """
    row = (
        await session.execute(
            select(ApiKey, User)
            .join(User, ApiKey.user_id == User.id)
            .where(ApiKey.user_id == user_id)
            .where(ApiKey.revoked_at.is_(None))
            .where(ApiKey.disabled_at.is_(None))
            # Cookie-side mirror of authenticate()'s account-status gate
            # (ACS-212): a rejected/pending owner gets no usable key, so the
            # workbench/loom generation paths can't outlive a rejection.
            .where(User.status == "approved")
            .order_by(ApiKey.created_at.desc())
        )
    ).first()
    if row is None:
        return None
    api_key, user = row
    if "completions" not in (api_key.scopes or []):
        return None

    usage_state = await _load_usage_state(
        session, key_id=api_key.id, user_id=api_key.user_id
    )

    return AuthedCaller(
        key_id=api_key.id,
        key_prefix=api_key.key_prefix,
        user_email=user.email,
        monthly_token_budget=api_key.monthly_token_budget,
        tokens_used_this_month=usage_state["tokens_used_this_month"],
        user_id=api_key.user_id,
        daily_token_budget=api_key.daily_token_budget,
        tokens_used_today=usage_state["tokens_used_today"],
        monthly_token_budget_total=user.monthly_token_budget_total,
        user_tokens_used_this_month=usage_state["user_tokens_used_this_month"],
        monthly_input_token_budget=api_key.monthly_input_token_budget,
        monthly_output_token_budget=api_key.monthly_output_token_budget,
        input_tokens_used_this_month=usage_state["input_tokens_used_this_month"],
        output_tokens_used_this_month=usage_state["output_tokens_used_this_month"],
        monthly_activation_budget=api_key.monthly_activation_budget,
        monthly_harvest_budget=api_key.monthly_harvest_budget,
    )


async def authed_caller_for_key(
    session: AsyncSession, user_id: uuid.UUID, key_id: uuid.UUID
) -> AuthedCaller | None:
    """Build an AuthedCaller for a *specific* key, but only if the logged-in
    user owns it and it's usable.

    Like :func:`primary_authed_caller`, but the caller picks the key (e.g. the
    workbench per-chat key selector) instead of defaulting to the newest one.
    Returns None — so the route falls back to the default key — when the key
    doesn't exist, isn't owned by ``user_id``, is revoked, or lacks the
    ``completions`` scope.
    """
    row = (
        await session.execute(
            select(ApiKey, User)
            .join(User, ApiKey.user_id == User.id)
            .where(ApiKey.id == key_id)
            .where(ApiKey.user_id == user_id)
            .where(ApiKey.revoked_at.is_(None))
            .where(ApiKey.disabled_at.is_(None))
            # Same account-status gate as primary_authed_caller (ACS-212).
            .where(User.status == "approved")
        )
    ).first()
    if row is None:
        return None
    api_key, user = row
    if "completions" not in (api_key.scopes or []):
        return None

    usage_state = await _load_usage_state(
        session, key_id=api_key.id, user_id=api_key.user_id
    )

    return AuthedCaller(
        key_id=api_key.id,
        key_prefix=api_key.key_prefix,
        user_email=user.email,
        monthly_token_budget=api_key.monthly_token_budget,
        tokens_used_this_month=usage_state["tokens_used_this_month"],
        user_id=api_key.user_id,
        daily_token_budget=api_key.daily_token_budget,
        tokens_used_today=usage_state["tokens_used_today"],
        monthly_token_budget_total=user.monthly_token_budget_total,
        user_tokens_used_this_month=usage_state["user_tokens_used_this_month"],
        monthly_input_token_budget=api_key.monthly_input_token_budget,
        monthly_output_token_budget=api_key.monthly_output_token_budget,
        input_tokens_used_this_month=usage_state["input_tokens_used_this_month"],
        output_tokens_used_this_month=usage_state["output_tokens_used_this_month"],
        monthly_activation_budget=api_key.monthly_activation_budget,
        monthly_harvest_budget=api_key.monthly_harvest_budget,
    )


def assert_admin(request: Request, admin_token: str) -> None:
    presented = request.headers.get("x-admin-token", "")
    if not presented or presented != admin_token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Admin auth required.", "code": "admin_auth"}},
        )


async def commit_usage(
    session: AsyncSession, key_id: uuid.UUID, prompt_tokens: int, completion_tokens: int
) -> None:
    """Increment this-month AND today's usage for a key in one transaction.

    Postgres `INSERT ... ON CONFLICT DO UPDATE` keeps each row atomic; two
    concurrent commits won't lose increments (though the auth-time read may be
    stale by one request, which the plan documents as accepted). Item 9 added
    the second write (``usage_daily``) — both happen on the same session, so
    the per-request commit either persists both or rolls both back.
    """
    from sqlalchemy.dialects.postgresql import insert

    monthly_stmt = (
        insert(UsageMonthly)
        .values(
            key_id=key_id,
            period_start=_current_period_start(),
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            request_count=1,
        )
        .on_conflict_do_update(
            index_elements=[UsageMonthly.key_id, UsageMonthly.period_start],
            set_={
                "tokens_prompt": UsageMonthly.tokens_prompt + prompt_tokens,
                "tokens_completion": UsageMonthly.tokens_completion + completion_tokens,
                "request_count": UsageMonthly.request_count + 1,
            },
        )
    )
    daily_stmt = (
        insert(UsageDaily)
        .values(
            key_id=key_id,
            period_start=_current_day_start(),
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            request_count=1,
        )
        .on_conflict_do_update(
            index_elements=[UsageDaily.key_id, UsageDaily.period_start],
            set_={
                "tokens_prompt": UsageDaily.tokens_prompt + prompt_tokens,
                "tokens_completion": UsageDaily.tokens_completion + completion_tokens,
                "request_count": UsageDaily.request_count + 1,
            },
        )
    )
    await session.execute(monthly_stmt)
    await session.execute(daily_stmt)
