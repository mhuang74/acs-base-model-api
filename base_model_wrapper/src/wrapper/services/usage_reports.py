"""Per-user usage report query service.

Builds the template context for the end-user ``/usage`` tab. The API enforces
budgets along several dimensions — a per-user monthly *total*, per-key monthly
input/output caps, and per-key *daily* ceilings (see ``auth._usage_snapshot``
and ``models.ApiKey``). This service surfaces all of them so the headline
figure is no longer the only thing a tester can see.

Privacy invariant: everything here is counts / metadata (token totals, request
counts, timestamps, latencies). No prompt or completion *text* is ever read or
returned.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import auth as authmod
from ..models import ApiKey, ApiRequest, UsageDaily, UsageMonthly, User

# How many days of history the daily chart covers.
_CHART_DAYS = 30


def _budget_row(used: int, budget: int | None, label: str) -> dict[str, Any] | None:
    """Build a progress-bar descriptor, or ``None`` when no budget is set.

    ``pct`` is clamped to [0, 100] for the bar width; ``over`` flags an
    exhausted budget. ``state`` drives the colour threshold in the template
    (ok < 80% ≤ warn < 100% ≤ over) — mirrors the Vercel/Stripe convention of
    going amber as a quota nears its limit.
    """
    if not budget or budget <= 0:
        return None
    raw_pct = 100.0 * used / budget
    pct = min(raw_pct, 100.0)
    remaining = max(budget - used, 0)
    if raw_pct >= 100:
        state = "over"
    elif raw_pct >= 80:
        state = "warn"
    else:
        state = "ok"
    return {
        "label": label,
        "used": used,
        "budget": budget,
        "remaining": remaining,
        "pct": pct,
        "pct_label": raw_pct,
        "state": state,
    }


async def build_usage_context(
    session: AsyncSession,
    user: User,
    *,
    selected_key: str | None = None,
) -> dict[str, Any]:
    """Build the template context for the per-user usage tab.

    ``selected_key`` is the raw ``?key=`` query value. When it parses to a UUID
    of one of the caller's own keys, the whole view (totals, chart, breakdowns,
    recent log, budgets) is scoped to that single key; otherwise the view falls
    back to the "All keys" aggregate. We never trust the value to belong to the
    user without checking — an unknown / other-user id silently degrades to the
    all-keys view rather than leaking or erroring.
    """
    period_start = authmod._current_period_start()
    today = authmod._current_day_start()
    chart_start = today - dt.timedelta(days=_CHART_DAYS)
    last_month_start = (period_start - dt.timedelta(days=1)).replace(day=1)

    key_rows = list(
        (
            await session.execute(
                select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    all_key_ids = [k.id for k in key_rows]

    # Resolve the optional ?key= filter against the caller's own keys only.
    selected_key_obj: ApiKey | None = None
    if selected_key:
        try:
            wanted = uuid.UUID(selected_key)
        except (ValueError, AttributeError):
            wanted = None
        if wanted is not None:
            selected_key_obj = next((k for k in key_rows if k.id == wanted), None)

    # The id set every aggregate below is scoped to: one key when filtered,
    # else all the caller's keys.
    if selected_key_obj is not None:
        scoped_ids = [selected_key_obj.id]
    else:
        scoped_ids = all_key_ids

    base = {
        "user": user,
        "month_budget_total": user.monthly_token_budget_total,
        "period_start": period_start,
        "today": today,
        "chart_days": _CHART_DAYS,
        # Selector state for the template.
        "key_options": [
            {"id": str(k.id), "name": k.name, "key_prefix": k.key_prefix}
            for k in key_rows
        ],
        "selected_key_id": str(selected_key_obj.id) if selected_key_obj else None,
        "selected_key_name": (
            (selected_key_obj.name or selected_key_obj.key_prefix)
            if selected_key_obj
            else None
        ),
    }
    if not all_key_ids:
        return {
            **base,
            "has_keys": False,
            "month_total": 0,
            "month_input": 0,
            "month_output": 0,
            "month_requests": 0,
            "today_total": 0,
            "budgets": [],
            "daily_series": [],
            "daily_max": 0,
            "per_key": [],
            "per_endpoint": [],
            "recent_requests": [],
        }

    # --- this-month totals, split by direction + request count ---------------
    monthly_agg = (
        await session.execute(
            select(
                func.coalesce(func.sum(UsageMonthly.tokens_prompt), 0),
                func.coalesce(func.sum(UsageMonthly.tokens_completion), 0),
                func.coalesce(func.sum(UsageMonthly.request_count), 0),
            ).where(
                UsageMonthly.key_id.in_(scoped_ids),
                UsageMonthly.period_start == period_start,
            )
        )
    ).one()
    month_input = int(monthly_agg[0] or 0)
    month_output = int(monthly_agg[1] or 0)
    month_requests = int(monthly_agg[2] or 0)
    month_total = month_input + month_output

    # --- today's total (for the daily-budget headroom bar) -------------------
    today_agg = (
        await session.execute(
            select(
                func.coalesce(func.sum(UsageDaily.tokens_prompt), 0),
                func.coalesce(func.sum(UsageDaily.tokens_completion), 0),
            ).where(
                UsageDaily.key_id.in_(scoped_ids),
                UsageDaily.period_start == today,
            )
        )
    ).one()
    today_total = int(today_agg[0] or 0) + int(today_agg[1] or 0)

    # --- budget progress bars ------------------------------------------------
    # When a single key is selected, show *that key's own* caps (these are the
    # ceilings the API actually clamps against per key). For the "All keys" view
    # the per-direction/daily caps live on the key not the user, so we sum them
    # across *active* keys (a revoked key can't accrue new usage) to present an
    # effective user-level ceiling, alongside the genuine per-user monthly total
    # cap. A bar is only shown when the relevant cap is actually set.
    if selected_key_obj is not None:
        monthly_total_budget = selected_key_obj.monthly_token_budget or None
        input_budget = selected_key_obj.monthly_input_token_budget or None
        output_budget = selected_key_obj.monthly_output_token_budget or None
        daily_budget = selected_key_obj.daily_token_budget or None
    else:
        active_keys = [k for k in key_rows if k.revoked_at is None]
        monthly_total_budget = user.monthly_token_budget_total
        input_budget = sum(
            (k.monthly_input_token_budget or 0) for k in active_keys
        ) or None
        output_budget = sum(
            (k.monthly_output_token_budget or 0) for k in active_keys
        ) or None
        daily_budget = sum((k.daily_token_budget or 0) for k in active_keys) or None

    # The "Monthly total" bar means two different caps depending on scope: the
    # account-wide user cap in the all-keys view vs. this key's own cap when
    # filtered. Qualify the label so the reused word isn't ambiguous, matching how
    # the Dashboard names the budget scope ("across all your keys" vs. per-key) —
    # ACS-346.
    total_scope = "this key" if selected_key_obj is not None else "all keys"
    budgets = [
        b
        for b in (
            _budget_row(month_total, monthly_total_budget, f"Monthly total ({total_scope})"),
            _budget_row(month_input, input_budget, "Monthly input"),
            _budget_row(month_output, output_budget, "Monthly output"),
            _budget_row(today_total, daily_budget, "Today"),
        )
        if b is not None
    ]

    # --- 30-day daily series, split by direction + request count -------------
    daily_rows = list(
        (
            await session.execute(
                select(
                    UsageDaily.period_start,
                    func.sum(UsageDaily.tokens_prompt),
                    func.sum(UsageDaily.tokens_completion),
                    func.sum(UsageDaily.request_count),
                )
                .where(
                    UsageDaily.key_id.in_(scoped_ids),
                    UsageDaily.period_start >= chart_start,
                )
                .group_by(UsageDaily.period_start)
                .order_by(UsageDaily.period_start.asc())
            )
        ).all()
    )
    daily_map = {
        row[0]: {
            "input": int(row[1] or 0),
            "output": int(row[2] or 0),
            "requests": int(row[3] or 0),
        }
        for row in daily_rows
    }
    daily_series = []
    for i in range(_CHART_DAYS + 1):
        day = chart_start + dt.timedelta(days=i)
        d = daily_map.get(day, {"input": 0, "output": 0, "requests": 0})
        daily_series.append(
            {
                "date": day,
                "input": d["input"],
                "output": d["output"],
                "tokens": d["input"] + d["output"],
                "requests": d["requests"],
            }
        )
    daily_max = max((s["tokens"] for s in daily_series), default=0)

    # --- per-key breakdown ---------------------------------------------------
    per_key = []
    for key in key_rows:
        this_row = (
            await session.execute(
                select(UsageMonthly).where(
                    UsageMonthly.key_id == key.id,
                    UsageMonthly.period_start == period_start,
                )
            )
        ).scalar_one_or_none()
        last_row = (
            await session.execute(
                select(UsageMonthly).where(
                    UsageMonthly.key_id == key.id,
                    UsageMonthly.period_start == last_month_start,
                )
            )
        ).scalar_one_or_none()
        lifetime = (
            await session.execute(
                select(
                    func.coalesce(
                        func.sum(UsageMonthly.tokens_prompt + UsageMonthly.tokens_completion),
                        0,
                    )
                ).where(UsageMonthly.key_id == key.id)
            )
        ).scalar_one()
        per_key.append(
            {
                "id": str(key.id),
                "key_prefix": key.key_prefix,
                "name": key.name,
                "revoked": key.revoked_at is not None,
                "disabled": key.disabled_at is not None,
                "this_month": (
                    this_row.tokens_prompt + this_row.tokens_completion if this_row else 0
                ),
                "this_month_input": this_row.tokens_prompt if this_row else 0,
                "this_month_output": this_row.tokens_completion if this_row else 0,
                "last_month": (
                    last_row.tokens_prompt + last_row.tokens_completion if last_row else 0
                ),
                "lifetime": int(lifetime or 0),
                "monthly_budget": key.monthly_token_budget or None,
            }
        )

    # --- per-endpoint breakdown (this month) ---------------------------------
    endpoint_rows = list(
        (
            await session.execute(
                select(
                    ApiRequest.endpoint,
                    func.count().label("requests"),
                    func.coalesce(func.sum(ApiRequest.n_prompt), 0),
                    func.coalesce(func.sum(ApiRequest.n_completion), 0),
                )
                .where(
                    ApiRequest.key_id.in_(scoped_ids),
                    ApiRequest.ts
                    >= dt.datetime.combine(
                        period_start,
                        dt.time.min,
                        tzinfo=dt.UTC,
                    ),
                )
                .group_by(ApiRequest.endpoint)
                .order_by(func.count().desc())
            )
        ).all()
    )
    per_endpoint = [
        {
            "endpoint": row[0],
            "requests": int(row[1]),
            "input": int(row[2]),
            "output": int(row[3]),
            "tokens": int(row[2]) + int(row[3]),
        }
        for row in endpoint_rows
    ]

    # --- recent requests -----------------------------------------------------
    key_prefix_by_id = {key.id: key.key_prefix for key in key_rows}
    recent_rows = list(
        (
            await session.execute(
                select(ApiRequest)
                .where(ApiRequest.key_id.in_(scoped_ids))
                .order_by(ApiRequest.ts.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    recent_requests = [
        {
            "ts": request.ts,
            "key_prefix": key_prefix_by_id.get(request.key_id, "-"),
            "endpoint": request.endpoint,
            "model": request.model,
            "input": request.n_prompt or 0,
            "output": request.n_completion or 0,
            "tokens": (request.n_prompt or 0) + (request.n_completion or 0),
            "latency_ms": request.latency_ms,
            "status": request.status,
        }
        for request in recent_rows
    ]

    return {
        **base,
        "has_keys": True,
        "month_total": month_total,
        "month_input": month_input,
        "month_output": month_output,
        "month_requests": month_requests,
        "today_total": today_total,
        "budgets": budgets,
        "daily_series": daily_series,
        "daily_max": daily_max,
        "per_key": per_key,
        "per_endpoint": per_endpoint,
        "recent_requests": recent_requests,
    }
