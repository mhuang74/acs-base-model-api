"""Capacity-probe admin routes."""

from __future__ import annotations

import asyncio
import datetime as dt

from fastapi import APIRouter, Depends, Form, Request
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ... import web_auth as webauth
from ...db import get_session
from ...lifespan import _probe_job_wrapper
from ...models import ProbeResult, ProbeSchedule, User
from ...services.availability import (
    compute_availability,
    list_models_with_recent_requests,
)
from .common import _uptime_redirect, log, templates

router = APIRouter()


# Per-model availability windows surfaced on /admin/uptime. 24h is the
# operator's "what's going on right now" view; 7d catches slower bleeds and
# is short enough that the aggregate query stays cheap even without a model+ts
# index (beta traffic volume is low — revisit if api_requests grows past ~10M).
_AVAILABILITY_WINDOWS: list[tuple[str, dt.timedelta]] = [
    ("24h", dt.timedelta(hours=24)),
    ("7d", dt.timedelta(days=7)),
]


@router.get("/admin/uptime")
async def admin_uptime_page(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Capacity-probe page — schedule, trigger, and recent results.

    Also surfaces per-model availability (ACS-51): uptime %, MTTR, and
    downtime windows derived from ``api_requests`` over 24h / 7d. No
    separate breaker-event log — every breaker trip that affected a real
    caller is already an ``error_kind='circuit_open'`` row on that table.
    """
    sched_row = (
        await session.execute(select(ProbeSchedule).where(ProbeSchedule.id == 1))
    ).scalar_one_or_none()
    cron_expression = (
        sched_row.cron_expression if sched_row is not None else "0 7,10,14,16,19,23 * * *"
    )
    probe_results = list(
        (await session.execute(select(ProbeResult).order_by(ProbeResult.fired_at.desc()).limit(20)))
        .scalars()
        .all()
    )

    # Build the availability matrix: one row per (model, window). The 7d
    # window is the source of truth for which models appear — a model that
    # only saw traffic >7d ago is stale enough to hide.
    longest_window = max(w for _, w in _AVAILABILITY_WINDOWS)
    models = await list_models_with_recent_requests(session, window=longest_window)
    availability_rows: list[dict] = []
    for model_id in models:
        per_window = {}
        for label, window in _AVAILABILITY_WINDOWS:
            per_window[label] = await compute_availability(
                session, model_id=model_id, window=window
            )
        availability_rows.append({"model_id": model_id, "per_window": per_window})

    # Data-volume tile (ACS-26 #4). One row from pg_total_relation_size +
    # COUNT(*); cheap enough that the page can run it inline. Postgres-only
    # — sqlite tests don't exercise this path (the dev-local-on-sqlite story
    # isn't a thing here, dev-local.sh boots Postgres). Failures are caught
    # so the page still renders if the role lacks the privilege. If
    # ``api_requests`` grows past ~10M rows the visibility-checked COUNT(*)
    # starts to drag this foreground page — swap to ``pg_class.reltuples``
    # (planner statistic, O(1)) when that becomes a concern.
    api_requests_volume: dict | None = None
    try:
        row = (
            await session.execute(
                text(
                    "SELECT COUNT(*) AS row_count, "
                    "pg_total_relation_size('api_requests') AS size_bytes "
                    "FROM api_requests"
                )
            )
        ).one()
        api_requests_volume = {
            "row_count": int(row.row_count),
            "size_bytes": int(row.size_bytes or 0),
        }
    except Exception as exc:  # noqa: BLE001 — observability tile, don't 500 the page
        log.warning("admin_uptime_data_volume_failed", error=str(exc))

    flash_message = request.query_params.get("msg") or None
    flash_error = request.query_params.get("err") or None
    return templates.TemplateResponse(
        request,
        "admin_uptime.html",
        {
            "user": admin,
            "probe": {
                "cron_expression": cron_expression,
                "results": probe_results,
            },
            "availability": {
                "window_labels": [label for label, _ in _AVAILABILITY_WINDOWS],
                "rows": availability_rows,
            },
            "data_volume": api_requests_volume,
            "flash_message": flash_message,
            "flash_error": flash_error,
        },
    )


@router.post("/admin/probe/schedule")
async def admin_probe_schedule(
    request: Request,
    cron_expression: str = Form(...),
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    from croniter import croniter as _croniter

    expr = cron_expression.strip()
    if not _croniter.is_valid(expr):
        log.info(
            "admin_probe_schedule_rejected",
            admin_id=str(admin.id),
            expression=expr,
        )
        return _uptime_redirect(err=f"Invalid cron expression: {expr!r}")

    row = (
        await session.execute(select(ProbeSchedule).where(ProbeSchedule.id == 1))
    ).scalar_one_or_none()
    if row is None:
        row = ProbeSchedule(id=1, cron_expression=expr, updated_by_user_id=admin.id)
        session.add(row)
    else:
        row.cron_expression = expr
        row.updated_at = dt.datetime.now(tz=dt.UTC)
        row.updated_by_user_id = admin.id

    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        try:
            from apscheduler.triggers.cron import CronTrigger

            scheduler.add_job(
                _probe_job_wrapper,
                CronTrigger.from_crontab(expr, timezone="UTC"),
                id="capacity_probe",
                replace_existing=True,
                kwargs={"app": request.app},
            )
        except Exception as exc:  # noqa: BLE001 — reschedule failure shouldn't block the DB write
            log.warning("admin_probe_reschedule_failed", error=str(exc))

    log.info(
        "admin_probe_schedule_updated",
        admin_id=str(admin.id),
        expression=expr,
    )
    return _uptime_redirect(msg=f"Probe schedule updated to {expr!r}.")


@router.post("/admin/probe/trigger")
async def admin_probe_trigger(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
):
    log.info("admin_probe_triggered", admin_id=str(admin.id))
    # Fire-and-forget: don't wait for the (potentially long) Modal round-trip.
    asyncio.create_task(_probe_job_wrapper(request.app))
    return _uptime_redirect(msg="Probe triggered — results will appear shortly.")


# --- admin: per-model warm windows ------------------------------------------
