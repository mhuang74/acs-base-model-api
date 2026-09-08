"""Per-model warm-window admin routes."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Form, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ... import web_auth as webauth
from ...db import get_session
from ...lifespan import _register_warm_window_jobs, _unregister_warm_window_jobs
from ...models import ModelWarmWindow, User
from .common import _admin_redirect, _lookup_model_or_404, log

router = APIRouter()


def _validate_warm_window_form(warm_cron: str, cool_cron: str, timezone_name: str) -> str | None:
    """Return an error message if any field is invalid, else None."""
    from croniter import croniter as _croniter
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    if not _croniter.is_valid(warm_cron):
        return f"Invalid warm cron expression: {warm_cron!r}"
    if not _croniter.is_valid(cool_cron):
        return f"Invalid cool cron expression: {cool_cron!r}"
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return f"Invalid timezone: {timezone_name!r}"
    return None


@router.post("/admin/models/{model_id}/warm-window")
async def admin_warm_window_upsert(
    request: Request,
    model_id: str,
    warm_cron: str = Form(...),
    cool_cron: str = Form(...),
    timezone: str = Form("UTC"),
    enabled: str | None = Form(None),
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    _lookup_model_or_404(request, model_id)
    warm_cron = warm_cron.strip()
    cool_cron = cool_cron.strip()
    timezone = timezone.strip() or "UTC"
    enabled_bool = enabled is not None and enabled not in ("", "false", "0", "off")

    err = _validate_warm_window_form(warm_cron, cool_cron, timezone)
    if err is not None:
        log.info(
            "admin_warm_window_rejected",
            admin_id=str(admin.id),
            model_id=model_id,
            warm_cron=warm_cron,
            cool_cron=cool_cron,
            timezone=timezone,
            error=err,
        )
        return _admin_redirect(err=err)

    row = (
        await session.execute(select(ModelWarmWindow).where(ModelWarmWindow.model_id == model_id))
    ).scalar_one_or_none()
    if row is None:
        row = ModelWarmWindow(
            model_id=model_id,
            warm_cron=warm_cron,
            cool_cron=cool_cron,
            timezone=timezone,
            enabled=enabled_bool,
            updated_by_user_id=admin.id,
        )
        session.add(row)
    else:
        row.warm_cron = warm_cron
        row.cool_cron = cool_cron
        row.timezone = timezone
        row.enabled = enabled_bool
        row.updated_at = dt.datetime.now(tz=dt.UTC)
        row.updated_by_user_id = admin.id

    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        if enabled_bool:
            try:
                _register_warm_window_jobs(scheduler, request.app, row)
            except Exception as exc:  # noqa: BLE001 — reschedule failure shouldn't block the DB write
                log.warning(
                    "admin_warm_window_reschedule_failed",
                    model_id=model_id,
                    error=str(exc),
                )
        else:
            _unregister_warm_window_jobs(scheduler, model_id)

    log.info(
        "admin_warm_window_updated",
        admin_id=str(admin.id),
        model_id=model_id,
        warm_cron=warm_cron,
        cool_cron=cool_cron,
        timezone=timezone,
        enabled=enabled_bool,
    )
    return _admin_redirect(msg=f"Warm window saved for {model_id}.")


@router.post("/admin/models/{model_id}/warm-window/delete")
async def admin_warm_window_delete(
    request: Request,
    model_id: str,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    _lookup_model_or_404(request, model_id)
    row = (
        await session.execute(select(ModelWarmWindow).where(ModelWarmWindow.model_id == model_id))
    ).scalar_one_or_none()
    if row is not None:
        await session.delete(row)

    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        _unregister_warm_window_jobs(scheduler, model_id)

    log.info(
        "admin_warm_window_deleted",
        admin_id=str(admin.id),
        model_id=model_id,
    )
    return _admin_redirect(msg=f"Warm window removed for {model_id}.")
