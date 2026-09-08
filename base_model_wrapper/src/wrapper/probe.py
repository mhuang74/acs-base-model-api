"""Capacity-probe scheduling glue.

The probe used to live as a ``modal.Cron`` inside ``serving/capacity_scheduled.py``.
We moved the schedule into Postgres (``probe_schedule.cron_expression``) and
run it from APScheduler on the wrapper side, so admins can edit the cadence
from /admin without redeploying Modal.

The Modal-side function is now an HTTP-triggered ``fastapi_endpoint``; this
module POSTs to it with a bearer token and writes the outcome to
``probe_results``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from .logging import get_logger
from .models import ProbeResult, ProbeSchedule

if TYPE_CHECKING:
    from .settings import Settings

log = get_logger()


async def run_probe(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> ProbeResult:
    """Fire the Modal HTTP probe endpoint and persist the outcome.

    Returns the inserted ``ProbeResult`` (with a populated ``id`` and ``fired_at``).
    Never raises — failures land in the row as ``ok=False`` with the exception
    text on ``error``.
    """
    t0 = time.monotonic()
    ok = False
    gpu_type: str | None = None
    cloud: str | None = None
    region: str | None = None
    gpu_count: int | None = None
    gpu_memory_total_mb: float | None = None
    gpu_memory_total_std_mb: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_used_std_mb: float | None = None
    gpu_utilization_pct: float | None = None
    gpu_utilization_std_pct: float | None = None
    gpu_temperature_c: float | None = None
    gpu_temperature_std_c: float | None = None
    gpu_power_w: float | None = None
    gpu_power_std_w: float | None = None
    driver_version: str | None = None
    error: str | None = None

    if not settings.modal_probe_url:
        error = "MODAL_PROBE_URL not configured"
    elif not settings.modal_probe_bearer:
        error = "MODAL_PROBE_BEARER not configured"
    else:
        try:
            async with httpx.AsyncClient(timeout=600) as http:
                r = await http.post(
                    settings.modal_probe_url,
                    headers={
                        "Authorization": f"Bearer {settings.modal_probe_bearer}"
                    },
                )
                r.raise_for_status()
                payload = r.json()
                ok = bool(payload.get("ok", True))
                gpu_type = payload.get("gpu_type")
                cloud = payload.get("cloud")
                region = payload.get("region")
                gpu_count = payload.get("gpu_count")
                gpu_memory_total_mb = payload.get("gpu_memory_total_mb")
                gpu_memory_total_std_mb = payload.get("gpu_memory_total_std_mb")
                gpu_memory_used_mb = payload.get("gpu_memory_used_mb")
                gpu_memory_used_std_mb = payload.get("gpu_memory_used_std_mb")
                gpu_utilization_pct = payload.get("gpu_utilization_pct")
                gpu_utilization_std_pct = payload.get("gpu_utilization_std_pct")
                gpu_temperature_c = payload.get("gpu_temperature_c")
                gpu_temperature_std_c = payload.get("gpu_temperature_std_c")
                gpu_power_w = payload.get("gpu_power_w")
                gpu_power_std_w = payload.get("gpu_power_std_w")
                driver_version = payload.get("driver_version")
                if not ok:
                    error = payload.get("error") or "probe reported ok=false"
        except Exception as exc:  # noqa: BLE001 — scheduler must never raise
            error = f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - t0
    row = ProbeResult(
        ok=ok,
        elapsed_s=elapsed,
        gpu_type=gpu_type,
        cloud=cloud,
        region=region,
        gpu_count=gpu_count,
        gpu_memory_total_mb=gpu_memory_total_mb,
        gpu_memory_total_std_mb=gpu_memory_total_std_mb,
        gpu_memory_used_mb=gpu_memory_used_mb,
        gpu_memory_used_std_mb=gpu_memory_used_std_mb,
        gpu_utilization_pct=gpu_utilization_pct,
        gpu_utilization_std_pct=gpu_utilization_std_pct,
        gpu_temperature_c=gpu_temperature_c,
        gpu_temperature_std_c=gpu_temperature_std_c,
        gpu_power_w=gpu_power_w,
        gpu_power_std_w=gpu_power_std_w,
        driver_version=driver_version,
        error=error,
    )
    async with session_factory() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    log.info(
        "probe_fired",
        ok=ok,
        elapsed_s=round(elapsed, 2),
        gpu_type=gpu_type,
        cloud=cloud,
        region=region,
        gpu_count=gpu_count,
        gpu_utilization_pct=gpu_utilization_pct,
        gpu_memory_used_mb=gpu_memory_used_mb,
        gpu_temperature_c=gpu_temperature_c,
        gpu_power_w=gpu_power_w,
        error=error,
    )
    return row


async def load_cron_expression(
    session_factory: async_sessionmaker[AsyncSession],
) -> str:
    """Return the current cron expression, or seed + return the default."""
    async with session_factory() as session:
        row = (
            await session.execute(select(ProbeSchedule).where(ProbeSchedule.id == 1))
        ).scalar_one_or_none()
        if row is None:
            row = ProbeSchedule(id=1, cron_expression="0 7,10,14,16,19,23 * * *")
            session.add(row)
            await session.commit()
            await session.refresh(row)
        return row.cron_expression
