"""Aggregated wrapper/backend health service."""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

from sqlalchemy import text


async def ping_database(engine) -> tuple[bool, int]:
    """Quick bounded ``SELECT 1`` round-trip. Returns ``(ok, latency_ms)``."""
    import time as _time

    t0 = _time.monotonic()
    try:
        async with asyncio.timeout(0.5):
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        return True, int((_time.monotonic() - t0) * 1000)
    except Exception:
        return False, int((_time.monotonic() - t0) * 1000)


def backend_health_entry(
    entry,
    breaker_status,
    last_completion_at,
    now: dt.datetime,
) -> dict[str, Any]:
    """Per-backend block for /health, derived from in-memory state."""
    last_at_iso = None
    seconds_ago = None
    warm_estimate = "cold"
    if last_completion_at is not None:
        last_at_iso = last_completion_at.isoformat()
        delta = now - last_completion_at
        seconds_ago = int(delta.total_seconds())
        # Per-model serving scaledown window (ACS-226), so warm_estimate flips
        # when this model's serving engine actually tears down rather than at a
        # global 30-min guess. /health iterates serving model ids, so the
        # activation window doesn't apply here.
        if delta < dt.timedelta(seconds=entry.scaledown_window_s):
            warm_estimate = "warm"
    return {
        "model_id": entry.model_id,
        "status": entry.status,
        "gpu_shape": entry.gpu_shape_label,
        "breaker_state": breaker_status.state,
        "breaker_consecutive_failures": breaker_status.consecutive_failures,
        "breaker_last_failure_kind": breaker_status.last_failure_kind or None,
        "last_completion_at": last_at_iso,
        "last_completion_seconds_ago": seconds_ago,
        "warm_estimate": warm_estimate,
    }


async def aggregate_health(app_state) -> tuple[int, dict[str, Any]]:
    """Build the /health response from in-memory state plus a single DB ping."""
    now = dt.datetime.now(tz=dt.UTC)
    boot_time = getattr(app_state, "boot_time", now)
    uptime_s = int((now - boot_time).total_seconds())

    db_ok, db_latency_ms = await ping_database(app_state.engine)

    registry = app_state.models
    breakers = app_state.breakers
    last_completion_map = app_state.last_completion_at

    backends: list[dict[str, Any]] = []
    open_backends: list[str] = []
    for model_id in sorted(registry.keys()):
        entry = registry[model_id]
        if entry.status == "disabled":
            continue
        snap = breakers.snapshot(model_id)
        backends.append(
            backend_health_entry(
                entry,
                snap,
                last_completion_map.get(model_id),
                now,
            )
        )
        if snap.state == "open":
            open_backends.append(model_id)

    if not db_ok:
        overall = "down"
        http_status = 503
    elif open_backends:
        overall = "degraded"
        http_status = 200
    else:
        overall = "ok"
        http_status = 200

    body = {
        "status": overall,
        "wrapper": {
            "uptime_seconds": uptime_s,
            "database": {
                "ok": db_ok,
                "latency_ms": db_latency_ms,
            },
        },
        "backends": backends,
        "degraded_backends": open_backends,
    }
    return http_status, body
