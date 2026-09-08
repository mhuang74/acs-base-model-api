"""Circuit-breaker admin APIs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ... import auth as authmod
from ... import breaker as breakermod
from ...dependencies import get_settings
from ...settings import Settings

router = APIRouter()


@router.get("/admin/breakers")
async def admin_list_breakers(request: Request, settings: Settings = Depends(get_settings)):
    """Read-only view of the per-backend circuit-breaker state.

    Admin-token gated (matches /admin/keys etc.) so on-call can see which
    backends are tripped without needing to log into the web UI. JSON
    response so it's tool-friendly (``jq``, scripts).
    """
    authmod.assert_admin(request, settings.admin_token)
    breakers: breakermod.BackendBreakers = request.app.state.breakers
    # Iterate over the registry rather than the breaker's internal dict so
    # models that haven't received any requests yet still surface (state =
    # "closed" by default — no failures recorded).
    registry = request.app.state.models
    items: list[dict[str, Any]] = []
    for model_id in sorted(registry.keys()):
        snap = breakers.snapshot(model_id)
        items.append(
            {
                "model_id": snap.model_id,
                "state": snap.state,
                "consecutive_failures": snap.consecutive_failures,
                "opens_until": snap.opens_until,
                "last_failure_kind": snap.last_failure_kind,
                "total_failures": snap.total_failures,
                "total_successes": snap.total_successes,
                "total_trips": snap.total_trips,
                "state_changed_at": snap.state_changed_at,
            }
        )
    return {
        "object": "list",
        "data": items,
        "policy": {
            "failure_threshold": breakermod.FAILURE_THRESHOLD,
            "open_duration_s": breakermod.OPEN_DURATION_S,
        },
    }


@router.post("/admin/breakers/{model_id}/reset")
async def admin_reset_breaker(
    model_id: str, request: Request, settings: Settings = Depends(get_settings)
):
    """Force-close a breaker. Counters preserved for audit. Use after fixing
    the underlying upstream issue so traffic resumes immediately instead of
    waiting out the half-open probe."""
    authmod.assert_admin(request, settings.admin_token)
    if model_id not in request.app.state.models:
        return JSONResponse(
            status_code=404,
            content={
                "error": {"code": "model_not_found", "message": f"Unknown model {model_id!r}"}
            },
        )
    breakers: breakermod.BackendBreakers = request.app.state.breakers
    await breakers.reset(model_id)
    return {"model_id": model_id, "state": "closed"}


# --- admin web UI (cookie session, role='admin') ----------------------------
