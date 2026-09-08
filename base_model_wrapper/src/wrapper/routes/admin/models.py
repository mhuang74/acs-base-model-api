"""Admin model lifecycle routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ... import modal_ops as modalops
from ... import web_auth as webauth
from ...db import get_session
from ...dependencies import get_settings
from ...models import ModelWarmWindow, User
from ...services.workbench import model_spec_for
from ...settings import Settings
from .common import (
    _admin_redirect,
    _lookup_model_or_404,
    _modal_ops_or_503,
    _modal_unavailable_msg,
    log,
    templates,
)

router = APIRouter()


@router.get("/admin/models")
async def admin_models_page(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    settings: Settings = request.app.state.settings
    registry = request.app.state.models
    modal_unavailable_msg = _modal_unavailable_msg(settings)
    model_rows = await _resolve_model_states(registry, modal_unavailable_msg)

    warm_window_rows = list((await session.execute(select(ModelWarmWindow))).scalars().all())
    warm_windows_by_id = {w.model_id: w for w in warm_window_rows}
    for row in model_rows:
        row["warm_window"] = warm_windows_by_id.get(row["id"])

    flash_message = request.query_params.get("msg") or None
    flash_error = request.query_params.get("err") or None
    return templates.TemplateResponse(
        request,
        "admin_models.html",
        {
            "user": admin,
            "models": model_rows,
            "modal_unavailable": modal_unavailable_msg,
            "flash_message": flash_message,
            "flash_error": flash_error,
        },
    )


async def _resolve_model_states(
    registry: dict[str, Any], modal_unavailable_msg: str | None
) -> list[dict[str, Any]]:
    """Fan out Modal state + live-container lookups for every registered model.

    Shared by the /admin HTML render and the /admin/models/status JSON
    poll. ``return_exceptions=True`` keeps one slow / failed RPC from
    breaking the whole render — the offending row surfaces as ``unknown``
    (state) and ``live_containers: None`` (live count unknown, NOT a
    confident 0 — see ACS-128).

    Per-row scaling params (``min_containers`` / ``max_containers`` /
    ``scaledown_window_s``) come from the shared ``acs_model_registry`` —
    they're baked into each model's Modal app at deploy time and are not
    carried on the wrapper-side ``ModelEntry``. Operators can override
    ``min_containers`` at runtime via Keep warm / Release; that override
    isn't reflected here (Modal doesn't expose it back over RPC). The
    warm-window summary on the page documents the scheduled toggles.
    """
    entries = list(registry.values())
    lookup_targets = [
        e for e in entries if e.modal_app_name is not None and not modal_unavailable_msg
    ]
    state_results: list[Any] = []
    runners_results: list[Any] = []
    if lookup_targets:
        state_results, runners_results = await asyncio.gather(
            asyncio.gather(
                *(modalops.get_app_state(e.modal_app_name) for e in lookup_targets),
                return_exceptions=True,
            ),
            asyncio.gather(
                *(modalops.get_active_runner_count(e.modal_app_name) for e in lookup_targets),
                return_exceptions=True,
            ),
        )
    lookup_state_by_id: dict[str, str] = {}
    live_count_by_id: dict[str, int | None] = {}
    for entry, state_result, runners_result in zip(
        lookup_targets, state_results, runners_results
    ):
        if isinstance(state_result, BaseException):
            log.warning(
                "admin_model_state_lookup_failed",
                model_id=entry.model_id,
                modal_app_name=entry.modal_app_name,
                error=str(state_result),
            )
            lookup_state_by_id[entry.model_id] = "unknown"
        else:
            lookup_state_by_id[entry.model_id] = state_result
        if isinstance(runners_result, BaseException):
            log.warning(
                "admin_model_runners_lookup_failed",
                model_id=entry.model_id,
                modal_app_name=entry.modal_app_name,
                error=str(runners_result),
            )
            live_count_by_id[entry.model_id] = None
        elif runners_result is None:
            # modal_ops couldn't determine the count (failed/timed-out fetch).
            # Keep it unknown rather than coercing to a misleading 0 (ACS-128).
            live_count_by_id[entry.model_id] = None
        else:
            live_count_by_id[entry.model_id] = int(runners_result)

    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.modal_app_name is None:
            state = "unconfigured"
        elif modal_unavailable_msg:
            state = "unavailable"
        else:
            state = lookup_state_by_id.get(entry.model_id, "unknown")
        spec = model_spec_for(entry.model_id)
        rows.append(
            {
                "id": entry.model_id,
                "modal_app_name": entry.modal_app_name,
                "gpu_shape_label": entry.gpu_shape_label,
                "state": state,
                # Not-looked-up rows (no modal_app_name, or Modal unavailable)
                # default to 0 — they have no app/containers. Looked-up rows
                # whose count couldn't be determined are None (unknown, ACS-128).
                "live_containers": live_count_by_id.get(entry.model_id, 0),
                "min_containers": spec.min_containers if spec else None,
                "max_containers": spec.max_containers if spec else None,
                "scaledown_window_s": spec.scaledown_window_s if spec else None,
            }
        )
    return rows


def _forget_warm_signal(request: Request, model_id: str) -> None:
    """Drop the model's last-completion timestamp so the workbench "usually
    warm" hint (ACS-98) and the ``/health`` warm-estimate reflect the now-cold
    state immediately after an admin Stop/Deploy kills or restarts the
    container. Without this, the heuristic keeps reading "recently used" for up
    to the scale-down window even though the admin just took the model down.
    The map is used only for warm estimation, not as an audit record.
    """
    last_seen = getattr(request.app.state, "last_completion_at", None)
    if isinstance(last_seen, dict):
        last_seen.pop(model_id, None)


@router.post("/admin/models/{model_id}/stop")
async def admin_model_stop(
    request: Request,
    model_id: str,
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
):
    _modal_ops_or_503(settings)
    entry = _lookup_model_or_404(request, model_id)
    try:
        await modalops.stop_app(entry.modal_app_name)
    except modalops.ModalOpsError as exc:
        log.warning("admin_model_stop_failed", model_id=model_id, error=str(exc))
        return _admin_redirect(err=f"Stop failed for {model_id}: {exc}")
    # Without this the dashboard can show "deployed" for up to 30 s after a
    # successful Stop (modal_ops._STATE_CACHE TTL), masking the state change
    # from admins doing a stop-then-verify dance.
    modalops.invalidate_state(entry.modal_app_name)
    _forget_warm_signal(request, model_id)
    log.info(
        "admin_model_stopped",
        admin_id=str(admin.id),
        model_id=model_id,
        modal_app_name=entry.modal_app_name,
    )
    return _admin_redirect(msg=f"Stopped {model_id}.")


@router.post("/admin/models/{model_id}/keep-warm")
async def admin_model_keep_warm(
    request: Request,
    model_id: str,
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
):
    _modal_ops_or_503(settings)
    entry = _lookup_model_or_404(request, model_id)
    try:
        # See modal_ops.set_min_containers — update_autoscaler is a no-op on a
        # stopped app, so we'd otherwise return "kept warm" while doing nothing.
        await modalops.assert_app_running(entry.modal_app_name)
        await modalops.set_min_containers(entry.modal_app_name, 1)
    except modalops.ModalOpsError as exc:
        log.warning("admin_model_keep_warm_failed", model_id=model_id, error=str(exc))
        return _admin_redirect(err=f"Keep-warm failed for {model_id}: {exc}")
    log.info(
        "admin_model_keep_warm",
        admin_id=str(admin.id),
        model_id=model_id,
        modal_app_name=entry.modal_app_name,
    )
    return _admin_redirect(msg=f"{model_id}: min_containers=1 (kept warm).")


@router.post("/admin/models/{model_id}/release")
async def admin_model_release(
    request: Request,
    model_id: str,
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
):
    _modal_ops_or_503(settings)
    entry = _lookup_model_or_404(request, model_id)
    try:
        # Release is meaningless on a stopped app — fail loudly so admins know
        # there's nothing to release vs. silently accepting the click.
        await modalops.assert_app_running(entry.modal_app_name)
        await modalops.set_min_containers(entry.modal_app_name, 0)
    except modalops.ModalOpsError as exc:
        log.warning("admin_model_release_failed", model_id=model_id, error=str(exc))
        return _admin_redirect(err=f"Release failed for {model_id}: {exc}")
    log.info(
        "admin_model_released",
        admin_id=str(admin.id),
        model_id=model_id,
        modal_app_name=entry.modal_app_name,
    )
    return _admin_redirect(msg=f"{model_id}: min_containers=0 (released).")


@router.post("/admin/models/{model_id}/deploy")
async def admin_model_deploy(
    request: Request,
    model_id: str,
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
):
    _modal_ops_or_503(settings)
    entry = _lookup_model_or_404(request, model_id)
    try:
        success, output = await modalops.deploy_app(model_id)
    except modalops.ModalOpsError as exc:
        log.warning("admin_model_deploy_failed", model_id=model_id, error=str(exc))
        return _admin_redirect(err=f"Deploy failed for {model_id}: {exc}")
    tail = output[-200:] if output else ""
    log.info(
        "admin_model_deployed",
        admin_id=str(admin.id),
        model_id=model_id,
        modal_app_name=entry.modal_app_name,
        success=success,
        output_tail=tail,
    )
    if not success:
        return _admin_redirect(err=f"Deploy failed for {model_id}: {tail}")
    modalops.invalidate_state(entry.modal_app_name)
    _forget_warm_signal(request, model_id)
    return _admin_redirect(msg=f"Deployed {model_id}.")


@router.get("/admin/models/status")
async def admin_models_status(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
):
    """JSON snapshot of per-model state for the dashboard auto-refresh poll.

    Reuses ``_resolve_model_states`` so the polling path and the HTML render
    can't drift. Polled every ~10 s from the dashboard; the 30 s state cache
    in ``modal_ops`` absorbs most calls.
    """
    settings: Settings = request.app.state.settings
    registry = request.app.state.models
    modal_unavailable_msg = _modal_unavailable_msg(settings)
    rows = await _resolve_model_states(registry, modal_unavailable_msg)
    return JSONResponse(
        {
            "models": rows,
            "modal_unavailable": modal_unavailable_msg,
        }
    )
