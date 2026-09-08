"""Shared backend warm-state helpers."""

from __future__ import annotations

import asyncio
import datetime as dt

from fastapi import Request

from . import modal_ops as modalops

# Modal scales each model's container to zero after this idle window. The
# workbench warm/cold dropdown indicator uses the same number so the label
# flips to "cold" exactly when Modal would have torn the container down.
MODAL_SCALEDOWN_WINDOW = dt.timedelta(minutes=30)

def _mark_model_warm(request: Request, model_id: str | None) -> None:
    """Record that ``model_id`` just served a successful upstream response.

    Called from the completion handlers; the workbench renderer reads this
    state to mark each model's dropdown entry warm vs cold.

    We also write through to ``modal_ops._RUNNERS_CACHE`` because a
    successful completion is ground-truth that a runner is alive — more
    accurate than waiting for Modal's ``get_current_stats`` to tell us,
    especially when the control plane is being slow. Without this, the
    warm pill can show "needs booting" for up to the cache TTL after a
    user has just successfully hit the model.
    """
    if not model_id:
        return
    request.app.state.last_completion_at[model_id] = dt.datetime.now(tz=dt.UTC)
    # For an activation breaker key (``<model_id>::activation``, ACS-199) the
    # ``last_completion_at`` write above still gives the activation engine its own
    # cold_hint, but ``models.get(...)`` is None (activation upstreams aren't
    # registry entries), so the runner-cache write-through is skipped. Harmless:
    # nothing reads the activation runner cache, and it's strictly better than
    # marking the *workbench* runner warm off an activation request.
    entry = request.app.state.models.get(model_id)
    if entry is not None and entry.modal_app_name:
        modalops.mark_runner_warm(entry.modal_app_name)


async def _is_model_warm(entry, last_at, now) -> bool:
    """Decide warm vs cold for the workbench dropdown — single-model variant.

    Source of truth is Modal's per-function runner count — that survives
    wrapper restarts and matches what the user experiences as "instant
    reply". We fall back to the in-process "last completion within
    MODAL_SCALEDOWN_WINDOW" heuristic only when we can't query Modal
    (no ``modal_app_name`` on the registry entry, RPC errors, etc.).

    Prefer ``_resolve_warm_flags`` when checking >1 model at once: this
    function is sequential per call site, and the workbench previously paid
    N× RPC latency on a cold ``_RUNNERS_CACHE`` because of the per-model
    ``for`` loop.
    """
    if entry.modal_app_name:
        try:
            count = await modalops.get_active_runner_count(entry.modal_app_name)
            if count is not None:
                return count > 0
            # count is None → unknown; fall through to the time heuristic.
        except Exception:
            pass
    return last_at is not None and (now - last_at) < MODAL_SCALEDOWN_WINDOW


async def _resolve_warm_flags(
    registry: dict,
    last_at_map: dict,
    now: dt.datetime,
) -> dict[str, bool]:
    """Parallel warm/cold flag lookup for every live model.

    Mirrors ``_resolve_workbench_model_states``: one ``asyncio.gather`` over
    every live model with a ``modal_app_name``, falling back to the
    time-heuristic on RPC failure. Returns ``{model_id: warm_bool}``.

    Without this, ``_render_chat`` and ``workbench_models_status`` paid N×
    Modal RPC latency on a cold cache because they awaited
    ``_is_model_warm`` per model inside a for loop. Combined with the
    parallel ``_resolve_workbench_model_states`` call this means the
    workbench pays at most one Modal round-trip per page render on cache
    miss, even with N models — matching what ``/admin`` already does.
    """
    targets = [m for m in registry.values() if m.status == "live" and m.modal_app_name]
    runner_counts: list = []
    if targets:
        runner_counts = await asyncio.gather(
            *(modalops.get_active_runner_count(m.modal_app_name) for m in targets),
            return_exceptions=True,
        )
    out: dict[str, bool] = {}
    runners_by_id: dict[str, int | BaseException] = {
        m.model_id: r for m, r in zip(targets, runner_counts)
    }
    for m in registry.values():
        if m.status != "live":
            continue
        result = runners_by_id.get(m.model_id)
        if isinstance(result, int):
            out[m.model_id] = result > 0
        else:
            # No modal_app_name, or RPC failed — fall back to time heuristic.
            last_at = last_at_map.get(m.model_id)
            out[m.model_id] = last_at is not None and (now - last_at) < MODAL_SCALEDOWN_WINDOW
    return out


