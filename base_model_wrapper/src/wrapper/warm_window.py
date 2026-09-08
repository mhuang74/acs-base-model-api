"""Per-model warm/cool window jobs.

Used to live as ``modal.Cron`` decorators inside ``modal_app.py`` (one pair
per model that needed conference-week scheduling). We moved the schedule
into Postgres (``model_warm_window``) and run it from APScheduler on the
wrapper side, so admins can edit each model's window from /admin without a
Modal redeploy. Mirrors the capacity-probe migration in ``probe.py``.

Errors (Modal app not deployed, transient RPC failure, etc.) are logged
but never raised — the APScheduler job must not crash the scheduler.
"""

from __future__ import annotations

from . import modal_ops as modalops
from .logging import get_logger

log = get_logger()


async def fire_warm(model_id: str, modal_app_name: str) -> None:
    """Set ``min_containers=1`` on the model's serve function."""
    await _apply_min_containers(model_id, modal_app_name, n=1, phase="warm")


async def fire_cool(model_id: str, modal_app_name: str) -> None:
    """Set ``min_containers=0`` on the model's serve function."""
    await _apply_min_containers(model_id, modal_app_name, n=0, phase="cool")


async def _apply_min_containers(
    model_id: str, modal_app_name: str, *, n: int, phase: str
) -> None:
    try:
        # update_autoscaler is a silent no-op on a stopped app: it writes the
        # autoscaler config but doesn't transition STOPPED → DEPLOYED, so the
        # cron would log fake success while the app stays dead. Guard with a
        # state check so a stopped app surfaces as warm_window_failed.
        await modalops.assert_app_running(modal_app_name)
        await modalops.set_min_containers(modal_app_name, n)
    except modalops.ModalOpsError as exc:
        log.warning(
            "warm_window_failed",
            model_id=model_id,
            modal_app_name=modal_app_name,
            phase=phase,
            min_containers=n,
            error=str(exc),
        )
        return
    except Exception as exc:  # noqa: BLE001 — scheduler must never raise
        log.warning(
            "warm_window_failed",
            model_id=model_id,
            modal_app_name=modal_app_name,
            phase=phase,
            min_containers=n,
            error=f"{type(exc).__name__}: {exc}",
        )
        return
    log.info(
        "warm_window_fired",
        model_id=model_id,
        modal_app_name=modal_app_name,
        phase=phase,
        min_containers=n,
    )
