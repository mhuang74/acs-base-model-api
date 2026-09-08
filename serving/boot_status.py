"""Publish authored cold-boot stage events to a shared Modal Dict (ACS-272).

The wrapper polls this Dict during a cold-boot wait and surfaces the current
stage to users (workbench banner, SSE keepalive comments, the per-model status
endpoint). Only the *authored* stage strings below ever leave the container —
never raw vLLM log lines, paths, or tracebacks — so nothing sensitive can leak
by construction.

Stages, in boot order:

    container_started -> weights_loading -> weights_loaded -> engine_ready
                      -> serving

plus the terminal ``failed`` (vLLM exited before the port opened). Stage
transitions are derived from the same lifetime events modal_app.py already
emits for the ACS-17 boot-time analysis; ``make_emitter`` wraps the lifetime
emitter so one callback feeds both sinks.

Dict layout: one shared Dict named ``BOOT_STATUS_DICT_NAME``; key = Modal app
name; value = ``{"stage", "detail", "ts", "container_id"}``. A new boot simply
overwrites the previous entry — the wrapper decides freshness from ``ts``.

Everything here is best-effort: a Dict outage must never break serving, so
every network call is wrapped and failures are logged to container stdout only.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, Callable

# The wrapper reads this Dict by name — keep in sync with
# base_model_wrapper/src/wrapper/boot_stage.py.
BOOT_STATUS_DICT_NAME = "acs-boot-status"

STAGE_ORDER: tuple[str, ...] = (
    "container_started",
    "weights_loading",
    "weights_loaded",
    "engine_ready",
    "serving",
)
STAGE_FAILED = "failed"

# Lifetime event -> user-facing stage. Events not listed (started, ended, …)
# are lifetime-CSV-only and publish nothing.
_EVENT_TO_STAGE: dict[str, str] = {
    "container_up": "container_started",
    "restored": "container_started",  # snapshot-restore path (ACS-200)
    "weights_load_start": "weights_loading",
    "weights_load_complete": "weights_loaded",
    "engine_init_complete": "engine_ready",
    "vllm_port_open": "serving",
}

# Events invented for boot-status only. ``make_emitter`` never forwards these
# to the lifetime CSV, so the ACS-17 analysis schema is unchanged.
SYNTHETIC_EVENTS: frozenset[str] = frozenset({"vllm_exited"})

# Written from two threads: the serve thread (container_up, weights_load_start,
# vllm_port_open) and the stdout-pump thread (marker + EOF events). The
# check-then-set on stage_idx is technically racy, but the events are causally
# ordered (port-open can't precede engine-init) and the monotonic guard drops
# any straggler, so a lock would buy nothing for this best-effort UX channel.
_state: dict[str, Any] = {
    "app_name": None,
    "dict": None,
    "stage_idx": -1,
    "failed": False,
}


def configure(app_name: str) -> None:
    """Bind the publisher to this container's app; resets per-boot state.

    Called next to ``lifecycle.configure_lifetime`` in every serve entrypoint —
    including the snapshot ``restore`` hook, whose module globals were captured
    at build time and must be re-armed for the restored container.
    """
    _state["app_name"] = app_name
    _state["stage_idx"] = -1
    _state["failed"] = False


def _resolve_dict() -> Any:
    if _state["dict"] is None:
        import modal

        _state["dict"] = modal.Dict.from_name(
            BOOT_STATUS_DICT_NAME, create_if_missing=True
        )
    return _state["dict"]


def publish_stage(stage: str, detail: str = "") -> None:
    """Write the current stage to the shared Dict. Best-effort; never raises."""
    app_name = _state["app_name"]
    if not app_name:
        return
    # Keep the monotonic index in sync for direct callers too (the snapshot
    # restore path publishes "serving" without going through publish_event).
    if stage in STAGE_ORDER:
        _state["stage_idx"] = max(_state["stage_idx"], STAGE_ORDER.index(stage))
    elif stage == STAGE_FAILED:
        _state["failed"] = True
    value = {
        "stage": stage,
        "detail": detail,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "container_id": os.environ.get("MODAL_TASK_ID", ""),
    }
    try:
        _resolve_dict().put(app_name, value)
        print(f"[boot-status] {stage}{f' ({detail})' if detail else ''}", flush=True)
    except Exception as exc:  # noqa: BLE001 - publishing must never break serving
        print(f"[boot-status] WARN: put '{stage}' failed: {exc!r}", flush=True)


def publish_event(event: str) -> None:
    """Map a lifetime/synthetic event to a stage and publish it.

    Monotonic: a stage earlier in ``STAGE_ORDER`` than the last published one
    is dropped (TP>1 workers can repeat weight-load lines after the engine is
    up). ``vllm_exited`` becomes ``failed`` only while the boot is still in
    progress — at shutdown of a serving container the pipe EOF is normal.
    """
    if _state["failed"]:
        return
    if event in SYNTHETIC_EVENTS:
        if event == "vllm_exited" and _state["stage_idx"] < STAGE_ORDER.index("serving"):
            publish_stage(STAGE_FAILED, detail="vLLM exited during startup")
        return
    stage = _EVENT_TO_STAGE.get(event)
    if stage is None:
        return
    idx = STAGE_ORDER.index(stage)
    if idx <= _state["stage_idx"]:
        return
    _state["stage_idx"] = idx
    publish_stage(stage)


def make_emitter(lifetime_emit: Callable[[str], None]) -> Callable[[str], None]:
    """Combine the lifetime-CSV emitter with boot-status publishing.

    Passed as ``emit_event`` to ``start_vllm_with_event_capture`` and used for
    the direct ``emit_lifetime_event`` call sites in modal_app.py, so every
    boot event reaches both sinks through one callback. Synthetic events are
    boot-status-only.
    """

    def emit(event: str) -> None:
        if event not in SYNTHETIC_EVENTS:
            lifetime_emit(event)
        publish_event(event)

    return emit
