"""Container lifetime logging and vLLM subprocess cleanup helpers."""

from __future__ import annotations

import atexit
import datetime as dt
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class LifetimeConfig:
    lifetime_dir: str
    lifetime_log_path: str
    lifetime_volume: object
    n_gpu: int
    gpu_type: str


_LIFETIME_CSV_HEADER = "ts_utc,container_id,event,n_gpu,gpu_type,cloud,region\n"
_LIFETIME_STATE: dict[str, object] = {"container_id": None, "ended": False}
_VLLM_PROC: subprocess.Popen | None = None
_CONFIG: LifetimeConfig | None = None
_HOOKS_REGISTERED = False


def configure_lifetime(config: LifetimeConfig) -> None:
    """Set per-model lifetime state before emitting events inside a container."""
    global _CONFIG
    _CONFIG = config


def set_vllm_proc(proc: subprocess.Popen) -> None:
    """Register the vLLM subprocess so shutdown hooks can kill its process group."""
    global _VLLM_PROC
    _VLLM_PROC = proc


def reset_lifetime_state() -> None:
    """Clear per-container lifetime state so the next boot re-derives it fresh.

    Needed by the GPU-snapshot serve path (ACS-200): Modal's snapshot captures
    this module's globals from the *build* container, so on every restore the
    container_id, "ended" flag, and hook-registration flag would otherwise be
    stale (they belong to the container that took the snapshot). Resetting here
    lets the restore container emit its own container_up/ended rows against its
    own MODAL_TASK_ID and re-register its own exit hooks.
    """
    global _HOOKS_REGISTERED
    _LIFETIME_STATE["container_id"] = None
    _LIFETIME_STATE["ended"] = False
    _HOOKS_REGISTERED = False


def _require_config() -> LifetimeConfig:
    if _CONFIG is None:
        raise RuntimeError("lifetime helpers used before configure_lifetime()")
    return _CONFIG


def kill_vllm_now() -> None:
    # On 405B TP=8, vLLM's own teardown (CUDA context release across 8 workers,
    # KV cache drop, worker joins) reliably exceeds Modal's 30 s shutdown
    # grace. SIGKILL'ing the process group on SIGTERM keeps us inside the
    # window so the container exits 0 instead of being recorded as "Failed".
    proc = _VLLM_PROC
    if proc is None:
        print("[lifetime] kill_vllm: no proc registered, skipping", flush=True)
        return
    if proc.poll() is not None:
        print(f"[lifetime] kill_vllm: proc already exited rc={proc.returncode}", flush=True)
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
        print(f"[lifetime] kill_vllm: SIGKILL pgid={pgid} pid={proc.pid}", flush=True)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        print(f"[lifetime] kill_vllm: killpg failed: {exc!r}", flush=True)


def emit_lifetime_event(event: str) -> None:
    """Append a single row to the lifetime CSV. Best-effort; never raises."""
    config = _require_config()
    try:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        cid = _LIFETIME_STATE.get("container_id")
        if cid is None:
            cid = os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex[:12]
            _LIFETIME_STATE["container_id"] = cid
        cloud = os.environ.get("MODAL_CLOUD_PROVIDER", "")
        region = os.environ.get("MODAL_REGION", "")
        os.makedirs(config.lifetime_dir, exist_ok=True)
        write_header = (
            not os.path.exists(config.lifetime_log_path)
            or os.path.getsize(config.lifetime_log_path) == 0
        )
        with open(config.lifetime_log_path, "a") as fh:
            if write_header:
                fh.write(_LIFETIME_CSV_HEADER)
            fh.write(
                f"{ts},{cid},{event},{config.n_gpu},{config.gpu_type},{cloud},{region}\n"
            )
        config.lifetime_volume.commit()
        print(
            f"[lifetime] {event} cid={cid} ts={ts} cloud={cloud} region={region}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - must never crash serve()
        print(f"[lifetime] WARN: emit '{event}' failed: {exc}", flush=True)


def emit_end_once(reason: str) -> None:
    if _LIFETIME_STATE.get("ended"):
        return
    _LIFETIME_STATE["ended"] = True
    emit_lifetime_event(f"ended:{reason}")


def register_lifetime_exit_hooks() -> None:
    """Register best-effort process cleanup and lifetime event hooks."""
    global _HOOKS_REGISTERED
    if _HOOKS_REGISTERED:
        return
    _HOOKS_REGISTERED = True

    def _on_exit() -> None:
        kill_vllm_now()
        emit_end_once("atexit")

    atexit.register(_on_exit)

    # Signal handlers are a fallback for paths where Modal does not override
    # them, such as local `modal serve` plus Ctrl-C.
    def _handler(signum: int, _frame: object) -> None:
        kill_vllm_now()
        emit_end_once(f"signal_{signum}")
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
