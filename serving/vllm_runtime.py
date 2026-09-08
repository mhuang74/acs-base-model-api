"""Runtime helpers for launching vLLM behind Modal web_server."""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from typing import Callable, Iterable


# Substrings emitted by vLLM 0.19.1 / uvicorn during startup. Each event has
# a tuple of acceptable substrings — the event fires the first time ANY of
# its substrings is seen on a stdout line, then no more matches for that event.
# Order matters: reader thread emits each lifetime event at most once.
#
# Markers verified against the vLLM 0.19.x release line:
#   - weights_load_complete:
#     * "Loading weights took" — sharded_state_loader.py:157 logs this once
#       per worker rank when `--load-format sharded_state` is used (the 405B
#       preshard path).
#     * "Loading model weights took" — gpu_model_runner.py / model_runner.py
#       logs this with a GB suffix when the default HF/safetensors loader
#       runs (llama-8b, trinity-base). Note: NOT a superset of the first
#       string — the "model" word is in different positions.
#     On TP > 1 each worker prints; reader dedupes so we record the FIRST
#     worker's completion. The slowest worker gates port-open, so first-vs-
#     last skew is bounded by per-rank load time variance and is fine for
#     cold-boot attribution.
#   - "Started server process" — uvicorn's standard startup log, printed
#     after vLLM's AsyncLLMEngine + OpenAI API server are ready to receive.
_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("weights_load_complete", ("Loading weights took", "Loading model weights took")),
    ("engine_init_complete", ("Started server process",)),
)

# Registry `vllm_extra_args` that would RE-ENABLE prefix-caching or chunked-prefill
# override the activation engine's pinned `--no-enable-*` (later flag wins) and
# reintroduce stale-KV capture corruption (ACS-160 §D). Surfaced by the 2026-07-01
# Trinity probe: trinity's extra_args include `--enable-prefix-caching`.
ACTIVATION_CONFLICTING_FLAGS = ("--enable-prefix-caching", "--enable-chunked-prefill")


def strip_activation_conflicts(extra_args: list[str]) -> tuple[list[str], list[str]]:
    """Split extra_args into (kept, dropped) for activation-engine serving.

    Shared by modal_app_activation.py and the scripts/activation/ harnesses so the
    probes exercise the exact flag set the engine ships.

    Prefix match on the underscore-normalized spelling (mirrors #178's engine
    strip) so `--enable-prefix-caching=true` / `--enable_prefix_caching` variants
    are caught too — for chunked-prefill the strip is the ONLY guard (there is no
    pinned `--no-enable-chunked-prefill` in the serve cmd; it's deferred).
    """

    def _conflicts(arg: str) -> bool:
        return arg.replace("_", "-").startswith(ACTIVATION_CONFLICTING_FLAGS)

    kept = [a for a in extra_args if not _conflicts(a)]
    dropped = [a for a in extra_args if _conflicts(a)]
    return kept, dropped


def activation_can_load_sharded(sharded_volume_name: str, activation_sharded_ok: bool) -> bool:
    """Whether the 0.19.1-pinned activation stack may load a model's sharded weights.

    Sharded-state weights are keyed to the parameter names of the vLLM that SAVED
    them, so loading shards under an incompatible vLLM dies in ShardedStateLoader
    with a KeyError AFTER a paid multi-GPU boot (the ACS-197 crash). This is
    **default-deny** (ACS-199 item 5): callers fall back to the HF-format
    ``local_model_path`` / repo download — always version-safe, exactly how the
    recorded 2026-07-02 Trinity PASS artifacts ran — unless a model is EXPLICITLY
    marked shard-loadable by the activation engine (``activation_sharded_ok``).

    Two layers, most-conservative first:
      1. **Hard block on ``-v023`` volumes.** These were re-presharded on 0.23's
         native model classes (ACS-197: 0.23's AfmoeForCausalLM names differ from
         0.19.1's trust-remote-code afmoe), so they are never 0.19.1-loadable,
         regardless of the opt-in flag — belt-and-suspenders against a mis-set
         ``activation_sharded_ok``.
      2. **Explicit per-model opt-in.** Only dense models whose shards are
         verified cross-version-loadable set ``activation_sharded_ok=True`` (405B,
         confirmed by the #188 capture-parity PASS). MoE models (Trinity) stay
         ``False`` and fall back to HF. Replaces the old brittle "guess from the
         volume-name suffix" heuristic (#180): a renamed volume can no longer
         silently flip a model to a crashing sharded load.
    """
    if sharded_volume_name.endswith("-v023"):
        return False
    return activation_sharded_ok


def wait_for_vllm_port(port: int, interval_s: float = 5.0) -> None:
    """Block until vLLM accepts a TCP connection on the local port."""
    t0 = time.monotonic()
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                print(
                    f"[serve] vLLM listening on :{port} after "
                    f"{time.monotonic() - t0:.1f}s",
                    flush=True,
                )
                return
        except (ConnectionRefusedError, OSError):
            print(
                f"[serve] T+{time.monotonic() - t0:.0f}s waiting for "
                f"vLLM to listen on :{port} - vLLM is still booting; "
                "the port isn't open yet.",
                flush=True,
            )
            time.sleep(interval_s)


def start_vllm_with_event_capture(
    cmd: list[str],
    *,
    emit_event: Callable[[str], None],
    markers: Iterable[tuple[str, tuple[str, ...]]] = _MARKERS,
    drop_patterns: Iterable[str] | None = None,
) -> subprocess.Popen:
    """Spawn vLLM and tee its stdout into Modal logs + a marker scanner.

    The subprocess is started in its own session (same as the previous
    plain-Popen path) so the lifecycle SIGKILL helper can still target the
    whole worker tree via os.killpg. stdout+stderr are merged onto a single
    PIPE so the scanner sees both vLLM's logger output and uvicorn's startup
    lines; a daemon thread reads the pipe line-by-line, mirrors each line
    back to this container's stdout (preserving `modal logs` visibility),
    and emits a lifetime event the first time each marker substring is
    seen.

    ``drop_patterns`` (opt-in, used by the GPU-snapshot serve path, ACS-200):
    stdout lines containing any of these substrings are NOT mirrored back to the
    container log. General-purpose noise filter for lines that transit THIS
    pump. NOTE: it does NOT catch the post-``/sleep`` c10d ``sendBytes`` spam it
    was originally aimed at — those are raw C++ writes to a grandchild's stderr
    that bypass this pipe entirely (verified ineffective on the ACS-200 staging
    boot; that noise is an unresolved known gap, see modal_app.py). Marker
    matching runs on the raw line first, so filtering can never swallow a
    lifetime event.
    """
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    pending = {event: tuple(needles) for event, needles in markers}
    drop = tuple(drop_patterns or ())

    def _pump() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if not (drop and any(pat in line for pat in drop)):
                    print(line, end="", flush=True)
                if not pending:
                    continue
                for event in list(pending):
                    if any(needle in line for needle in pending[event]):
                        del pending[event]
                        try:
                            emit_event(event)
                        except Exception as exc:  # pragma: no cover
                            print(
                                f"[serve] WARN: emit '{event}' failed: {exc!r}",
                                flush=True,
                            )
            # Pipe EOF = the vLLM process died (crash mid-boot or normal
            # teardown). Emit the synthetic ``vllm_exited`` event so the
            # boot-status publisher can flag a failed boot (ACS-272). The
            # boot_status emitter keeps this OUT of the lifetime CSV, and
            # ignores it entirely once the boot reached ``serving``.
            try:
                emit_event("vllm_exited")
            except Exception as exc:  # pragma: no cover
                print(f"[serve] WARN: emit 'vllm_exited' failed: {exc!r}", flush=True)
        except Exception as exc:  # pragma: no cover - pump must never crash serve()
            print(f"[serve] WARN: stdout pump exited: {exc!r}", flush=True)

    threading.Thread(target=_pump, name="vllm-stdout-pump", daemon=True).start()
    return proc
