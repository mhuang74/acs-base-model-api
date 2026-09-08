"""Unit tests for cold-boot marker detection in serving.vllm_runtime.

The markers split the cold-boot timeline into per-stage events written to
the lifetime CSV. PR #27 shipped one substring per event, but vLLM 0.19.1's
sharded_state_loader (used by 405B) and gpu_model_runner (used by 8B / Trinity)
log slightly different phrasings when weight load completes — so the original
single-substring matcher silently dropped weights_load_complete on 405B.

These tests pin the substrings against real log lines captured from cold
boots on 2026-06-26 (acs-lifetime-log + Modal logs), so a future vLLM
upgrade that changes either phrasing fails loudly here rather than in prod.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import threading

from serving import vllm_runtime


SHARDED_STATE_LINE = (
    "(Worker_TP0 pid=54) INFO 06-26 09:00:53 [sharded_state_loader.py:157] "
    "Loading weights took 137.72 seconds"
)
GPU_MODEL_RUNNER_LINE = (
    "(EngineCore pid=16) INFO 06-26 08:46:32 "
    "[gpu_model_runner.py:4842] Loading model weights took 14.99 GB"
)
UVICORN_STARTED_LINE = "(APIServer pid=4) INFO:     Started server process [4]"
NOISE_LINE = "(EngineCore pid=16) INFO 06-26 08:46:31 [parallel_state.py:1400] world_size=1"


def _pump_lines(lines: list[str]) -> list[str]:
    """Run start_vllm_with_event_capture's pump against a fake `cat <lines>`
    process so we exercise the real Popen + thread + matching path end-to-end."""
    payload = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    proc = subprocess.Popen(
        ["cat"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(payload)
    proc.stdin.close()

    events: list[str] = []
    lock = threading.Lock()

    def _emit(ev: str) -> None:
        with lock:
            events.append(ev)

    # Reproduce start_vllm_with_event_capture's pump logic directly against
    # the already-running `cat` proc (skip subprocess.Popen — we made our own).
    pending = {event: tuple(needles) for event, needles in vllm_runtime._MARKERS}

    def _pump() -> None:
        for line in proc.stdout:
            if not pending:
                continue
            for event in list(pending):
                if any(needle in line for needle in pending[event]):
                    del pending[event]
                    _emit(event)

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    proc.wait(timeout=5)
    t.join(timeout=2)
    return events


def test_sharded_state_loader_fires_weights_load_complete():
    """405B path: --load-format sharded_state."""
    events = _pump_lines([NOISE_LINE, SHARDED_STATE_LINE, UVICORN_STARTED_LINE])
    assert events == ["weights_load_complete", "engine_init_complete"]


def test_gpu_model_runner_fires_weights_load_complete():
    """8B / Trinity path: default loader."""
    events = _pump_lines([NOISE_LINE, GPU_MODEL_RUNNER_LINE, UVICORN_STARTED_LINE])
    assert events == ["weights_load_complete", "engine_init_complete"]


def test_each_event_fires_at_most_once():
    """TP > 1 prints the weights line per worker rank; only first should emit."""
    events = _pump_lines([SHARDED_STATE_LINE, SHARDED_STATE_LINE, SHARDED_STATE_LINE])
    assert events == ["weights_load_complete"]


def test_noise_does_not_fire_anything():
    events = _pump_lines([NOISE_LINE, NOISE_LINE])
    assert events == []


def _run_real_capture(
    lines: list[str], drop_patterns: tuple[str, ...]
) -> tuple[str, list[str]]:
    """Drive the REAL start_vllm_with_event_capture against a `printf` process
    and capture what it mirrors to stdout, so the drop_patterns filter (ACS-200
    NCCL-noise fallback) is exercised exactly as it runs in the container."""
    payload = "".join(ln if ln.endswith("\n") else ln + "\n" for ln in lines)
    events: list[str] = []
    lock = threading.Lock()

    def _emit(ev: str) -> None:
        with lock:
            events.append(ev)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        proc = vllm_runtime.start_vllm_with_event_capture(
            ["printf", "%s", payload],
            emit_event=_emit,
            drop_patterns=drop_patterns,
        )
        proc.wait(timeout=5)
        for thread in threading.enumerate():
            if thread.name == "vllm-stdout-pump":
                thread.join(timeout=2)
    return buf.getvalue(), events


NCCL_NOISE_LINE = (
    "[rank0] WARNING ProcessGroupNCCL.cpp HeartbeatMonitor: "
    "TCPStore ... Broken pipe c10::Error"
)


def test_drop_patterns_suppress_nccl_noise_but_keep_normal_lines():
    """Post-/sleep NCCL heartbeat spam is filtered from the log; real lines stay."""
    drop = ("HeartbeatMonitor", "TCPStore", "Broken pipe", "c10::Error")
    out, _ = _run_real_capture([NOISE_LINE, NCCL_NOISE_LINE], drop_patterns=drop)
    assert "HeartbeatMonitor" not in out
    assert NOISE_LINE in out  # a non-matching line is still mirrored


def test_drop_patterns_never_swallow_a_marker_event():
    """A line matching BOTH a drop pattern and a marker still fires the event —
    matching runs on the raw line before the print filter."""
    drop = ("ProcessGroupNCCL",)
    combined = "ProcessGroupNCCL noise ... Loading model weights took 14.99 GB"
    out, events = _run_real_capture([combined], drop_patterns=drop)
    # Trailing vllm_exited: pipe EOF always fires the synthetic event (ACS-272).
    assert events == ["weights_load_complete", "vllm_exited"]
    assert "ProcessGroupNCCL" not in out  # line was dropped from the log


def test_default_polling_interval_is_short_enough_for_attribution():
    """The 15s default added ~10s of pure measurement noise to every cold boot
    (researchlog 2026-06-26). Keep this guard so a future bump above ~5s is
    caught — anything higher and per-stage deltas get fuzzy again."""
    import inspect

    sig = inspect.signature(vllm_runtime.wait_for_vllm_port)
    assert sig.parameters["interval_s"].default <= 5.0


# --- activation_can_load_sharded guard (ACS-199 item 5) ---------------------
#
# The 0.19.1-pinned activation engine must never sharded_state-load weights it
# can't read (crashes AFTER a paid multi-GPU boot). The guard is default-deny:
# only an explicit per-model opt-in flag enables it, and any ``-v023`` volume is
# hard-blocked regardless.


def test_guard_hard_blocks_v023_even_if_opted_in():
    # 0.23-format shards are never 0.19.1-loadable — the flag can't override it.
    assert vllm_runtime.activation_can_load_sharded("acs-trinity-sharded-v023", True) is False


def test_guard_default_denies_without_opt_in():
    # A non-v023 volume still requires the explicit flag; default is HF fallback.
    assert vllm_runtime.activation_can_load_sharded("acs-sharded", False) is False


def test_guard_allows_opted_in_non_v023():
    assert vllm_runtime.activation_can_load_sharded("acs-sharded", True) is True


def test_trinity_and_405b_specs_resolve_correctly():
    from acs_model_registry import get_model_spec

    t = get_model_spec("trinity-truebase")
    assert vllm_runtime.activation_can_load_sharded(
        t.sharded_volume_name, t.activation_sharded_ok
    ) is False, "Trinity (MoE, -v023) must fall back to HF"

    b = get_model_spec("llama-405b")
    assert vllm_runtime.activation_can_load_sharded(
        b.sharded_volume_name, b.activation_sharded_ok
    ) is True, "405B (dense, verified) may sharded-load in the activation engine"


def test_no_moe_spec_is_marked_activation_sharded_ok():
    """Registry invariant: MoE models (expert-parallel) MUST NOT opt into
    activation sharded-load — 0.23 renamed their params, so a 0.19.1 sharded_state
    load would crash. Default-deny makes this the natural state; this test stops a
    future edit from flipping an MoE model True."""
    from acs_model_registry import SPECS

    offenders = [
        mid
        for mid, spec in SPECS.items()
        if spec.enable_expert_parallel and spec.activation_sharded_ok
    ]
    assert not offenders, f"MoE specs must not set activation_sharded_ok=True: {offenders}"


def test_activation_sharded_ok_is_an_explicit_vetted_allowlist():
    """Robust catch-all for the opt-in (PR #198 review note 2): only models
    whose shards are EXPLICITLY vetted as 0.19.1-loadable may set the flag. This
    catches architecturally-MoE models that lack the ``--enable-expert-parallel``
    serving flag (e.g. kimi-k2-base) — the `enable_expert_parallel` invariant
    above would miss those. Any new opt-in must consciously extend VETTED."""
    from acs_model_registry import SPECS

    # Dense models whose current shards are confirmed loadable by vLLM 0.19.1.
    VETTED = {"llama-405b"}  # #188 capture-parity PASS on acs-sharded
    opted_in = {mid for mid, spec in SPECS.items() if spec.activation_sharded_ok}
    assert opted_in <= VETTED, (
        "activation_sharded_ok=True must be explicitly vetted (dense + shards "
        f"confirmed 0.19.1-loadable): unexpected {opted_in - VETTED}"
    )


def test_real_pump_emits_vllm_exited_on_eof():
    """ACS-272: pipe EOF (vLLM process died) fires the synthetic ``vllm_exited``
    event after any marker events. At normal teardown the boot_status emitter
    ignores it (stage already ``serving``); mid-boot it becomes ``failed``."""
    _, events = _run_real_capture(
        [NOISE_LINE, GPU_MODEL_RUNNER_LINE, UVICORN_STARTED_LINE], drop_patterns=None
    )
    assert events == ["weights_load_complete", "engine_init_complete", "vllm_exited"]
