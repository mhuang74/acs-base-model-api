"""
ACS activation-harvesting/steering engine (ACS-156).

A SEPARATE Modal app from the workbench (`modal_app.py`) on purpose:
  - vLLM-Lens auto-registers on `import vllm` and force-sets `enforce_eager`, so it
    must live in its OWN image — co-installing it in the vanilla serving image would
    disable CUDA graphs for every workbench request. See docs/design/
    activation-architecture.md and ACS-160/ACS-161.
  - Warm-container floor is per-model from the registry (`activation_min_containers`):
    all models scale to zero ($0 when idle). Snapshot-capable models (llama-8b) serve
    through a GPU-memory-snapshot lifecycle (ACS-218, pattern from ACS-200) so an idle
    wake is seconds, not the ~2-min cold boot that made ACS-223 keep a warm floor.
    Mounts the SAME weights Volume as the workbench, so it adds no duplicate
    weights — the point of the chosen architecture.

Capabilities (via vLLM-Lens over the OpenAI HTTP API, `vllm_xargs`):
  - encode / capture-during-generation: `{"output_residual_stream": true}` (capture ALL
    layers) or a `json.dumps`'d string list of layer indices (capture only those — the
    per-layer LIST over HTTP was fixed in vLLM-Lens v1.2.0, now pinned; ACS-266). A bare
    list still 400s at vLLM's vllm_xargs validation, so a subset is sent as a string.
  - steering: `{"apply_steering_vectors": "<json.dumps([SteeringVector...])>"}`. **Additive**
    (`norm_match=False`) is validated mechanically correct on Llama-8B (ACS-160 §B, S1–S6,
    2026-07-01). `norm_match=True` runs and steers (differs from additive + baseline) but is
    not correctness-gated. The earlier "norm_match buggy on fused-residual" note was a
    secondary claim not found in vLLM-Lens source — re-check on MoE (Trinity/Kimi) when probed.

Determinism (ACS-160 §D): the shipped config is the capture-parity-validated set —
  prefix-caching OFF (stale-KV corruption), greedy seed, and `enforce_eager` (forced
  by the plugin, required for hooks). The stricter bit-exact knobs (chunked-prefill
  OFF, fixed attention backend, TF32 off, batch-invariant) are DEFERRED — see the
  image .env note (VLLM_BATCH_INVARIANT hung boot on 0.19.1/L40S).

Usage:
    # Weights come from the shared acs-hf-cache Volume (staged by modal_app.py).
    MODEL_ID=llama-8b modal deploy modal_app_activation.py
    MODEL_ID=llama-8b modal serve  modal_app_activation.py   # dev, ephemeral

Correctness gate before a model is declared live: scripts/activation/parity_check.py
(capture vs HF eager gold, ACS-160 §A). Validated for llama-8b 2026-07-01.
"""

import json
import os
import subprocess
import time

import modal

from acs_model_registry import default_model_id, get_model_config, get_model_spec
from serving.modal_resources import (
    VLLM_PORT,
    create_volumes,
    serve_volumes,
    sharded_dir_for_model,
)
from serving import boot_status, snapshot_runtime
from serving.vllm_runtime import (
    activation_can_load_sharded,
    start_vllm_with_event_capture,
    strip_activation_conflicts,
    wait_for_vllm_port,
)

MINUTES = 60

# --------------------------------------------------------------------------- #
# Model selection (registry-driven, same selector as modal_app.py).
# --------------------------------------------------------------------------- #
MODEL_ID = os.environ.get("MODEL_ID", default_model_id(dev_mode=True))
_SPEC = get_model_spec(MODEL_ID)
_CFG = get_model_config(MODEL_ID)
MODEL_NAME = _CFG["hf_repo"]
SERVED_MODEL_NAME = _CFG["served_model_name"]
N_GPU = _CFG["n_gpu"]
GPU_TYPE = _CFG["gpu_type"]
MAX_MODEL_LEN = _CFG["max_model_len"]
DTYPE = _CFG["dtype"]
PRESHARD = _CFG["preshard"]
TRUST_REMOTE_CODE = _CFG.get("trust_remote_code", False)
VLLM_EXTRA_ARGS: list[str] = _CFG.get("vllm_extra_args", [])
LOCAL_MODEL_PATH: str | None = _CFG.get("local_model_path")

# Activation apps are named distinctly so they never collide with the workbench
# app (acs-<id>) and can scale independently. ``ACT_APP_NAME`` overrides for
# staging deploys (e.g. acs-llama-8b-activation-snapstage) so the snapshot
# lifecycle can be validated without touching the production app/URL.
APP_NAME = os.environ.get("ACT_APP_NAME", f"acs-{MODEL_ID}-activation")

# GPU-memory-snapshot lifecycle (ACS-218 cost work, pattern from ACS-200):
# snapshot-capable models (registry ``snapshot=True`` — single-GPU only, a GPU
# snapshot captures one device's memory) serve through the class-based
# warm→sleep→snapshot→wake path below and scale to zero; multi-GPU models keep
# the plain function path. N_GPU==1 is re-asserted here as a belt-and-braces
# guard against a registry entry flipping ``snapshot`` on a TP>1 model.
SNAPSHOT = bool(_CFG.get("snapshot", False)) and N_GPU == 1
SHARDED_DIR = sharded_dir_for_model(model_name=MODEL_NAME, n_gpu=N_GPU)

# Scaledown window, container cap, and warm-container floor are per-model in the
# shared registry (cheap models keep a warmer, wider engine; the 8×H200 models
# stay tight for cost). Env ``ACT_SCALEDOWN_S`` / ``ACT_MAX_CONTAINERS`` /
# ``ACT_MIN_CONTAINERS`` override for one-off deploys. NOTE: if you set
# ``ACT_SCALEDOWN_S`` at deploy, mirror it in the registry's
# ``activation_scaledown_window_s`` — the wrapper's cold-hint reads only the
# static registry value, so an override here silently desyncs it from the real
# teardown (ACS-226).
SCALEDOWN_WINDOW_S = int(
    os.environ.get("ACT_SCALEDOWN_S", _CFG.get("activation_scaledown_window_s", 5 * MINUTES))
)
MAX_CONTAINERS = int(
    os.environ.get("ACT_MAX_CONTAINERS", _CFG.get("activation_max_containers", 1))
)
MIN_CONTAINERS = int(
    os.environ.get("ACT_MIN_CONTAINERS", _CFG.get("activation_min_containers", 0))
)
# Per-container request concurrency (``@modal.concurrent``). Registry-driven
# (``activation_max_inputs``, default 8) so it can be tuned per model after a load
# test without editing this file — each concurrent capture holds an all-layers
# hidden-state blob, so the safe ceiling is memory/prompt-length bound (ACS-249).
# Env ``ACT_MAX_INPUTS`` overrides for one-off deploys.
MAX_INPUTS = int(
    os.environ.get("ACT_MAX_INPUTS", _CFG.get("activation_max_inputs", 8))
)

# --------------------------------------------------------------------------- #
# Image — vanilla vLLM + vLLM-Lens ONLY (kept out of the workbench image).
# enforce_eager (forced by the plugin) skips torch.compile, so no CUDA-devel /
# nvcc base is needed (unlike the old Prism image). Confirmed by the 2026-06-29
# smoke on debian_slim.
# --------------------------------------------------------------------------- #
activation_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.19.1",
        # Pinned to 1.2.0 (ACS-266): it fixes per-layer `output_residual_stream`
        # list filtering via vllm_xargs (commit caba27a9) — the bug that forced
        # capture-all-then-filter. Unpinned previously; pin so the fix is
        # deterministic. vllm-lens's own pin is vllm>=0.16.0, so 0.19.1 is in range.
        "vllm-lens==1.2.0",  # auto-registers as a vLLM plugin; pulls zstandard for decode
        "huggingface_hub[hf_transfer]",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "MODEL_ID": MODEL_ID,
            # NOTE — the ACS-160 §D *bit-exact-reproducibility* knobs
            # (VLLM_BATCH_INVARIANT, a forced VLLM_ATTENTION_BACKEND, TF32/cuBLAS
            # determinism, chunked-prefill OFF) are DEFERRED, not baked in:
            # `VLLM_BATCH_INVARIANT=1` hung vLLM startup on 0.19.1 / L40S (the port
            # never bound; 2026-07-01 smoke). They must be enabled and
            # boot-verified ONE AT A TIME in a follow-up. The config below matches
            # the capture-parity-validated setup (ACS-160 §A) that boots cleanly.
        }
    )
    .add_local_python_source("acs_model_registry", "serving")
)

_VOLUMES = create_volumes()
_SERVE_VOLUMES = serve_volumes(spec=_SPEC, volumes=_VOLUMES)

app = modal.App(APP_NAME)


def _build_activation_cmd(*, enable_sleep_mode: bool = False) -> list[str]:
    """vLLM launch command for the activation engine (shared by both lifecycles).

    Model-path selection mirrors serve() (pre-sharded → local path → HF repo),
    EXCEPT that 0.23-format (-v023) shards are skipped — this stack pins vLLM
    0.19.1 and cannot load them (see activation_can_load_sharded).
    """
    use_sharded = (
        PRESHARD
        and activation_can_load_sharded(_SPEC.sharded_volume_name, _SPEC.activation_sharded_ok)
        and os.path.isdir(SHARDED_DIR)
        and os.listdir(SHARDED_DIR)
    )
    if use_sharded:
        model_path = SHARDED_DIR
    elif LOCAL_MODEL_PATH:
        model_path = LOCAL_MODEL_PATH
    else:
        model_path = MODEL_NAME

    cmd = [
        "vllm",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--dtype",
        DTYPE,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--served-model-name",
        SERVED_MODEL_NAME,
    ]
    if use_sharded:
        cmd += ["--load-format", "sharded_state"]
    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    filtered_extra, dropped = strip_activation_conflicts(VLLM_EXTRA_ARGS)
    if dropped:
        print(f"[activation] dropped conflicting extra args: {dropped}", flush=True)
    cmd.extend(filtered_extra)
    # Capture-parity-validated pins (ACS-160 §A), appended AFTER extra_args so they
    # win on last-flag-wins argparse ordering no matter how a registry entry spells
    # a conflicting flag (the strip above is a courtesy warning, not the safety
    # mechanism). enforce-eager is required for vLLM-Lens hooks (the plugin forces
    # it too); prefix-caching OFF avoids stale-KV capture corruption; greedy seed
    # for reproducible token paths. (chunked-prefill OFF + batch-invariant etc.
    # deferred — see image .env note.)
    cmd += ["--enforce-eager", "--no-enable-prefix-caching", "--seed", "0"]
    if enable_sleep_mode:
        # Unlocks /sleep + /wake_up (with VLLM_SERVER_DEV_MODE=1) for the
        # snapshot lifecycle. Snapshot path only — the plain path stays byte-
        # identical to the pre-ACS-218 command.
        cmd += ["--enable-sleep-mode"]
    return cmd


def _warm_capture(*, port: int, served_model_name: str) -> None:
    """Fire one capture request so the vLLM-Lens residual-stream hooks are hot
    before the snapshot. Best-effort: a failure here must not abort the build
    (the plain warmup already gated readiness), so it's logged, not raised."""
    import urllib.error
    import urllib.request

    key = os.environ.get("VLLM_API_KEY")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(
        {
            "model": served_model_name,
            "prompt": "The capital of France is",
            "max_tokens": 1,
            "temperature": 0,
            "vllm_xargs": {"output_residual_stream": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            ok = "activations" in json.loads(r.read())
        print(f"[activation-snap] capture warmup ok (activations={ok})", flush=True)
    except (urllib.error.URLError, ValueError, KeyError) as e:
        print(f"[activation-snap] capture warmup skipped: {e!r}", flush=True)


if SNAPSHOT:
    # ----------------------------------------------------------------------- #
    # Snapshot lifecycle (single-GPU models, currently llama-8b): boot vLLM +
    # vLLM-Lens once per build, warm it, /sleep (weights GPU→CPU), let Modal
    # GPU-snapshot that state; every later cold start restores the snapshot and
    # /wake_up's in seconds instead of re-booting for ~2 min. This is what makes
    # activation_min_containers=0 acceptable for interactive steering (ACS-218,
    # pattern from ACS-200).
    #
    # The web label is pinned to f"{APP_NAME}-serve-activation" — exactly the
    # auto-label the plain serve_activation() function had — so the production
    # URL (and the wrapper's activation_upstream_url) does not change on
    # cutover. No coexisting -snap endpoint, unlike the modal_app.py flip: the
    # activation engine is isolated behind its own circuit breaker, so revert
    # is "redeploy previous main", not a wrapper URL flip.
    # ----------------------------------------------------------------------- #
    _snapshot_env_secret = modal.Secret.from_dict(
        {
            # Unlocks vLLM's /sleep + /wake_up dev endpoints.
            "VLLM_SERVER_DEV_MODE": "1",
            # Canonical snapshot hygiene: no NCCL heartbeat-monitor thread across
            # the checkpoint (benign at TP=1; see modal_app.py KNOWN GAP note).
            "TORCH_NCCL_ENABLE_MONITORING": "0",
        }
    )

    @app.cls(
        image=activation_image,
        gpu=f"{GPU_TYPE}:{N_GPU}",
        # Registry-driven floor/cap (llama-8b: 1 warm / 5 max since ACS-231).
        min_containers=MIN_CONTAINERS,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_S,
        timeout=60 * MINUTES,
        volumes=_SERVE_VOLUMES,
        secrets=[
            modal.Secret.from_name("huggingface-secret"),
            modal.Secret.from_name("vllm-api"),  # vLLM reads VLLM_API_KEY → bearer auth
            _snapshot_env_secret,
        ],
        enable_memory_snapshot=True,
        experimental_options={"enable_gpu_snapshot": True},
    )
    @modal.concurrent(max_inputs=MAX_INPUTS)  # heavier per request than plain gen; registry-tunable (ACS-249)
    class ActivationSnap:
        # Class-level default so a restored container that never ran startup()
        # (snap=True runs once at BUILD time) has a defined attribute — stop()
        # can then narrow its except to the real "already gone" cases instead of
        # swallowing AttributeError, which would hide a genuinely-unset process.
        process: subprocess.Popen | None = None

        @modal.enter(snap=True)
        def startup(self) -> None:
            """Pre-snapshot: start vLLM+Lens, warm it, /sleep. Once per build."""
            t0 = time.monotonic()
            print(f"[activation-snap] build boot, MODEL_ID={MODEL_ID}", flush=True)
            cmd = _build_activation_cmd(enable_sleep_mode=True)
            print(f"[activation-snap] launching: {' '.join(cmd)}", flush=True)
            # start_new_session so the whole worker tree is one process group.
            self.process = subprocess.Popen(cmd, start_new_session=True)
            # wait_ready polls /health (retrying through connection-refused while
            # the port comes up) AND fast-fails if the subprocess dies, so it
            # subsumes wait_for_vllm_port — no separate port wait needed. Gating
            # on /health (not just a TCP accept) matters: a 503 "engine loading"
            # would abort the snapshot build (see the modal_app.py VllmSnap note).
            snapshot_runtime.wait_ready(self.process, port=VLLM_PORT)
            # Warm BOTH paths this engine exists for before snapshotting: a plain
            # completion primes the sampler, and a capture request exercises the
            # vLLM-Lens residual-stream hooks so the sleep→snapshot→wake cycle is
            # validated for harvesting/steering, not just generation. Empirically
            # verified post-restore (capture shape/finiteness + steering identity)
            # in the ACS-218 cutover smoke, but warming it keeps the first live
            # request off the cold hook path.
            snapshot_runtime.warmup(port=VLLM_PORT, served_model_name=SERVED_MODEL_NAME)
            _warm_capture(port=VLLM_PORT, served_model_name=SERVED_MODEL_NAME)
            snapshot_runtime.sleep(port=VLLM_PORT, level=1)
            print(
                f"[activation-snap] startup(snap=True) done in "
                f"{time.monotonic() - t0:.1f}s — ready for snapshot",
                flush=True,
            )

        @modal.enter(snap=False)
        def restore(self) -> None:
            """Post-restore (and first build boot): /wake_up, then serve."""
            # Boot stages (ACS-276): restore is the only boot users wait on
            # (startup(snap=True) runs at deploy build). Snapshot restores skip
            # weight-load entirely, so publish container_started → serving.
            boot_status.configure(APP_NAME)
            boot_status.publish_event("container_up")
            dt = snapshot_runtime.wake_up(port=VLLM_PORT)
            # Keep the same readiness marker the plain path prints, so cold-boot
            # timing greps (docs/dependency-version-census.md) still find 8b.
            print(f"[activation] vLLM listening after snapshot wake in {dt:.1f}s", flush=True)
            boot_status.publish_stage("serving")

        @modal.web_server(
            port=VLLM_PORT,
            startup_timeout=40 * MINUTES,
            label=f"{APP_NAME}-serve-activation",
        )
        def serve(self) -> None:
            """No-op: vLLM is already listening on VLLM_PORT from startup()."""
            print("[activation-snap] web_server hook: vLLM already listening", flush=True)

        @modal.exit()
        def stop(self) -> None:
            import signal

            if self.process is None:
                return
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                # Already reaped, or the group is gone — nothing to clean up.
                pass

else:

    @app.function(
        image=activation_image,
        gpu=f"{GPU_TYPE}:{N_GPU}",
        # Warm-container floor from the registry (``activation_min_containers``).
        # Default 0 = scale-to-zero ($0 when no research runs).
        min_containers=MIN_CONTAINERS,
        # Per-model cap from the registry (``activation_max_containers``). The
        # 8×H200 models stay at 1 (a second concurrent container doubles burn for
        # a bursty workload that can queue instead).
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_S,
        timeout=60 * MINUTES,
        volumes=_SERVE_VOLUMES,
        secrets=[
            modal.Secret.from_name("huggingface-secret"),
            modal.Secret.from_name("vllm-api"),  # vLLM reads VLLM_API_KEY → bearer auth
        ],
    )
    @modal.concurrent(max_inputs=MAX_INPUTS)  # heavier per request than plain gen; registry-tunable (ACS-249)
    @modal.web_server(port=VLLM_PORT, startup_timeout=40 * MINUTES)
    def serve_activation():
        t0 = time.monotonic()
        print(f"[activation] container up, MODEL_ID={MODEL_ID}", flush=True)
        # User-facing boot stages (ACS-272/276). Unlike modal_app.py there is
        # no lifetime CSV here, so events feed boot_status only — same Dict,
        # keyed on THIS app's name, which is what the wrapper's activation
        # BackendContext carries (modal_ops.resolve_activation_app).
        boot_status.configure(APP_NAME)
        boot_status.publish_event("container_up")
        cmd = _build_activation_cmd()
        print(
            f"[activation] launching at T+{time.monotonic() - t0:.1f}s: {' '.join(cmd)}",
            flush=True,
        )
        boot_status.publish_event("weights_load_start")
        # start_new_session so the whole worker tree is one process group; the
        # capture wrapper additionally tees stdout (preserving `modal logs`)
        # and fires the weight-load / engine-init / EOF-failure stage events.
        start_vllm_with_event_capture(cmd, emit_event=boot_status.publish_event)
        wait_for_vllm_port(VLLM_PORT)
        boot_status.publish_event("vllm_port_open")
        print(f"[activation] vLLM listening at T+{time.monotonic() - t0:.1f}s", flush=True)


@app.local_entrypoint()
def main():
    print(f"App:      {APP_NAME}")
    print(f"Model:    {MODEL_NAME}  (id={MODEL_ID})")
    _warm = "always-on" if MIN_CONTAINERS > 0 else "scale-to-zero"
    print(f"GPUs:     {N_GPU}x{GPU_TYPE}   {_warm} (min_containers={MIN_CONTAINERS})")
    print(
        "Endpoint: vLLM OpenAI API + vLLM-Lens vllm_xargs (output_residual_stream / apply_steering_vectors)"
    )
    print()
    print("Deploy:  MODEL_ID=%s modal deploy modal_app_activation.py" % MODEL_ID)
    print("Correctness gate: scripts/activation/parity_check.py (ACS-160 §A)")
