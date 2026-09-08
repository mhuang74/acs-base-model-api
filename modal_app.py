"""
ACS Infra — Modal + vLLM pilot (Week 1).

Design doc: design/week1-modal-vllm-pilot.md
Wiki:       wiki/common-projects/base-model-hosting/

Usage:
    # One-time: download model weights into the Volume (CPU, ~30-60 min for 405B).
    modal run modal_app.py::stage_weights

    # One-time: pre-shard to TP=N on N GPUs (skip for small dev models).
    modal run modal_app.py::preshard

    # Deploy the server for collaborators to hit over HTTPS.
    modal deploy modal_app.py

    # Dev: ephemeral serve, torn down when you Ctrl-C.
    modal serve modal_app.py

Cost discipline (see design doc §Cost discipline during dev):
  - For plumbing work, set DEV_MODE=1 to run the 1B dev model on 1xH100.
  - `modal serve` (not `modal deploy`) during dev — tears down on Ctrl-C.
  - FAST_BOOT=True skips torch.compile — faster cold starts, lower throughput.
"""

import os
import subprocess
import time

import modal

# Beta API for multi-node deployment (see kimi-k2-base-and-chinese-base-alternatives.md §H).
# `modal.experimental` is NOT auto-loaded by `import modal`, so we have to
# import it explicitly. Names re-exported under private aliases to avoid
# polluting the module's public surface.
from modal.experimental import (
    clustered as _modal_clustered,
    get_cluster_info as _modal_get_cluster_info,
)

from serving import modal_lifecycle as lifecycle
from serving.modal_lifecycle import LifetimeConfig
from serving.modal_config import (
    MODELS,
    default_model_id,
    get_model_config,
    get_model_spec,
)
from serving.modal_resources import (
    HF_CACHE_DIR,
    LIFETIME_DIR,
    LIFETIME_LOG_PATH,
    PROD_TORCH_CUDA_INDEX,
    PROD_VLLM_VERSION,
    VLLM_PORT,
    build_vllm_image,
    clustered_serve_volumes,
    copy_metadata_volumes,
    create_volumes,
    preshard_volumes,
    serve_volumes,
    sharded_dir_for_model,
    sharded_volume_for,
)
from serving.staging import copy_metadata_impl, preshard_impl, stage_weights_impl
from serving.vllm_runtime import start_vllm_with_event_capture, wait_for_vllm_port
from serving import boot_status  # noqa: E402 - grouped with its serving siblings
from serving import snapshot_runtime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# This file deploys ONE model per `modal deploy` invocation. The model to
# deploy is selected by ``MODEL_ID`` (preferred) or ``DEV_MODE`` (legacy):
#
#     MODEL_ID=llama-8b modal deploy modal_app.py        → 8B dev model on 1×L40S
#     MODEL_ID=llama-405b modal deploy modal_app.py      → 405B prod model on 8×H200
#
# Multiple models can be served simultaneously by running the deploy command
# once per ``MODEL_ID`` — each produces a separately-named Modal app with its
# own URL. The wrapper (base_model_wrapper) keeps a registry env var
# ``MODELS_REGISTRY_JSON`` that maps short model ids → upstream URL and routes
# per-request to the right app.
#
# Legacy ``DEV_MODE`` is kept as a default-selector: ``DEV_MODE=1`` (the
# default) selects ``llama-8b``; ``DEV_MODE=0`` selects ``llama-405b``. This
# preserves existing CI / docs that say ``DEV_MODE=0 modal deploy modal_app.py``.
#
# Forgetting the prefix on a 405B deploy silently fires an 8xH200 run at
# ~$36/hr — the default is deliberately the cheap dev path.
#
_DEV = os.environ.get("DEV_MODE", "1")
DEV_MODE = _DEV != "0"
_DEFAULT_MODEL_ID = default_model_id(DEV_MODE)
MODEL_ID = os.environ.get("MODEL_ID", _DEFAULT_MODEL_ID)
_SPEC = get_model_spec(MODEL_ID)
_CFG = get_model_config(MODEL_ID)
MODEL_NAME = _CFG["hf_repo"]
SERVED_MODEL_NAME = _CFG["served_model_name"]
N_GPU = _CFG["n_gpu"]
GPU_TYPE = _CFG["gpu_type"]
MAX_MODEL_LEN = _CFG["max_model_len"]
APP_NAME = _CFG["app_name"]
PRESHARD = _CFG["preshard"]
DTYPE = _CFG["dtype"]

# Multi-node clustered-deployment fields (all optional; default to single-node).
# When N_NODES > 1, use serve_clustered() instead of serve() — see §H of
# the wiki plan doc. RDMA + Ray + per-node TP + cross-node PP is the canonical
# 16-GPU topology for Kimi-K2-Base.
N_NODES = _CFG.get("n_nodes", 1)
RDMA = _CFG.get("rdma", False)
COORDINATION = _CFG.get("coordination", "mp")  # "mp" (single-node) | "ray" (multi-node)
TOPOLOGY = _CFG.get("topology", "tp_only")  # "tp_only" | "tp8_pp2" | etc.
TRUST_REMOTE_CODE = _CFG.get("trust_remote_code", False)
VLLM_EXTRA_ARGS: list[str] = _CFG.get("vllm_extra_args", [])
# Max top_logprobs / prompt_logprobs vLLM will return, passed as ``--max-logprobs``.
# Default ``-1`` = UNCAPPED: vLLM returns whatever the request asks for, including
# full-vocab ``prompt_logprobs=-1`` (ACS-191, Tier 2 — enabled on the 0.23 cutover;
# ``-1`` needs the V1 engine, which 0.23 uses by default). The wrapper — not this
# flag — is the guardrail: ``schemas.MAX_LOGPROBS`` caps positive top-k requests,
# the full-vocab prompt-length gate bounds ``prompt_len × vocab``, and the
# output-work bound caps generated positions. Deploying vLLM uncapped means it
# never 400s something the wrapper already allowed. The invariant is "wrapper cap
# ≤ server cap", satisfied trivially here (see ``tests/test_logprobs_cap_lockstep``).
# A positive override (``MAX_LOGPROBS_CAP=200000 modal deploy``) re-imposes a
# server-side per-position ceiling for direct-to-Modal callers if ever wanted.
MAX_LOGPROBS_CAP = int(os.environ.get("MAX_LOGPROBS_CAP", "-1"))
# When set, serve() passes this local filesystem path to `vllm serve` instead
# of the HF repo id. Used by trinity-truebase (weights pre-staged into the
# acs-trinity-cache Volume); set None for HF-cache-based models.
LOCAL_MODEL_PATH: str | None = _CFG.get("local_model_path")
IS_CLUSTERED = N_NODES > 1

# Per-model autoscaler config. Defaults preserve the pre-2026-05-20 behavior
# (min=0, max=4, max_inputs=128, target=max) for any entry that doesn't
# override. See researchlog 2026-05-20 for the conference rationale.
MIN_CONTAINERS = _CFG.get("min_containers", 0)
MAX_CONTAINERS = _CFG.get("max_containers", 4)
SCALEDOWN_WINDOW_S = _CFG.get("scaledown_window_s", 30 * 60)
MAX_INPUTS = _CFG.get("max_inputs", 128)
TARGET_INPUTS = _CFG.get("target_inputs", MAX_INPUTS)

# ACS-200: serve this model through the class-based GPU-memory-snapshot
# lifecycle (VllmSnap below) instead of the shared function-based serve().
# ONLY True for single-GPU models (llama-8b) — a GPU snapshot captures one
# device's memory, so TP>1 models are snapshot-incompatible. When False (every
# multi-GPU model), the VllmSnap class is not even defined and serve() is
# unchanged.
SNAPSHOT = _CFG.get("snapshot", False)

# FAST_BOOT=True: skip torch.compile + CUDA graphs. Saves ~30s on cold start,
# costs ~10-20% throughput. Leave on during dev; turn off once the compile
# cache Volume is warm and we care about throughput.
FAST_BOOT = True

MINUTES = 60

# T1 ACS-17: read at module-import time so the value bakes into the image env.
# Modal does not forward the shell env into the container — without this, a
# `LOAD_FORMAT_DUMMY=1 modal serve` invocation silently runs in normal mode.
_LOAD_FORMAT_DUMMY = os.environ.get("LOAD_FORMAT_DUMMY", "0")

# Prod default is PROD_VLLM_VERSION (0.23.0 since the 2026-07-03 cutover). An env
# override (`VLLM_VERSION=0.19.1 modal deploy modal_app.py`) builds a different
# version — rollback or a future candidate — without changing the default.
# Build-time value: read locally at image construction, so this is not a
# container-env trap.
_VLLM_VERSION = os.environ.get("VLLM_VERSION", PROD_VLLM_VERSION)
# Must track the vLLM version's CUDA (cu130 for 0.23.0, cu128 for 0.19.1).
_TORCH_CUDA_INDEX = os.environ.get("TORCH_CUDA_INDEX", PROD_TORCH_CUDA_INDEX)

# Optional vLLM tuning env baked into the candidate image. Empty for prod.
# VLLM_USE_FLASHINFER_SAMPLER=0 is required for 0.23 on debian_slim: its default
# FlashInfer sampler JIT-compiles a kernel needing nvcc (absent here), which
# kills the engine core. Native sampler avoids the JIT (and the boot delay).
_EXTRA_VLLM_ENV = {
    k: os.environ[k]
    for k in (
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_PORT_WAIT_INTERVAL_S",
        "MAX_LOGPROBS_CAP",
    )
    if k in os.environ
}

# Cold-boot heartbeat cadence for wait_for_vllm_port. Default 5s keeps prod's
# per-stage cold-boot attribution sharp (guard: test_vllm_runtime_markers). A
# staging deploy can widen it (e.g. VLLM_PORT_WAIT_INTERVAL_S=30) so vLLM's own
# boot lines survive Modal's small log buffer during debugging. Read in the
# container; baked into the candidate image via the passthrough above.
_PORT_WAIT_INTERVAL_S = float(os.environ.get("VLLM_PORT_WAIT_INTERVAL_S", "5"))

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
vllm_image = build_vllm_image(
    dev_mode_env=_DEV,
    model_id=MODEL_ID,
    load_format_dummy_env=_LOAD_FORMAT_DUMMY,
    vllm_version=_VLLM_VERSION,
    torch_cuda_index=_TORCH_CUDA_INDEX,
    extra_vllm_env=_EXTRA_VLLM_ENV,
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
# Three Volumes: HF snapshot (weights), vLLM's torch.compile / CUDA-graph
# cache, and pre-sharded weights. The sharded copy needs its own Volume
# because HF cache (~810 GB for 405B) + sharded copy (~810 GB) exceeds
# Modal's 1 TB per-Volume cap — preshard OOMs when both live on one.
_VOLUMES = create_volumes()
hf_cache_vol = _VOLUMES.hf_cache
# Per-model sharded-weights Volume (acs-sharded by default; acs-trinity-sharded
# for Trinity — see ModelSpec.sharded_volume_name). preshard/copy_metadata/serve
# all read/write the SAME Volume at SHARDED_CACHE_DIR via the helpers below.
sharded_vol = sharded_volume_for(_SPEC, _VOLUMES)
# acs-kimi-cache — separate ~1 TB Volume for Kimi-K2-Base block-FP8 weights.
# At Modal's per-Volume 1 TB cap; if the model grows we'd need to shard across
# two volumes. Mounted by serve_clustered() on both rank-0 and rank-1 containers.
# acs-trinity-cache — dedicated Volume for Trinity-Large-TrueBase BF16 weights
# (~743 GB across 31 shards). Populated by serving/prestage_trinity_weights.py.
# acs-lifetime-log — append-only CSV of container start/end events for the
# serve function, so post-run analysis can compute container-seconds (and
# therefore $) without scraping the Modal dashboard. Pull with
# `modal volume get acs-lifetime-log /lifetime.csv ./_attachments/lifetime.csv`.
lifetime_vol = _VOLUMES.lifetime

SHARDED_DIR = sharded_dir_for_model(model_name=MODEL_NAME, n_gpu=N_GPU)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Lifetime tracker — writes one row per container event (start/end) to a
# small Volume so post-run analysis can compute container-seconds (× $/hr)
# without scraping the Modal dashboard. Used by `benchmarks/billing.py`.
#
# Coverage: best-effort. SIGTERM (graceful scaledown) fires the atexit + the
# signal handler. SIGKILL (modal preemption, timeout) bypasses both — those
# containers' end timestamps will be missing and the analysis script falls
# back to the dashboard's "Finished" column for those.
# ---------------------------------------------------------------------------
def _configure_lifetime() -> None:
    lifecycle.configure_lifetime(
        LifetimeConfig(
            lifetime_dir=LIFETIME_DIR,
            lifetime_log_path=LIFETIME_LOG_PATH,
            lifetime_volume=lifetime_vol,
            n_gpu=N_GPU,
            gpu_type=GPU_TYPE,
        )
    )
    # User-facing boot-stage publishing (ACS-272) is armed alongside the
    # lifetime CSV so every serve entrypoint (incl. snapshot restore, whose
    # globals were captured at build time) re-binds to this container.
    boot_status.configure(APP_NAME)


# One emitter feeds both sinks: the ACS-17 lifetime CSV and the ACS-272
# boot-status Dict the wrapper polls to show users cold-boot progress.
_emit = boot_status.make_emitter(lifecycle.emit_lifetime_event)


# ---------------------------------------------------------------------------
# Phase 1 — stage weights
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=120 * MINUTES,
)
def stage_weights():
    """Download MODEL_NAME into the HF cache Volume. Idempotent."""
    stage_weights_impl(
        model_name=MODEL_NAME,
        hf_cache_dir=HF_CACHE_DIR,
        hf_cache_vol=hf_cache_vol,
    )


# ---------------------------------------------------------------------------
# Phase 2 — pre-shard to TP=N (skip for small dev models)
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    volumes=preshard_volumes(spec=_SPEC, volumes=_VOLUMES),
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * MINUTES,
    ephemeral_disk=2 * 1024 * 1024,  # 2 TiB, in MiB. Default is 512 GiB; preshard's
    # 8 workers writing ~100 GB each fill local disk via FUSE before flush. Modal docs:
    # https://modal.com/docs/guide/resources
)
def preshard():
    """Pre-shard MODEL_NAME to TP=N_GPU using vLLM's save_sharded_state.

    One-time op. Writes to SHARDED_DIR. Skips if shards already exist or if
    the selected model's ``preshard`` flag is False (e.g. dev 1B). For MoE
    models (Trinity) the LLM is built with the model's serve-time
    expert-parallel config so the saved shards match what serve() loads —
    save_sharded_state has no EP awareness of its own.
    """
    preshard_impl(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        n_gpu=N_GPU,
        preshard=PRESHARD,
        sharded_dir=SHARDED_DIR,
        sharded_vol=sharded_vol,
        source_model_path=LOCAL_MODEL_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        enable_expert_parallel=_SPEC.enable_expert_parallel,
    )


# ---------------------------------------------------------------------------
# Phase 2b — copy metadata to SHARDED_DIR
# ---------------------------------------------------------------------------
# Cheap CPU function to copy non-weight metadata (config.json, tokenizer.*,
# generation_config.json, …) from the HF cache into SHARDED_DIR. vLLM's
# sharded_state loader needs these alongside the shard files.
#
# Split out of preshard() because preshard's snapshot_download() call
# contended with the async flush of the freshly-written ~800 GB of shards on
# the same FUSE mount — fetch verification ran at 20+ s/file, see researchlog
# 2026-05-12. Running this on a separate CPU container after the shards are
# committed avoids the contention.
@app.function(
    image=vllm_image,
    volumes=copy_metadata_volumes(spec=_SPEC, volumes=_VOLUMES),
    timeout=10 * MINUTES,
)
def copy_metadata():
    copy_metadata_impl(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        preshard=PRESHARD,
        hf_cache_dir=HF_CACHE_DIR,
        sharded_dir=SHARDED_DIR,
        sharded_vol=sharded_vol,
        # Locally-staged models (Trinity) keep config/tokenizer/afmoe-.py in
        # their staged Volume dir, not the HF-hub cache layout.
        metadata_source_dir=LOCAL_MODEL_PATH,
    )


# ---------------------------------------------------------------------------
# Phase 3 — serve
# ---------------------------------------------------------------------------
# Auth (Week 1 dev): vLLM's native --api-key. Set VLLM_API_KEY in the
# "vllm-api" Modal Secret. Every request must carry:
#   Authorization: Bearer <VLLM_API_KEY>
#
# Per-collaborator keys / request-level logging come later via a FastAPI
# shim in front of vLLM. Tracked as a follow-up — see design doc §Auth.
# Volumes mounted into serve() depend on the active model. Built at import
# time off MODEL_ID so each `modal deploy` mounts exactly what that model
# needs and nothing else. acs-trinity-cache only attaches when serving Trinity.
_SERVE_VOLUMES = serve_volumes(spec=_SPEC, volumes=_VOLUMES)


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_S,
    timeout=60 * MINUTES,
    volumes=_SERVE_VOLUMES,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("vllm-api"),
    ],
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=40 * MINUTES)
def serve():
    # Progress visibility during cold boot. `modal serve` prints
    # `Running app...` and then nothing for minutes while Modal
    # provisions the container and vLLM does its imports. The prints
    # below (all `flush=True`) plus wait_for_vllm_port's heartbeat
    # give a wall-clock anchor through that silent window.
    t0 = time.monotonic()
    print("[serve] container up, building command ...", flush=True)

    _configure_lifetime()
    _emit("started")
    _emit("container_up")
    lifecycle.register_lifetime_exit_hooks()

    # Model-path selection priority (highest first):
    #   1) Pre-sharded path (PRESHARD=True + shards on disk)        — llama-405b
    #   2) LOCAL_MODEL_PATH (weights pre-staged into a Volume)      — trinity-truebase
    #   3) HF repo id (vllm pulls via HF cache)                     — llama-8b
    # On Modal Volume FUSE, sharded loading saves ~30 s end-to-end vs the
    # HF-cache path — bottleneck is volume read throughput, not the loader
    # (see researchlog 2026-05-12). Pre-staged local paths skip the HF
    # download entirely.
    use_sharded = PRESHARD and os.path.isdir(SHARDED_DIR) and os.listdir(SHARDED_DIR)
    if use_sharded:
        model_path = SHARDED_DIR
    elif LOCAL_MODEL_PATH:
        model_path = LOCAL_MODEL_PATH
    else:
        model_path = MODEL_NAME
    print(
        f"[serve] use_sharded={use_sharded} model_path={model_path} "
        f"FAST_BOOT={FAST_BOOT} T+{time.monotonic() - t0:.1f}s",
        flush=True,
    )

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
        "--max-logprobs",
        str(MAX_LOGPROBS_CAP),
    ]
    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    cmd.extend(VLLM_EXTRA_ARGS)

    # `--load-format` is single-valued in vLLM's argparser; sharded_state and
    # dummy are mutually exclusive. Dummy wins when set so the T1 engine-init
    # baseline measurement is uncontaminated by sharded-loader path differences.
    dummy_load = os.environ.get("LOAD_FORMAT_DUMMY") == "1"
    if dummy_load:
        cmd += ["--load-format", "dummy"]
        print("[serve] LOAD_FORMAT_DUMMY=1 — engine-init baseline run", flush=True)
    elif use_sharded:
        cmd += ["--load-format", "sharded_state"]

    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--cuda-graph-sizes", "1,2,4,8,16,32,64"]

    print(
        f"[serve] launching at T+{time.monotonic() - t0:.1f}s: {' '.join(cmd)}",
        flush=True,
    )
    _emit("weights_load_start")
    # start_new_session=True puts vLLM in its own process group so the
    # lifecycle hooks can SIGKILL the whole TP=N worker tree atomically via
    # os.killpg. stdout is piped so the marker scanner can split weights-load
    # from engine-init; the scanner re-prints every line so `modal logs`
    # stays equivalent to the old plain-Popen path.
    lifecycle.set_vllm_proc(
        start_vllm_with_event_capture(cmd, emit_event=_emit)
    )
    wait_for_vllm_port(VLLM_PORT, interval_s=_PORT_WAIT_INTERVAL_S)
    _emit("vllm_port_open")


# ---------------------------------------------------------------------------
# VllmSnap — class-based GPU-memory-snapshot serve (ACS-200, single-GPU only)
# ---------------------------------------------------------------------------
# A PARALLEL serve lifecycle for snapshot models (llama-8b), defined ONLY when
# SNAPSHOT is True so multi-GPU deploys (405B/Trinity/Kimi) never register it
# and the shared function-based serve() above is completely unaffected.
#
# Why a class and not a function: Modal GPU memory snapshots need the
# @modal.enter(snap=True/False) split — warm+sleep BEFORE the snapshot, wake
# AFTER restore — which only @app.cls exposes. A single GPU snapshot captures
# one device's memory, so this is inherently single-GPU (TP=1) only.
#
# Endpoint URL: this class method gets a DISTINCT pinned web label
# (f"{APP_NAME}-snap") so it does not collide with the co-registered serve()
# function's auto-label (f"{APP_NAME}-serve") within the same Modal app. The
# prod flip therefore changes the wrapper's upstream_url for llama-8b — see
# docs/design/gpu-snapshot-8b-cold-boot.md for the human-gated cutover sequence.
def _build_snapshot_serve_cmd() -> list[str]:
    """vLLM launch command for the snapshot path, mirroring serve()'s flags.

    Deliberately kept in lockstep with serve() (lines building `cmd` above) for
    the single-GPU case so the snapshot path has no silent flag drift — most
    importantly `--max-logprobs MAX_LOGPROBS_CAP` (ACS-84), which the wrapper
    advertises in lockstep; omitting it would 400 any top_logprobs>20 request.
    Snapshot models are never presharded (single-GPU), so only the
    HF-repo / local-path model source applies (no sharded_state branch).
    Adds `--enable-sleep-mode` (unlocks /sleep + /wake_up) on top.
    """
    model_path = LOCAL_MODEL_PATH or MODEL_NAME
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
        "--max-logprobs",
        str(MAX_LOGPROBS_CAP),
    ]
    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    cmd.extend(VLLM_EXTRA_ARGS)
    # FAST_BOOT parity with serve(): --enforce-eager also means less GPU state to
    # snapshot and a faster warmup. Snapshot models keep FAST_BOOT on.
    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--cuda-graph-sizes", "1,2,4,8,16,32,64"]
    cmd += ["--enable-sleep-mode"]
    return cmd


# NCCL/torch-distributed noise post-restore (KNOWN GAP — benign at TP=1): after
# /sleep + snapshot the frozen torch.distributed TCPStore socket dies, so the
# NCCL watchdog logs a c10d "Exception raised from sendBytes at ...
# c10d/Utils.hpp:653" + a C++ stack-frame dump ~1×/s. Harmless — single-GPU
# inference uses no collectives and completions verified correct — but on the
# ACS-200 staging boot it was ~75% of all log lines. NOT SUPPRESSED: env knobs
# (TORCH_NCCL_ENABLE_MONITORING=0, GLOG_minloglevel=3, TORCH_CPP_LOG_LEVEL=FATAL)
# and the stdout drop-filter below were all verified ineffective — the dumps are
# raw C++ writes to the EngineCore grandchild's stderr, straight to container
# logs, bypassing both glog gating and our Popen pipe. A real fix (tear down the
# process group before snapshot, or a Modal-side log filter) is a follow-up.
# The drop_patterns wiring is kept (tested, reusable) for any noise that DOES
# transit the pump; the patterns are targeted so a genuine crash trace is never
# swallowed (NOT matching "frame #"/"Exception raised from"; Python-level
# EngineDead/Traceback stay visible).
_SNAPSHOT_NCCL_DROP_PATTERNS = (
    "sendBytes",
    "Utils.hpp:653",
    "distributed/c10d",
)

if SNAPSHOT:
    # VLLM_SERVER_DEV_MODE=1 unlocks vLLM's /sleep + /wake_up dev endpoints.
    # TORCH_NCCL_ENABLE_MONITORING=0 disables the NCCL heartbeat *monitor* thread
    # (canonical snapshot hygiene); it does NOT stop the separate sendBytes spam
    # above (see KNOWN GAP). Scoped to this snapshot Secret so it never affects
    # the shared serve() path. Injected as a Secret because build_vllm_image()
    # ends with .add_local_python_source(...), so .env() can't be chained after.
    _snapshot_env_secret = modal.Secret.from_dict(
        {
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCH_NCCL_ENABLE_MONITORING": "0",
        }
    )

    @app.cls(
        image=vllm_image,
        gpu=f"{GPU_TYPE}:{N_GPU}",
        min_containers=MIN_CONTAINERS,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_S,
        timeout=60 * MINUTES,
        volumes=_SERVE_VOLUMES,
        secrets=[
            modal.Secret.from_name("huggingface-secret"),
            modal.Secret.from_name("vllm-api"),
            _snapshot_env_secret,
        ],
        enable_memory_snapshot=True,
        experimental_options={"enable_gpu_snapshot": True},
    )
    @modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
    class VllmSnap:
        @modal.enter(snap=True)
        def startup(self) -> None:
            """Pre-snapshot: start vLLM, warm it, /sleep. Runs once per build."""
            t0 = time.monotonic()
            print("[snap] startup(snap=True): building command ...", flush=True)
            _configure_lifetime()
            _emit("started")
            _emit("container_up")
            lifecycle.register_lifetime_exit_hooks()

            cmd = _build_snapshot_serve_cmd()
            print(f"[snap] launching: {' '.join(cmd)}", flush=True)
            _emit("weights_load_start")
            self.process = start_vllm_with_event_capture(
                cmd,
                emit_event=_emit,
                drop_patterns=_SNAPSHOT_NCCL_DROP_PATTERNS,
            )
            lifecycle.set_vllm_proc(self.process)
            wait_for_vllm_port(VLLM_PORT, interval_s=_PORT_WAIT_INTERVAL_S)
            _emit("vllm_port_open")
            # Port-open is a TCP accept; on vLLM V1 the HTTP port can open before
            # the engine can serve (503 "engine loading"). warmup() POSTs
            # /v1/completions with raise_for_status and no retry, so gate it on
            # /health first or a race would 503 and abort the snapshot build.
            snapshot_runtime.wait_ready(self.process, port=VLLM_PORT)
            snapshot_runtime.warmup(port=VLLM_PORT, served_model_name=SERVED_MODEL_NAME)
            snapshot_runtime.sleep(port=VLLM_PORT, level=1)
            print(
                f"[snap] startup(snap=True) done in {time.monotonic() - t0:.1f}s "
                "— ready for snapshot",
                flush=True,
            )

        @modal.enter(snap=False)
        def restore(self) -> None:
            """Post-restore (and first build boot): /wake_up, then serve.

            The snapshot captured this module's globals from the build
            container, so reset lifetime state and re-register hooks against
            THIS container's id before emitting its container_up row.
            """
            lifecycle.reset_lifetime_state()
            _configure_lifetime()
            _emit("restored")
            _emit("container_up")
            lifecycle.set_vllm_proc(self.process)
            lifecycle.register_lifetime_exit_hooks()
            snapshot_runtime.wake_up(port=VLLM_PORT)
            # Restored containers skip the weight-load/engine-init stages —
            # vLLM is already up after /wake_up. Publish the terminal stage
            # directly (Dict-only; no extra lifetime-CSV row, ACS-272).
            boot_status.publish_stage("serving")

        @modal.web_server(
            port=VLLM_PORT,
            startup_timeout=40 * MINUTES,
            label=f"{APP_NAME}-snap",
        )
        def serve(self) -> None:
            """No-op: vLLM is already listening on VLLM_PORT from startup()."""
            print("[snap] web_server hook: vLLM already listening", flush=True)

        @modal.exit()
        def stop(self) -> None:
            lifecycle.kill_vllm_now()
            lifecycle.emit_end_once("modal_exit")


# ---------------------------------------------------------------------------
# serve_clustered — multi-node Kimi-K2-Base via @modal.experimental.clustered
# ---------------------------------------------------------------------------
# 2026-05-18 eve — added by kimi-k2-multinode research team for the Kimi-K2-Base
# entry in MODELS. Coexists with serve() above; per Modal-deploy semantics,
# both functions exist in the deployed app but only the one matching the
# active MODEL_ID's topology gets traffic from the wrapper.
#
# Topology TP=8 within node + PP=2 between nodes:
#   - rank 0: starts a Ray head, launches `vllm serve` with PP=2 TP=8,
#             exposes the HTTP endpoint on port 8000 (this rank's URL is the
#             public-facing one routed to by the wrapper)
#   - rank 1: starts a Ray worker, joins the rank-0 head, then `--block`s to
#             keep the container alive while vLLM uses it as a PP stage
#
# IMPORTANT: at the time of writing (2026-05-18 eve), no public example
# composes @modal.web_server with @modal.experimental.clustered. A 30-min
# smoke test (deploy a trivial clustered function with web_server, confirm
# rank-0 URL is publicly reachable) MUST PASS before relying on this code.
# See plan doc §H.2 Risk #0.
#
# Only registered when the active MODEL_ID is a multi-node entry. For
# single-node models the import-time @modal.experimental.clustered(size=1)
# would either be a no-op or a validation error; in either case there's no
# benefit to declaring it.
if IS_CLUSTERED:

    @app.function(
        image=vllm_image,
        gpu=f"{GPU_TYPE}:{N_GPU}",
        min_containers=0,
        max_containers=N_NODES,  # Must be a multiple of cluster_size (Modal counts individual containers, not clusters). N_NODES=2 means one cluster of size 2.
        scaledown_window=30 * MINUTES,
        timeout=120
        * MINUTES,  # 1 TB FP8 weights load — be generous on cold-boot timeout
        volumes=clustered_serve_volumes(spec=_SPEC, volumes=_VOLUMES),
        secrets=[
            modal.Secret.from_name("huggingface-secret"),
            modal.Secret.from_name("vllm-api"),
        ],
        # `efa_enabled` was claimed by the modal-app-architect researcher to be
        # required for @clustered(rdma=True) to actually use RDMA, but I couldn't
        # find it in Modal's official multi-node-training docs (which just say
        # "RDMA is enabled with the rdma parameter"). Kept here defensively — if
        # it's a no-op the cost is zero; if it's actually required the cost of
        # omitting it would be 3-5× slowdown. Verify in the smoke test by checking
        # NCCL_DEBUG=INFO output for NET/IB/GDRDMA vs NET/Socket.
        experimental_options={"efa_enabled": True},
    )
    # broadcast=True is the only supported value in Modal SDK 1.4.2 (the signature
    # accepts broadcast=False but the body asserts True — "not implemented yet").
    # This makes the smoke test (plan §H.5.2) load-bearing: we need to confirm
    # Modal's web_server proxy doesn't fan HTTP requests to BOTH containers (rank 1
    # has no listener on port 8000 — would 502). If it does fan, fallback to the
    # two-function head+worker design (architect Risk #1 contingency).
    @_modal_clustered(size=N_NODES, rdma=RDMA)
    @modal.concurrent(max_inputs=128)
    @modal.web_server(port=VLLM_PORT, startup_timeout=60 * MINUTES)
    def serve_clustered():
        """Multi-node serving via Modal's experimental clustered API.

        Only invoked when MODEL_ID's entry has n_nodes > 1. For single-node
        models (llama-8b, llama-405b), the regular serve() function above is
        used.
        """
        t0 = time.monotonic()
        print(
            f"[serve_clustered] container up at T+0s, MODEL_ID={MODEL_ID}", flush=True
        )

        # Modal's cluster-info — provides rank assignment + peer IPs.
        # NOTE: modal.experimental.get_cluster_info() is Beta; signature could shift.
        info = _modal_get_cluster_info()
        rank = info.rank
        n_ranks = N_NODES
        head_ip = info.peer_ips[0]
        print(
            f"[serve_clustered] rank={rank}/{n_ranks - 1}  head_ip={head_ip}  "
            f"peer_ips={info.peer_ips}  T+{time.monotonic() - t0:.1f}s",
            flush=True,
        )

        _configure_lifetime()
        lifecycle.emit_lifetime_event(f"started_rank{rank}")
        if rank == 0:
            _emit("container_up")
        lifecycle.register_lifetime_exit_hooks()

        n_gpus_per_node = N_GPU  # 8

        if rank == 0:
            # Rank 0 starts the Ray head, then launches vLLM with PP=2 TP=8.
            # vLLM bringtup auto-spawns Ray workers across the cluster via the
            # ray:// scheme — the rank-1 container's Ray worker registers with
            # this head via the RDMA peer network.
            print("[serve_clustered] rank 0: starting Ray head ...", flush=True)
            ray_head_cmd = [
                "ray",
                "start",
                "--head",
                "--port=6379",
                "--num-gpus",
                str(n_gpus_per_node),
                "--block=false",
            ]
            subprocess.run(ray_head_cmd, check=True)

            # Parse PP/TP from TOPOLOGY string. Currently only "tp16" supported.
            # Per Moonshot's deploy guide + the vLLM Kimi-K2 recipe, TP=16 is the
            # canonical 16-GPU shape (with RDMA between nodes). PP=2/TP=8 was the
            # initial design but rejected after the research-team synthesis —
            # Moonshot's own guide omits PP at this scale.
            if TOPOLOGY == "tp16":
                tp_size, pp_size = 16, 1
            elif TOPOLOGY == "tp8_pp2":
                tp_size, pp_size = 8, 2
            else:
                raise RuntimeError(
                    f"Unsupported topology {TOPOLOGY!r} for clustered serve. "
                    "Supported: 'tp16' (default), 'tp8_pp2' (fallback if TP=16 fails)."
                )

            cmd = [
                "vllm",
                "serve",
                MODEL_NAME,
                "--host",
                "0.0.0.0",
                "--port",
                str(VLLM_PORT),
                "--tensor-parallel-size",
                str(tp_size),
                "--pipeline-parallel-size",
                str(pp_size),
                "--distributed-executor-backend",
                "ray",
                "--dtype",
                DTYPE,
                "--max-model-len",
                str(MAX_MODEL_LEN),
                "--served-model-name",
                SERVED_MODEL_NAME,
                "--max-logprobs",
                str(MAX_LOGPROBS_CAP),
            ]
            if TRUST_REMOTE_CODE:
                cmd.append("--trust-remote-code")
            cmd.extend(VLLM_EXTRA_ARGS)

            if FAST_BOOT:
                cmd.append("--enforce-eager")

            if os.environ.get("LOAD_FORMAT_DUMMY") == "1":
                cmd += ["--load-format", "dummy"]
                print(
                    "[serve_clustered] LOAD_FORMAT_DUMMY=1 — engine-init baseline run",
                    flush=True,
                )

            print(
                f"[serve_clustered] rank 0: launching vLLM at T+{time.monotonic() - t0:.1f}s: "
                f"{' '.join(cmd)}",
                flush=True,
            )
            _emit("weights_load_start")
            lifecycle.set_vllm_proc(
                start_vllm_with_event_capture(
                    cmd, emit_event=_emit
                )
            )
            wait_for_vllm_port(VLLM_PORT, interval_s=_PORT_WAIT_INTERVAL_S)
            _emit("vllm_port_open")
        else:
            # Rank 1+: join the Ray head on rank-0 in the background, then RETURN.
            # Modal's @web_server requires every rank's function to return within
            # `startup_timeout` (otherwise the rank is considered failed-to-init
            # and the whole cluster tears down — confirmed by 2026-05-18 eve smoke
            # test logs: "Runner failed with exception: Runner has been initializing
            # for too long: 600 seconds. ... use a non-blocking call such as
            # subprocess.Popen"). The Popen'd `ray start --block` process keeps
            # running in the background as a Ray worker; Modal sees the function
            # returned cleanly + keeps the container alive because the cluster
            # is alive.
            print(
                f"[serve_clustered] rank {rank}: joining Ray head at {head_ip}:6379 (Popen, non-blocking) ...",
                flush=True,
            )
            ray_worker_cmd = [
                "ray",
                "start",
                "--address",
                f"{head_ip}:6379",
                "--num-gpus",
                str(n_gpus_per_node),
                "--block",  # ray-internal flag — keeps Ray's daemon in the foreground of ITS process
            ]
            # NOTE: Popen here is non-blocking from the function's POV — function
            # returns immediately, but the Popen'd `ray start --block` runs forever
            # inside its own process group. Modal keeps the container alive because
            # the @clustered decorator pins it to the cluster lifetime.
            subprocess.Popen(
                ray_worker_cmd,
                start_new_session=True,
            )
            # Briefly wait so Ray's worker has registered with the head before this
            # function returns, otherwise rank 0's `vllm serve` may try to recruit
            # workers before they're ready.
            import time as _time

            _time.sleep(15)
            print(
                f"[serve_clustered] rank {rank}: Ray worker spawned in background; returning",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Local entrypoint — prints how to test.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print(f"App:     {APP_NAME}")
    print(f"Model:   {MODEL_NAME}  (id={MODEL_ID})")
    print(f"GPUs:    {N_GPU}x{GPU_TYPE}")
    print(f"Preshard: {PRESHARD}")
    print()
    print("Available models:", ", ".join(sorted(MODELS)))
    print("Override with MODEL_ID=<id> on any modal run/deploy/serve command.")
    print()
    print("Run sequence:")
    print(f"  1) MODEL_ID={MODEL_ID} modal run modal_app.py::stage_weights")
    if PRESHARD:
        print(f"  2) MODEL_ID={MODEL_ID} modal run modal_app.py::preshard")
        print(f"  3) MODEL_ID={MODEL_ID} modal run modal_app.py::copy_metadata")
    print(f"  4) MODEL_ID={MODEL_ID} modal serve modal_app.py     # dev, ephemeral")
    print(f"     MODEL_ID={MODEL_ID} modal deploy modal_app.py    # persistent")
    print()
    print("ACS-17 baseline (engine-init only, skips weight I/O):")
    print(
        f"  LOAD_FORMAT_DUMMY=1 MODEL_ID={MODEL_ID} modal serve modal_app.py"
    )
    print()
    print("Smoke test:")
    print("  BASE_URL=<modal-url> API_KEY=<vllm-api-key> python scripts/validate.py")
