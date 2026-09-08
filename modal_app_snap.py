"""ACS-193 SPIKE — llama-8b cold-boot via Modal GPU memory snapshots.

Standalone staging app (``acs-llama-8b-snap``). Does NOT touch the prod
``serve()`` path or the prod ``acs-llama-8b`` app. Reuses the existing weights
Volume (``acs-hf-cache``) — no re-download.

Idea (Modal GPU memory snapshots, alpha; only viable on single-GPU):
  - ``enable_memory_snapshot=True`` + ``experimental_options={"enable_gpu_snapshot": True}``
    make Modal capture CPU **and** GPU memory state of a warmed container.
  - vLLM ``--enable-sleep-mode`` exposes ``/sleep`` + ``/wake_up`` dev endpoints
    (gated behind ``VLLM_SERVER_DEV_MODE=1``).
  - ``@modal.enter(snap=True)`` starts vLLM, warms it, then ``/sleep`` (level 1 —
    offload weights GPU→CPU RAM). Modal snapshots that state.
  - ``@modal.enter(snap=False)`` runs on restore: ``/wake_up`` (CPU→GPU) so the
    already-listening server can serve immediately.

Reference: Modal's ``06_gpu_and_ml/llm-serving/lfm_snapshot.py`` (uses the newer
``@app.server`` API; we're on Modal 1.4.3 so we use ``@app.cls`` + ``@modal.web_server``).

Usage:
    # 0) Verify the pinned vLLM actually supports sleep mode (cheap, no GPU):
    modal run modal_app_snap.py::check_flags

    # 1) Deploy (snapshots ONLY engage under `modal deploy`, not run/serve):
    modal deploy modal_app_snap.py

    # 2) Boot a few times (snapshot activates after a handful of cold starts,
    #    usually < 5), force-stopping the container between boots, and confirm a
    #    RESTORE actually happened in the logs before trusting any timing.

    # 3) Tear down when done:
    modal app stop acs-llama-8b-snap --yes
"""

from __future__ import annotations

import subprocess
import time

import modal

from serving.modal_resources import (
    HF_CACHE_DIR,
    VLLM_CACHE_DIR,
    VLLM_PORT,
    build_vllm_image,
    create_volumes,
)

MINUTES = 60

# --- llama-8b, hardcoded (mirrors acs_model_registry llama-8b spec) -----------
MODEL_NAME = "meta-llama/Llama-3.1-8B"
SERVED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
GPU_TYPE = "L40S"
N_GPU = 1
MAX_MODEL_LEN = 8192
DTYPE = "bfloat16"
APP_NAME = "acs-llama-8b-snap"

# --- image: this POC was validated on vllm==0.19.1 (the ~97s→~32s result is
# specific to that version's sleep-mode). Pin it explicitly so the prod-default
# flip to 0.23 (2026-07-03) doesn't silently change what this POC builds — snapshot
# behaviour on 0.23 sleep-mode is untested. Re-validate before assuming 0.23 here.
# build_vllm_image ends with .add_local_python_source(...), so we can't chain
# .env() after it. Inject VLLM_SERVER_DEV_MODE=1 (which unlocks vLLM's /sleep +
# /wake_up dev endpoints) via a Secret attached to the function instead.
vllm_image = build_vllm_image(
    dev_mode_env="1",
    model_id="llama-8b",
    vllm_version="0.19.1",
    torch_cuda_index="https://download.pytorch.org/whl/cu128",
)
_dev_mode_secret = modal.Secret.from_dict({"VLLM_SERVER_DEV_MODE": "1"})

_VOLUMES = create_volumes()
hf_cache_vol = _VOLUMES.hf_cache
vllm_cache_vol = _VOLUMES.vllm_cache

app = modal.App(APP_NAME)

with vllm_image.imports():
    import requests


# ---------------------------------------------------------------------------
# Sleep/wake helpers (localhost, unauthenticated — staging app carries no
# --api-key, so warmup/sleep/wake and the external probe all skip auth).
# ---------------------------------------------------------------------------
def _check_running(p: subprocess.Popen) -> None:
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def _wait_ready(process: subprocess.Popen, timeout: int = 35 * MINUTES) -> None:
    deadline = time.time() + timeout
    t0 = time.time()
    while time.time() < deadline:
        try:
            _check_running(process)
            requests.get(
                f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5
            ).raise_for_status()
            print(f"[snap] vLLM healthy after {time.time() - t0:.1f}s", flush=True)
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.ReadTimeout,
        ):
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready within {timeout}s")


def _warmup() -> None:
    payload = {
        "model": SERVED_MODEL_NAME,
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0.0,
    }
    for _ in range(2):
        requests.post(
            f"http://127.0.0.1:{VLLM_PORT}/v1/completions", json=payload, timeout=60
        ).raise_for_status()
    print("[snap] warmup complete", flush=True)


def _sleep(level: int = 1) -> None:
    requests.post(
        f"http://127.0.0.1:{VLLM_PORT}/sleep?level={level}", timeout=120
    ).raise_for_status()
    print(f"[snap] vLLM asleep (level={level})", flush=True)


def _wake_up() -> None:
    t0 = time.time()
    requests.post(
        f"http://127.0.0.1:{VLLM_PORT}/wake_up", timeout=120
    ).raise_for_status()
    print(f"[snap] vLLM woke up in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Pre-flight: confirm the pinned vLLM exposes sleep mode (no GPU needed).
# ---------------------------------------------------------------------------
@app.function(image=vllm_image, timeout=5 * MINUTES)
def check_flags() -> None:
    import vllm

    print(f"[check_flags] vLLM version: {vllm.__version__}", flush=True)

    # 1) EngineArgs dataclass field for sleep mode
    try:
        import dataclasses
        from vllm.engine.arg_utils import EngineArgs

        fields = [f.name for f in dataclasses.fields(EngineArgs)]
        sleep_fields = [f for f in fields if "sleep" in f.lower()]
        print(f"[check_flags] EngineArgs sleep fields: {sleep_fields}", flush=True)
    except Exception as exc:
        print(f"[check_flags] EngineArgs introspection failed: {exc!r}", flush=True)

    # 2) Grep the full CLI help (stdout+stderr) for the flag literally
    out = subprocess.run(["vllm", "serve", "--help"], capture_output=True, text=True)
    hay = out.stdout + out.stderr
    hits = [ln.strip() for ln in hay.splitlines() if "sleep" in ln.lower()]
    print(f"[check_flags] '--help' sleep lines: {hits}", flush=True)

    # 3) Grep the WHOLE installed vllm package for /sleep + /wake_up route defs
    grep = subprocess.run(
        [
            "grep",
            "-rn",
            "-E",
            r'"/(sleep|wake_up)"|/wake_up|reset_prefix_cache',
            "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/",
        ],
        capture_output=True,
        text=True,
    )
    print("[check_flags] route grep in entrypoints/:", flush=True)
    print(grep.stdout[:2000] or "  (no matches)", flush=True)

    # 4) Does the in-process LLM engine expose sleep()/wake_up() methods?
    try:
        from vllm import LLM

        print(
            f"[check_flags] LLM.sleep exists: {hasattr(LLM, 'sleep')}; "
            f"LLM.wake_up exists: {hasattr(LLM, 'wake_up')}",
            flush=True,
        )
    except Exception as exc:
        print(f"[check_flags] LLM introspection failed: {exc!r}", flush=True)


# ---------------------------------------------------------------------------
# Snapshot-enabled serve.
# ---------------------------------------------------------------------------
@app.cls(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    volumes={HF_CACHE_DIR: hf_cache_vol, VLLM_CACHE_DIR: vllm_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret"), _dev_mode_secret],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    min_containers=0,  # must be 0 to observe a cold boot (prod spec is 1/always-on)
    max_containers=1,
    scaledown_window=2 * MINUTES,  # short so it scales down fast between measurements
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=32)
class VllmSnap:
    @modal.enter(snap=True)
    def startup(self) -> None:
        """Runs BEFORE the snapshot: start vLLM, warm it, put it to sleep."""
        t0 = time.monotonic()
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
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
            "--enforce-eager",  # match prod FAST_BOOT; less to snapshot, faster warmup
            "--enable-sleep-mode",
        ]
        print(f"[snap] startup(snap=True) launching: {' '.join(cmd)}", flush=True)
        self.process = subprocess.Popen(cmd)
        _wait_ready(self.process)
        _warmup()
        _sleep(level=1)
        print(
            f"[snap] startup(snap=True) done in {time.monotonic() - t0:.1f}s "
            "— ready for snapshot",
            flush=True,
        )

    @modal.enter(snap=False)
    def restore(self) -> None:
        """Runs AFTER snapshot restore (and on the very first build boot)."""
        print("[snap] restore(snap=False): waking vLLM ...", flush=True)
        _wake_up()

    @modal.web_server(port=VLLM_PORT, startup_timeout=40 * MINUTES)
    def serve(self) -> None:
        """No-op: vLLM is already listening on VLLM_PORT from startup()."""
        print("[snap] web_server hook: vLLM already listening", flush=True)

    @modal.exit()
    def stop(self) -> None:
        try:
            self.process.terminate()
        except Exception:
            pass
