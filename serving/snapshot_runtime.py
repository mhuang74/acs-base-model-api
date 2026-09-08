"""Sleep/wake helpers for the class-based GPU-memory-snapshot serve path.

Ported from the ACS-193 POC (``modal_app_snap.py``) so the productionized
snapshot lifecycle in ``modal_app.py`` reuses the same, tested localhost
sleep/wake/warmup calls instead of duplicating them (ACS-200).

The functions talk to a locally-running vLLM (started with
``--enable-sleep-mode`` and ``VLLM_SERVER_DEV_MODE=1``, which unlocks the
``/sleep`` + ``/wake_up`` dev endpoints):

  - ``sleep(level=1)``  offloads weights GPU→CPU RAM so Modal's GPU snapshot
    captures a small, restorable state.
  - ``wake_up()``       reloads weights CPU→GPU on snapshot restore; the vLLM
    server was already listening, so it can serve immediately after.

All calls are unauthenticated: they hit 127.0.0.1 inside the container, and
``/sleep`` + ``/wake_up`` are not exposed through vLLM's ``--api-key`` auth.

``requests`` is imported lazily inside each function so this module stays
importable outside the vLLM image (e.g. in unit tests / local `import` proofs).
"""

from __future__ import annotations

import os
import subprocess
import time

MINUTES = 60


def _auth_headers() -> dict[str, str]:
    """Bearer header when the container carries a VLLM_API_KEY.

    The productionized snapshot serve inherits the ``vllm-api`` Secret (unlike
    the unauthenticated ACS-193 POC), so vLLM enforces its ``--api-key`` auth on
    /v1/completions. The localhost warmup must present the key or it 401s. Dev
    endpoints (/sleep, /wake_up) and /health are not key-gated, but sending the
    header there is harmless. Empty dict when no key is set, so the helpers also
    work against an unauthenticated local/POC vLLM.
    """
    key = os.environ.get("VLLM_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def check_running(proc: subprocess.Popen) -> None:
    """Raise if the vLLM subprocess has already exited."""
    if (rc := proc.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=proc.args)


def wait_ready(process: subprocess.Popen, *, port: int, timeout: int = 35 * MINUTES) -> None:
    """Block until vLLM answers /health, or the subprocess dies / times out."""
    import requests

    deadline = time.time() + timeout
    t0 = time.time()
    while time.time() < deadline:
        # Outside the try: a dead vLLM subprocess must fail the boot immediately
        # (CalledProcessError propagates) instead of being retried until the
        # 35-min timeout — code-review finding on PR #185.
        check_running(process)
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=5).raise_for_status()
            print(f"[snap] vLLM healthy after {time.time() - t0:.1f}s", flush=True)
            return
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.ReadTimeout,
        ):
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready within {timeout}s")


def warmup(*, port: int, served_model_name: str, rounds: int = 2) -> None:
    """Fire a couple of tiny completions so caches/graphs are hot pre-snapshot."""
    import requests

    payload = {
        "model": served_model_name,
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0.0,
    }
    headers = _auth_headers()
    for _ in range(rounds):
        requests.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json=payload,
            headers=headers,
            timeout=60,
        ).raise_for_status()
    print("[snap] warmup complete", flush=True)


def sleep(*, port: int, level: int = 1) -> None:
    """Put vLLM to sleep (level 1 = offload weights GPU→CPU RAM)."""
    import requests

    requests.post(
        f"http://127.0.0.1:{port}/sleep?level={level}",
        headers=_auth_headers(),
        timeout=120,
    ).raise_for_status()
    print(f"[snap] vLLM asleep (level={level})", flush=True)


def wake_up(*, port: int) -> float:
    """Wake vLLM (reload weights CPU→GPU). Returns wall-clock wake seconds."""
    import requests

    t0 = time.time()
    requests.post(
        f"http://127.0.0.1:{port}/wake_up", headers=_auth_headers(), timeout=120
    ).raise_for_status()
    elapsed = time.time() - t0
    print(f"[snap] vLLM woke up in {elapsed:.1f}s", flush=True)
    return elapsed
