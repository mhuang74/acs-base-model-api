"""Smoke test: @modal.experimental.clustered + @modal.web_server composition.

Answers the single question:
    Can we serve HTTP traffic from a 2-container Modal cluster (rdma=True)
    where one rank is a "head" and the other is a silent worker?

Why we need this: nothing in Modal's public examples composes these two
decorators (every clustered example is for training). The Kimi-K2-Base
multi-node plan rests on the assumption that they compose; if rank 1 (which
has nothing listening on :8000) starts receiving HTTP requests and 502s, we
have to fall back to a two-function head+worker design coordinated via
`modal.Dict`.

Each rank starts a tiny stdlib HTTP server on :8000 and logs its rank +
container ID on every request. After deploy, hit the URL ~50 times and:
  - confirm every request returns 200 (not 502)
  - via `modal app logs`, check the rank distribution of handled requests
    — ideally all served by rank 0; if rank 1 ever responds, that's the
    "fan to both" failure mode

Cost: 8×H200 × 2 nodes × ~5 min of cluster life ≈ $5.

Run:
    modal deploy serving/experiments/clustered_smoke.py
    # wait ~6 min for cold boot
    for i in {1..50}; do curl -s <URL>/healthz; done
    .venv/bin/modal app logs acs-clustered-smoke | grep -E "rank|HEALTHZ"
    .venv/bin/modal app stop acs-clustered-smoke -y
"""

from __future__ import annotations

import os
import subprocess

import modal
from modal.experimental import (
    clustered as _modal_clustered,
    get_cluster_info as _modal_get_cluster_info,
)


app = modal.App("acs-clustered-smoke")

image = modal.Image.debian_slim(python_version="3.12")


@app.function(
    image=image,
    gpu="H200:8",
    timeout=10 * 60,
    min_containers=0,
    max_containers=2,  # Must be a multiple of cluster_size for clustered fns
    scaledown_window=60,  # Quick teardown after we stop the app
    # `efa_enabled` is claimed by some sources to be required for rdma=True
    # to actually engage RDMA. Modal's official docs don't mention it.
    # Including it defensively for the smoke test.
    experimental_options={"efa_enabled": True},
)
@_modal_clustered(
    size=2, rdma=False
)  # rdma=True errors with "RDMA is not supported on this workspace" — needs Modal-side enablement
@modal.web_server(port=8000, startup_timeout=10 * 60)
def smoke():
    """Each rank starts a tiny HTTP server on :8000.

    Logs every request with rank + container ID so post-run analysis can
    see who handled what. Modal's web_server proxy picks one response per
    public-URL request — the goal is to learn which rank wins.
    """
    info = _modal_get_cluster_info()
    rank = info.rank
    container_id = os.environ.get("MODAL_TASK_ID", "<unknown>")
    print(
        f"[SMOKE] rank={rank}/{len(info.container_ips) - 1} "
        f"container_id={container_id} container_ips={info.container_ips}",
        flush=True,
    )

    # nvidia-smi to confirm 8×H200 visible
    try:
        nv = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print(f"[SMOKE rank={rank}] nvidia-smi:\n{nv.stdout}", flush=True)
    except Exception as exc:
        print(f"[SMOKE rank={rank}] nvidia-smi failed: {exc!r}", flush=True)

    # Tiny stdlib HTTP server on :8000. We avoid Flask to keep the image trivial.
    handler_code = f"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            body = json.dumps({{
                "rank": {rank},
                "world_size": {len(info.container_ips)},
                "container_id": "{container_id}",
                "path": self.path,
            }}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            sys.stdout.write(f"[HEALTHZ rank={rank} container={container_id}] 200\\n")
            sys.stdout.flush()
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *args, **kw): pass


srv = ThreadingHTTPServer(("0.0.0.0", 8000), _H)
sys.stdout.write(f"[SMOKE rank={rank}] HTTP server listening on :8000\\n")
sys.stdout.flush()
srv.serve_forever()
"""

    # Spawn the HTTP server in the background with Popen — Modal's @web_server
    # requires this function to RETURN (not block) after the port is listening,
    # otherwise the startup-check times out at 600s and the cluster tears down.
    # The Popen'd subprocess keeps running in the background; Modal sees the
    # function returned cleanly + the port is listening, and routes HTTP traffic.
    print(
        f"[SMOKE rank={rank}] launching stdlib HTTP server (Popen, non-blocking) ...",
        flush=True,
    )
    proc = subprocess.Popen(
        ["python", "-c", handler_code],
        start_new_session=True,
        env={**os.environ, "NCCL_DEBUG": "INFO"},
    )

    # Wait briefly so the port is bound before the function returns. If we
    # return before the port is open, Modal's web_server proxy may briefly
    # 503 on early requests.
    import socket as _socket
    import time as _time

    deadline = _time.monotonic() + 30
    while _time.monotonic() < deadline:
        try:
            with _socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
                print(
                    f"[SMOKE rank={rank}] HTTP port :8000 is live; returning",
                    flush=True,
                )
                return
        except OSError:
            _time.sleep(0.5)
    raise RuntimeError(
        f"[SMOKE rank={rank}] HTTP server failed to bind :8000 within 30s; subprocess pid={proc.pid} returncode={proc.returncode}"
    )


@app.local_entrypoint()
def main():
    print("Run:")
    print("  modal deploy serving/experiments/clustered_smoke.py")
    print("  # wait ~6 min for cluster cold boot")
    print("  # hit https://<workspace>--acs-clustered-smoke-smoke.example.modal.run/healthz")
    print("  # then: modal app logs acs-clustered-smoke")
    print("  # cleanup: modal app stop acs-clustered-smoke -y")
