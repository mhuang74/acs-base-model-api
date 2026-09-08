from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_real_uvicorn_flushes_json_keepalive_chunks():
    port = _free_port()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{root}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.keepalive_uvicorn_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                if httpx.get(f"{base}/ready", timeout=0.2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                stdout, stderr = proc.communicate(timeout=1)
                raise AssertionError(f"uvicorn did not start: {stdout}\n{stderr}")
            time.sleep(0.02)

        arrivals: list[tuple[float, bytes]] = []
        started = time.monotonic()
        with httpx.stream("GET", f"{base}/probe", timeout=2.0) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("application/json")
            for chunk in response.iter_raw():
                arrivals.append((time.monotonic() - started, chunk))

        body = b"".join(chunk for _, chunk in arrivals)
        assert json.loads(body.strip())["choices"][0]["text"] == "Paris"
        whitespace_chunks = [chunk for _, chunk in arrivals[:-1] if not chunk.strip()]
        assert len(whitespace_chunks) >= 2
        assert arrivals[0][0] < 0.1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
