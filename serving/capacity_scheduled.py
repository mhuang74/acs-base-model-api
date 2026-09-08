"""
ACS Modal capacity probe — HTTP-triggered edition

The schedule used to live in this file as a ``modal.Cron``. It now lives in
the wrapper's Postgres (``probe_schedule.cron_expression``), with the wrapper's
APScheduler firing this endpoint on the configured cadence. Admins can edit
the cron expression from /admin without redeploying Modal.

Deploy:
  modal deploy serving/capacity_scheduled.py

Endpoint URL is printed at deploy time — put it in Railway as MODAL_PROBE_URL.

Auth: a bearer token from the ``probe-bearer`` Modal Secret. Create it once
per workspace before deploying:
  modal secret create probe-bearer PROBE_BEARER=<long-random>

Stop it when you're done:
  modal app stop acs-capacity-probe
"""

import hmac
import os
import json
import time

import modal
from fastapi import Request

from serving.capacity_probe import (
    TELEMETRY_EMPTY,
    gpu_type_from_inventory,
    query_gpu_inventory,
    query_gpu_telemetry,
    summarize_gpu_telemetry,
)

app = modal.App("acs-capacity-probe")

# fastapi[standard] is needed because @modal.fastapi_endpoint defers FastAPI
# import-time work into the container.
image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]")
    .add_local_python_source("serving")
)

PROBE_BEARER_SECRET = modal.Secret.from_name("probe-bearer")


@app.function(image=image, gpu="H200:8", timeout=600)
def probe_h200():
    inventory = query_gpu_inventory().strip()
    telemetry = summarize_gpu_telemetry(query_gpu_telemetry())
    return {
        "inventory": inventory,
        "gpu_type": gpu_type_from_inventory(inventory),
        "cloud": os.environ.get("MODAL_CLOUD_PROVIDER") or None,
        "region": os.environ.get("MODAL_REGION") or None,
        **telemetry,
    }


@app.local_entrypoint()
def main():
    """Run one visible paid probe from the Modal CLI."""
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] requesting 8xH200 capacity probe...")
    payload = probe_h200.remote()
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] probe completed in {elapsed:.1f}s")
    print(json.dumps(payload, indent=2, sort_keys=True))


def scheduled_probe_handler(request):
    """Bearer-check, then call probe_h200 and return the structured outcome.

    Separated out from the decorated entry point so the bearer-check has
    direct access to ``request.headers`` (cleaner than FastAPI Header DI
    once we're inside a fastapi_endpoint wrapper).
    """
    from fastapi import HTTPException

    expected_secret = os.environ.get("PROBE_BEARER", "")
    if not expected_secret:
        raise HTTPException(
            status_code=500, detail="PROBE_BEARER not set in probe-bearer Modal secret"
        )
    auth_header = request.headers.get("authorization", "")
    # Constant-time compare so a per-byte timing side-channel can't leak the token.
    if not hmac.compare_digest(auth_header, f"Bearer {expected_secret}"):
        raise HTTPException(status_code=401, detail="unauthorized")

    t0 = time.time()
    try:
        output = probe_h200.remote()
        elapsed = time.time() - t0
        if isinstance(output, dict):
            gpu_type = output.get("gpu_type") or gpu_type_from_inventory(
                str(output.get("inventory") or "")
            )
            telemetry = {
                key: output.get(key)
                for key in (
                    "gpu_count",
                    "gpu_memory_total_mb",
                    "gpu_memory_total_std_mb",
                    "gpu_memory_used_mb",
                    "gpu_memory_used_std_mb",
                    "gpu_utilization_pct",
                    "gpu_utilization_std_pct",
                    "gpu_temperature_c",
                    "gpu_temperature_std_c",
                    "gpu_power_w",
                    "gpu_power_std_w",
                    "driver_version",
                )
            }
            cloud = output.get("cloud")
            region = output.get("region")
        else:
            gpu_type = gpu_type_from_inventory(str(output))
            telemetry = dict(TELEMETRY_EMPTY)
            cloud = None
            region = None
        return {
            "ok": True,
            "gpu_type": gpu_type,
            "elapsed_s": elapsed,
            "cloud": cloud,
            "region": region,
            **telemetry,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "ok": False,
            "gpu_type": None,
            "elapsed_s": elapsed,
            "cloud": None,
            "region": None,
            **TELEMETRY_EMPTY,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.function(
    image=image,
    secrets=[PROBE_BEARER_SECRET],
    timeout=600,
)
@modal.fastapi_endpoint(method="POST", label="probe")
def scheduled_probe(request: Request):
    return scheduled_probe_handler(request)
