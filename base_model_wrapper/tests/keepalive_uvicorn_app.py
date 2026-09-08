from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")

from fastapi import FastAPI, Request

from wrapper import proxy as proxymod
from wrapper.routes import api as api_routes


class _Session:
    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


async def _slow_completion(**_kwargs):
    await asyncio.sleep(0.15)
    return 200, {"choices": [{"text": "Paris"}], "usage": {}}, {}


api_routes.run_completion_nonstream = _slow_completion
api_routes.PUBLIC_RESPONSE_GRACE_S = 0.01
api_routes.PUBLIC_KEEPALIVE_INTERVAL_S = 0.03

app = FastAPI()


@app.get("/ready")
async def ready():
    return {"ok": True}


@app.get("/probe")
async def probe(request: Request):
    return await api_routes._serve_nonstream_with_keepalive(
        request_id="req_uvicorn",
        caller=SimpleNamespace(key_id="key"),
        ip=None,
        upstream_url="https://up/v1/completions",
        body={"prompt": "x"},
        settings=SimpleNamespace(vllm_api_key="key", upstream_timeout_s=1200.0),
        session=_Session(),
        http=SimpleNamespace(),
        model_id="llama-405b",
        request=request,
        key_semaphore=asyncio.Semaphore(0),
        backend_ctx=proxymod.BackendContext(model_id="llama-405b", cold_hint=True),
        breakers=SimpleNamespace(),
        effective_timeout=1200.0,
        tokenizer_repo="gpt2",
    )
