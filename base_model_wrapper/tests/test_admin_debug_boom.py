"""ACS-40: the admin /admin/debug/boom verify route — admin-gated, and its
raised error flows through the catch-all handler into a generic 500."""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from wrapper.dependencies import get_settings
from wrapper.routes.admin import debug as debug_routes
from wrapper.settings import Settings


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(debug_routes.router)
    stub = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="sekret",
    )
    app.dependency_overrides[get_settings] = lambda: stub

    # Mirror main.py's catch-all so we can assert the boom → generic-500 contract
    # without standing up the whole app + DB.
    @app.exception_handler(Exception)
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(getattr(request, "state", None), "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
        return JSONResponse(
            status_code=500,
            content={
                "error": {"message": "Internal server error.", "type": "internal_error", "code": "internal_error"},
                "request_id": rid,
            },
        )

    return app


def test_boom_requires_admin_token():
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.post("/admin/debug/boom")  # no X-Admin-Token
    assert r.status_code == 401
    assert "admin_auth" in r.text


def test_boom_triggers_generic_500_when_admin():
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.post("/admin/debug/boom", headers={"X-Admin-Token": "sekret"})
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "Internal server error."
    assert body["request_id"].startswith("req_")
    # The deliberate error text must not leak to the client.
    assert "deliberate test error" not in r.text
