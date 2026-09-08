"""Loud 400 for /v1/chat/completions, no DB or upstream needed.

Uses FastAPI's TestClient against just the route — the endpoint doesn't touch
the DB or upstream Modal so we can drive it with a stub Settings.
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("HF_TOKEN", "")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from wrapper.main import chat_completions_unsupported
from wrapper.settings import Settings


def _app() -> FastAPI:
    app = FastAPI()
    app.state.settings = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
    )
    app.post("/v1/chat/completions")(chat_completions_unsupported)
    return app


def test_chat_completions_returns_loud_400():
    client = TestClient(_app())
    r = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "chat_completions_unsupported"
    assert "/v1/completions" in body["error"]["message"]
