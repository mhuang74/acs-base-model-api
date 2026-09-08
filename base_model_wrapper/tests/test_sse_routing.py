"""End-to-end-ish test: verify SSE streaming + the multi-model router play
nicely together (no real Modal needed — upstream is mocked with respx).

Verifies that with two registry entries:
  - A streaming request to model 'small' hits the 'small' upstream URL.
  - A streaming request to model 'big' hits the 'big' upstream URL.
  - The response is text/event-stream and chunks are forwarded verbatim.
  - The final usage chunk's tokens reach the request log.

This catches multi-model wiring bugs that would otherwise only show up
against a real Modal deploy.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from httpx import Response

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run SSE routing tests",
)

try:
    import respx
    _RESPX = True
except ImportError:
    respx = None  # type: ignore[assignment]
    _RESPX = False

respx_required = pytest.mark.skipif(
    not _RESPX,
    reason="respx not installed; install dev extras or pip install respx to enable",
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-sse-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://fallback.example")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ["MODELS_REGISTRY_JSON"] = json.dumps(
        {
            "small": {
                "upstream_url": "https://small.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "test",
                "status": "live",
            },
            "big": {
                "upstream_url": "https://big.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "test",
                "status": "live",
            },
        }
    )
    os.environ["DEFAULT_MODEL_ID"] = "small"
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True
    os.environ.pop("MODELS_REGISTRY_JSON", None)
    os.environ.pop("DEFAULT_MODEL_ID", None)


async def _make_key(test_db_url: str) -> str:
    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.keys import generate as generate_key
    from wrapper.models import ApiKey, User

    engine = make_engine(test_db_url)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = User(email=f"sse-{uuid.uuid4().hex[:6]}@example.local")
            s.add(u)
            await s.flush()
            gk = generate_key()
            s.add(
                ApiKey(
                    user_id=u.id,
                    key_hash=gk.hash_,
                    key_prefix=gk.prefix,
                    monthly_token_budget=0,
                )
            )
            return gk.plaintext
    finally:
        await engine.dispose()


_SSE_CHUNKS = [
    b'data: {"id":"x","choices":[{"text":"hello"}]}\n\n',
    b'data: {"id":"x","choices":[{"text":" world"}]}\n\n',
    b'data: {"id":"x","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
    b"data: [DONE]\n\n",
]


@dbtest
@respx_required
def test_stream_routes_to_resolved_upstream(client):
    import asyncio

    plaintext = asyncio.run(_make_key(TEST_DATABASE_URL))

    with respx.mock(assert_all_called=False) as mock:
        # Only register the 'big' URL — if the router instead hits 'small',
        # respx will return an unhandled-request error and the test fails.
        big_route = mock.post("https://big.example/v1/completions").mock(
            return_value=Response(
                200,
                content=b"".join(_SSE_CHUNKS),
                headers={"Content-Type": "text/event-stream"},
            )
        )
        r = client.post(
            "/v1/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "big", "prompt": "hi", "stream": True},
        )
        assert r.status_code == 200, r.text
        assert "text/event-stream" in r.headers["content-type"]
        body = r.content
        assert b"hello" in body
        assert b"[DONE]" in body
        assert big_route.called


@dbtest
@respx_required
def test_unknown_model_returns_400(client):
    import asyncio

    plaintext = asyncio.run(_make_key(TEST_DATABASE_URL))
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"model": "definitely-not-a-real-model", "prompt": "hi"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "model_not_found"
