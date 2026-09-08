"""``GET /v1/models`` advertises per-model activation support (ACS-199 PR4).

A model with an ``activation_upstream_url`` reports ``activations: true`` plus the
activation caps; a model without one reports ``activations: false`` and omits the
caps. DB-gated like ``test_multi_model.py`` (needs a migrated Postgres via
``TEST_DATABASE_URL``).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run this test",
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-act-cap-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ["MODELS_REGISTRY_JSON"] = json.dumps(
        {
            "act": {
                "upstream_url": "https://act-workbench.example",
                "activation_upstream_url": "https://act-activation.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "test",
                "status": "live",
            },
            "plain": {
                "upstream_url": "https://plain-workbench.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "test",
                "status": "live",
            },
        }
    )
    os.environ["DEFAULT_MODEL_ID"] = "act"
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True
    os.environ.pop("MODELS_REGISTRY_JSON", None)
    os.environ.pop("DEFAULT_MODEL_ID", None)


def _make_key() -> str:
    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.keys import generate as generate_key
    from wrapper.models import ApiKey, User

    async def _setup() -> str:
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                u = User(email=f"cap-{uuid.uuid4().hex[:6]}@example.local")
                s.add(u)
                await s.flush()
                gk = generate_key()
                s.add(ApiKey(user_id=u.id, key_hash=gk.hash_, key_prefix=gk.prefix))
                return gk.plaintext
        finally:
            await engine.dispose()

    return asyncio.run(_setup())


@dbtest
def test_v1_models_advertises_activation_support(client):
    from wrapper.schemas import DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS, MAX_STEERING_VECTORS

    key = _make_key()
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200, r.text
    caps = {m["id"]: m["capabilities"] for m in r.json()["data"]}

    # Activation-capable model advertises the flag + caps.
    assert caps["act"]["activations"] is True
    assert caps["act"]["max_steering_vectors"] == MAX_STEERING_VECTORS
    # "act" doesn't set activation_max_prompt_tokens → the wrapper default.
    assert caps["act"]["max_activation_prompt_tokens"] == DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS

    # Plain model: flag false, caps omitted (not advertised).
    assert caps["plain"]["activations"] is False
    assert "max_activation_prompt_tokens" not in caps["plain"]
    assert "max_steering_vectors" not in caps["plain"]
