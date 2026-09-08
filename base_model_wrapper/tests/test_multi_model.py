"""Tests for the multi-model registry + routing layer (Phase 5).

Two slices:
  - Pure unit tests for ``Settings.parsed_models_registry`` (no FastAPI app).
  - DB-gated end-to-end tests that the /v1/models route and the model-id
    validation behave correctly under a real app.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from wrapper.settings import Settings


# ---- unit: registry parsing ------------------------------------------------


def _base_env(**overrides) -> dict[str, str]:
    env = {
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "MODAL_BASE_URL": "https://upstream.example",
        "VLLM_API_KEY": "vllm-key",
        "ADMIN_TOKEN": "admin-tok",
        "SERVED_MODEL_NAME": "meta-llama/Llama-3.1-405B",
    }
    env.update(overrides)
    return env


def _settings_from(env: dict[str, str]) -> Settings:
    return Settings(**{k.lower(): v for k, v in env.items()})


def test_registry_falls_back_to_single_entry_when_unset():
    s = _settings_from(_base_env())
    reg = s.parsed_models_registry()
    assert list(reg) == ["meta-llama/Llama-3.1-405B"]
    entry = reg["meta-llama/Llama-3.1-405B"]
    assert entry.upstream_url == "https://upstream.example"
    assert entry.tokenizer_repo == "meta-llama/Llama-3.1-405B"
    assert entry.status == "live"


def test_registry_parses_multi_entry_json():
    payload = {
        "llama-1b": {
            "upstream_url": "https://small.example",
            "served_model_name": "meta-llama/Llama-3.2-1B",
            "tokenizer_repo": "meta-llama/Llama-3.2-1B",
            "gpu_shape_label": "1×H100",
            "status": "live",
        },
        "llama-405b": {
            "upstream_url": "https://big.example",
            "served_model_name": "meta-llama/Llama-3.1-405B",
            "tokenizer_repo": "meta-llama/Llama-3.1-405B",
            "gpu_shape_label": "8×H200",
            "status": "live",
        },
    }
    s = _settings_from(_base_env(MODELS_REGISTRY_JSON=json.dumps(payload)))
    reg = s.parsed_models_registry()
    assert sorted(reg) == ["llama-1b", "llama-405b"]
    assert reg["llama-1b"].upstream_url == "https://small.example"
    assert reg["llama-405b"].upstream_url == "https://big.example"


def test_registry_known_model_entry_can_inherit_shared_defaults():
    payload = {
        "llama-405b": {
            "upstream_url": "https://big.example",
        }
    }
    s = _settings_from(_base_env(MODELS_REGISTRY_JSON=json.dumps(payload)))
    reg = s.parsed_models_registry()
    entry = reg["llama-405b"]
    assert entry.upstream_url == "https://big.example"
    assert entry.served_model_name == "meta-llama/Llama-3.1-405B"
    assert entry.tokenizer_repo == "meta-llama/Llama-3.1-405B"
    assert entry.gpu_shape_label == "8xH200"
    assert entry.modal_app_name == "acs-llama-405b"
    assert entry.max_model_len == 32768


def test_registry_unknown_model_entry_still_requires_full_shape():
    payload = {
        "custom": {
            "upstream_url": "https://custom.example",
        }
    }
    s = _settings_from(_base_env(MODELS_REGISTRY_JSON=json.dumps(payload)))
    with pytest.raises(RuntimeError, match="missing required field"):
        s.parsed_models_registry()


def test_registry_rejects_malformed_json():
    s = _settings_from(_base_env(MODELS_REGISTRY_JSON="not-json"))
    with pytest.raises(RuntimeError, match="not valid JSON"):
        s.parsed_models_registry()


def test_registry_rejects_missing_required_field():
    payload = {
        "llama-1b": {
            "upstream_url": "https://x.example",
            # missing served_model_name + tokenizer_repo
        }
    }
    s = _settings_from(_base_env(MODELS_REGISTRY_JSON=json.dumps(payload)))
    with pytest.raises(RuntimeError, match="missing required field"):
        s.parsed_models_registry()


def test_resolve_default_picks_configured_id():
    payload = {
        "small": {
            "upstream_url": "https://a.example",
            "served_model_name": "a",
            "tokenizer_repo": "a",
        },
        "big": {
            "upstream_url": "https://b.example",
            "served_model_name": "b",
            "tokenizer_repo": "b",
        },
    }
    s = _settings_from(_base_env(MODELS_REGISTRY_JSON=json.dumps(payload), DEFAULT_MODEL_ID="big"))
    reg = s.parsed_models_registry()
    assert s.resolve_default_model_id(reg) == "big"


def test_resolve_default_falls_back_to_first_live():
    payload = {
        "first": {
            "upstream_url": "https://a.example",
            "served_model_name": "a",
            "tokenizer_repo": "a",
        },
        "second": {
            "upstream_url": "https://b.example",
            "served_model_name": "b",
            "tokenizer_repo": "b",
        },
    }
    s = _settings_from(_base_env(MODELS_REGISTRY_JSON=json.dumps(payload)))
    reg = s.parsed_models_registry()
    assert s.resolve_default_model_id(reg) == "first"


def test_resolve_default_skips_disabled():
    payload = {
        "disabled-one": {
            "upstream_url": "https://a.example",
            "served_model_name": "a",
            "tokenizer_repo": "a",
            "status": "disabled",
        },
        "live-one": {
            "upstream_url": "https://b.example",
            "served_model_name": "b",
            "tokenizer_repo": "b",
            "status": "live",
        },
    }
    s = _settings_from(_base_env(MODELS_REGISTRY_JSON=json.dumps(payload)))
    reg = s.parsed_models_registry()
    assert s.resolve_default_model_id(reg) == "live-one"


# ---- DB-gated end-to-end ----------------------------------------------------

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run multi-model end-to-end tests",
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-multi-model-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example")
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
            "medium": {
                "upstream_url": "https://medium.example",
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


@dbtest
def test_v1_models_returns_registry(client):
    from wrapper.keys import generate as generate_key
    from wrapper.models import ApiKey, User
    from wrapper.db import make_engine, make_session_factory, session_scope
    import asyncio

    async def _setup() -> str:
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                u = User(email=f"mm-{uuid.uuid4().hex[:6]}@example.local")
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

    plaintext = asyncio.run(_setup())
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200, r.text
    body = r.json()
    ids = sorted(m["id"] for m in body["data"])
    assert ids == ["medium", "small"]
