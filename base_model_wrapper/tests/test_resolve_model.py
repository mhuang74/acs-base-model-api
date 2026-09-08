"""Unit tests for ``_resolve_model`` (strict short-id lookup) and the
'did you mean' hint in ``_unknown_model_response`` when a client pastes
the upstream HF served name instead of the canonical short id."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")

from wrapper.main import _resolve_model, _unknown_model_response
from wrapper.settings import ModelEntry


def _entry(model_id: str, served: str, *, status: str = "live") -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        upstream_url=f"https://{model_id}.example",
        served_model_name=served,
        tokenizer_repo=served,
        gpu_shape_label="test",
        status=status,
    )


def _request(registry: dict, *, default: str = "") -> SimpleNamespace:
    state = SimpleNamespace(models=registry, default_model_id=default)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_resolve_by_short_id():
    registry = {"llama-8b": _entry("llama-8b", "meta-llama/Llama-3.1-8B")}
    result = _resolve_model(_request(registry), "llama-8b")
    assert result is not None
    model_id, entry = result
    assert model_id == "llama-8b"


def test_resolve_default_when_blank():
    registry = {"llama-8b": _entry("llama-8b", "meta-llama/Llama-3.1-8B")}
    result = _resolve_model(_request(registry, default="llama-8b"), None)
    assert result is not None
    assert result[0] == "llama-8b"


def test_resolve_rejects_hf_served_name():
    """HF served name is upstream-only — the public API is the short id."""
    registry = {"llama-8b": _entry("llama-8b", "meta-llama/Llama-3.1-8B")}
    assert _resolve_model(_request(registry), "meta-llama/Llama-3.1-8B") is None


def test_resolve_unknown_returns_none():
    registry = {"llama-8b": _entry("llama-8b", "meta-llama/Llama-3.1-8B")}
    assert _resolve_model(_request(registry), "does-not-exist") is None


def test_resolve_disabled_returns_none():
    registry = {"llama-8b": _entry("llama-8b", "meta-llama/Llama-3.1-8B", status="disabled")}
    assert _resolve_model(_request(registry), "llama-8b") is None


def _body(response) -> dict:
    return json.loads(bytes(response.body))


def test_unknown_response_lists_available_ids():
    registry = {
        "llama-8b": _entry("llama-8b", "meta-llama/Llama-3.1-8B"),
        "trinity-truebase": _entry("trinity-truebase", "arcee-ai/Trinity-Large-TrueBase"),
    }
    response = _unknown_model_response(_request(registry), "definitely-not-a-model")
    assert response.status_code == 400
    body = _body(response)
    assert body["error"]["code"] == "model_not_found"
    msg = body["error"]["message"]
    assert "llama-8b" in msg and "trinity-truebase" in msg
    assert "Did you mean" not in msg  # no HF match, no hint


def test_unknown_response_suggests_short_id_for_hf_name():
    """Pasting the HF repo id should surface a 'did you mean' hint."""
    registry = {
        "llama-8b": _entry("llama-8b", "meta-llama/Llama-3.1-8B"),
        "trinity-truebase": _entry("trinity-truebase", "arcee-ai/Trinity-Large-TrueBase"),
    }
    response = _unknown_model_response(_request(registry), "arcee-ai/Trinity-Large-TrueBase")
    assert response.status_code == 400
    msg = _body(response)["error"]["message"]
    assert "Did you mean 'trinity-truebase'?" in msg


def test_unknown_response_blank_request():
    registry = {"llama-8b": _entry("llama-8b", "meta-llama/Llama-3.1-8B")}
    response = _unknown_model_response(_request(registry), None)
    assert response.status_code == 400
    msg = _body(response)["error"]["message"]
    assert "No model specified" in msg
