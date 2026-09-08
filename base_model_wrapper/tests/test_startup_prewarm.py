"""Verify the lifespan pre-warm hook calls modal_ops.prewarm_caches.

The lifespan in ``wrapper.main`` is supposed to schedule
``modal_ops.prewarm_caches(app_names)`` after the scheduler block, so the
first workbench render doesn't pay the cold-cache cost. These tests pin:

1. ``prewarm_caches`` is invoked exactly once with the registry's
   ``modal_app_name`` values (entries without one are filtered out).
2. If ``prewarm_caches`` raises, lifespan completes without crashing —
   startup never blocks on this hook.

Both tests run lifespan in-process with ``disable_scheduler = True`` so we
don't touch APScheduler or Postgres.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

import pytest
from fastapi import FastAPI

from wrapper import main as main_module
from wrapper import modal_ops as modalops


def _registry_json() -> str:
    """Two entries with modal_app_name (in insertion order) + one without."""
    return json.dumps(
        {
            "trinity-truebase": {
                "upstream_url": "https://trinity.example",
                "served_model_name": "trinity",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "8xH200",
                "modal_app_name": "acs-trinity-base",
            },
            "compass-base": {
                "upstream_url": "https://compass.example",
                "served_model_name": "compass",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "8xH100",
                "modal_app_name": "acs-compass-base",
            },
            "legacy-no-modal": {
                "upstream_url": "https://legacy.example",
                "served_model_name": "legacy",
                "tokenizer_repo": "gpt2",
            },
        }
    )


@pytest.fixture
def registry_env(monkeypatch):
    monkeypatch.setenv("MODELS_REGISTRY_JSON", _registry_json())
    # DEFAULT_MODEL_ID has to resolve to a live entry in the registry.
    monkeypatch.setenv("DEFAULT_MODEL_ID", "trinity-truebase")


async def test_lifespan_invokes_prewarm_caches_with_modal_app_names(
    monkeypatch, registry_env
):
    """prewarm_caches is called once with the registry's modal_app_name values
    (in insertion order), with entries lacking modal_app_name filtered out.
    """
    calls: list[list[str]] = []

    def fake_prewarm(app_names):
        calls.append(list(app_names))

    monkeypatch.setattr(modalops, "prewarm_caches", fake_prewarm, raising=False)

    app = FastAPI()
    app.state.disable_scheduler = True
    async with main_module.lifespan(app):
        pass

    assert calls == [["acs-trinity-base", "acs-compass-base"]]


async def test_lifespan_survives_prewarm_caches_raising(monkeypatch, registry_env):
    """If prewarm_caches raises, lifespan still completes (the wrapper boots
    even if the helper isn't present / blows up).
    """

    def raising_prewarm(app_names):
        raise RuntimeError("modal_ops not ready")

    monkeypatch.setattr(modalops, "prewarm_caches", raising_prewarm, raising=False)

    app = FastAPI()
    app.state.disable_scheduler = True
    # Should not raise.
    async with main_module.lifespan(app):
        pass


async def test_lifespan_survives_prewarm_caches_missing(monkeypatch):
    """If modal_ops has no ``prewarm_caches`` attribute at all, the try/except
    guard in lifespan must still let startup complete. Covers the case where
    Agent A's refactor renames the helper.
    """
    monkeypatch.setenv("MODELS_REGISTRY_JSON", _registry_json())
    monkeypatch.setenv("DEFAULT_MODEL_ID", "trinity-truebase")
    monkeypatch.delattr(modalops, "prewarm_caches", raising=False)

    app = FastAPI()
    app.state.disable_scheduler = True
    async with main_module.lifespan(app):
        pass
