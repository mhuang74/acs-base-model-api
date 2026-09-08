"""Guards against drift between Modal serving config and wrapper registry fields."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from wrapper import settings as settings_mod
from wrapper.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
MODAL_CONFIG_PATH = REPO_ROOT / "serving" / "modal_config.py"
SHARED_REGISTRY_PATH = REPO_ROOT / "acs_model_registry" / "__init__.py"

EXPECTED_WRAPPER_MODAL_IDENTITIES = {
    "llama-8b": {
        "modal_app_name": "acs-llama-8b",
        "served_model_name": "meta-llama/Llama-3.1-8B",
    },
    "llama-405b": {
        "modal_app_name": "acs-llama-405b",
        "served_model_name": "meta-llama/Llama-3.1-405B",
    },
    "trinity-truebase": {
        "modal_app_name": "acs-trinity-base",
        "served_model_name": "arcee-ai/Trinity-Large-TrueBase",
    },
    "kimi-k2-base": {
        "modal_app_name": "acs-kimi-k2-base",
        "served_model_name": "moonshotai/Kimi-K2-Base",
    },
}


def _load_modal_config():
    spec = importlib.util.spec_from_file_location("modal_config_under_test", MODAL_CONFIG_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_shared_registry():
    spec = importlib.util.spec_from_file_location(
        "acs_model_registry_under_test", SHARED_REGISTRY_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["acs_model_registry_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_modal_config_matches_wrapper_registry_identity_expectations():
    modal_config = _load_modal_config()
    shared_registry = _load_shared_registry()

    # Only LIVE models are wrapper-facing. Staging siblings (-v023, -snapprod)
    # live in the registry for isolated deploys but never appear in the wrapper's
    # live listing (GET /v1/models over status=="live"), so exclude them here.
    live_ids = {
        model_id
        for model_id, spec in shared_registry.SPECS.items()
        if spec.status == "live"
    }
    actual = {
        model_id: {
            "modal_app_name": cfg["app_name"],
            "served_model_name": cfg["served_model_name"],
        }
        for model_id, cfg in modal_config.MODELS.items()
        if model_id in live_ids
    }
    assert actual == EXPECTED_WRAPPER_MODAL_IDENTITIES
    assert actual == {
        model_id: {
            "modal_app_name": cfg["app_name"],
            "served_model_name": cfg["served_model_name"],
        }
        for model_id, cfg in shared_registry.MODELS.items()
        if model_id in live_ids
    }

    registry_payload = {
        model_id: {
            "upstream_url": f"https://example.invalid/{expected['modal_app_name']}/serve",
            "served_model_name": expected["served_model_name"],
            "tokenizer_repo": expected["served_model_name"],
            "gpu_shape_label": "test",
            "modal_app_name": expected["modal_app_name"],
        }
        for model_id, expected in EXPECTED_WRAPPER_MODAL_IDENTITIES.items()
    }
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        modal_base_url="https://legacy.example.invalid",
        vllm_api_key="test-vllm-key",
        admin_token="test-admin-token",
        models_registry_json=json.dumps(registry_payload),
    )
    registry = settings.parsed_models_registry()

    assert sorted(registry) == sorted(EXPECTED_WRAPPER_MODAL_IDENTITIES)
    for model_id, expected in EXPECTED_WRAPPER_MODAL_IDENTITIES.items():
        entry = registry[model_id]
        assert entry.model_id == model_id
        assert entry.modal_app_name == expected["modal_app_name"]
        assert entry.served_model_name == expected["served_model_name"]


def test_wrapper_registry_can_inherit_known_model_defaults():
    payload = {
        "llama-8b": {
            "upstream_url": "https://example.invalid/acs-llama-8b/serve",
        },
        "trinity-truebase": {
            "upstream_url": "https://example.invalid/acs-trinity-base/serve",
            "status": "staging",
        },
    }
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        modal_base_url="https://legacy.example.invalid",
        vllm_api_key="test-vllm-key",
        admin_token="test-admin-token",
        models_registry_json=json.dumps(payload),
    )
    registry = settings.parsed_models_registry()

    assert registry["llama-8b"].served_model_name == "meta-llama/Llama-3.1-8B"
    assert registry["llama-8b"].tokenizer_repo == "meta-llama/Llama-3.1-8B"
    assert registry["llama-8b"].modal_app_name == "acs-llama-8b"
    assert registry["llama-8b"].max_model_len == 8192
    # ACS-226: per-model scaledown windows inherited from the shared registry so
    # the wrapper's cold-hint matches the real Modal teardown per breaker key.
    assert registry["llama-8b"].scaledown_window_s == 30 * 60
    assert registry["llama-8b"].activation_scaledown_window_s == 10 * 60
    # ACS-250: 8B capture-prompt cap raised to 256 (streamed response ~68 MB) now
    # that streaming removed the wrapper-RAM wall; big models inherit None → the
    # wrapper's conservative DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS (405B is ~4 MB/tok).
    assert registry["llama-8b"].activation_max_prompt_tokens == 256
    assert registry["trinity-truebase"].modal_app_name == "acs-trinity-base"
    assert registry["trinity-truebase"].status == "staging"
    assert registry["trinity-truebase"].activation_scaledown_window_s == 10 * 60
    assert registry["trinity-truebase"].activation_max_prompt_tokens is None


def test_wrapper_registry_rejects_explicit_null_window():
    """An explicit ``null`` window must raise, not silently fall to the default.

    ACS-226: ``{**defaults, **raw_entry}`` means a `"scaledown_window_s": null`
    in the JSON would otherwise override the registry-inherited value with the
    hardcoded default — judged warm long past the real teardown. A window is
    never legitimately "unset"; omitting the key inherits the default instead.
    """
    payload = {
        "llama-8b": {
            "upstream_url": "https://example.invalid/acs-llama-8b/serve",
            "scaledown_window_s": None,
        },
    }
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        modal_base_url="https://legacy.example.invalid",
        vllm_api_key="test-vllm-key",
        admin_token="test-admin-token",
        models_registry_json=json.dumps(payload),
    )
    with pytest.raises(RuntimeError, match="scaledown_window_s=null"):
        settings.parsed_models_registry()


def test_wrapper_default_windows_match_shared_registry():
    """The wrapper's local default-window constants mirror the shared registry.

    ``settings.py`` keeps local copies of the two scaledown-window defaults
    (the registry package isn't reliably importable at its module scope), so
    lock them to the registry here — otherwise a change to
    ``DEFAULT_SCALEDOWN_WINDOW_S`` / ``DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S`` in
    the registry would silently drift from the wrapper's fallback (ACS-226).
    """
    shared_registry = _load_shared_registry()
    assert (
        settings_mod.DEFAULT_SCALEDOWN_WINDOW_S
        == shared_registry.DEFAULT_SCALEDOWN_WINDOW_S
    )
    assert (
        settings_mod.DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S
        == shared_registry.DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S
    )


def test_llama_autoscaler_scaledown_contracts_are_explicit():
    modal_config = _load_modal_config()

    llama_8b = modal_config.MODELS["llama-8b"]
    # ACS-223: back on an always-on warm container (min=1), reversing the
    # ACS-17/#193 scale-to-zero — the snapshot restore is still too disruptive
    # for interactive workbench use. snapshot=True still routes it to the
    # class-based serve path.
    assert llama_8b["min_containers"] == 1
    assert llama_8b["snapshot"] is True
    assert llama_8b["scaledown_window_s"] == 30 * 60

    llama_405b = modal_config.MODELS["llama-405b"]
    assert llama_405b["min_containers"] == 0
    # ACS-315 (Modal cost review): serve idle window tightened 30 min → 5 min so a
    # scale-to-zero 8×H200 stops billing 25 min sooner after a spiky burst.
    assert llama_405b["scaledown_window_s"] == 5 * 60

    # Trinity serving back to always-on (operator decision 2026-07-13): a warm
    # 8×H200 avoids the ~150-180s cold boot. Its ACTIVATION engine stays
    # scale-to-zero (asserted in the activation-contract test below). ACS-315
    # tightened its serve idle window to 5 min (governs the burst 2nd container
    # today; becomes the full window if min_containers is ever dropped to 0).
    assert modal_config.MODELS["trinity-truebase"]["min_containers"] == 1
    assert modal_config.MODELS["trinity-truebase"]["scaledown_window_s"] == 5 * 60


def test_activation_engine_scaledown_and_container_contracts():
    """Per-model activation-engine runtime (modal_app_activation.py reads these).

    Cheap 1×L40S llama-8b scales to zero with burst headroom to 5; the 8×H200
    models stay tight (10-min window, single container, scale-to-zero) for cost.
    """
    modal_config = _load_modal_config()

    llama_8b = modal_config.MODELS["llama-8b"]
    # 10-min warm tail governs the scale-up containers (2..5); the min=1 floor
    # never scales below one warm.
    assert llama_8b["activation_scaledown_window_s"] == 10 * 60
    # Interactive burst headroom: 80 concurrent captures (5×16); only the 4 extra
    # containers bill while scaled (ACS-249).
    assert llama_8b["activation_max_containers"] == 5
    # Per-container concurrency, load-tested to 16 (ACS-249, 2026-07-20: no OOM
    # through 32 concurrent, p50 latency knee past 24). Big models keep default 8.
    assert llama_8b["activation_max_inputs"] == 16
    # One always-on warm L40S (≈ $1.4k/mo) so the first capture/steer is instant —
    # re-armed 2026-07-21 for daily interactive research, reversing the ACS-248
    # scale-to-zero that followed the ACS-214 workshop.
    assert llama_8b["activation_min_containers"] == 1

    for big in ("llama-405b", "trinity-truebase"):
        entry = modal_config.MODELS[big]
        assert entry["activation_scaledown_window_s"] == 10 * 60, big
        assert entry["activation_max_containers"] == 1, big
        assert entry["activation_max_inputs"] == 8, big
        # 8×H200 stays scale-to-zero — too expensive to keep warm.
        assert entry["activation_min_containers"] == 0, big
