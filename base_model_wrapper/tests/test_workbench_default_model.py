"""ACS-145: the workbench picker defaults to the largest always-on model.

A new chat must never pre-select an on-demand model (a tester could otherwise
trigger an expensive cold boot by accident — Theia's picker defaulted to 405B).
"""

from __future__ import annotations

from wrapper.routes import workbench as wb


def _am(model_id: str, always_on: bool) -> dict:
    return {"id": model_id, "startup_always_on": always_on}


def test_existing_chat_keeps_its_saved_model():
    models = [_am("llama-8b", True), _am("trinity-truebase", True)]
    # Even though 405b isn't always-on, an explicit prior choice is respected.
    assert wb._select_workbench_model_id("llama-405b", models, "trinity-truebase") == "llama-405b"


def test_new_chat_prefers_largest_always_on(monkeypatch):
    sizes = {"llama-8b": 1, "trinity-truebase": 8, "llama-405b": 8}
    monkeypatch.setattr(wb, "_model_total_gpus", lambda mid: sizes.get(mid, 0))
    models = [_am("llama-8b", True), _am("trinity-truebase", True), _am("llama-405b", False)]
    # 405b is on-demand → excluded; trinity (8 GPUs) beats llama-8b (1 GPU).
    assert wb._select_workbench_model_id(None, models, "llama-8b") == "trinity-truebase"


def test_new_chat_falls_back_to_configured_default_when_none_always_on():
    # dev-local single-model registry: nothing always-on, no spec.
    models = [_am("gpt2", False)]
    assert wb._select_workbench_model_id(None, models, "gpt2") == "gpt2"


def test_empty_chat_model_is_treated_as_new(monkeypatch):
    # A chat row with model=None (legacy/untouched) must still get the default.
    monkeypatch.setattr(wb, "_model_total_gpus", lambda mid: {"trinity-truebase": 8}.get(mid, 0))
    models = [_am("llama-8b", True), _am("trinity-truebase", True)]
    assert wb._select_workbench_model_id(None, models, "llama-8b") == "trinity-truebase"
