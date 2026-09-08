"""Unit tests for the static workbench model-status label (ACS-80).

The label is the *simple* version chosen for beta — sourced from each model's
keep-warm config (registry ``min_containers`` + enabled ``ModelWarmWindow``
rows), not the live warm/cold signal that ACS-62 tracks separately. These
tests pin the policy decision so a future change can't silently regress the
label users see in the workbench dropdown.
"""

from __future__ import annotations

import datetime as dt
import os

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

from wrapper.models import ModelWarmWindow
from wrapper.services import workbench as workbench_svc
from wrapper.settings import ModelEntry


def _entry(model_id: str, modal_app_name: str | None = None) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        upstream_url=f"https://{model_id}.example",
        served_model_name=model_id,
        tokenizer_repo="gpt2",
        gpu_shape_label="—",
        modal_app_name=modal_app_name,
    )


def _warm_window(model_id: str, enabled: bool) -> ModelWarmWindow:
    row = ModelWarmWindow(
        model_id=model_id,
        warm_cron="0 8 * * *",
        cool_cron="0 18 * * *",
        timezone="UTC",
        enabled=enabled,
    )
    return row


def test_known_always_on_small_model_is_always_on():
    """Llama-8B is kept always-on for the beta (min_containers=1 in the shared
    registry, ACS-81) → "Always on" label."""
    registry = {"llama-8b": _entry("llama-8b", modal_app_name="acs-llama-8b")}
    labels = workbench_svc.compute_model_startup_labels(registry, [])
    row = labels["llama-8b"]
    assert row["always_on"] is True
    assert row["label"] == "Always on"


def test_known_big_model_with_min_containers_zero_is_needs_startup_big():
    """405B (n_gpu=8) → "Needs startup" with the big-model estimate (~2-10 min)."""
    registry = {"llama-405b": _entry("llama-405b", modal_app_name="acs-llama-405b")}
    labels = workbench_svc.compute_model_startup_labels(registry, [])
    row = labels["llama-405b"]
    assert row["always_on"] is False
    assert row["label"].startswith("Needs startup")
    assert "min" in row["label"]


def test_enabled_warm_window_promotes_model_to_always_on():
    """The static label respects admin warm-window overrides: an enabled row
    means the model is being kept warm on a schedule, so the label flips to
    "Always on" (treated as 24/7 for the beta label even if the cron only
    covers a window — refine if confusing for users)."""
    registry = {"llama-405b": _entry("llama-405b", modal_app_name="acs-llama-405b")}
    labels = workbench_svc.compute_model_startup_labels(
        registry, [_warm_window("llama-405b", enabled=True)]
    )
    assert labels["llama-405b"]["always_on"] is True
    assert labels["llama-405b"]["label"] == "Always on"


def test_disabled_warm_window_does_not_promote():
    """A saved-but-disabled warm window must NOT count as keep-warm. Uses an
    on-demand model (405b, min_containers=0) so the warm window is the only
    possible promotion path."""
    registry = {"llama-405b": _entry("llama-405b", modal_app_name="acs-llama-405b")}
    labels = workbench_svc.compute_model_startup_labels(
        registry, [_warm_window("llama-405b", enabled=False)]
    )
    assert labels["llama-405b"]["always_on"] is False


def test_unknown_model_falls_back_to_needs_startup_small():
    """A model not in the shared registry (e.g. the dev-local single-model
    fallback using ``gpt2``) defaults to "Needs startup" rather than claiming
    always-on. Under-promise, never over-promise."""
    registry = {"gpt2": _entry("gpt2")}
    labels = workbench_svc.compute_model_startup_labels(registry, [])
    assert labels["gpt2"]["always_on"] is False
    assert "Needs startup" in labels["gpt2"]["label"]
    # Unknown → small estimate (under-promise); preserves the ~30s coverage that
    # came from llama-8b before it became always-on (ACS-81).
    assert "30s" in labels["gpt2"]["label"]


_NOW = dt.datetime(2026, 6, 24, 12, 0, tzinfo=dt.UTC)

# Derive the warm-hint window from the model's ACTUAL scale-down window rather
# than hard-coding minutes — the label logic reads `spec.scaledown_window_s`
# (workbench.compute_model_startup_labels), so a registry change (e.g. ACS-315
# tightened 405b 30 min -> 5 min) must not silently break these tests.
_W405B_WINDOW_S = workbench_svc.model_spec_for("llama-405b").scaledown_window_s


def test_recent_completion_shows_usually_warm_hint():
    """ACS-98: an on-demand model used within its scale-down window gets the
    soft "recently used - usually warm" hint — sourced from the in-memory
    last_completion_at map, no Modal RPC. Uses half the model's real window so
    it stays inside it regardless of the configured value."""
    registry = {"llama-405b": _entry("llama-405b", modal_app_name="acs-llama-405b")}
    labels = workbench_svc.compute_model_startup_labels(
        registry,
        [],
        last_completion_at={"llama-405b": _NOW - dt.timedelta(seconds=_W405B_WINDOW_S // 2)},
        now=_NOW,
    )
    row = labels["llama-405b"]
    assert row["always_on"] is False
    assert row["warm_hint"] is True
    assert "usually warm" in row["label"].lower()


def test_stale_completion_past_window_is_needs_startup():
    """Used longer ago than the scale-down window → back to "Needs startup"."""
    registry = {"llama-405b": _entry("llama-405b", modal_app_name="acs-llama-405b")}
    labels = workbench_svc.compute_model_startup_labels(
        registry,
        [],
        last_completion_at={"llama-405b": _NOW - dt.timedelta(seconds=_W405B_WINDOW_S * 2)},
        now=_NOW,
    )
    row = labels["llama-405b"]
    assert row["warm_hint"] is False
    assert row["label"].startswith("Needs startup")


def test_no_completion_is_needs_startup():
    """No completion recorded (default empty map) → no warm hint."""
    registry = {"llama-405b": _entry("llama-405b", modal_app_name="acs-llama-405b")}
    labels = workbench_svc.compute_model_startup_labels(registry, [])
    assert labels["llama-405b"]["warm_hint"] is False
    assert labels["llama-405b"]["label"].startswith("Needs startup")


def test_always_on_model_ignores_recent_completion():
    """A recent completion must NOT downgrade an always-on model to the soft
    hint — it stays "Always on"."""
    registry = {"llama-8b": _entry("llama-8b", modal_app_name="acs-llama-8b")}
    labels = workbench_svc.compute_model_startup_labels(
        registry,
        [],
        last_completion_at={"llama-8b": _NOW - dt.timedelta(minutes=1)},
        now=_NOW,
    )
    row = labels["llama-8b"]
    assert row["always_on"] is True
    assert row["warm_hint"] is False
    assert row["label"] == "Always on"


def test_every_registered_model_gets_a_label():
    """The renderer indexes the result by model_id without a defensive check
    for non-live entries; the helper must return one row per registry entry."""
    registry = {
        "llama-8b": _entry("llama-8b", modal_app_name="acs-llama-8b"),
        "llama-405b": _entry("llama-405b", modal_app_name="acs-llama-405b"),
        "trinity-truebase": _entry("trinity-truebase", modal_app_name="acs-trinity-base"),
        "unknown-id": _entry("unknown-id"),
    }
    labels = workbench_svc.compute_model_startup_labels(registry, [])
    assert set(labels.keys()) == set(registry.keys())
    for row in labels.values():
        assert row["label"]  # non-empty
        assert row["tooltip"]
        assert isinstance(row["always_on"], bool)
