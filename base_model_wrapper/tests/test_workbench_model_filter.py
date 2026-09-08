"""Unit tests for the workbench model-list filter + SSE error event.

PR 3 changes that survived the rebase onto the earlier upstream change:
- _resolve_workbench_model_states: parallel Modal state lookup shared by
  _render_chat and /workbench/models/status. Replaces the per-page serial
  for-loop that paid N× cost on a slow Modal day.
- Dropdown filter: models in confirmed "stopped" state are omitted from
  available_models / /workbench/models/status. Unknown / missing
  modal_app_name / RPC failure all keep the model visible (so Modal
  control-plane hiccups can't empty the dropdown).
- _friendly_upstream_error: module-level helper used by chat_stream as the
  fallback message when the upstream body isn't OpenAI-shaped JSON. Maps 404
  (most common: model app stopped) to "contact infra@…".

Note: the SSE event-frame plumbing (_sse_error definition + emission in
chat_stream, JS handler in workbench.html) was already shipped by that
earlier change; we only kept _friendly_upstream_error and rewired the
existing fallback to use it.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

from wrapper import main as main_module
from wrapper.settings import ModelEntry


def _registry() -> dict[str, ModelEntry]:
    """Three live models + one staging entry to exercise the filter.

    deployed/stopped/initializing covers the three "what the dropdown does
    with each modal state" cases; the staging entry is filtered earlier by
    ``m.status != "live"``.
    """
    return {
        "alive-deployed": ModelEntry(
            model_id="alive-deployed",
            upstream_url="https://a.example",
            served_model_name="gpt2",
            tokenizer_repo="gpt2",
            gpu_shape_label="1×H100",
            modal_app_name="acs-alive",
        ),
        "stopped-one": ModelEntry(
            model_id="stopped-one",
            upstream_url="https://b.example",
            served_model_name="big-model",
            tokenizer_repo="gpt2",
            gpu_shape_label="8×H200",
            modal_app_name="acs-stopped",
        ),
        "warming-up": ModelEntry(
            model_id="warming-up",
            upstream_url="https://c.example",
            served_model_name="warm-model",
            tokenizer_repo="gpt2",
            gpu_shape_label="8×H200",
            modal_app_name="acs-warming",
        ),
        "no-modal-name": ModelEntry(
            model_id="no-modal-name",
            upstream_url="https://d.example",
            served_model_name="local",
            tokenizer_repo="gpt2",
            gpu_shape_label="—",
            modal_app_name=None,
        ),
        "still-cooking": ModelEntry(
            model_id="still-cooking",
            upstream_url="https://e.example",
            served_model_name="future",
            tokenizer_repo="gpt2",
            gpu_shape_label="?",
            status="staging",
            modal_app_name="acs-future",
        ),
    }


def test_llama_405b_is_workbench_visible_by_default():
    assert "llama-405b" not in main_module._UI_HIDDEN_MODEL_IDS


async def _stub_runners_zero(_name):
    """Default stub for modalops.get_active_runner_count — keeps warm checks
    silent during dropdown-filter tests so we exercise only the filter path."""
    return 0


async def test_resolve_workbench_model_states_parallel(monkeypatch):
    """All Modal lookups must be in flight concurrently — verified by counting
    concurrent calls inside the patched get_app_state. The /admin equivalent
    has shipped parallel since day 1; the workbench used to be serial."""
    import asyncio as _asyncio

    in_flight = 0
    max_in_flight = 0

    async def fake_get_state(name):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        # Yield so other coroutines can join the in-flight count.
        await _asyncio.sleep(0.005)
        in_flight -= 1
        return {
            "acs-alive": "deployed",
            "acs-stopped": "stopped",
            "acs-warming": "initializing",
        }.get(name, "unknown")

    monkeypatch.setattr(main_module.modalops, "get_app_state", fake_get_state)
    states = await main_module._resolve_workbench_model_states(_registry())

    # 3 live models with modal_app_name → 3 concurrent RPCs.
    assert max_in_flight == 3
    assert states == {
        "alive-deployed": "deployed",
        "stopped-one": "stopped",
        "warming-up": "initializing",
    }


async def test_resolve_workbench_model_states_skips_rpc_failure(monkeypatch):
    """An RPC failure must be logged and omitted from the result map — callers
    treat absence as "no veto", which keeps the model visible during outages."""

    async def fake_get_state(name):
        if name == "acs-stopped":
            raise RuntimeError("simulated modal outage")
        return "deployed"

    monkeypatch.setattr(main_module.modalops, "get_app_state", fake_get_state)
    states = await main_module._resolve_workbench_model_states(_registry())
    assert "stopped-one" not in states
    assert states.get("alive-deployed") == "deployed"


async def test_workbench_models_status_filters_ui_hidden_models(monkeypatch):
    """/workbench/models/status returns only models that should be visible in
    the workbench dropdown. UI_HIDDEN_MODEL_IDS entries are excluded when the
    set is configured; staging models are excluded; everything else is included.
    No Modal RPCs — the endpoint is fully decoupled from Modal latency."""

    def boom_state(_name):
        raise AssertionError(
            "workbench_models_status must not call get_app_state — the "
            "endpoint is decoupled from Modal control plane"
        )

    def boom_runners(_name):
        raise AssertionError(
            "workbench_models_status must not call get_active_runner_count"
        )

    monkeypatch.setattr(main_module.modalops, "get_app_state", boom_state)
    monkeypatch.setattr(
        main_module.modalops, "get_active_runner_count", boom_runners
    )
    # Force one of our test models to be in the hidden set so we can assert
    # it's filtered. _UI_HIDDEN_MODEL_IDS is a frozenset so we monkeypatch
    # the whole binding rather than mutate.
    monkeypatch.setattr(
        main_module, "_UI_HIDDEN_MODEL_IDS", frozenset({"stopped-one"})
    )

    from fastapi import FastAPI
    app = FastAPI()
    app.state.models = _registry()
    app.state.last_completion_at = {}

    class _StubRequest:
        def __init__(self, app):
            self.app = app

    from wrapper.models import User

    user = User()
    user.id = uuid.uuid4()
    user.email = "u@example.local"
    user.role = "user"
    user.status = "approved"

    resp = await main_module.workbench_models_status(
        request=_StubRequest(app),  # type: ignore[arg-type]
        user=user,
    )
    body = resp.body
    import json as _json
    payload = _json.loads(body)
    ids = [m["id"] for m in payload["models"]]
    # UI-hidden model excluded.
    assert "stopped-one" not in ids
    # Live models stay visible regardless of Modal state.
    assert "alive-deployed" in ids
    assert "warming-up" in ids
    assert "no-modal-name" in ids
    # Staging entry is filtered upstream by status != "live".
    assert "still-cooking" not in ids
    # No warm key on the payload — Modal-dependent UX is gone.
    for m in payload["models"]:
        assert "warm" not in m


async def test_workbench_models_status_does_not_call_modal_on_outage(monkeypatch):
    """A Modal control-plane outage must not affect /workbench/models/status.
    The endpoint is now pure registry iteration — it never reaches Modal,
    so the dropdown is unaffected by upstream health."""

    def boom(_name):
        raise RuntimeError("modal control plane down — should never be called")

    monkeypatch.setattr(main_module.modalops, "get_app_state", boom)
    monkeypatch.setattr(main_module.modalops, "get_active_runner_count", boom)

    from fastapi import FastAPI
    app = FastAPI()
    app.state.models = _registry()
    app.state.last_completion_at = {}

    class _StubRequest:
        def __init__(self, app):
            self.app = app

    from wrapper.models import User

    user = User()
    user.id = uuid.uuid4()
    user.email = "u@example.local"
    user.role = "user"
    user.status = "approved"

    resp = await main_module.workbench_models_status(
        request=_StubRequest(app),  # type: ignore[arg-type]
        user=user,
    )
    import json as _json
    payload = _json.loads(resp.body)
    ids = {m["id"] for m in payload["models"]}
    assert ids == {"alive-deployed", "stopped-one", "warming-up", "no-modal-name"}


# --- SSE error event UX -----------------------------------------------------

def test_friendly_upstream_error_404_names_stopped_model():
    """404 is the case users actually hit when a model app is stopped. The
    message must point them somewhere useful — not raw 'Upstream error 404'."""
    msg = main_module._friendly_upstream_error(404)
    assert "disabled" in msg.lower()
    assert "infra@acsresearch.org" in msg


def test_friendly_upstream_error_other_codes_stay_generic():
    """5xx / unknown codes get a generic message — we don't surface vLLM /
    Modal internal error bodies to end users."""
    for code in (500, 502, 503, 504):
        msg = main_module._friendly_upstream_error(code)
        assert f"({code})" in msg
        assert "try again" in msg.lower()


# --- _resolve_warm_flags: warm-pill lookups run in parallel ----------------

async def test_resolve_warm_flags_runs_in_parallel(monkeypatch):
    """Every workbench page render used to await _is_model_warm in a per-model
    for loop, paying N× Modal RPC latency on a cold _RUNNERS_CACHE. The
    replacement helper must fan out via asyncio.gather. Verified by counting
    concurrent calls inside the patched get_active_runner_count.
    """
    import asyncio as _asyncio
    import datetime as _dt

    in_flight = 0
    max_in_flight = 0

    async def fake_runner_count(name):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await _asyncio.sleep(0.005)
        in_flight -= 1
        return {
            "acs-alive": 1,
            "acs-stopped": 0,
            "acs-warming": 0,
        }.get(name, 0)

    monkeypatch.setattr(
        main_module.modalops, "get_active_runner_count", fake_runner_count
    )
    flags = await main_module._resolve_warm_flags(
        _registry(),
        last_at_map={},
        now=_dt.datetime.now(tz=_dt.UTC),
    )
    # 3 live models with modal_app_name → 3 concurrent RPCs.
    assert max_in_flight == 3
    assert flags["alive-deployed"] is True
    assert flags["stopped-one"] is False
    assert flags["warming-up"] is False
    # No modal_app_name + no last_completion_at → cold by time-heuristic.
    assert flags["no-modal-name"] is False


async def test_resolve_warm_flags_rpc_failure_falls_back_to_time_heuristic(monkeypatch):
    """If get_active_runner_count raises for one model, that model's warm
    flag falls back to the in-process last_completion_at time heuristic
    rather than vanishing or defaulting to True."""
    import datetime as _dt

    now = _dt.datetime.now(tz=_dt.UTC)
    recent = now - _dt.timedelta(minutes=1)

    async def fake_runner_count(name):
        if name == "acs-stopped":
            raise RuntimeError("modal control plane down")
        return 1 if name == "acs-alive" else 0

    monkeypatch.setattr(
        main_module.modalops, "get_active_runner_count", fake_runner_count
    )
    flags = await main_module._resolve_warm_flags(
        _registry(),
        last_at_map={"stopped-one": recent},  # within MODAL_SCALEDOWN_WINDOW
        now=now,
    )
    # RPC raised but recent completion → fall back says "warm".
    assert flags["stopped-one"] is True
    # Unaffected model still uses Modal answer.
    assert flags["alive-deployed"] is True
