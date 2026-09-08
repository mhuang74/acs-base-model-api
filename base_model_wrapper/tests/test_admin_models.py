"""Unit tests for the per-model admin lifecycle endpoints.

Hits the real FastAPI app via TestClient but stubs ``modal_ops`` so we don't
touch Modal, and overrides ``require_admin`` so we don't need a Postgres-backed
cookie session. The endpoints don't touch the DB themselves (they log + call
``modal_ops``), so we don't spin one up here.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from wrapper import modal_ops as modalops
from wrapper.main import (
    admin_model_deploy,
    admin_model_keep_warm,
    admin_model_release,
    admin_model_stop,
    admin_models_status,
)
from wrapper.models import User
from wrapper.settings import ModelEntry, Settings


def _make_admin() -> User:
    u = User()
    u.id = uuid.uuid4()
    u.email = "admin@example.local"
    u.role = "admin"
    u.status = "approved"
    return u


def _build_app(*, modal_configured: bool = True) -> FastAPI:
    """Tiny FastAPI app wiring just the three model-lifecycle routes."""
    app = FastAPI()
    settings_kwargs = dict(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
    )
    if modal_configured:
        settings_kwargs["modal_token_id"] = "tok-id"
        settings_kwargs["modal_token_secret"] = "tok-secret"
    app.state.settings = Settings(**settings_kwargs)
    app.state.models = {
        "llama-1b": ModelEntry(
            model_id="llama-1b",
            upstream_url="https://small.example",
            served_model_name="gpt2",
            tokenizer_repo="gpt2",
            gpu_shape_label="1xH100",
            modal_app_name="acs-llama-1b",
        ),
        "no-modal": ModelEntry(
            model_id="no-modal",
            upstream_url="https://x.example",
            served_model_name="gpt2",
            tokenizer_repo="gpt2",
            gpu_shape_label="—",
            modal_app_name=None,
        ),
    }
    app.post("/admin/models/{model_id}/stop")(admin_model_stop)
    app.post("/admin/models/{model_id}/keep-warm")(admin_model_keep_warm)
    app.post("/admin/models/{model_id}/release")(admin_model_release)
    app.post("/admin/models/{model_id}/deploy")(admin_model_deploy)
    app.get("/admin/models/status")(admin_models_status)

    from wrapper import web_auth as webauth

    app.dependency_overrides[webauth.require_admin] = _make_admin
    return app


async def _anoop(*_a, **_kw):
    """Async no-op stand-in for ``set_min_containers`` (now a coroutine)."""
    return None


@contextmanager
def _patch_modal_ops(monkeypatch, *, calls):
    """Record calls into ``calls`` dict rather than letting real Modal RPCs fire.

    ``stop_app``, ``get_app_state`` and ``set_min_containers`` are all async in
    production (they drive the Modal ``_Client`` stub RPCs directly); the fakes
    mirror that so ``await`` at the call sites behaves like prod.
    """

    async def fake_stop_app(name):
        calls.setdefault("stop", []).append(name)

    async def fake_set_min(name, n):
        calls.setdefault("set_min", []).append((name, n))

    async def fake_get_state(name):
        return "deployed"

    async def fake_assert_running(name):
        # Default to "running" so the keep-warm / release happy-path tests
        # don't have to wire it up. The handful of tests that exercise the
        # stopped-app guard re-patch this with a raising fake.
        return None

    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    monkeypatch.setattr(modalops, "set_min_containers", fake_set_min)
    monkeypatch.setattr(modalops, "get_app_state", fake_get_state)
    monkeypatch.setattr(modalops, "assert_app_running", fake_assert_running)
    try:
        yield calls
    finally:
        pass


# ---- admin gating ----------------------------------------------------------

def test_stop_requires_admin(monkeypatch):
    """Without the override the route's require_admin redirects to /login."""
    with _patch_modal_ops(monkeypatch, calls={}):
        app = _build_app()
        # Drop the override so the real require_admin dependency runs.
        from wrapper import web_auth as webauth

        app.dependency_overrides.pop(webauth.require_admin, None)
        # And drop get_session: require_admin pulls current_user which needs it.
        from wrapper.db import get_session

        async def _no_session():
            yield None

        app.dependency_overrides[get_session] = _no_session
        client = TestClient(app)
        r = client.post("/admin/models/llama-1b/stop", follow_redirects=False)
    # Unauthenticated → 303 to /login (HTTPException headers carry Location).
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


# ---- happy paths -----------------------------------------------------------

def test_stop_happy_path(monkeypatch):
    calls: dict = {}
    with _patch_modal_ops(monkeypatch, calls=calls):
        client = TestClient(_build_app())
        r = client.post("/admin/models/llama-1b/stop", follow_redirects=False)
    assert r.status_code == 303, r.text
    assert calls.get("stop") == ["acs-llama-1b"]


def test_keep_warm_calls_update_autoscaler_min_1(monkeypatch):
    calls: dict = {}
    with _patch_modal_ops(monkeypatch, calls=calls):
        client = TestClient(_build_app())
        r = client.post("/admin/models/llama-1b/keep-warm", follow_redirects=False)
    assert r.status_code == 303
    assert calls.get("set_min") == [("acs-llama-1b", 1)]


def test_release_calls_update_autoscaler_min_0(monkeypatch):
    calls: dict = {}
    with _patch_modal_ops(monkeypatch, calls=calls):
        client = TestClient(_build_app())
        r = client.post("/admin/models/llama-1b/release", follow_redirects=False)
    assert r.status_code == 303
    assert calls.get("set_min") == [("acs-llama-1b", 0)]


def test_stop_idempotent_when_modal_returns_notfound(monkeypatch):
    """``modal_ops.stop_app`` already swallows NotFoundError; route still 303s."""
    calls: dict = {}

    async def fake_stop_app(name):
        calls.setdefault("stop", []).append(name)

    async def fake_get_state(*_):
        return "stopped"

    async def fake_assert_running(*_):
        return None

    async def fake_set_min(*_a, **_kw):
        return None

    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    monkeypatch.setattr(modalops, "set_min_containers", fake_set_min)
    monkeypatch.setattr(modalops, "get_app_state", fake_get_state)
    monkeypatch.setattr(modalops, "assert_app_running", fake_assert_running)
    client = TestClient(_build_app())
    r = client.post("/admin/models/llama-1b/stop", follow_redirects=False)
    assert r.status_code == 303


def test_stop_invalidates_state_cache(monkeypatch):
    """After a successful Stop, the dashboard must reflect the stop within the
    next render rather than showing stale "deployed" for up to 30 s (the
    _STATE_CACHE TTL). Verify invalidate_state was called with the app name.
    """
    invalidated: list[str] = []
    monkeypatch.setattr(
        modalops,
        "invalidate_state",
        lambda name: invalidated.append(name),
    )
    with _patch_modal_ops(monkeypatch, calls={}):
        client = TestClient(_build_app())
        r = client.post("/admin/models/llama-1b/stop", follow_redirects=False)
    assert r.status_code == 303, r.text
    assert invalidated == ["acs-llama-1b"]


def test_stop_clears_warm_signal(monkeypatch):
    """ACS-98: a Stop drops the model's last-completion timestamp so the
    workbench "usually warm" hint and the /health warm-estimate go cold
    immediately — only for the stopped model, not its neighbours."""
    import datetime as dt

    with _patch_modal_ops(monkeypatch, calls={}):
        app = _build_app()
        app.state.last_completion_at = {
            "llama-1b": dt.datetime.now(tz=dt.UTC),
            "no-modal": dt.datetime.now(tz=dt.UTC),
        }
        client = TestClient(app)
        r = client.post("/admin/models/llama-1b/stop", follow_redirects=False)
    assert r.status_code == 303, r.text
    assert "llama-1b" not in app.state.last_completion_at
    # Unrelated models keep their warm signal.
    assert "no-modal" in app.state.last_completion_at


def test_keep_warm_rejects_stopped_app(monkeypatch):
    """Keep-warm on a stopped app must surface as an admin-redirect error
    instead of silently succeeding (the underlying update_autoscaler call is a
    no-op on stopped apps — see modal_ops.assert_app_running)."""
    set_min_called: list = []

    async def stopped_assert(name):
        raise modalops.ModalOpsError(
            f"Modal app {name!r} is not running (state=stopped)"
        )

    def boom_set_min(*_a, **_kw):
        set_min_called.append(True)

    async def fake_stop_app(*_):
        return None

    async def fake_get_state(*_):
        return "stopped"

    monkeypatch.setattr(modalops, "assert_app_running", stopped_assert)
    monkeypatch.setattr(modalops, "set_min_containers", boom_set_min)
    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    monkeypatch.setattr(modalops, "get_app_state", fake_get_state)
    client = TestClient(_build_app())
    r = client.post("/admin/models/llama-1b/keep-warm", follow_redirects=False)
    assert r.status_code == 303
    location = r.headers.get("location", "")
    assert "err=" in location
    assert set_min_called == [], "set_min_containers must not run on a stopped app"


def test_release_rejects_stopped_app(monkeypatch):
    """Same guard for Release — releasing a stopped app is a no-op masquerading
    as success without this check."""

    async def stopped_assert(name):
        raise modalops.ModalOpsError(
            f"Modal app {name!r} is not running (state=stopped)"
        )

    set_min_called: list = []

    def boom_set_min(*_a, **_kw):
        set_min_called.append(True)

    async def fake_stop_app(*_):
        return None

    async def fake_get_state(*_):
        return "stopped"

    monkeypatch.setattr(modalops, "assert_app_running", stopped_assert)
    monkeypatch.setattr(modalops, "set_min_containers", boom_set_min)
    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    monkeypatch.setattr(modalops, "get_app_state", fake_get_state)
    client = TestClient(_build_app())
    r = client.post("/admin/models/llama-1b/release", follow_redirects=False)
    assert r.status_code == 303
    assert "err=" in r.headers.get("location", "")
    assert set_min_called == []


# ---- error / config paths --------------------------------------------------

def test_unknown_model_id_returns_404(monkeypatch):
    with _patch_modal_ops(monkeypatch, calls={}):
        client = TestClient(_build_app())
        r = client.post("/admin/models/nope/stop", follow_redirects=False)
    assert r.status_code == 404


def test_model_without_modal_app_name_returns_400(monkeypatch):
    with _patch_modal_ops(monkeypatch, calls={}):
        client = TestClient(_build_app())
        r = client.post("/admin/models/no-modal/stop", follow_redirects=False)
    assert r.status_code == 400


def test_modal_unconfigured_returns_503(monkeypatch):
    """When MODAL_TOKEN_ID is unset the route 503s rather than crashing."""
    with _patch_modal_ops(monkeypatch, calls={}):
        client = TestClient(_build_app(modal_configured=False))
        r = client.post("/admin/models/llama-1b/stop", follow_redirects=False)
    assert r.status_code == 503
    body = r.json()
    # _http_exception_handler isn't wired on this subapp, so default {detail: {...}}.
    detail = body.get("detail", body)
    if isinstance(detail, dict) and "error" in detail:
        assert detail["error"]["code"] == "modal_not_configured"


def test_modal_ops_error_redirects_with_flash(monkeypatch):
    async def boom(*_a, **_kw):
        # set_min_containers is async (drives the stub RPCs); mirror a prod raise.
        raise modalops.ModalOpsError("simulated")

    async def fake_stop_app(*_):
        return None

    async def fake_get_state(*_):
        return "deployed"

    async def fake_assert_running(*_):
        return None

    monkeypatch.setattr(modalops, "set_min_containers", boom)
    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    monkeypatch.setattr(modalops, "get_app_state", fake_get_state)
    monkeypatch.setattr(modalops, "assert_app_running", fake_assert_running)
    client = TestClient(_build_app())
    r = client.post("/admin/models/llama-1b/keep-warm", follow_redirects=False)
    assert r.status_code == 303
    assert "err=" in r.headers.get("location", "")


# ---- deploy endpoint -------------------------------------------------------

def _stage_modal_source(tmp_path: Path, *, missing_rel_path: str) -> None:
    for rel_path in modalops.MODAL_REQUIRED_SOURCE_FILES:
        if rel_path == missing_rel_path:
            continue
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# staged for test\n", encoding="utf-8")


def test_deploy_source_file_contract_includes_modal_entrypoint_and_config():
    assert modalops.MODAL_SOURCE_DIR == "/app/modal_source"
    assert "modal_app.py" in modalops.MODAL_REQUIRED_SOURCE_FILES
    assert os.path.join("serving", "modal_config.py") in modalops.MODAL_REQUIRED_SOURCE_FILES


@pytest.mark.parametrize(
    "missing_rel_path",
    [
        "modal_app.py",
        os.path.join("serving", "modal_config.py"),
    ],
)
async def test_deploy_preflight_missing_required_source_file_fails_clearly(
    monkeypatch,
    tmp_path,
    missing_rel_path,
):
    _stage_modal_source(tmp_path, missing_rel_path=missing_rel_path)
    monkeypatch.setattr(modalops, "MODAL_SOURCE_DIR", str(tmp_path))
    monkeypatch.setenv("MODAL_TOKEN_ID", "tok-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "tok-secret")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("modal deploy subprocess must not run after preflight failure")

    monkeypatch.setattr(modalops.subprocess, "run", fail_run)

    with pytest.raises(modalops.ModalOpsError) as excinfo:
        await modalops.deploy_app("llama-1b")

    msg = str(excinfo.value)
    assert "modal source file(s) missing" in msg
    assert str(tmp_path) in msg
    assert missing_rel_path in msg


def test_deploy_requires_admin(monkeypatch):
    """Without the override the route's require_admin redirects to /login."""
    with _patch_modal_ops(monkeypatch, calls={}):
        app = _build_app()
        from wrapper import web_auth as webauth

        app.dependency_overrides.pop(webauth.require_admin, None)
        from wrapper.db import get_session

        async def _no_session():
            yield None

        app.dependency_overrides[get_session] = _no_session
        client = TestClient(app)
        r = client.post("/admin/models/llama-1b/deploy", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_deploy_happy_path(monkeypatch):
    calls: dict = {}

    async def fake_deploy(model_id):
        calls.setdefault("deploy", []).append(model_id)
        return True, "App deployed in 1.6s"

    monkeypatch.setattr(modalops, "deploy_app", fake_deploy)
    monkeypatch.setattr(modalops, "set_min_containers", _anoop)

    async def fake_stop_app(*_):
        return None

    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    client = TestClient(_build_app())
    r = client.post("/admin/models/llama-1b/deploy", follow_redirects=False)
    assert r.status_code == 303, r.text
    assert calls.get("deploy") == ["llama-1b"]
    assert "msg=" in r.headers.get("location", "")


def test_deploy_unknown_model_id(monkeypatch):
    async def fake_deploy(model_id):
        raise AssertionError("deploy_app should not be called for unknown model")

    monkeypatch.setattr(modalops, "deploy_app", fake_deploy)
    monkeypatch.setattr(modalops, "set_min_containers", _anoop)

    async def fake_stop_app(*_):
        return None

    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    client = TestClient(_build_app())
    r = client.post("/admin/models/nope/deploy", follow_redirects=False)
    assert r.status_code == 404


def test_deploy_subprocess_failure(monkeypatch):
    async def fake_deploy(model_id):
        return False, "error: not authenticated"

    monkeypatch.setattr(modalops, "deploy_app", fake_deploy)
    monkeypatch.setattr(modalops, "set_min_containers", _anoop)

    async def fake_stop_app(*_):
        return None

    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    client = TestClient(_build_app())
    r = client.post("/admin/models/llama-1b/deploy", follow_redirects=False)
    assert r.status_code == 303
    location = r.headers.get("location", "")
    assert "err=" in location
    assert "not+authenticated" in location or "not%20authenticated" in location


def test_deploy_modal_ops_error(monkeypatch):
    async def fake_deploy(model_id):
        raise modalops.ModalOpsError("modal_app.py not found")

    monkeypatch.setattr(modalops, "deploy_app", fake_deploy)
    monkeypatch.setattr(modalops, "set_min_containers", _anoop)

    async def fake_stop_app(*_):
        return None

    monkeypatch.setattr(modalops, "stop_app", fake_stop_app)
    client = TestClient(_build_app())
    r = client.post("/admin/models/llama-1b/deploy", follow_redirects=False)
    assert r.status_code == 303
    location = r.headers.get("location", "")
    assert "err=" in location
    assert "modal_app.py" in location.replace("+", " ").replace("%20", " ")


# ---- /admin/models/status JSON endpoint -----------------------------------

def test_models_status_requires_admin(monkeypatch):
    """Without the admin override the GET 303s to /login."""
    with _patch_modal_ops(monkeypatch, calls={}):
        app = _build_app()
        from wrapper import web_auth as webauth

        app.dependency_overrides.pop(webauth.require_admin, None)
        from wrapper.db import get_session

        async def _no_session():
            yield None

        app.dependency_overrides[get_session] = _no_session
        client = TestClient(app)
        r = client.get("/admin/models/status", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_models_status_returns_per_model_state(monkeypatch):
    """JSON body lists every registered model with its resolved state."""
    with _patch_modal_ops(monkeypatch, calls={}):
        client = TestClient(_build_app())
        r = client.get("/admin/models/status")
    assert r.status_code == 200
    body = r.json()
    assert body.get("modal_unavailable") is None
    models = body.get("models")
    assert isinstance(models, list)
    by_id = {m["id"]: m for m in models}
    assert "llama-1b" in by_id
    assert "no-modal" in by_id
    # llama-1b has a modal_app_name and tokens are configured, fake_get_state -> deployed.
    assert by_id["llama-1b"]["state"] == "deployed"
    assert by_id["llama-1b"]["modal_app_name"] == "acs-llama-1b"
    # no-modal has no modal_app_name; state surfaces as "unconfigured".
    assert by_id["no-modal"]["state"] == "unconfigured"
    assert by_id["no-modal"]["modal_app_name"] is None


def test_models_status_surfaces_live_containers_and_scaling(monkeypatch):
    """ACS-62: each row carries live container count + deploy-time scaling params.

    Live containers come from ``modal_ops.get_active_runner_count`` (mocked
    here to avoid Modal RPCs). Scaling params come from the shared
    ``acs_model_registry`` — the test models (``llama-1b`` / ``no-modal``)
    aren't in the registry, so those fields surface as ``None`` rather than
    fake values. We use a registry-backed id (``llama-405b``) to assert the
    populated path.
    """
    runner_calls: list[str] = []

    async def fake_get_state(*_):
        return "deployed"

    async def fake_get_runners(name):
        runner_calls.append(name)
        return 2 if name == "acs-llama-1b" else 0

    monkeypatch.setattr(modalops, "get_app_state", fake_get_state)
    monkeypatch.setattr(modalops, "get_active_runner_count", fake_get_runners)

    app = _build_app()
    # Add a registry-backed entry so we can assert scaling params are filled.
    app.state.models["llama-405b"] = ModelEntry(
        model_id="llama-405b",
        upstream_url="https://stub",
        served_model_name="meta-llama/Llama-3.1-405B",
        tokenizer_repo="meta-llama/Llama-3.1-405B",
        gpu_shape_label="8xH200",
        modal_app_name="acs-llama-405b",
    )
    client = TestClient(app)
    r = client.get("/admin/models/status")
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["models"]}

    # Live container count is fanned in alongside state. Calls happen for
    # every entry with a modal_app_name set.
    assert "acs-llama-1b" in runner_calls
    assert by_id["llama-1b"]["live_containers"] == 2
    # llama-1b isn't in the shared registry → scaling fields are None
    # (rather than fake defaults that would mislead admins).
    assert by_id["llama-1b"]["min_containers"] is None
    assert by_id["llama-1b"]["max_containers"] is None
    assert by_id["llama-1b"]["scaledown_window_s"] is None

    # llama-405b is in the registry; scaling values reflect the spec.
    from acs_model_registry import get_model_spec

    spec_405 = get_model_spec("llama-405b")
    assert by_id["llama-405b"]["live_containers"] == 0
    assert by_id["llama-405b"]["min_containers"] == spec_405.min_containers
    assert by_id["llama-405b"]["max_containers"] == spec_405.max_containers
    assert by_id["llama-405b"]["scaledown_window_s"] == spec_405.scaledown_window_s

    # no-modal has no modal_app_name → no RPC fires for it, live_containers
    # defaults to 0. The fake gets called exactly twice — once each for
    # llama-1b and llama-405b, both of which have a modal_app_name.
    assert set(runner_calls) == {"acs-llama-1b", "acs-llama-405b"}
    assert by_id["no-modal"]["live_containers"] == 0


def test_models_status_does_not_call_runner_rpc_when_modal_unavailable(monkeypatch):
    """ACS-62: when MODAL_TOKEN_ID is missing, skip both state and runner RPCs."""

    async def boom_get_state(*_):
        raise AssertionError("get_app_state must not run when modal unavailable")

    async def boom_get_runners(*_):
        raise AssertionError(
            "get_active_runner_count must not run when modal unavailable"
        )

    monkeypatch.setattr(modalops, "get_app_state", boom_get_state)
    monkeypatch.setattr(modalops, "get_active_runner_count", boom_get_runners)
    client = TestClient(_build_app(modal_configured=False))
    r = client.get("/admin/models/status")
    assert r.status_code == 200
    body = r.json()
    assert body["modal_unavailable"]
    for row in body["models"]:
        # When Modal is unavailable, live_containers defaults to 0; scaling
        # params still come from the registry (no RPC needed for those).
        assert row["live_containers"] == 0


def test_models_status_runner_rpc_failure_degrades_to_unknown(monkeypatch):
    """ACS-128: a failed ``get_active_runner_count`` RPC must not break the
    page, and must surface as ``live_containers=None`` (unknown) — NOT a
    misleading 0 that would render as "scaled to zero" for a possibly-warm
    model.
    """

    async def fake_get_state(*_):
        return "deployed"

    async def flaky_get_runners(name):
        raise modalops.ModalOpsError(f"simulated control-plane outage for {name}")

    monkeypatch.setattr(modalops, "get_app_state", fake_get_state)
    monkeypatch.setattr(modalops, "get_active_runner_count", flaky_get_runners)

    client = TestClient(_build_app())
    r = client.get("/admin/models/status")
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["models"]}
    assert by_id["llama-1b"]["state"] == "deployed"
    assert by_id["llama-1b"]["live_containers"] is None


def test_models_status_runner_count_none_surfaces_as_unknown(monkeypatch):
    """ACS-128: when ``get_active_runner_count`` returns None (unknown — a
    failed/timed-out fetch with no prior value), the row carries
    ``live_containers=None``, distinct from a genuine 0."""

    async def fake_get_state(*_):
        return "deployed"

    async def unknown_get_runners(name):
        return None

    monkeypatch.setattr(modalops, "get_app_state", fake_get_state)
    monkeypatch.setattr(modalops, "get_active_runner_count", unknown_get_runners)

    client = TestClient(_build_app())
    r = client.get("/admin/models/status")
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["models"]}
    assert by_id["llama-1b"]["live_containers"] is None


def test_models_status_surfaces_modal_unavailable(monkeypatch):
    """When MODAL_TOKEN_ID is missing the JSON flags it; no RPCs fire."""

    async def boom_get_state(*_):
        raise AssertionError("get_app_state must not be called when modal unconfigured")

    monkeypatch.setattr(modalops, "get_app_state", boom_get_state)
    client = TestClient(_build_app(modal_configured=False))
    r = client.get("/admin/models/status")
    assert r.status_code == 200
    body = r.json()
    assert body.get("modal_unavailable")
    states = {m["id"]: m["state"] for m in body["models"]}
    assert states["llama-1b"] == "unavailable"
    assert states["no-modal"] == "unconfigured"


# ---- modal_ops state-cache SWR behaviour ----------------------------------

async def test_get_app_state_returns_stale_value_immediately_when_expired(
    monkeypatch,
):
    """Stale cache entry: return the stale value, never block on the fresh
    fetch. The refresh runs in the background; we assert it was *scheduled*
    (via the dedupe set) but don't await it before reading the return value.
    """
    import asyncio
    import time

    from modal_proto import api_pb2

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    modalops._clear_state_cache()

    # Plant a stale entry: timestamp older than the TTL.
    stale_ts = time.monotonic() - (modalops._STATE_CACHE_TTL_S + 10)
    modalops._STATE_CACHE["acs-llama-1b"] = (stale_ts, "stopped")

    resolve_calls: list[str] = []

    async def fake_resolve(name):
        resolve_calls.append(name)
        # Sleep to make the refresh non-trivially slow — proves the caller
        # didn't await it.
        await asyncio.sleep(0.05)
        return "ap-1", api_pb2.APP_STATE_DEPLOYED

    monkeypatch.setattr(modalops, "_resolve_app", fake_resolve)

    state = await modalops.get_app_state("acs-llama-1b")

    # Stale value returned immediately — NOT the fresh "deployed" value.
    assert state == "stopped"
    # Background refresh was scheduled (dedupe marker set).
    assert "acs-llama-1b" in modalops._STATE_REFRESHING

    # Let the background task drain so we don't leak it into the next test.
    for _ in range(20):
        if "acs-llama-1b" not in modalops._STATE_REFRESHING:
            break
        await asyncio.sleep(0.02)
    assert "acs-llama-1b" not in modalops._STATE_REFRESHING
    # After refresh completes, cache holds the fresh value.
    assert modalops._STATE_CACHE["acs-llama-1b"][1] == "deployed"
    assert resolve_calls == ["acs-llama-1b"]


async def test_get_app_state_blocks_on_first_ever_read(monkeypatch):
    """Cold miss path: no cache entry → block on the real fetch and populate
    the cache. The SWR fast-path only kicks in once there's something stale
    to serve.
    """
    from modal_proto import api_pb2

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    modalops._clear_state_cache()

    async def fake_resolve(name):
        return "ap-1", api_pb2.APP_STATE_DEPLOYED

    monkeypatch.setattr(modalops, "_resolve_app", fake_resolve)

    state = await modalops.get_app_state("acs-llama-1b")
    assert state == "deployed"
    assert "acs-llama-1b" in modalops._STATE_CACHE
    assert modalops._STATE_CACHE["acs-llama-1b"][1] == "deployed"


async def test_invalidate_state_clears_runners_cache_too(monkeypatch):
    """invalidate_state is called from admin Stop / Deploy etc. and must
    flush BOTH caches so the next render sees the post-action reality —
    otherwise the workbench would still flag the app warm right after Stop.
    """
    import time

    modalops._clear_state_cache()
    now = time.monotonic()
    modalops._STATE_CACHE["acs-llama-1b"] = (now, "deployed")
    modalops._RUNNERS_CACHE["acs-llama-1b"] = (now, 3)
    modalops._STATE_REFRESHING.add("acs-llama-1b")
    modalops._RUNNERS_REFRESHING.add("acs-llama-1b")

    modalops.invalidate_state("acs-llama-1b")

    assert "acs-llama-1b" not in modalops._STATE_CACHE
    assert "acs-llama-1b" not in modalops._RUNNERS_CACHE
    assert "acs-llama-1b" not in modalops._STATE_REFRESHING
    assert "acs-llama-1b" not in modalops._RUNNERS_REFRESHING


async def test_prewarm_caches_schedules_background_tasks(monkeypatch):
    """prewarm_caches is the lifespan-hook entry: sync, fire-and-forget,
    schedules one state + one runners refresh per app. After the tasks
    drain, both caches should be populated.
    """
    import asyncio

    from modal_proto import api_pb2

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    modalops._clear_state_cache()

    resolve_calls: list[str] = []

    async def fake_resolve(name):
        resolve_calls.append(name)
        return "ap-1", api_pb2.APP_STATE_DEPLOYED

    # Runner count now goes through the internal async client
    # (_ModalAsyncClient.from_env().stub.FunctionGet → FunctionGetCurrentStats),
    # not the high-level modal.Function API.
    fn_get_calls: list[str] = []

    class _Stub:
        async def FunctionGet(self, req):
            fn_get_calls.append(req.app_name)
            return type("R", (), {"function_id": f"fn-{req.app_name}"})()

        async def FunctionGetCurrentStats(self, req):
            return type("R", (), {"num_total_tasks": 1})()

    class _FakeClient:
        stub = _Stub()

    async def fake_from_env():
        return _FakeClient()

    monkeypatch.setattr(modalops, "_resolve_app", fake_resolve)
    monkeypatch.setattr(
        modalops._ModalAsyncClient, "from_env", staticmethod(fake_from_env)
    )

    # Sync return — must not raise, must not await.
    result = modalops.prewarm_caches(["a", "b"])
    assert result is None

    # Drain scheduled tasks.
    for _ in range(50):
        if (
            "a" in modalops._STATE_CACHE
            and "b" in modalops._STATE_CACHE
            and "a" in modalops._RUNNERS_CACHE
            and "b" in modalops._RUNNERS_CACHE
        ):
            break
        await asyncio.sleep(0.02)

    assert modalops._STATE_CACHE["a"][1] == "deployed"
    assert modalops._STATE_CACHE["b"][1] == "deployed"
    assert modalops._RUNNERS_CACHE["a"][1] == 1
    assert modalops._RUNNERS_CACHE["b"][1] == 1
    assert sorted(resolve_calls) == ["a", "b"]
    assert sorted(fn_get_calls) == ["a", "b"]
    # Dedupe markers cleared after tasks complete.
    assert "a" not in modalops._STATE_REFRESHING
    assert "b" not in modalops._STATE_REFRESHING
    assert "a" not in modalops._RUNNERS_REFRESHING
    assert "b" not in modalops._RUNNERS_REFRESHING


# ---- get_active_runner_count: caches failures as unknown, not 0 -----------

async def test_get_active_runner_count_caches_rpc_failure_as_unknown(monkeypatch):
    """ACS-128: a failing Modal RPC on cache miss must cache *unknown* (None),
    NOT a confident 0. Caching 0 rendered as "scaled to zero" and misreported
    warm always-on models as cold. None is still cached (so the page stays
    fast) but as unknown — and it self-heals: a None hit always kicks a
    background retry, and ``mark_runner_warm`` write-through corrects it the
    moment a real completion succeeds.
    """

    # The internal-client runner fetch fails (simulated control-plane timeout).
    class _Stub:
        async def FunctionGet(self, req):
            raise RuntimeError("simulated modal control-plane timeout")

        async def FunctionGetCurrentStats(self, req):  # pragma: no cover
            raise RuntimeError("unreachable")

    class _FakeClient:
        stub = _Stub()

    async def fake_from_env():
        return _FakeClient()

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    modalops._clear_state_cache()

    monkeypatch.setattr(
        modalops._ModalAsyncClient, "from_env", staticmethod(fake_from_env)
    )

    count = await modalops.get_active_runner_count("acs-llama-1b")
    assert count is None
    # None (unknown) IS cached so subsequent reads stay fast — but it is NOT a
    # confident 0, so callers render "unavailable", not "scaled to zero".
    assert "acs-llama-1b" in modalops._RUNNERS_CACHE
    assert modalops._RUNNERS_CACHE["acs-llama-1b"][1] is None


# ---- mark_runner_warm -----------------------------------------------------

def test_mark_runner_warm_writes_through_to_cache(monkeypatch):
    """A successful completion is ground-truth that the runner is alive.
    mark_runner_warm bypasses Modal entirely so the workbench warm pill
    reflects reality immediately, even when Modal's control plane is too
    slow to confirm via get_current_stats."""

    modalops._clear_state_cache()
    assert "acs-llama-1b" not in modalops._RUNNERS_CACHE

    modalops.mark_runner_warm("acs-llama-1b")
    cached = modalops._RUNNERS_CACHE["acs-llama-1b"]
    assert cached[1] == 1
    # And it clears any in-flight refresh marker so a late stale-fetch can't
    # overwrite the freshly-written-through value.
    modalops._RUNNERS_REFRESHING.add("acs-llama-1b")
    modalops.mark_runner_warm("acs-llama-1b", count=3)
    assert modalops._RUNNERS_CACHE["acs-llama-1b"][1] == 3
    assert "acs-llama-1b" not in modalops._RUNNERS_REFRESHING


# ---- assert_app_running ---------------------------------------------------

async def test_assert_app_running_passes_when_deployed(monkeypatch):
    from modal_proto import api_pb2

    async def fake_resolve(name):
        return "ap-1", api_pb2.APP_STATE_DEPLOYED

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    monkeypatch.setattr(modalops, "_resolve_app", fake_resolve)
    # No raise.
    await modalops.assert_app_running("acs-llama-1b")


async def test_assert_app_running_raises_on_stopped(monkeypatch):
    from modal_proto import api_pb2

    async def fake_resolve(name):
        return "ap-1", api_pb2.APP_STATE_STOPPED

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    monkeypatch.setattr(modalops, "_resolve_app", fake_resolve)
    with pytest.raises(modalops.ModalOpsError, match="not running"):
        await modalops.assert_app_running("acs-llama-1b")


async def test_assert_app_running_raises_on_unspecified(monkeypatch):
    """No previous_app_id from Modal (e.g. app name never existed) — treated
    as not-running so we don't issue updates against ghost names."""
    from modal_proto import api_pb2

    async def fake_resolve(name):
        return None, api_pb2.APP_STATE_UNSPECIFIED

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    monkeypatch.setattr(modalops, "_resolve_app", fake_resolve)
    with pytest.raises(modalops.ModalOpsError):
        await modalops.assert_app_running("acs-ghost")


# ---- set_min_containers: async stub path (ACS-202) ------------------------
#
# Regression cover for the Release/Keep-warm hang: the old implementation
# called the synchronicity-wrapped modal.Function API from an asyncio.to_thread
# worker, which under the 1.5.x pin issued its RPCs "outside of task context"
# and hung for minutes (browser spun → 499). set_min_containers now drives
# FunctionGet → FunctionUpdateSchedulingParams over the async stub itself, is
# awaited inline, and is bounded by a timeout.

async def test_set_min_containers_drives_stub_rpcs(monkeypatch):
    """Happy path: resolve function_id via FunctionGet, then write min_containers
    via FunctionUpdateSchedulingParams with the right function_id + settings."""
    get_reqs: list[tuple[str, str]] = []
    update_reqs: list[tuple[str, int]] = []

    class _Stub:
        async def FunctionGet(self, req):
            get_reqs.append((req.app_name, req.object_tag))
            return type("R", (), {"function_id": f"fn-{req.app_name}"})()

        async def FunctionUpdateSchedulingParams(self, req):
            update_reqs.append((req.function_id, req.settings.min_containers))
            return type("R", (), {})()

    class _FakeClient:
        stub = _Stub()

    async def fake_from_env():
        return _FakeClient()

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    monkeypatch.setattr(
        modalops._ModalAsyncClient, "from_env", staticmethod(fake_from_env)
    )

    await modalops.set_min_containers("acs-llama-1b", 0)

    assert get_reqs == [("acs-llama-1b", modalops.SERVE_FUNCTION_NAME)]
    assert update_reqs == [("fn-acs-llama-1b", 0)]


async def test_set_min_containers_notfound_raises_modalopserror(monkeypatch):
    """A missing serve Function surfaces as ModalOpsError, not a raw SDK error."""
    import modal

    class _Stub:
        async def FunctionGet(self, req):
            raise modal.exception.NotFoundError("no such function")

    class _FakeClient:
        stub = _Stub()

    async def fake_from_env():
        return _FakeClient()

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    monkeypatch.setattr(
        modalops._ModalAsyncClient, "from_env", staticmethod(fake_from_env)
    )

    with pytest.raises(modalops.ModalOpsError, match="not found"):
        await modalops.set_min_containers("acs-ghost", 1)


async def test_set_min_containers_times_out_instead_of_hanging(monkeypatch):
    """A hung control-plane RPC must surface as a bounded ModalOpsError rather
    than blocking the request forever (the 499 spinner). Shrink the cap so the
    test is fast."""
    import asyncio

    class _Stub:
        async def FunctionGet(self, req):
            await asyncio.sleep(10)  # simulate the >2min control-plane hang

    class _FakeClient:
        stub = _Stub()

    async def fake_from_env():
        return _FakeClient()

    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    monkeypatch.setattr(modalops, "_AUTOSCALER_UPDATE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(
        modalops._ModalAsyncClient, "from_env", staticmethod(fake_from_env)
    )

    with pytest.raises(modalops.ModalOpsError, match="[Tt]imed out"):
        await modalops.set_min_containers("acs-llama-1b", 0)
