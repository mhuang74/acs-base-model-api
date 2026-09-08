"""Unit tests for the per-model warm-window admin endpoints + scheduler job.

Mirrors ``test_admin_models.py`` and ``test_admin_probe.py``:

- Validation (bad cron / bad timezone) is pure unit; uses the same null-session
  stub the probe tests use, so the route can resolve its session dependency
  even though it short-circuits before any DB write.
- Upsert / delete / re-register / disable run against a sqlite ``model_warm_window``
  table created on the fly (no live Postgres required), with ``modal_ops`` and
  APScheduler patched at the module boundary.
- ``fire_warm`` / ``fire_cool`` are checked against a fake ``modal_ops``.
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

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from wrapper import main as main_module
from wrapper import modal_ops as modalops
from wrapper import warm_window as warmwin
from wrapper import web_auth as webauth
from wrapper.db import get_session
from wrapper.main import (
    admin_models_page,
    admin_warm_window_delete,
    admin_warm_window_upsert,
)
from wrapper.models import ModelWarmWindow, User
from wrapper.settings import ModelEntry, Settings


def _make_admin() -> User:
    u = User()
    u.id = uuid.uuid4()
    u.email = "admin@example.local"
    u.role = "admin"
    u.status = "approved"
    return u


class _FakeScheduler:
    """Captures add_job / remove_job for assertion."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def add_job(self, func, trigger, *, id, replace_existing=True, kwargs=None, **_):
        self.jobs[id] = {"trigger": trigger, "kwargs": kwargs}

    def remove_job(self, job_id):
        if job_id in self.jobs:
            del self.jobs[job_id]
        else:
            raise LookupError(job_id)


def _build_app(*, scheduler: _FakeScheduler | None = None) -> FastAPI:
    """Wire just the warm-window routes + state needed by the handlers."""
    app = FastAPI()
    app.state.settings = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        modal_token_id="tok-id",
        modal_token_secret="tok-secret",
    )
    app.state.models = {
        "trinity-truebase": ModelEntry(
            model_id="trinity-truebase",
            upstream_url="https://trinity.example",
            served_model_name="trinity",
            tokenizer_repo="gpt2",
            gpu_shape_label="8×H200",
            modal_app_name="acs-trinity-base",
        ),
    }
    app.state.scheduler = scheduler
    app.post("/admin/models/{model_id}/warm-window")(admin_warm_window_upsert)
    app.post("/admin/models/{model_id}/warm-window/delete")(admin_warm_window_delete)
    app.dependency_overrides[webauth.require_admin] = _make_admin
    return app


def _create_warm_window_table_sync(conn) -> None:
    ModelWarmWindow.__table__.create(conn, checkfirst=True)


async def _sqlite_session_factory(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/wwindow.db"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(_create_warm_window_table_sync)
    return async_sessionmaker(engine, expire_on_commit=False), engine


def _bind_session(app: FastAPI, factory) -> None:
    async def _dep():
        async with factory() as s:
            yield s
            await s.commit()

    app.dependency_overrides[get_session] = _dep


# ---- GET /admin renders warm windows --------------------------------------

async def test_get_dashboard_renders_warm_windows(monkeypatch, tmp_path):
    """The admin models handler reads ModelWarmWindow rows and attaches them to model_rows.

    Sqlite-backed test: create just the tables the models page reads
    (model_warm_window), seed one warm window, and patch the model-state
    resolver + TemplateResponse so we can inspect the context dict passed to
    the template. (Model controls moved from /admin to /admin/models.)
    """
    db_url = f"sqlite+aiosqlite:///{tmp_path}/dash.db"
    engine = create_async_engine(db_url)

    async with engine.begin() as conn:
        await conn.run_sync(ModelWarmWindow.__table__.create)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            s.add(
                ModelWarmWindow(
                    model_id="trinity-truebase",
                    warm_cron="15 8 * * 3-6,0",
                    cool_cron="0 17 * * 3-6,0",
                    timezone="America/Los_Angeles",
                    enabled=True,
                )
            )
            await s.commit()

        captured: dict = {}

        async def fake_resolve(registry, msg):
            return [
                {
                    "id": "trinity-truebase",
                    "modal_app_name": "acs-trinity-base",
                    "gpu_shape_label": "8×H200",
                    "state": "deployed",
                }
            ]

        def fake_render(request, name, ctx, **_):
            captured["ctx"] = ctx
            from starlette.responses import Response

            return Response("ok")

        monkeypatch.setattr(main_module, "_resolve_model_states", fake_resolve)
        monkeypatch.setattr(main_module.templates, "TemplateResponse", fake_render)

        from starlette.requests import Request

        app = _build_app()
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/admin/models",
            "headers": [],
            "query_string": b"",
            "app": app,
        }
        request = Request(scope)
        async with factory() as session:
            await admin_models_page(
                request=request, admin=_make_admin(), session=session
            )
        ctx = captured["ctx"]
        models = ctx["models"]
        assert len(models) == 1
        assert models[0]["warm_window"] is not None
        assert models[0]["warm_window"].warm_cron == "15 8 * * 3-6,0"
        assert models[0]["warm_window"].enabled is True
    finally:
        await engine.dispose()


# ---- admin gating ----------------------------------------------------------

def test_post_warm_window_requires_admin(monkeypatch):
    """Without an admin override the POST 303s to /login."""
    app = _build_app()
    app.dependency_overrides.pop(webauth.require_admin, None)

    async def _no_session():
        yield None

    app.dependency_overrides[get_session] = _no_session
    client = TestClient(app)
    r = client.post(
        "/admin/models/trinity-truebase/warm-window",
        data={
            "warm_cron": "* * * * *",
            "cool_cron": "* * * * *",
            "timezone": "UTC",
            "enabled": "true",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


# ---- validation ------------------------------------------------------------

async def test_post_warm_window_validates_cron(monkeypatch, tmp_path):
    """Invalid cron → 303 with err= flash; no row written."""
    factory, engine = await _sqlite_session_factory(tmp_path)
    try:
        scheduler = _FakeScheduler()
        app = _build_app(scheduler=scheduler)
        _bind_session(app, factory)
        client = TestClient(app)
        r = client.post(
            "/admin/models/trinity-truebase/warm-window",
            data={
                "warm_cron": "not-a-cron",
                "cool_cron": "0 17 * * 3-6,0",
                "timezone": "America/Los_Angeles",
                "enabled": "true",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "err=" in r.headers["location"]
        assert scheduler.jobs == {}
        async with factory() as s:
            rows = list((await s.execute(select(ModelWarmWindow))).scalars().all())
            assert rows == []
    finally:
        await engine.dispose()


async def test_post_warm_window_validates_timezone(monkeypatch, tmp_path):
    """Invalid timezone → 303 with err= flash; no row written."""
    factory, engine = await _sqlite_session_factory(tmp_path)
    try:
        scheduler = _FakeScheduler()
        app = _build_app(scheduler=scheduler)
        _bind_session(app, factory)
        client = TestClient(app)
        r = client.post(
            "/admin/models/trinity-truebase/warm-window",
            data={
                "warm_cron": "15 8 * * *",
                "cool_cron": "0 17 * * *",
                "timezone": "Mars/Olympus_Mons",
                "enabled": "true",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "err=" in r.headers["location"]
        async with factory() as s:
            rows = list((await s.execute(select(ModelWarmWindow))).scalars().all())
            assert rows == []
    finally:
        await engine.dispose()


# ---- upsert + re-register --------------------------------------------------

async def test_post_warm_window_upserts_and_reregisters(monkeypatch, tmp_path):
    """Valid POST writes the row and calls add_job twice (warm + cool)."""
    factory, engine = await _sqlite_session_factory(tmp_path)
    try:
        scheduler = _FakeScheduler()
        app = _build_app(scheduler=scheduler)
        _bind_session(app, factory)
        client = TestClient(app)
        r = client.post(
            "/admin/models/trinity-truebase/warm-window",
            data={
                "warm_cron": "15 8 * * 3-6,0",
                "cool_cron": "0 17 * * 3-6,0",
                "timezone": "America/Los_Angeles",
                "enabled": "true",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303, r.text
        assert "msg=" in r.headers["location"]
        assert set(scheduler.jobs) == {
            "warm_window:trinity-truebase:warm",
            "warm_window:trinity-truebase:cool",
        }
        async with factory() as s:
            row = (
                await s.execute(
                    select(ModelWarmWindow).where(
                        ModelWarmWindow.model_id == "trinity-truebase"
                    )
                )
            ).scalar_one()
            assert row.warm_cron == "15 8 * * 3-6,0"
            assert row.cool_cron == "0 17 * * 3-6,0"
            assert row.timezone == "America/Los_Angeles"
            assert row.enabled is True
    finally:
        await engine.dispose()


async def test_post_warm_window_disable_unregisters_jobs(monkeypatch, tmp_path):
    """When enabled is omitted (unchecked) the two jobs get removed."""
    factory, engine = await _sqlite_session_factory(tmp_path)
    try:
        scheduler = _FakeScheduler()
        # Pre-populate both jobs and a DB row so the disable path has work to do.
        scheduler.jobs["warm_window:trinity-truebase:warm"] = {"trigger": None, "kwargs": {}}
        scheduler.jobs["warm_window:trinity-truebase:cool"] = {"trigger": None, "kwargs": {}}
        async with factory() as s:
            s.add(
                ModelWarmWindow(
                    model_id="trinity-truebase",
                    warm_cron="15 8 * * *",
                    cool_cron="0 17 * * *",
                    timezone="UTC",
                    enabled=True,
                )
            )
            await s.commit()

        app = _build_app(scheduler=scheduler)
        _bind_session(app, factory)
        client = TestClient(app)
        # Note: no ``enabled`` field in the body (HTML checkbox unchecked).
        r = client.post(
            "/admin/models/trinity-truebase/warm-window",
            data={
                "warm_cron": "15 8 * * *",
                "cool_cron": "0 17 * * *",
                "timezone": "UTC",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303, r.text
        assert scheduler.jobs == {}
        async with factory() as s:
            row = (
                await s.execute(
                    select(ModelWarmWindow).where(
                        ModelWarmWindow.model_id == "trinity-truebase"
                    )
                )
            ).scalar_one()
            assert row.enabled is False
    finally:
        await engine.dispose()


# ---- delete ----------------------------------------------------------------

async def test_delete_warm_window(monkeypatch, tmp_path):
    """POST /delete drops the row and unregisters both jobs."""
    factory, engine = await _sqlite_session_factory(tmp_path)
    try:
        scheduler = _FakeScheduler()
        scheduler.jobs["warm_window:trinity-truebase:warm"] = {"trigger": None, "kwargs": {}}
        scheduler.jobs["warm_window:trinity-truebase:cool"] = {"trigger": None, "kwargs": {}}
        async with factory() as s:
            s.add(
                ModelWarmWindow(
                    model_id="trinity-truebase",
                    warm_cron="15 8 * * *",
                    cool_cron="0 17 * * *",
                    timezone="UTC",
                    enabled=True,
                )
            )
            await s.commit()

        app = _build_app(scheduler=scheduler)
        _bind_session(app, factory)
        client = TestClient(app)
        r = client.post(
            "/admin/models/trinity-truebase/warm-window/delete",
            follow_redirects=False,
        )
        assert r.status_code == 303, r.text
        assert scheduler.jobs == {}
        async with factory() as s:
            rows = list((await s.execute(select(ModelWarmWindow))).scalars().all())
            assert rows == []
    finally:
        await engine.dispose()


# ---- fire_warm / fire_cool --------------------------------------------------

async def _patch_assert_running_ok(monkeypatch):
    """Default: assert_app_running passes — the app is in a runnable state."""

    async def fake_assert(name):
        return None

    monkeypatch.setattr(warmwin.modalops, "assert_app_running", fake_assert)


async def test_fire_warm_calls_set_min_containers_with_1(monkeypatch):
    await _patch_assert_running_ok(monkeypatch)
    calls: list[tuple[str, int]] = []

    async def fake_set_min(name, n):
        calls.append((name, n))

    monkeypatch.setattr(warmwin.modalops, "set_min_containers", fake_set_min)
    await warmwin.fire_warm("trinity-truebase", "acs-trinity-base")
    assert calls == [("acs-trinity-base", 1)]


async def test_fire_cool_calls_set_min_containers_with_0(monkeypatch):
    await _patch_assert_running_ok(monkeypatch)
    calls: list[tuple[str, int]] = []

    async def fake_set_min(name, n):
        calls.append((name, n))

    monkeypatch.setattr(warmwin.modalops, "set_min_containers", fake_set_min)
    await warmwin.fire_cool("trinity-truebase", "acs-trinity-base")
    assert calls == [("acs-trinity-base", 0)]


async def test_fire_warm_swallows_modal_ops_error_no_raise(monkeypatch):
    """A ModalOpsError inside set_min_containers must be swallowed (logged, not raised)."""

    await _patch_assert_running_ok(monkeypatch)

    async def boom(name, n):
        raise modalops.ModalOpsError("modal function not found")

    monkeypatch.setattr(warmwin.modalops, "set_min_containers", boom)
    # No raise — function returns None.
    result = await warmwin.fire_warm("trinity-truebase", "acs-trinity-base")
    assert result is None


async def test_fire_warm_skips_set_min_when_app_stopped(monkeypatch):
    """When the app is stopped, assert_app_running raises and we never call
    set_min_containers — that prevents the fake-success log path where the cron
    cheerfully reported warm_window_fired against a stopped app.
    """

    async def stopped_assert(name):
        raise modalops.ModalOpsError(
            f"Modal app {name!r} is not running (state=stopped)"
        )

    def boom(*_a, **_kw):
        raise AssertionError(
            "set_min_containers must not be called when the app is stopped"
        )

    monkeypatch.setattr(warmwin.modalops, "assert_app_running", stopped_assert)
    monkeypatch.setattr(warmwin.modalops, "set_min_containers", boom)
    # No raise — _apply_min_containers swallows ModalOpsError and logs it.
    result = await warmwin.fire_warm("llama-405b", "acs-llama-405b")
    assert result is None


async def test_fire_warm_unknown_model_logs_warning_no_raise(monkeypatch):
    """If the model isn't in app.state.models, the job wrapper logs and exits.

    Covers the not-in-registry branch of ``_warm_window_job_wrapper``: it must
    never raise, must never call ``modal_ops`` (because we have no app name
    to call it on), and must short-circuit on ``model_id`` not present.
    """

    def boom(*_a, **_kw):
        raise AssertionError("set_min_containers must not be called for unknown model")

    monkeypatch.setattr(warmwin.modalops, "set_min_containers", boom)

    app = _build_app()
    # Empty registry — nothing matches "ghost".
    app.state.models = {}
    # No raise; just exits.
    await main_module._warm_window_job_wrapper(app=app, model_id="ghost", phase="warm")


# ---- registry gating -------------------------------------------------------

def test_post_warm_window_unknown_model_returns_404(monkeypatch, tmp_path):
    """An admin POST for a model_id not in the registry must 404 (mirrors stop / keep-warm)."""
    app = _build_app()
    # No DB needed — the route 404s before touching get_session, but the
    # dependency still needs to resolve, so install a no-op.

    async def _no_session():
        yield None

    app.dependency_overrides[get_session] = _no_session
    client = TestClient(app)
    r = client.post(
        "/admin/models/not-in-registry/warm-window",
        data={
            "warm_cron": "* * * * *",
            "cool_cron": "* * * * *",
            "timezone": "UTC",
            "enabled": "true",
        },
        follow_redirects=False,
    )
    assert r.status_code == 404
