"""Unit + DB-gated tests for the capacity-probe admin endpoints.

The validation slice (cron + redirect on invalid expression) is pure unit so
it always runs. The schedule + trigger slice walks the real DB via the wrapper
TestClient, so it is skipped without ``TEST_DATABASE_URL`` (same convention as
``test_admin_ui.py``).
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

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wrapper.models import User


def _make_admin() -> User:
    u = User()
    u.id = uuid.uuid4()
    u.email = "admin@example.local"
    u.role = "admin"
    u.status = "approved"
    return u


# ---- pure-unit: cron validation rejects nonsense ---------------------------

def test_schedule_rejects_invalid_cron(monkeypatch):
    """The route's croniter.is_valid gate redirects with an error flash, no DB write."""
    from wrapper.main import admin_probe_schedule
    from wrapper import web_auth as webauth

    app = FastAPI()
    app.post("/admin/probe/schedule")(admin_probe_schedule)
    app.dependency_overrides[webauth.require_admin] = _make_admin

    # Stub the session dependency so the route can resolve it (it never runs
    # against the DB on the invalid-cron path because we redirect first).
    class _NullSession:
        async def execute(self, *_a, **_kw):
            class _R:
                def scalar_one_or_none(self):
                    return None

            return _R()

        def add(self, *_a, **_kw):
            pass

    from wrapper.db import get_session

    async def _fake_session():
        yield _NullSession()

    app.dependency_overrides[get_session] = _fake_session

    client = TestClient(app)
    r = client.post(
        "/admin/probe/schedule",
        data={"cron_expression": "not-a-cron"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "err=" in r.headers["location"]


def test_schedule_accepts_valid_cron_expression(monkeypatch):
    """Validate that the croniter check accepts standard 5-field expressions."""
    from croniter import croniter

    assert croniter.is_valid("0 7,10,14,16,19,23 * * *")
    assert croniter.is_valid("*/15 * * * *")
    assert not croniter.is_valid("not-a-cron")


# ---- pure-unit: run_probe writes a row with error when URL is unset --------

async def test_run_probe_logs_error_when_url_unset(tmp_path):
    """Without ``MODAL_PROBE_URL`` set, run_probe must still write a row (ok=False)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from wrapper.models import ProbeResult
    from wrapper.probe import run_probe
    from wrapper.settings import Settings

    db_url = f"sqlite+aiosqlite:///{tmp_path}/probe.db"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(_create_probe_only, ProbeResult.__table__)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    settings = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        modal_probe_url=None,
        modal_probe_bearer=None,
    )
    row = await run_probe(session_factory=factory, settings=settings)
    assert row.ok is False
    assert "MODAL_PROBE_URL not configured" in (row.error or "")

    await engine.dispose()


def _create_probe_only(conn, table):
    """Create only the probe_results table on the sqlite test DB."""
    table.create(conn, checkfirst=True)


# ---- pure-unit: run_probe with a stubbed httpx returns ok ------------------

async def test_run_probe_records_success(tmp_path, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from wrapper.models import ProbeResult
    from wrapper import probe as probemod
    from wrapper.settings import Settings

    db_url = f"sqlite+aiosqlite:///{tmp_path}/probe.db"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(_create_probe_only, ProbeResult.__table__)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "ok": True,
                "gpu_type": "NVIDIA H200",
                "elapsed_s": 1.2,
                "cloud": "aws",
                "region": "us-east-1",
                "gpu_count": 8,
                "gpu_memory_total_mb": 143771.0,
                "gpu_memory_total_std_mb": 0.0,
                "gpu_memory_used_mb": 200.0,
                "gpu_memory_used_std_mb": 100.0,
                "gpu_utilization_pct": 25.0,
                "gpu_utilization_std_pct": 25.0,
                "gpu_temperature_c": 33.0,
                "gpu_temperature_std_c": 2.0,
                "gpu_power_w": 71.0,
                "gpu_power_std_w": 0.5,
                "driver_version": "550.54.15",
            }

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_a, **_kw):
            return _FakeResp()

    monkeypatch.setattr(probemod.httpx, "AsyncClient", _FakeClient)

    settings = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        modal_probe_url="https://probe.example/run",
        modal_probe_bearer="secret",
    )
    row = await probemod.run_probe(session_factory=factory, settings=settings)
    assert row.ok is True
    assert row.gpu_type == "NVIDIA H200"
    assert row.cloud == "aws"
    assert row.region == "us-east-1"
    assert row.gpu_count == 8
    assert row.gpu_memory_total_mb == 143771.0
    assert row.gpu_memory_total_std_mb == 0.0
    assert row.gpu_memory_used_mb == 200.0
    assert row.gpu_memory_used_std_mb == 100.0
    assert row.gpu_utilization_pct == 25.0
    assert row.gpu_utilization_std_pct == 25.0
    assert row.gpu_temperature_c == 33.0
    assert row.gpu_temperature_std_c == 2.0
    assert row.gpu_power_w == 71.0
    assert row.gpu_power_std_w == 0.5
    assert row.driver_version == "550.54.15"
    assert row.error is None

    await engine.dispose()


async def test_run_probe_records_legacy_success_without_metadata(tmp_path, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from wrapper.models import ProbeResult
    from wrapper import probe as probemod
    from wrapper.settings import Settings

    db_url = f"sqlite+aiosqlite:///{tmp_path}/probe.db"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(_create_probe_only, ProbeResult.__table__)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "gpu_type": "NVIDIA H200", "elapsed_s": 1.2}

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_a, **_kw):
            return _FakeResp()

    monkeypatch.setattr(probemod.httpx, "AsyncClient", _FakeClient)

    settings = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        modal_probe_url="https://probe.example/run",
        modal_probe_bearer="secret",
    )
    row = await probemod.run_probe(session_factory=factory, settings=settings)
    assert row.ok is True
    assert row.gpu_type == "NVIDIA H200"
    assert row.cloud is None
    assert row.region is None
    assert row.gpu_utilization_pct is None
    assert row.driver_version is None
    assert row.error is None

    await engine.dispose()


# ---- DB-gated: full /admin/probe/* round-trip ------------------------------

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run probe endpoint tests",
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-admin-probe-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("EMAIL_ENABLED", "false")
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    # Disable rate limiter and skip APScheduler so tests don't fire real probes.
    app.state.limiter.enabled = False
    app.state.disable_scheduler = True
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


async def _make_user(*, role: str = "user", status: str = "approved", password: str = "test-pw-12345"):
    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.models import User
    from wrapper.web_auth import hash_password

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = f"probe-{uuid.uuid4().hex[:8]}@example.local"
        async with session_scope(factory) as s:
            u = User(
                email=email,
                password_hash=hash_password(password),
                role=role,
                status=status,
            )
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str = "test-pw-12345"):
    r = client.post(
        "/login", data={"email": email, "password": password}, follow_redirects=False
    )
    assert r.status_code == 303, r.text[:500]


@dbtest
def test_schedule_requires_admin(client):
    r = client.post(
        "/admin/probe/schedule",
        data={"cron_expression": "* * * * *"},
        follow_redirects=False,
    )
    # 303 to /login when no cookie at all.
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


@dbtest
async def test_schedule_updates_row_and_reschedules_job(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.post(
        "/admin/probe/schedule",
        data={"cron_expression": "*/30 * * * *"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text[:500]
    assert "err=" not in r.headers["location"]

    from sqlalchemy import select
    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.models import ProbeSchedule

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = (
                await s.execute(select(ProbeSchedule).where(ProbeSchedule.id == 1))
            ).scalar_one()
            assert row.cron_expression == "*/30 * * * *"
    finally:
        await engine.dispose()


@dbtest
async def test_admin_landing_renders_menu(client):
    """GET /admin renders the admin menu linking to every subpage (no probe UI)."""
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.get("/admin")
    assert r.status_code == 200, r.text[:500]
    body = r.text
    for href in (
        "/admin/users",
        "/admin/models",
        "/admin/uptime",
        "/admin/feedback",
        "/admin/emails",
    ):
        assert f'href="{href}"' in body, f"missing link to {href}"
    # The probe UI (schedule/trigger forms) moved off /admin onto /admin/uptime.
    assert 'action="/admin/probe/schedule"' not in body
    assert 'action="/admin/probe/trigger"' not in body


@dbtest
async def test_uptime_page_renders_probe_ui(client):
    """GET /admin/uptime renders the capacity-probe controls."""
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.get("/admin/uptime")
    assert r.status_code == 200, r.text[:500]
    body = r.text
    assert "Capacity probe" in body
    assert 'action="/admin/probe/schedule"' in body
    assert 'action="/admin/probe/trigger"' in body


@dbtest
async def test_uptime_page_blocks_non_admin(client):
    """GET /admin/uptime is admin-gated (403 for a non-admin user)."""
    _, email = await _make_user(role="user")
    _login(client, email)

    r = client.get("/admin/uptime", follow_redirects=False)
    assert r.status_code == 403


@dbtest
def test_uptime_page_requires_login(client):
    """GET /admin/uptime redirects to /login when unauthenticated."""
    r = client.get("/admin/uptime", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


@dbtest
async def test_trigger_writes_probe_result(client, monkeypatch):
    from wrapper import probe as probemod
    from wrapper.models import ProbeResult
    from sqlalchemy import select
    from wrapper.db import make_engine, make_session_factory, session_scope

    async def fake_run_probe(*, session_factory, settings):
        async with session_factory() as s:
            row = ProbeResult(
                ok=True,
                elapsed_s=0.5,
                gpu_type="fake",
                cloud="aws",
                region="us-east-1",
                error=None,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return row

    monkeypatch.setattr(probemod, "run_probe", fake_run_probe)

    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.post("/admin/probe/trigger", follow_redirects=False)
    assert r.status_code == 303

    # asyncio.create_task scheduled it; let it run.
    import asyncio
    await asyncio.sleep(0.2)

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = (
                await s.execute(
                    select(ProbeResult).order_by(ProbeResult.fired_at.desc()).limit(1)
                )
            ).scalars().all()
            assert any(r.gpu_type == "fake" for r in rows)
    finally:
        await engine.dispose()
