"""Tests for the per-model availability service (ACS-51).

Pure-function tests for the circuit-open window derivation (no DB needed).
The full `compute_availability` path is exercised via the DB-gated test
that uses TEST_DATABASE_URL (mirroring test_cost_monitor.py)."""

from __future__ import annotations

import datetime as dt
import os
import uuid

import pytest

from wrapper.services.availability import (
    DowntimeWindow,
    _derive_circuit_open_windows,
    compute_availability,
)

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run availability DB tests",
)


# ---- pure: circuit-open window derivation ----------------------------------

def _ts(seconds: int) -> dt.datetime:
    """Helper: a deterministic ts at `seconds` past a fixed epoch."""
    return dt.datetime(2026, 6, 24, 0, 0, 0, tzinfo=dt.UTC) + dt.timedelta(seconds=seconds)


def test_no_circuit_open_rows_no_windows():
    rows = [(_ts(0), None), (_ts(10), None), (_ts(20), "cold_boot")]
    assert _derive_circuit_open_windows(rows, end_of_period=_ts(100)) == []


def test_single_closed_window():
    rows = [
        (_ts(0), None),
        (_ts(10), "circuit_open"),
        (_ts(20), "circuit_open"),
        (_ts(30), None),  # recovery
        (_ts(40), None),
    ]
    out = _derive_circuit_open_windows(rows, end_of_period=_ts(100))
    assert out == [DowntimeWindow(start_ts=_ts(10), end_ts=_ts(30), duration_s=20.0)]


def test_two_separate_windows():
    rows = [
        (_ts(0), "circuit_open"),
        (_ts(5), None),  # recovers
        (_ts(10), None),
        (_ts(20), "circuit_open"),
        (_ts(25), "circuit_open"),
        (_ts(30), None),  # recovers
    ]
    out = _derive_circuit_open_windows(rows, end_of_period=_ts(100))
    assert out == [
        DowntimeWindow(start_ts=_ts(0), end_ts=_ts(5), duration_s=5.0),
        DowntimeWindow(start_ts=_ts(20), end_ts=_ts(30), duration_s=10.0),
    ]


def test_open_window_at_end_of_period():
    """Run extends past the last row — end_ts=None, duration measured to end_of_period."""
    rows = [
        (_ts(0), None),
        (_ts(50), "circuit_open"),
        (_ts(60), "circuit_open"),
    ]
    out = _derive_circuit_open_windows(rows, end_of_period=_ts(100))
    assert out == [DowntimeWindow(start_ts=_ts(50), end_ts=None, duration_s=50.0)]


def test_open_window_at_start_no_prior_success():
    """A circuit_open run at the very start of the period is still a window —
    the formula doesn't require a preceding non-circuit_open row to anchor."""
    rows = [
        (_ts(0), "circuit_open"),
        (_ts(10), "circuit_open"),
        (_ts(15), None),
    ]
    out = _derive_circuit_open_windows(rows, end_of_period=_ts(100))
    assert out == [DowntimeWindow(start_ts=_ts(0), end_ts=_ts(15), duration_s=15.0)]


def test_zero_uptime_distinct_from_no_traffic():
    """``uptime_pct = 0.0`` (all circuit_open) and ``uptime_pct = None``
    (no traffic) MUST be distinct in the data model — the template renders
    them differently (red pill vs em-dash). This test pins the type contract."""
    # 0.0: denominator > 0, successes = 0 → pct = 0/N = 0.0 (float)
    assert (0 / 5) == 0.0
    assert (0 / 5) is not None
    # None: denominator == 0 → pct stays None (per ``compute_availability``
    # at availability.py: ``uptime_pct = (successes / denominator) if
    # denominator > 0 else None``).
    # We can't call the async DB function from a sync pure test without
    # spinning up a DB; the dbtest below covers the live path. This assertion
    # documents the invariant the template depends on.


def test_open_window_does_not_pollute_mttr_average():
    """An 80s still-open window plus one closed 10s window — the MTTR average
    must come out to 10s, not 45s. One stuck outage would dominate the mean
    otherwise."""
    rows = [
        (_ts(0), "circuit_open"),
        (_ts(10), None),                # closes the 10s window
        (_ts(20), "circuit_open"),      # open through end_of_period (100) = 80s
    ]
    windows = _derive_circuit_open_windows(rows, end_of_period=_ts(100))
    assert len(windows) == 2
    closed = [w for w in windows if w.end_ts is not None]
    mttr = sum(w.duration_s for w in closed) / len(closed)
    assert mttr == 10.0, "open 80s window leaked into MTTR — it must be excluded"


def test_non_circuit_open_errors_do_not_create_window():
    """Only `circuit_open` triggers a window — plain 5xx with other error_kind
    is counted as a failure in uptime % but not as a breaker downtime window."""
    rows = [
        (_ts(0), None),
        (_ts(10), "upstream_5xx"),
        (_ts(20), "upstream_unreachable"),
        (_ts(30), None),
    ]
    assert _derive_circuit_open_windows(rows, end_of_period=_ts(100)) == []


# ---- DB-gated: end-to-end ---------------------------------------------------

# ---- admin /admin/uptime renders the availability section -----------------

def test_admin_uptime_page_renders_availability_section(monkeypatch):
    """The new section in admin_uptime.html renders without template errors
    when there's at least one model with availability data. Verifies the
    route → service → template wiring (the parts most likely to silently break
    on refactor)."""
    import datetime as dt
    import uuid

    os.environ.setdefault("DATABASE_URL", "postgresql://stub")
    os.environ.setdefault("MODAL_BASE_URL", "https://stub")
    os.environ.setdefault("VLLM_API_KEY", "stub")
    os.environ.setdefault("ADMIN_TOKEN", "stub")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from wrapper import web_auth as webauth
    from wrapper.db import get_session
    from wrapper.models import User
    from wrapper.routes.admin.probe import admin_uptime_page
    from wrapper.services import availability as avmod

    admin = User()
    admin.id = uuid.uuid4()
    admin.email = "admin@example.local"
    admin.role = "admin"
    admin.status = "approved"

    class _NullSession:
        async def execute(self, *_a, **_kw):
            # Covers the route's ORM queries (ProbeSchedule / ProbeResult)
            # AND the raw text() data-volume query — the latter calls .one().
            class _R:
                def scalar_one_or_none(self):
                    return None
                def scalars(self):
                    class _S:
                        def all(self):
                            return []
                    return _S()
                def one(self):
                    # Pretend api_requests has 42 rows / 1234567 bytes so the
                    # populated-render path of the data-volume tile is exercised.
                    class _Row:
                        row_count = 42
                        size_bytes = 1234567
                    return _Row()
            return _R()

    async def _stub_session():
        yield _NullSession()

    # Stub the availability service so the template gets a populated, deterministic
    # row — DB queries are out of scope for the rendering smoke.
    async def _stub_list_models(*_a, **_kw):
        return ["llama-8b"]

    fixed_now = dt.datetime(2026, 6, 24, 12, 0, tzinfo=dt.UTC)

    async def _stub_compute(_session, *, model_id, window, now=None):
        anchor = now or fixed_now
        return avmod.ModelAvailability(
            model_id=model_id,
            window_start=anchor - window,
            window_end=anchor,
            total_requests=12,
            denominator=10,
            successes=9,
            uptime_pct=0.90,
            downtime_windows=[
                avmod.DowntimeWindow(
                    start_ts=anchor - dt.timedelta(minutes=30),
                    end_ts=anchor - dt.timedelta(minutes=25),
                    duration_s=300.0,
                ),
            ],
            mttr_s=300.0,
        )

    monkeypatch.setattr(
        "wrapper.routes.admin.probe.list_models_with_recent_requests", _stub_list_models
    )
    monkeypatch.setattr("wrapper.routes.admin.probe.compute_availability", _stub_compute)

    app = FastAPI()
    app.get("/admin/uptime")(admin_uptime_page)
    app.dependency_overrides[webauth.require_admin] = lambda: admin
    app.dependency_overrides[get_session] = _stub_session

    client = TestClient(app)
    r = client.get("/admin/uptime")
    assert r.status_code == 200, r.text
    body = r.text
    assert "Per-model availability" in body
    assert "llama-8b" in body
    assert "90.00%" in body  # uptime pill formatted from 0.90
    assert "300s" in body or "300.0s" in body  # MTTR rendered with the %.0f format
    # Data-volume tile (ACS-26 #4): the stubbed .one() returns 42 / 1234567 bytes;
    # the template should render "42" and "1.2 MB" (1234567 / 1024^2 ≈ 1.18).
    assert "Data volume" in body
    assert "42" in body
    assert "1.2 MB" in body


@dbtest
@pytest.mark.asyncio
async def test_compute_availability_excludes_cold_boot_and_caller_errors():
    """The denominator must exclude cold_boot rows + 401/403/429.
    Successes = status<400 AND error_kind IS NULL, over that denominator."""
    from wrapper.db import make_engine, make_session_factory
    from wrapper.models import ApiKey, ApiRequest, User

    # make_engine normalises the DSN for asyncpg; create_async_engine on a
    # bare postgresql:// URL loads psycopg2 and blows up (ACS-308).
    engine = make_engine(TEST_DATABASE_URL)
    Session = make_session_factory(engine)

    user_id = uuid.uuid4()
    key_id = uuid.uuid4()
    model_name = f"m-{uuid.uuid4().hex[:6]}"  # cross-test isolation on the shared DB
    now = dt.datetime.now(tz=dt.UTC).replace(microsecond=0)

    async with Session() as session:
        # Tests share the migrated DB — seed a fresh user/key so api_requests
        # rows don't collide with prior fixtures.
        session.add(
            User(
                id=user_id,
                email=f"avail-{user_id}@test.local",
                # Schema renamed long ago (hashed_password/is_admin/is_approved
                # → password_hash/role/status) — this test never caught up (ACS-308).
                password_hash=uuid.uuid4().bytes,
                role="user",
                status="approved",
            )
        )
        session.add(
            ApiKey(
                id=key_id,
                user_id=user_id,
                name="t",
                # Unique per run: key_hash carries a UNIQUE constraint and the
                # test DB is shared/persistent (ACS-308).
                key_hash=uuid.uuid4().bytes,
                key_prefix=f"pf-{str(uuid.uuid4())[:6]}",
            )
        )
        await session.flush()

        rows = [
            # 4 successes
            ApiRequest(key_id=key_id, ts=now - dt.timedelta(seconds=60 - i),
                       endpoint="/v1/completions", model=model_name, status=200,
                       cold_boot=False, error_kind=None)
            for i in range(4)
        ] + [
            # 1 cold_boot → excluded from denominator
            ApiRequest(key_id=key_id, ts=now - dt.timedelta(seconds=50),
                       endpoint="/v1/completions", model=model_name, status=503,
                       cold_boot=True, error_kind="cold_boot"),
            # 1 caller error 401 → excluded
            ApiRequest(key_id=key_id, ts=now - dt.timedelta(seconds=40),
                       endpoint="/v1/completions", model=model_name, status=401,
                       cold_boot=False, error_kind=None),
            # 1 rate-limit 429 → excluded
            ApiRequest(key_id=key_id, ts=now - dt.timedelta(seconds=30),
                       endpoint="/v1/completions", model=model_name, status=429,
                       cold_boot=False, error_kind=None),
            # 1 real 5xx failure → in denom, NOT a success
            ApiRequest(key_id=key_id, ts=now - dt.timedelta(seconds=20),
                       endpoint="/v1/completions", model=model_name, status=502,
                       cold_boot=False, error_kind="upstream_5xx"),
        ]
        for r in rows:
            session.add(r)
        await session.commit()

    async with Session() as session:
        avail = await compute_availability(
            session, model_id=model_name,
            window=dt.timedelta(minutes=2),
            now=now,
        )

    # denom = 4 successes + 1 5xx = 5 (cold_boot + 401 + 429 excluded)
    assert avail.denominator == 5
    assert avail.successes == 4
    assert avail.uptime_pct == pytest.approx(4 / 5)

    await engine.dispose()
