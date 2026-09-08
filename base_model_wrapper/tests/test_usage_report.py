"""Tests for the monthly per-model usage report (CSV download, issue #10).

Service seam (``build_usage_report`` / ``usage_report_months`` /
``default_report_month``): row grain, grouping + ordering, the ``NULL``-model
bucket, offered months, default month, revoked-key inclusion, the
tokenized-success filter, and month-window boundaries.

Route seam (TestClient ``GET /usage/report.csv``): auth redirect, attachment
headers + filename, UTF-8 BOM, parsed CSV rows, ``?key=`` filter ignored —
and the /usage page rendering the month picker.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate as generate_key
from wrapper.models import ApiKey, ApiRequest, UsageMonthly, User
from wrapper.services.usage_reports import (
    _month_bounds,
    _parse_month,
    _previous_month_start,
    build_usage_report,
    default_report_month,
)
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run lifecycle tests",
)

CSV_HEADER = [
    "key_name",
    "key_prefix",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "requests",
]


# --- pure unit tests (no DB) -------------------------------------------------


def test_parse_month_accepts_month_values_only():
    assert _parse_month("2026-08") == dt.date(2026, 8, 1)
    assert _parse_month(" 2026-08 ") == dt.date(2026, 8, 1)
    assert _parse_month(dt.date(2026, 8, 1)) == dt.date(2026, 8, 1)
    # Anything not exactly YYYY-MM yields None → callers fall back to the
    # default month instead of erroring (issue #10).
    assert _parse_month("garbage") is None
    assert _parse_month("2026-13") is None
    assert _parse_month("2026-08-01") is None
    assert _parse_month(None) is None


def test_month_bounds_half_open_and_december_wrap():
    start, end = _month_bounds(dt.date(2026, 2, 1))
    assert start == dt.datetime(2026, 2, 1, tzinfo=dt.UTC)
    assert end == dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
    start, end = _month_bounds(dt.date(2026, 12, 1))
    assert end == dt.datetime(2027, 1, 1, tzinfo=dt.UTC)


def test_default_report_month_is_previous_month():
    from wrapper.auth import _current_period_start

    assert default_report_month() == _previous_month_start(_current_period_start())


# --- DB-gated tests ----------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with lifespan started + the same envvars as test_chat_history."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-usage-report-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("RATE_LIMIT_LOGIN_PER_IP", "1000/minute")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


def _ts(month: dt.date, day: int, *, hour: int = 12) -> dt.datetime:
    return dt.datetime(month.year, month.month, day, hour, tzinfo=dt.UTC)


async def _seed_report_user() -> dict[str, object]:
    """One user, two keys, request-log rows across three months.

    key-a: named, active, created first. key-b: unnamed, revoked mid-month,
    created later — its usage must still appear in the report.

    Months relative to now: ``prev`` (the default report month) holds the
    main fixture; ``prev2`` and ``prev3`` prove window boundaries and that
    token-less-only months are not offered.
    """
    prev = default_report_month()
    prev2 = _previous_month_start(prev)
    prev3 = _previous_month_start(prev2)
    current = _month_bounds(prev)[1].date()  # first day of the current month

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            email = f"usage-report-{uuid.uuid4().hex[:8]}@example.local"
            user = User(email=email, password_hash=hash_password("test-pw-12345"))
            s.add(user)
            await s.flush()

            gk_a, gk_b = generate_key(), generate_key()
            key_a = ApiKey(
                user_id=user.id,
                key_hash=gk_a.hash_,
                key_prefix=gk_a.prefix,
                name="key-a",
                created_at=_ts(prev3, 1),
            )
            key_b = ApiKey(
                user_id=user.id,
                key_hash=gk_b.hash_,
                key_prefix=gk_b.prefix,
                name=None,
                created_at=_ts(prev3, 2),
                revoked_at=_ts(prev, 5),
            )
            s.add_all([key_a, key_b])
            await s.flush()

            def req(key, when, model, n_prompt, n_completion, status=200):
                return ApiRequest(
                    key_id=key.id,
                    ts=when,
                    endpoint="/v1/completions",
                    model=model,
                    n_prompt=n_prompt,
                    n_completion=n_completion,
                    status=status,
                )

            s.add_all(
                [
                    # prev month: alpha ties zeta on total → alphabetical order;
                    # the NULL-model row buckets separately.
                    req(key_a, _ts(prev, 10), "alpha-1b", 1000, 800),
                    req(key_a, _ts(prev, 11), "zeta-9b", 700, 500),
                    req(key_a, _ts(prev, 11, hour=13), "zeta-9b", 500, 100),
                    req(key_a, _ts(prev, 12), None, 50, 25),
                    # Exactly the first instant of the month: inside prev's
                    # window, and exactly at prev2's exclusive end.
                    req(key_a, _ts(prev, 1, hour=0), "boundary-model", 10, 10),
                    # Tokenized-success filter: zero-token row contributes nothing.
                    req(key_a, _ts(prev, 13), "excluded-model", 0, 0),
                    # Revoked key's usage still counts.
                    req(key_b, _ts(prev, 14), "gpt2-4b", 10, 5),
                    # Harvest-style token-less row: never contributes.
                    req(key_b, _ts(prev, 14, hour=13), None, None, None, status=202),
                    # Lands in the current (partial) month, not prev.
                    req(key_a, _ts(current, 1, hour=0), "next-month-model", 10, 10),
                    # prev2 window: boundary + aggregation for that month.
                    req(key_a, _ts(prev2, 15), "prev2-model", 20, 20),
                    # prev3: log rows but no rollup → month not offered.
                    req(key_a, _ts(prev3, 20), "stale-model", 999, 999),
                ]
            )
            # Rollups make prev + prev2 "months with usage" for the picker.
            s.add(
                UsageMonthly(
                    key_id=key_a.id,
                    period_start=prev,
                    tokens_prompt=1,
                    tokens_completion=1,
                    request_count=1,
                )
            )
            s.add(
                UsageMonthly(
                    key_id=key_a.id,
                    period_start=prev2,
                    tokens_prompt=1,
                    tokens_completion=1,
                    request_count=1,
                )
            )

        return {
            "email": email,
            "key_a_id": key_a.id,
            "key_b_id": key_b.id,
            "prefix_a": gk_a.prefix,
            "prefix_b": gk_b.prefix,
            "prev": prev,
            "prev2": prev2,
            "prev3": prev3,
            "current": current,
        }
    finally:
        await engine.dispose()


EXPECTED_PREV_ROWS = [
    # key-a rows first (oldest key); alpha before zeta (tie → alphabetical);
    # the NULL-model bucket (75) before boundary-model (20); the zero-token
    # row and the harvest row appear nowhere; revoked key-b still counted.
    ("key-a", "alpha-1b", 1000, 800, 1800, 1),
    ("key-a", "zeta-9b", 1200, 600, 1800, 2),
    ("key-a", None, 50, 25, 75, 1),
    ("key-a", "boundary-model", 10, 10, 20, 1),
    # Unnamed key: the service returns the raw name; the CSV layer renders
    # it "Unnamed key" (asserted in the route test below).
    (None, "gpt2-4b", 10, 5, 15, 1),
]


@dbtest
async def test_build_usage_report_rows_months_and_default():
    """Service contract: row grain, ordering, (unknown) bucket, month list."""
    seed = await _seed_report_user()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            user = (await s.execute(select(User).where(User.email == seed["email"]))).scalar_one()

            report = await build_usage_report(s, user)
            # Default month is the previous month even though the current
            # (partial) month also has usage.

            got = [
                (
                    r["key_name"],
                    r["model"],
                    r["prompt_tokens"],
                    r["completion_tokens"],
                    r["total_tokens"],
                    r["requests"],
                )
                for r in report["rows"]
            ]
            assert got == EXPECTED_PREV_ROWS
            # Prefixes ride along on every row (model attribution per story 7).
            assert all(r["key_prefix"] for r in report["rows"])
    finally:
        await engine.dispose()


@dbtest
async def test_build_usage_report_other_months():
    """prev2 aggregation excludes the prev boundary row and the stale month."""
    seed = await _seed_report_user()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            user = (await s.execute(select(User).where(User.email == seed["email"]))).scalar_one()

            report = await build_usage_report(s, user, month=seed["prev2"])
            assert report["month"] == seed["prev2"]
            assert [
                (r["key_name"], r["model"], r["total_tokens"], r["requests"])
                for r in report["rows"]
            ] == [("key-a", "prev2-model", 40, 1)]

            # The current (partial) month aggregates too.
            report = await build_usage_report(s, user, month=seed["current"])
            assert report["month"] == seed["current"]
            assert [
                (r["key_name"], r["model"], r["total_tokens"], r["requests"])
                for r in report["rows"]
            ] == [("key-a", "next-month-model", 20, 1)]
    finally:
        await engine.dispose()


@dbtest
async def test_build_usage_report_bad_month_falls_back_to_default():
    """Garbage and not-offered months fall back to the default, never 400."""
    seed = await _seed_report_user()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            user = (await s.execute(select(User).where(User.email == seed["email"]))).scalar_one()

            for bad in ("garbage", "1999-01", "2026-13"):
                report = await build_usage_report(s, user, month=bad)
                assert report["month"] == seed["prev"]
    finally:
        await engine.dispose()


@dbtest
async def test_build_usage_report_no_keys_header_only():
    """A user with keys but no usage yields an empty row set (header-only CSV)."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            email = f"usage-report-empty-{uuid.uuid4().hex[:8]}@example.local"
            user = User(email=email, password_hash=hash_password("test-pw-12345"))
            s.add(user)
            await s.flush()
            gk = generate_key()
            s.add(
                ApiKey(
                    user_id=user.id,
                    key_hash=gk.hash_,
                    key_prefix=gk.prefix,
                    name="lonely",
                )
            )
            await s.flush()
            uid = user.id

        async with session_scope(factory) as s:
            user = (await s.execute(select(User).where(User.id == uid))).scalar_one()
            report = await build_usage_report(s, user)
            assert report["month"] == default_report_month()
            assert report["rows"] == []
            assert report["months"] == [
                _month_bounds(default_report_month())[1].date(),
                default_report_month(),
            ]
    finally:
        await engine.dispose()


# --- route tests (TestClient) ------------------------------------------------


@dbtest
async def test_report_requires_login(client):
    r = client.get("/usage/report.csv", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


@dbtest
async def test_report_csv_headers_filename_and_rows(client):
    seed = await _seed_report_user()
    _login(client, seed["email"], "test-pw-12345")

    r = client.get("/usage/report.csv")
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/csv; charset=utf-8"
    fname = f"usage-report-{seed['prev']:%Y-%m}.csv"
    assert r.headers["content-disposition"] == f'attachment; filename="{fname}"'
    # UTF-8 BOM so Excel/Sheets open it without an import dialog.
    assert r.content[:3] == b"\xef\xbb\xbf"

    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0] == CSV_HEADER
    pa, pb = seed["prefix_a"], seed["prefix_b"]
    assert rows[1:] == [
        ["key-a", pa, "alpha-1b", "1000", "800", "1800", "1"],
        ["key-a", pa, "zeta-9b", "1200", "600", "1800", "2"],
        ["key-a", pa, "(unknown)", "50", "25", "75", "1"],
        ["key-a", pa, "boundary-model", "10", "10", "20", "1"],
        ["Unnamed key", pb, "gpt2-4b", "10", "5", "15", "1"],
    ]
    # No grand-total row (issue #10, story 14).
    assert len(rows) == 6


@dbtest
async def test_report_month_param_selects_and_falls_back(client):
    seed = await _seed_report_user()
    _login(client, seed["email"], "test-pw-12345")
    pa = seed["prefix_a"]

    # An offered month is honoured, with its own filename.
    r = client.get(f"/usage/report.csv?month={seed['prev2']:%Y-%m}")
    assert r.headers["content-disposition"] == (
        f'attachment; filename="usage-report-{seed["prev2"]:%Y-%m}.csv"'
    )
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[1:] == [["key-a", pa, "prev2-model", "20", "20", "40", "1"]]

    # Garbage / not-offered months fall back to the default (previous month).
    for bad in ("garbage", "1999-01"):
        r = client.get(f"/usage/report.csv?month={bad}")
        assert r.headers["content-disposition"] == (
            f'attachment; filename="usage-report-{seed["prev"]:%Y-%m}.csv"'
        )
        rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
        assert rows[0] == CSV_HEADER
        assert len(rows) == 6


@dbtest
async def test_report_ignores_page_key_filter(client):
    """The download always covers ALL the user's keys, filter or not."""
    seed = await _seed_report_user()
    _login(client, seed["email"], "test-pw-12345")

    r = client.get(f"/usage/report.csv?key={seed['key_b_id']}")
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    prefixes = {row[1] for row in rows[1:]}
    assert prefixes == {seed["prefix_a"], seed["prefix_b"]}
    # key-a rows are present even though the filter named key-b.
    assert any(row[0] == "key-a" for row in rows[1:])


@dbtest
async def test_usage_page_offers_report_picker(client):
    seed = await _seed_report_user()
    _login(client, seed["email"], "test-pw-12345")

    r = client.get("/usage")
    assert r.status_code == 200
    assert 'action="/usage/report.csv"' in r.text
    # Default month pre-selected; other offered months present as options.
    assert f'value="{seed["prev"]:%Y-%m}" selected' in r.text
    assert f'value="{seed["prev2"]:%Y-%m}"' in r.text
    assert f'value="{seed["current"]:%Y-%m}"' in r.text
    assert "Download CSV" in r.text
