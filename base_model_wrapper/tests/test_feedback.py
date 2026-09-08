"""DB-gated tests for the in-app feedback feature.

Mirrors ``test_admin_ui.py``: skip unless TEST_DATABASE_URL points at a
migrated Postgres. Covers user submission (incl. anonymous + screenshot
validation), admin triage gating, resolve/reopen, and the admin-only
screenshot serve route.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import Feedback, FeedbackScreenshot, User
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run feedback tests",
)

# 1x1 transparent PNG — a real, tiny image so content-type sniffing isn't needed.
_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f8d0000000049454e44ae426082"
)


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-feedback-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("DEFAULT_MONTHLY_TOKEN_BUDGET_TOTAL", "500000")
    os.environ.setdefault("DEFAULT_PER_KEY_BUDGET", "500000")
    os.environ.setdefault("EMAIL_ENABLED", "false")
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


def _ue() -> str:
    return f"fb-{uuid.uuid4().hex[:8]}@example.local"


async def _make_user(
    *, role: str = "user", status: str = "approved", password: str = "test-pw-12345"
) -> tuple[uuid.UUID, str]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(
                email=email, password_hash=hash_password(password), role=role, status=status
            )
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post(
        "/login", data={"email": email, "password": password}, follow_redirects=False
    )
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


async def _latest_feedback() -> Feedback:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            return (
                await s.execute(select(Feedback).order_by(Feedback.created_at.desc()))
            ).scalars().first()
    finally:
        await engine.dispose()


# ---- user submission --------------------------------------------------------

@dbtest
async def test_submit_creates_feedback_row(client):
    uid, email = await _make_user()
    _login(client, email, "test-pw-12345")

    r = client.post(
        "/feedback",
        data={
            "category": "bug",
            "description": "Something is broken on the workbench.",
            "is_anonymous": "false",
            "page_path": "/workbench",
        },
    )
    assert r.status_code == 200, r.text[:300]
    assert r.json() == {"ok": True}

    fb = await _latest_feedback()
    assert fb.category == "bug"
    assert fb.description == "Something is broken on the workbench."
    assert fb.user_id == uid
    assert fb.is_anonymous is False
    assert fb.status == "open"
    assert fb.page_path == "/workbench"
    assert fb.user_agent is not None  # TestClient sends a UA header


@dbtest
async def test_anonymous_submit_stores_null_user(client):
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")

    r = client.post(
        "/feedback",
        data={"category": "general", "description": "Anon note", "is_anonymous": "true"},
    )
    assert r.status_code == 200
    fb = await _latest_feedback()
    assert fb.is_anonymous is True
    assert fb.user_id is None


@dbtest
def test_submit_requires_login(client):
    r = client.post("/feedback", data={"category": "bug", "description": "x"})
    assert r.status_code == 401


@dbtest
async def test_submit_with_screenshot(client):
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    r = client.post(
        "/feedback",
        data={"category": "feature", "description": "with image"},
        files=[("screenshots", ("shot.png", _PNG_1PX, "image/png"))],
    )
    assert r.status_code == 200, r.text[:300]
    fb = await _latest_feedback()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            shots = (
                await s.execute(
                    select(FeedbackScreenshot).where(FeedbackScreenshot.feedback_id == fb.id)
                )
            ).scalars().all()
            assert len(shots) == 1
            assert shots[0].content_type == "image/png"
            assert shots[0].image_bytes == _PNG_1PX
    finally:
        await engine.dispose()


@dbtest
async def test_submit_rejects_bad_category(client):
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    r = client.post("/feedback", data={"category": "spam", "description": "x"})
    assert r.status_code == 400
    assert "error" in r.json()


@dbtest
async def test_submit_rejects_non_image(client):
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    r = client.post(
        "/feedback",
        data={"category": "bug", "description": "x"},
        files=[("screenshots", ("note.txt", b"hello", "text/plain"))],
    )
    assert r.status_code == 400


@dbtest
async def test_submit_rejects_oversize_image(client):
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (2 * 1024 * 1024 + 10)
    r = client.post(
        "/feedback",
        data={"category": "bug", "description": "x"},
        files=[("screenshots", ("big.png", big, "image/png"))],
    )
    assert r.status_code == 400


@dbtest
async def test_submit_rejects_too_many_images(client):
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    files = [
        ("screenshots", (f"s{i}.png", _PNG_1PX, "image/png")) for i in range(6)
    ]
    r = client.post(
        "/feedback", data={"category": "bug", "description": "x"}, files=files
    )
    assert r.status_code == 400


# ---- admin triage gating ----------------------------------------------------

@dbtest
async def test_admin_feedback_blocks_non_admin(client):
    _, email = await _make_user(role="user")
    _login(client, email, "test-pw-12345")
    r = client.get("/admin/feedback", follow_redirects=False)
    assert r.status_code == 403


@dbtest
def test_admin_feedback_requires_login(client):
    r = client.get("/admin/feedback", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


# ---- resolve / reopen -------------------------------------------------------

@dbtest
async def test_resolve_sets_status_and_notes(client):
    _, user_email = await _make_user()
    _login(client, user_email, "test-pw-12345")
    client.post("/feedback", data={"category": "bug", "description": "fix me"})
    fb = await _latest_feedback()
    client.cookies.clear()

    admin_id, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")
    r = client.post(
        f"/admin/feedback/{fb.id}/resolve",
        data={"admin_notes": "fixed in 0011"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = (await s.execute(select(Feedback).where(Feedback.id == fb.id))).scalar_one()
            assert row.status == "resolved"
            assert row.resolved_at is not None
            assert row.resolved_by_user_id == admin_id
            assert row.admin_notes == "fixed in 0011"
    finally:
        await engine.dispose()


@dbtest
async def test_reopen_clears_resolution(client):
    _, user_email = await _make_user()
    _login(client, user_email, "test-pw-12345")
    client.post("/feedback", data={"category": "bug", "description": "fix me"})
    fb = await _latest_feedback()
    client.cookies.clear()

    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")
    client.post(
        f"/admin/feedback/{fb.id}/resolve", data={"admin_notes": "x"}, follow_redirects=False
    )
    r = client.post(f"/admin/feedback/{fb.id}/reopen", follow_redirects=False)
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = (await s.execute(select(Feedback).where(Feedback.id == fb.id))).scalar_one()
            assert row.status == "open"
            assert row.resolved_at is None
            assert row.resolved_by_user_id is None
    finally:
        await engine.dispose()


# ---- screenshot serve -------------------------------------------------------

@dbtest
async def test_screenshot_serve_requires_admin_and_returns_bytes(client):
    _, user_email = await _make_user()
    _login(client, user_email, "test-pw-12345")
    client.post(
        "/feedback",
        data={"category": "bug", "description": "with image"},
        files=[("screenshots", ("shot.png", _PNG_1PX, "image/png"))],
    )
    fb = await _latest_feedback()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            shot = (
                await s.execute(
                    select(FeedbackScreenshot).where(FeedbackScreenshot.feedback_id == fb.id)
                )
            ).scalars().one()
            shot_id = shot.id
    finally:
        await engine.dispose()

    # Non-admin (the submitter) cannot fetch the raw bytes.
    r_user = client.get(f"/admin/feedback/screenshots/{shot_id}", follow_redirects=False)
    assert r_user.status_code == 403
    client.cookies.clear()

    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")
    r_admin = client.get(f"/admin/feedback/screenshots/{shot_id}")
    assert r_admin.status_code == 200
    assert r_admin.content == _PNG_1PX
    assert r_admin.headers["content-type"].startswith("image/png")

    # Missing screenshot → 404.
    r_404 = client.get(f"/admin/feedback/screenshots/{uuid.uuid4()}")
    assert r_404.status_code == 404
