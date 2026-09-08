"""DB-gated tests for the self-service password-reset flow.

Mirrors ``test_admin_emails.py`` / ``test_feedback.py``: skip unless
TEST_DATABASE_URL points at a migrated Postgres (must reach 0013). Covers:

- forgot-password for a real email creates a PasswordReset row AND records an
  EmailLog with kind='password_reset' whose body does NOT contain the token
  (redaction works);
- forgot-password for an unknown email returns the SAME neutral response and
  creates NO row (no account enumeration);
- a valid token sets the new password (user can then log in with it);
- an expired token is rejected;
- an already-used token is rejected (single-use);
- the login page contains the "Forgot password?" link.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import EmailLog, PasswordReset, User
from wrapper.web_auth import hash_password, hash_reset_token

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run password-reset tests",
)


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-pwreset-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("DEFAULT_MONTHLY_TOKEN_BUDGET_TOTAL", "500000")
    os.environ.setdefault("DEFAULT_PER_KEY_BUDGET", "500000")
    # Email disabled — the flow still records a 'skipped' EmailLog row (audit).
    os.environ.setdefault("EMAIL_ENABLED", "false")
    os.environ.setdefault("PUBLIC_BASE_URL", "https://app.example.test")
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


def _ue() -> str:
    return f"pwr-{uuid.uuid4().hex[:8]}@example.local"


async def _make_user(*, password: str = "test-pw-12345") -> tuple[uuid.UUID, str]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password), status="approved")
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


async def _make_passwordless_user() -> tuple[uuid.UUID, str]:
    """An approved account that never set a password (e.g. admin-created)."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=None, status="approved")
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


async def _reset_rows(user_id: uuid.UUID) -> list[PasswordReset]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            return list(
                (
                    await s.execute(
                        select(PasswordReset).where(PasswordReset.user_id == user_id)
                    )
                ).scalars().all()
            )
    finally:
        await engine.dispose()


async def _insert_reset_token(
    user_id: uuid.UUID, token: str, *, expires_in_minutes: int = 60, used: bool = False
) -> None:
    """Insert a PasswordReset directly so tests can control expiry / used state."""
    now = dt.datetime.now(tz=dt.UTC)
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            s.add(
                PasswordReset(
                    user_id=user_id,
                    token_hash=hash_reset_token(token),
                    expires_at=now + dt.timedelta(minutes=expires_in_minutes),
                    used_at=now if used else None,
                )
            )
    finally:
        await engine.dispose()


# ---- login page link --------------------------------------------------------

@dbtest
def test_login_page_has_forgot_link(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert "/forgot-password" in r.text
    assert "Forgot password?" in r.text


# ---- forgot-password: real email --------------------------------------------

@dbtest
async def test_forgot_password_real_email_creates_row_and_redacted_log(client):
    user_id, email = await _make_user()

    r = client.post("/forgot-password", data={"email": email})
    assert r.status_code == 200
    assert "If an account exists for that email" in r.text

    rows = await _reset_rows(user_id)
    assert len(rows) == 1
    pr = rows[0]
    assert pr.used_at is None
    assert pr.expires_at > dt.datetime.now(tz=dt.UTC)

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            elog = (
                await s.execute(
                    select(EmailLog).where(
                        EmailLog.user_id == user_id, EmailLog.kind == "password_reset"
                    )
                )
            ).scalars().one()
            # Email disabled in tests → suppressed but still logged.
            assert elog.send_status == "skipped"
            # Redaction: the persisted body must NOT contain a live reset link.
            assert "/reset-password/" not in elog.body_html
            assert "/reset-password/" not in elog.body_text
            assert "redacted" in elog.body_html.lower()
    finally:
        await engine.dispose()


# ---- forgot-password: unknown email (no enumeration) ------------------------

@dbtest
def test_forgot_password_unknown_email_same_response_no_row(client):
    unknown = _ue()
    r = client.post("/forgot-password", data={"email": unknown})
    assert r.status_code == 200
    # Identical neutral copy as the real-email case.
    assert "If an account exists for that email" in r.text

    # And it created nothing — assert no EmailLog row for that recipient.
    import anyio

    async def _check() -> int:
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                rows = (
                    await s.execute(
                        select(EmailLog).where(EmailLog.to_email == unknown)
                    )
                ).scalars().all()
                return len(rows)
        finally:
            await engine.dispose()

    assert anyio.run(_check) == 0


# ---- valid token resets the password ----------------------------------------

@dbtest
async def test_reset_with_valid_token_sets_new_password(client):
    user_id, email = await _make_user(password="old-pw-12345")
    token = "valid-token-" + uuid.uuid4().hex
    await _insert_reset_token(user_id, token)

    # GET shows the form (not the invalid page).
    r_form = client.get(f"/reset-password/{token}")
    assert r_form.status_code == 200
    assert "Set a new password" in r_form.text

    new_pw = "brand-new-pw-9876"
    r = client.post(
        f"/reset-password/{token}",
        data={"password": new_pw, "confirm_password": new_pw},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")

    # New password works; old one doesn't.
    r_new = client.post(
        "/login", data={"email": email, "password": new_pw}, follow_redirects=False
    )
    assert r_new.status_code == 303
    r_old = client.post(
        "/login", data={"email": email, "password": "old-pw-12345"}, follow_redirects=False
    )
    assert r_old.status_code == 401


# ---- expired token rejected -------------------------------------------------

@dbtest
async def test_reset_with_expired_token_rejected(client):
    user_id, _ = await _make_user()
    token = "expired-token-" + uuid.uuid4().hex
    await _insert_reset_token(user_id, token, expires_in_minutes=-5)

    r_get = client.get(f"/reset-password/{token}")
    assert r_get.status_code == 400
    assert "invalid or expired" in r_get.text.lower()

    r_post = client.post(
        f"/reset-password/{token}",
        data={"password": "another-pw-12345", "confirm_password": "another-pw-12345"},
        follow_redirects=False,
    )
    assert r_post.status_code == 400


# ---- already-used token rejected (single-use) -------------------------------

@dbtest
async def test_reset_with_used_token_rejected(client):
    user_id, _ = await _make_user()
    token = "used-token-" + uuid.uuid4().hex
    await _insert_reset_token(user_id, token, used=True)

    r_get = client.get(f"/reset-password/{token}")
    assert r_get.status_code == 400

    r_post = client.post(
        f"/reset-password/{token}",
        data={"password": "another-pw-12345", "confirm_password": "another-pw-12345"},
        follow_redirects=False,
    )
    assert r_post.status_code == 400


# ---- first-time set-password copy (ACS-99) ----------------------------------

@dbtest
async def test_first_time_set_password_shows_set_copy(client):
    """A token whose user has no password yet is an initial set-password, not a
    reset, so the page shows 'Set your password' copy (ACS-99)."""
    user_id, _ = await _make_passwordless_user()
    token = "firsttime-" + uuid.uuid4().hex
    await _insert_reset_token(user_id, token)

    r = client.get(f"/reset-password/{token}")
    assert r.status_code == 200
    assert "Set your password" in r.text
    assert "Welcome" in r.text
    # The "reset" framing must not appear for a first-time set.
    assert "Choose a new password for your account" not in r.text


@dbtest
async def test_first_time_set_password_completes_and_redirects(client):
    """The first-time flow actually sets the password and the success redirect
    uses 'Password set' (not 'reset') wording (ACS-99)."""
    user_id, email = await _make_passwordless_user()
    token = "firsttime2-" + uuid.uuid4().hex
    await _insert_reset_token(user_id, token)

    new_pw = "first-password-12345"
    r = client.post(
        f"/reset-password/{token}",
        data={"password": new_pw, "confirm_password": new_pw},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "Password+set" in r.headers["location"]

    # The just-set password works for login.
    r_login = client.post(
        "/login", data={"email": email, "password": new_pw}, follow_redirects=False
    )
    assert r_login.status_code == 303


@dbtest
async def test_reset_for_existing_password_keeps_reset_copy(client):
    """A user who already has a password sees the normal 'reset' copy."""
    user_id, _ = await _make_user()
    token = "resetcopy-" + uuid.uuid4().hex
    await _insert_reset_token(user_id, token)

    r = client.get(f"/reset-password/{token}")
    assert r.status_code == 200
    assert "Choose a new password for your account" in r.text
