"""Unit + lifecycle tests for the email + password web auth.

Pure tests (hashing, cookie sign/verify) always run. DB-touching lifecycle
tests require ``TEST_DATABASE_URL`` and a migrated Postgres, same pattern as
``test_auth_lifecycle.py``.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import User, UserSession
from wrapper.web_auth import (
    COOKIE_NAME,
    authenticate_user,
    create_user_session,
    generate_temp_password,
    hash_password,
    revoke_user_session,
    set_password,
    sign_session_cookie,
    verify_password,
    verify_session_cookie,
)

SECRET = "test-only-secret-please-do-not-use-in-prod"


# --- pure unit tests -------------------------------------------------------

def test_hash_password_returns_bytes():
    h = hash_password("hunter22-pw")
    assert isinstance(h, bytes)
    # Bcrypt hashes start with $2b$ (or $2a$); the encoded bytes contain that.
    assert h.startswith(b"$2")


def test_verify_password_round_trip():
    h = hash_password("hunter22-pw")
    assert verify_password("hunter22-pw", h) is True
    assert verify_password("hunter3", h) is False


def test_verify_password_rejects_none_hash():
    # Legacy users with no password set — must never authenticate.
    assert verify_password("anything", None) is False


def test_verify_password_rejects_garbage_hash():
    assert verify_password("hunter22-pw", b"not-a-bcrypt-hash") is False


def test_each_hash_is_unique_same_input():
    # Bcrypt salt means same input → different hash, both verify.
    a = hash_password("hunter22-pw")
    b = hash_password("hunter22-pw")
    assert a != b
    assert verify_password("hunter22-pw", a)
    assert verify_password("hunter22-pw", b)


def test_hash_password_rejects_too_short():
    with pytest.raises(ValueError, match="shorter than"):
        hash_password("short")  # 5 chars < 8


def test_hash_password_rejects_too_long():
    with pytest.raises(ValueError, match="exceeds"):
        hash_password("x" * 100)


def test_generate_temp_password_shape():
    pw = generate_temp_password()
    assert isinstance(pw, str)
    assert len(pw) >= 16  # token_urlsafe(16) gives ≥ 22 chars in practice


def test_temp_passwords_are_unique():
    pws = {generate_temp_password() for _ in range(50)}
    assert len(pws) == 50


def test_sign_verify_session_cookie_round_trip():
    sid = uuid.uuid4()
    cookie = sign_session_cookie(sid, SECRET)
    assert verify_session_cookie(cookie, SECRET) == sid


def test_verify_rejects_tampered_cookie():
    cookie = sign_session_cookie(uuid.uuid4(), SECRET)
    tampered = cookie[:-2] + ("AA" if cookie[-2:] != "AA" else "BB")
    assert verify_session_cookie(tampered, SECRET) is None


def test_verify_rejects_wrong_secret():
    cookie = sign_session_cookie(uuid.uuid4(), SECRET)
    assert verify_session_cookie(cookie, "different-secret") is None


def test_verify_rejects_garbage_cookie():
    assert verify_session_cookie("", SECRET) is None
    assert verify_session_cookie("not-even-base64", SECRET) is None


def test_cookie_name_stable():
    # Renaming the cookie invalidates every active session — keep this test as
    # a tripwire so the rename is a deliberate decision.
    assert COOKIE_NAME == "acs_session"


# --- DB lifecycle (gated on TEST_DATABASE_URL) -----------------------------

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run lifecycle tests",
)


@pytest.fixture
def engine():
    eng = make_engine(TEST_DATABASE_URL)
    yield eng
    # Use a fresh loop for teardown — Python 3.12's asyncio.get_event_loop()
    # raises when no loop is current, which happens after pytest-asyncio
    # closes the per-test loop. Cheap workaround: spin up a one-shot loop
    # just for the dispose() coroutine.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(eng.dispose())
    finally:
        loop.close()


@dbtest
async def test_authenticate_user_happy_path(engine):
    factory = make_session_factory(engine)
    email = f"auth-test-{uuid.uuid4().hex[:8]}@example.local"

    async with session_scope(factory) as s:
        user = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(user)
        await s.flush()
        user_id = user.id

    async with session_scope(factory) as s:
        user = await authenticate_user(s, email, "hunter22-pw")
        assert user is not None
        assert user.id == user_id


@dbtest
async def test_authenticate_user_wrong_password(engine):
    factory = make_session_factory(engine)
    email = f"auth-test-{uuid.uuid4().hex[:8]}@example.local"

    async with session_scope(factory) as s:
        s.add(User(email=email, password_hash=hash_password("hunter22-pw")))

    async with session_scope(factory) as s:
        assert await authenticate_user(s, email, "wrong") is None


@dbtest
async def test_authenticate_unknown_email(engine):
    factory = make_session_factory(engine)
    async with session_scope(factory) as s:
        assert await authenticate_user(s, "does-not-exist@nowhere", "anything") is None


@dbtest
async def test_authenticate_legacy_user_without_password(engine):
    factory = make_session_factory(engine)
    email = f"auth-test-{uuid.uuid4().hex[:8]}@example.local"

    async with session_scope(factory) as s:
        s.add(User(email=email))  # no password_hash

    async with session_scope(factory) as s:
        assert await authenticate_user(s, email, "anything") is None


@dbtest
async def test_session_create_lookup_revoke(engine):
    factory = make_session_factory(engine)
    email = f"sess-test-{uuid.uuid4().hex[:8]}@example.local"

    async with session_scope(factory) as s:
        user = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(user)
        await s.flush()
        us = await create_user_session(s, user.id, user_agent="pytest", ip="127.0.0.1")
        sid = us.id

    async with session_scope(factory) as s:
        us = (
            await s.execute(select(UserSession).where(UserSession.id == sid))
        ).scalar_one()
        assert us.revoked_at is None

    async with session_scope(factory) as s:
        await revoke_user_session(s, sid)

    async with session_scope(factory) as s:
        us = (
            await s.execute(select(UserSession).where(UserSession.id == sid))
        ).scalar_one()
        assert us.revoked_at is not None
        first_revoked = us.revoked_at

    # Re-revoke is idempotent.
    async with session_scope(factory) as s:
        await revoke_user_session(s, sid)

    async with session_scope(factory) as s:
        us = (
            await s.execute(select(UserSession).where(UserSession.id == sid))
        ).scalar_one()
        assert us.revoked_at == first_revoked


@dbtest
async def test_set_password_persists(engine):
    factory = make_session_factory(engine)
    email = f"setpw-test-{uuid.uuid4().hex[:8]}@example.local"

    async with session_scope(factory) as s:
        u = User(email=email)  # no password
        s.add(u)
        await s.flush()

    async with session_scope(factory) as s:
        u = (await s.execute(select(User).where(User.email == email))).scalar_one()
        await set_password(s, u, "new-password-1")

    async with session_scope(factory) as s:
        user = await authenticate_user(s, email, "new-password-1")
        assert user is not None
