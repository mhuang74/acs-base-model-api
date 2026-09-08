"""DB-gated tests for the /me/* self-service surface (item 5).

Exercises the routes via FastAPI's TestClient against a real Postgres so the
full middleware + cookie + ORM stack runs.

The `client` fixture wraps `TestClient(app)` in a `with` block — that's what
triggers FastAPI's lifespan startup. Without it, `app.state.settings` isn't
populated and every dependency that reads from app.state crashes.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate as generate_key
from wrapper.models import ApiKey, User
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run lifecycle tests",
)


@pytest.fixture
def client():
    """FastAPI TestClient with lifespan started + minimum env vars set."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-self-service-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    # TestClient always uses the same fake IP, which would trip slowapi after
    # 5 login POSTs across the test suite. Bump it generously for tests.
    os.environ.setdefault("RATE_LIMIT_LOGIN_PER_IP", "1000/minute")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    # gpt2 tokenizer is small + ungated, keeps lifespan warmup fast.
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    # HF_TOKEN unset (not empty-string!): empty string makes HF send
    # `Authorization: Bearer ` which httpx 0.28+ rejects with LocalProtocolError.
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    # slowapi's /login limit (5/min) is a literal in the decorator, not read
    # from env. Tests share the testclient IP, so disable rate limiting for
    # the suite to avoid spurious 429s when multiple tests log in.
    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


async def _make_user(password: str = "test-pw-12345") -> tuple[uuid.UUID, str]:
    """Create a user with a password. Returns (user_id, email)."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = f"selfsrv-{uuid.uuid4().hex[:8]}@example.local"
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password))
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


def _scope():
    """Async session scope on a fresh engine (mirrors the per-call pattern used
    throughout this module so each block disposes its own engine)."""
    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)

    class _Ctx:
        async def __aenter__(self):
            self._cm = session_scope(factory)
            return await self._cm.__aenter__()

        async def __aexit__(self, *exc):
            try:
                return await self._cm.__aexit__(*exc)
            finally:
                await engine.dispose()

    return _Ctx()


async def _one_key_id(user_id: uuid.UUID) -> uuid.UUID:
    async with _scope() as s:
        return (
            await s.execute(select(ApiKey.id).where(ApiKey.user_id == user_id))
        ).scalar_one()


async def _make_bob_key(bob_user_id: uuid.UUID) -> uuid.UUID:
    from wrapper.keys import generate as generate_key

    async with _scope() as s:
        gk = generate_key()
        bob_key = ApiKey(
            user_id=bob_user_id, key_hash=gk.hash_, key_prefix=gk.prefix,
            monthly_token_budget=1000,
        )
        s.add(bob_key)
        await s.flush()
        return bob_key.id


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post(
        "/login", data={"email": email, "password": password}, follow_redirects=False
    )
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


# ---- create key -----------------------------------------------------------

@dbtest
async def test_me_create_key_happy_path(client):
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")

    r = client.post(
        "/me/keys",
        data={"name": "laptop", "budget": "5000000"},
        follow_redirects=False,
    )
    assert r.status_code == 200, r.text[:500]
    body = r.text
    assert "acs-bm-" in body
    assert "Copy this key now" in body

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = (
                await s.execute(select(ApiKey).where(ApiKey.user_id == user_id))
            ).scalars().all()
            assert len(rows) == 1
            assert rows[0].name == "laptop"
            assert rows[0].monthly_token_budget == 5000000
    finally:
        await engine.dispose()


@dbtest
def test_me_create_key_requires_login(client):
    r = client.post("/me/keys", data={"name": "x"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


# ---- revoke key -----------------------------------------------------------

@dbtest
async def test_me_revoke_own_key(client):
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")
    client.post("/me/keys", data={"name": "rev-test", "budget": "1000"}, follow_redirects=False)

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            k = (await s.execute(select(ApiKey).where(ApiKey.user_id == user_id))).scalar_one()
            key_id = k.id
    finally:
        await engine.dispose()

    r = client.post(f"/me/keys/{key_id}/revoke", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard"

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            k = (await s.execute(select(ApiKey).where(ApiKey.id == key_id))).scalar_one()
            assert k.revoked_at is not None
    finally:
        await engine.dispose()


@dbtest
async def test_me_revoke_someone_elses_key_returns_404(client):
    """Revoking another user's key must 404 — same shape as a non-existent key,
    so attackers can't enumerate other users' key ids.
    """
    _, alice_email = await _make_user()
    bob_user_id, _ = await _make_user()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            gk = generate_key()
            bob_key = ApiKey(
                user_id=bob_user_id,
                key_hash=gk.hash_,
                key_prefix=gk.prefix,
                monthly_token_budget=1000,
            )
            s.add(bob_key)
            await s.flush()
            bob_key_id = bob_key.id
    finally:
        await engine.dispose()

    _login(client, alice_email, "test-pw-12345")
    r = client.post(f"/me/keys/{bob_key_id}/revoke", follow_redirects=False)
    assert r.status_code == 404

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            k = (await s.execute(select(ApiKey).where(ApiKey.id == bob_key_id))).scalar_one()
            assert k.revoked_at is None  # bob's key untouched
    finally:
        await engine.dispose()


# ---- rename key -----------------------------------------------------------

@dbtest
async def test_me_rename_own_key(client):
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")
    client.post("/me/keys", data={"name": "before", "budget": "1000"}, follow_redirects=False)

    key_id = await _one_key_id(user_id)
    r = client.post(f"/me/keys/{key_id}/rename", data={"name": "after"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard"

    async with _scope() as s:
        k = (await s.execute(select(ApiKey).where(ApiKey.id == key_id))).scalar_one()
        assert k.name == "after"


@dbtest
async def test_me_rename_someone_elses_key_404(client):
    _, alice_email = await _make_user()
    bob_user_id, _ = await _make_user()
    bob_key_id = await _make_bob_key(bob_user_id)

    _login(client, alice_email, "test-pw-12345")
    r = client.post(f"/me/keys/{bob_key_id}/rename", data={"name": "hax"}, follow_redirects=False)
    assert r.status_code == 404
    async with _scope() as s:
        k = (await s.execute(select(ApiKey).where(ApiKey.id == bob_key_id))).scalar_one()
        assert k.name is None  # untouched


# ---- auth-error rendering (regression: ACS-113) ---------------------------

def test_raise_auth_error_survives_partial_scope():
    """``_raise_auth_error`` must render a clean ``HTTPException`` even when the
    request scope is partial (no ``path``).

    Regression pin for ACS-113: the helper used to read ``request.url.path``,
    and building ``request.url`` hard-indexes ``scope['path']`` — so an early
    auth rejection on a scope without ``path`` raised ``KeyError: 'path'`` *while
    rendering the rejection*, turning a 401 into a 500. This test reproduces the
    exact partial scope and asserts a clean 401 (it raised ``KeyError`` before
    the fix). Not DB-gated, so it always runs.
    """
    from fastapi import HTTPException, Request

    from wrapper.auth import _raise_auth_error

    req = Request({"type": "http", "headers": []})  # no "path" — the trigger
    with pytest.raises(HTTPException) as ei:
        _raise_auth_error(
            req,
            status_code=401,
            error_kind="key_disabled",
            message="key is paused",
            code="key_disabled",
        )
    assert ei.value.status_code == 401
    assert ei.value.detail["error"]["code"] == "key_disabled"


# ---- pause / resume -------------------------------------------------------

@dbtest
async def test_me_pause_makes_key_unusable_then_resume_restores(client):
    """A paused key must be rejected on the bearer-auth path (server-side), and
    resuming it must restore access. Mirrors the revoked-key rejection."""
    from fastapi import HTTPException, Request

    from wrapper.auth import authenticate
    from wrapper.keys import generate as generate_key

    user_id, email = await _make_user()
    # Create a known-plaintext key directly so we can present it as a bearer.
    gk = generate_key()
    async with _scope() as s:
        s.add(ApiKey(
            user_id=user_id, key_hash=gk.hash_, key_prefix=gk.prefix,
            name="pause-test", monthly_token_budget=0,
        ))
        await s.flush()
        key_id = (
            await s.execute(select(ApiKey.id).where(ApiKey.user_id == user_id))
        ).scalar_one()

    req = Request({
        "type": "http",
        "method": "POST",
        "path": "/v1/completions",
        "headers": [(b"authorization", f"Bearer {gk.plaintext}".encode())],
    })

    # Usable before pause.
    async with _scope() as s:
        caller = await authenticate(req, s)
        assert caller.key_id == key_id

    # Pause via the route.
    _login(client, email, "test-pw-12345")
    r = client.post(f"/me/keys/{key_id}/pause", follow_redirects=False)
    assert r.status_code == 303
    async with _scope() as s:
        k = (await s.execute(select(ApiKey).where(ApiKey.id == key_id))).scalar_one()
        assert k.disabled_at is not None
        assert k.revoked_at is None  # pause is NOT revoke

    # Rejected while paused.
    async with _scope() as s:
        with pytest.raises(HTTPException) as ei:
            await authenticate(req, s)
        assert ei.value.status_code == 401
        assert ei.value.detail["error"]["code"] == "key_disabled"

    # Resume via the route → usable again.
    r = client.post(f"/me/keys/{key_id}/resume", follow_redirects=False)
    assert r.status_code == 303
    async with _scope() as s:
        k = (await s.execute(select(ApiKey).where(ApiKey.id == key_id))).scalar_one()
        assert k.disabled_at is None
    async with _scope() as s:
        caller = await authenticate(req, s)
        assert caller.key_id == key_id


@dbtest
async def test_me_revoke_still_works_as_trash(client):
    """Revoke (the trash control) remains a permanent soft-delete, distinct from
    pause: it sets revoked_at, not disabled_at."""
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")
    client.post("/me/keys", data={"name": "trash-me", "budget": "1000"}, follow_redirects=False)
    key_id = await _one_key_id(user_id)

    r = client.post(f"/me/keys/{key_id}/revoke", follow_redirects=False)
    assert r.status_code == 303
    async with _scope() as s:
        k = (await s.execute(select(ApiKey).where(ApiKey.id == key_id))).scalar_one()
        assert k.revoked_at is not None
        assert k.disabled_at is None


# ---- per-key budget vs account limit (item 8) -----------------------------

@dbtest
async def test_me_create_key_unbounded_by_default(client):
    """Default create UX is 'Unbounded' → monthly_token_budget == 0 (the legacy
    unlimited sentinel; the account aggregate still governs)."""
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")
    # Mimic the form: the Unbounded checkbox is checked, budget input disabled.
    r = client.post("/me/keys", data={"name": "unb", "unbounded": "true"}, follow_redirects=False)
    assert r.status_code == 200
    async with _scope() as s:
        k = (await s.execute(select(ApiKey).where(ApiKey.user_id == user_id))).scalar_one()
        assert k.monthly_token_budget == 0


@dbtest
async def test_me_create_key_explicit_budget_capped_at_account_limit(client):
    """An explicit per-key budget above the account aggregate is clamped down —
    a per-key cap above the account total is meaningless and must not let a key
    escape the aggregate."""
    user_id, email = await _make_user()
    async with _scope() as s:
        u = (await s.execute(select(User).where(User.id == user_id))).scalar_one()
        u.monthly_token_budget_total = 10_000

    _login(client, email, "test-pw-12345")
    r = client.post(
        "/me/keys",
        data={"name": "capped", "unbounded": "false", "budget": "999999999"},
        follow_redirects=False,
    )
    assert r.status_code == 200
    async with _scope() as s:
        k = (await s.execute(select(ApiKey).where(ApiKey.user_id == user_id))).scalar_one()
        assert k.monthly_token_budget == 10_000  # clamped to account limit


# ---- removed sentences (items 3 & 7) --------------------------------------

@dbtest
async def test_change_password_sentence_removed(client):
    """Item 7: the bcrypt-72-bytes sentence must be gone from the dashboard."""
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    r = client.get("/dashboard")
    assert "bcrypt only sees the first 72 bytes" not in r.text


@dbtest
async def test_key_created_prefix_sentence_removed(client):
    """Item 3: the 'Only its prefix is stored on the server.' sentence must be
    gone from the post-create key screen."""
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    r = client.post("/me/keys", data={"name": "x", "unbounded": "true"}, follow_redirects=False)
    assert r.status_code == 200
    assert "Only its prefix is stored on the server." not in r.text


# ---- change password ------------------------------------------------------

@dbtest
async def test_me_password_happy_path(client):
    _, email = await _make_user(password="old-password-1")
    _login(client, email, "old-password-1")

    r = client.post(
        "/me/password",
        data={
            "current_password": "old-password-1",
            "new_password": "new-password-2",
            "confirm_password": "new-password-2",
        },
        follow_redirects=False,
    )
    assert r.status_code == 200
    assert "Password updated" in r.text

    # Old password no longer works.
    client.cookies.clear()
    r_bad = client.post(
        "/login", data={"email": email, "password": "old-password-1"}, follow_redirects=False
    )
    assert r_bad.status_code == 401

    r_good = client.post(
        "/login", data={"email": email, "password": "new-password-2"}, follow_redirects=False
    )
    assert r_good.status_code == 303


@dbtest
async def test_me_password_wrong_current(client):
    _, email = await _make_user(password="real-password-1")
    _login(client, email, "real-password-1")

    r = client.post(
        "/me/password",
        data={
            "current_password": "wrong-password",
            "new_password": "new-password-2",
            "confirm_password": "new-password-2",
        },
        follow_redirects=False,
    )
    assert r.status_code == 401
    assert "Current password is incorrect" in r.text


@dbtest
async def test_me_password_mismatched_confirm(client):
    _, email = await _make_user(password="orig-password-1")
    _login(client, email, "orig-password-1")

    r = client.post(
        "/me/password",
        data={
            "current_password": "orig-password-1",
            "new_password": "new-password-2",
            "confirm_password": "different-confirm-3",
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    # Jinja auto-escapes the apostrophe in "don't" to "don&#39;t".
    assert "don&#39;t match" in r.text


@dbtest
async def test_me_password_too_short(client):
    _, email = await _make_user(password="orig-password-1")
    _login(client, email, "orig-password-1")

    r = client.post(
        "/me/password",
        data={
            "current_password": "orig-password-1",
            "new_password": "short",
            "confirm_password": "short",
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "shorter" in r.text or "Password" in r.text
