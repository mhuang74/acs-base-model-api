"""DB-gated tests: bearer auth requires the key's owner to be approved (ACS-212).

Rejecting (or otherwise un-approving) a user used to block only their web
login — any API key they had already minted kept working. The auth layer now
refuses keys whose owner's ``status != 'approved'`` with 401
``account_not_approved``, and ``admin_reject_user`` additionally revokes the
user's live keys (belt-and-braces; see test in ``test_admin_ui.py``).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate as generate_key
from wrapper.models import ApiKey, User

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run auth-status tests",
)


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-status-gate-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
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


def _mint_key_for(status: str) -> str:
    """Create a user with ``status`` holding one live key; return the plaintext."""

    async def _setup() -> str:
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                u = User(
                    email=f"gate-{status}-{uuid.uuid4().hex[:6]}@example.local",
                    status=status,
                )
                s.add(u)
                await s.flush()
                gk = generate_key()
                s.add(
                    ApiKey(
                        user_id=u.id,
                        key_hash=gk.hash_,
                        key_prefix=gk.prefix,
                        monthly_token_budget=0,
                    )
                )
                return gk.plaintext
        finally:
            await engine.dispose()

    return asyncio.run(_setup())


@dbtest
def test_approved_user_key_authenticates(client):
    key = _mint_key_for("approved")
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200, r.text


@dbtest
@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        ("rejected", "account_not_approved"),
        ("pending", "account_not_approved"),
        # Suspension gets its own code + copy (ACS-353): it's reversible and
        # usually scheduled, so "not approved" would read as a verdict.
        ("suspended", "account_suspended"),
    ],
)
def test_unapproved_user_key_is_refused(client, status, expected_code):
    """A live key whose owner isn't approved must 401 regardless of key flags."""
    key = _mint_key_for(status)
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 401, r.text
    # main.py's HTTPException handler unwraps OpenAI-shaped bodies to top level.
    assert r.json()["error"]["code"] == expected_code


@dbtest
@pytest.mark.parametrize("status", ["rejected", "pending", "suspended"])
def test_cookie_side_key_helpers_gate_on_status(client, status):
    """primary_authed_caller / authed_caller_for_key (the workbench/loom
    cookie->key paths, which never touch authenticate()) must also refuse a
    non-approved owner — otherwise a rejected user with a live web session
    could keep generating via the UI (#211 review finding)."""
    import asyncio

    from wrapper import auth as authmod

    async def _check() -> tuple[object, object, object]:
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                u = User(
                    email=f"cookie-{status}-{uuid.uuid4().hex[:6]}@example.local",
                    status=status,
                )
                s.add(u)
                await s.flush()
                gk = generate_key()
                ak = ApiKey(
                    user_id=u.id,
                    key_hash=gk.hash_,
                    key_prefix=gk.prefix,
                    monthly_token_budget=0,
                )
                s.add(ak)
                await s.flush()
                primary = await authmod.primary_authed_caller(s, u.id)
                specific = await authmod.authed_caller_for_key(s, u.id, ak.id)
                return primary, specific, u.id
        finally:
            await engine.dispose()

    primary, specific, _ = asyncio.run(_check())
    assert primary is None
    assert specific is None
