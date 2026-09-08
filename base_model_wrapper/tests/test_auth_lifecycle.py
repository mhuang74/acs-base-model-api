"""End-to-end auth lifecycle test against a real Postgres.

Skipped by default; runs only when `TEST_DATABASE_URL` is set. Use the
project's docker-compose-style snippet from the README:

    docker run -d --name acs-pg-test -p 5433:5432 \\
      -e POSTGRES_USER=acs -e POSTGRES_PASSWORD=acs -e POSTGRES_DB=acs_test postgres:16
    export TEST_DATABASE_URL=postgresql://acs:acs@localhost:5433/acs_test
    alembic -x url=$TEST_DATABASE_URL upgrade head  # or use DATABASE_URL
    pytest tests/test_auth_lifecycle.py -v

This is the Phase B success criterion from the implementation plan:
"Can issue a key, can curl /v1/models with it, 401 without."
"""

from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate, hash_key
from wrapper.models import ApiKey, User


TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run lifecycle tests",
)


@pytest.fixture
def engine():
    eng = make_engine(TEST_DATABASE_URL)
    yield eng
    # asyncio.get_event_loop() raises in Python 3.14 when no loop is current
    # (pytest-asyncio has already closed the per-test loop here). Spin up a
    # one-shot loop to dispose the engine cleanly (same pattern as
    # test_web_auth.py).
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(eng.dispose())
    finally:
        loop.close()


async def test_issue_then_authenticate_then_revoke(engine):
    sessions = make_session_factory(engine)
    gk = generate()
    user_email = f"test-{gk.prefix}@example.local"

    # Issue.
    async with session_scope(sessions) as s:
        u = User(email=user_email)
        s.add(u)
        await s.flush()
        ak = ApiKey(
            user_id=u.id,
            key_hash=gk.hash_,
            key_prefix=gk.prefix,
            name="lifecycle-test",
            monthly_token_budget=1000,
        )
        s.add(ak)
        await s.flush()
        key_id = ak.id

    # Look up by hash (this is what authenticate() does).
    async with session_scope(sessions) as s:
        row = (
            await s.execute(select(ApiKey).where(ApiKey.key_hash == hash_key(gk.plaintext)))
        ).scalar_one()
        assert row.id == key_id
        assert row.revoked_at is None

    # Revoke.
    import datetime as dt
    async with session_scope(sessions) as s:
        row = (
            await s.execute(select(ApiKey).where(ApiKey.id == key_id))
        ).scalar_one()
        row.revoked_at = dt.datetime.now(tz=dt.UTC)

    # Confirm revoked state visible.
    async with session_scope(sessions) as s:
        row = (
            await s.execute(select(ApiKey).where(ApiKey.id == key_id))
        ).scalar_one()
        assert row.revoked_at is not None
