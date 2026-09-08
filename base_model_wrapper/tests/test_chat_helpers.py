"""Tests for the chat-workbench helpers added in item 3 of the MVP plan.

- ``run_completion_nonstream`` (extracted from the old `/v1/completions` body):
  the budget-clamp / upstream-call / usage-commit pipeline.
- ``primary_authed_caller``: build an AuthedCaller from a user's owned key.

Unit tests run always (mocking httpx for the upstream call). The
``primary_authed_caller`` tests are DB-gated since the function exercises real
SQL joins.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from wrapper.auth import AuthedCaller, primary_authed_caller
from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import ApiKey, User
from wrapper.web_auth import hash_password


# --- run_completion_nonstream unit tests -----------------------------------

@pytest.fixture
def fake_caller() -> AuthedCaller:
    return AuthedCaller(
        key_id=uuid.uuid4(),
        key_prefix="testpfx0",
        user_email="test@example.local",
        monthly_token_budget=1_000_000,
        tokens_used_this_month=100,
    )


@pytest.fixture
def fake_settings():
    s = MagicMock()
    s.modal_base_url = "https://upstream.example/"
    s.vllm_api_key = "vllm-test"
    s.upstream_timeout_s = 30.0
    s.served_model_name = "meta-llama/Llama-3.1-405B"
    s.hf_token = None
    s.log_ip = False
    return s


async def test_run_completion_nonstream_happy_path(monkeypatch, fake_caller, fake_settings):
    """Happy path: returns (200, payload), commits usage, records request."""
    from wrapper import main as mainmod

    # Stub the upstream call.
    fake_payload = {
        "id": "cmpl-test",
        "choices": [{"text": " continuation"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
    }
    monkeypatch.setattr(
        mainmod.proxymod, "post_nonstream",
        AsyncMock(return_value=(200, fake_payload, 1234)),
    )
    # Don't monkeypatch clamp_max_tokens — its real return shape (original,
    # clamped) is part of the contract the caller now relies on.

    # Stub the tokenizer.
    counter = MagicMock()
    counter.count = lambda txt: len(txt.split())
    monkeypatch.setattr(mainmod, "get_token_counter", lambda *a, **kw: counter)

    # Stub usage commit + _record_request (no DB).
    monkeypatch.setattr(mainmod.authmod, "commit_usage", AsyncMock())
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    fake_db = MagicMock()
    fake_http = MagicMock()
    status, payload, extras = await mainmod.run_completion_nonstream(
        session=fake_db, http=fake_http, settings=fake_settings,
        caller=fake_caller, body={"prompt": "The capital of France is", "max_tokens": 8},
        ip=None, endpoint="/v1/completions",
    )
    assert status == 200
    assert payload["choices"][0]["text"] == " continuation"
    # Happy path: no clamp applied, extras stays empty.
    assert extras == {}
    mainmod.authmod.commit_usage.assert_awaited_once()
    mainmod._record_request.assert_awaited_once()


async def test_run_completion_nonstream_budget_exhausted(monkeypatch, fake_settings):
    """Caller with remaining_budget=0 short-circuits to 429 without calling upstream."""
    from wrapper import main as mainmod

    caller = AuthedCaller(
        key_id=uuid.uuid4(),
        key_prefix="exhaustd",
        user_email="exhausted@example.local",
        monthly_token_budget=1000,
        tokens_used_this_month=1000,  # remaining = 0
    )
    upstream_mock = AsyncMock()
    monkeypatch.setattr(mainmod.proxymod, "post_nonstream", upstream_mock)
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    fake_db = MagicMock()
    fake_http = MagicMock()
    status, payload, _extras = await mainmod.run_completion_nonstream(
        session=fake_db, http=fake_http, settings=fake_settings,
        caller=caller, body={"prompt": "x"}, ip=None, endpoint="/v1/completions",
    )
    assert status == 429
    assert payload["error"]["code"] == "budget_exceeded"
    upstream_mock.assert_not_called()  # never reached the upstream


async def test_run_completion_nonstream_prompt_alone_exceeds(monkeypatch, fake_settings):
    """If the prompt's token count alone exceeds remaining budget, return 429."""
    from wrapper import main as mainmod

    caller = AuthedCaller(
        key_id=uuid.uuid4(), key_prefix="tightbud", user_email="t@e.l",
        monthly_token_budget=10, tokens_used_this_month=5,  # remaining = 5
    )
    counter = MagicMock()
    counter.count = lambda txt: 100  # any prompt is "100 tokens"
    monkeypatch.setattr(mainmod, "get_token_counter", lambda *a, **kw: counter)
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())
    upstream_mock = AsyncMock()
    monkeypatch.setattr(mainmod.proxymod, "post_nonstream", upstream_mock)

    status, payload, _extras = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=caller, body={"prompt": "long prompt"}, ip=None,
        endpoint="/v1/completions",
    )
    assert status == 429
    assert payload["error"]["code"] == "budget_exceeded"
    upstream_mock.assert_not_called()


async def test_run_completion_nonstream_unlimited_budget_skips_clamp(monkeypatch, fake_settings):
    """Caller with monthly_token_budget=0 (unlimited) skips the tokenizer call."""
    from wrapper import main as mainmod

    caller = AuthedCaller(
        key_id=uuid.uuid4(), key_prefix="unltdtbg", user_email="u@e.l",
        monthly_token_budget=0, tokens_used_this_month=0,  # remaining = None
    )
    counter_calls = []

    def fake_counter(*a, **kw):
        counter_calls.append((a, kw))
        return MagicMock(count=lambda t: 1)

    monkeypatch.setattr(mainmod, "get_token_counter", fake_counter)
    monkeypatch.setattr(mainmod.proxymod, "post_nonstream",
                        AsyncMock(return_value=(200, {"choices": [{"text": "ok"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, 100)))
    monkeypatch.setattr(mainmod.authmod, "commit_usage", AsyncMock())
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status, _payload, _extras = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=caller, body={"prompt": "test"}, ip=None,
        endpoint="/v1/completions",
    )
    assert status == 200
    assert counter_calls == []  # tokenizer not invoked for unlimited budget


# --- primary_authed_caller DB lifecycle tests -------------------------------

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run lifecycle tests",
)


@pytest.fixture
def engine():
    eng = make_engine(TEST_DATABASE_URL)
    yield eng
    # asyncio.get_event_loop() raises in Python 3.14 when no loop is current
    # (pytest-asyncio has already closed the per-test loop by this point).
    # Spin up a one-shot loop to dispose the engine cleanly. Same pattern as
    # the fixture in test_web_auth.py.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(eng.dispose())
    finally:
        loop.close()


@dbtest
async def test_primary_authed_caller_no_user(engine):
    """Unknown user id → None."""
    factory = make_session_factory(engine)
    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, uuid.uuid4())
        assert caller is None


@dbtest
async def test_primary_authed_caller_no_keys(engine):
    """User exists but has no API keys → None."""
    factory = make_session_factory(engine)
    email = f"nokey-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(u)
        await s.flush()
        user_id = u.id

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        assert caller is None


@dbtest
async def test_primary_authed_caller_only_revoked_keys(engine):
    """User has keys but all are revoked → None."""
    import datetime as dt

    factory = make_session_factory(engine)
    email = f"revoked-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(u)
        await s.flush()
        s.add(ApiKey(
            user_id=u.id,
            key_hash=uuid.uuid4().bytes * 2,  # 32 bytes, doesn't matter for this test
            key_prefix="revoked0",
            monthly_token_budget=1000,
            revoked_at=dt.datetime.now(tz=dt.UTC),
        ))
        await s.flush()
        user_id = u.id

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        assert caller is None


@dbtest
async def test_primary_authed_caller_happy_path(engine):
    """User has an active key → AuthedCaller with correct fields, remaining_budget computed."""
    factory = make_session_factory(engine)
    email = f"happy-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(u)
        await s.flush()
        s.add(ApiKey(
            user_id=u.id,
            key_hash=uuid.uuid4().bytes * 2,
            key_prefix="happykey",
            monthly_token_budget=10000,
        ))
        await s.flush()
        user_id = u.id

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        assert caller is not None
        assert caller.user_email == email
        assert caller.key_prefix == "happykey"
        assert caller.monthly_token_budget == 10000
        assert caller.tokens_used_this_month == 0
        assert caller.remaining_budget == 10000


@dbtest
async def test_primary_authed_caller_picks_newest_active(engine):
    """User has two active keys → returns the most recently created one."""
    import time

    factory = make_session_factory(engine)
    email = f"newest-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(u)
        await s.flush()
        s.add(ApiKey(
            user_id=u.id,
            key_hash=uuid.uuid4().bytes * 2,
            key_prefix="oldkey00",
            monthly_token_budget=5000,
        ))
        await s.flush()
        user_id = u.id
    time.sleep(0.05)  # ensure created_at differs
    async with session_scope(factory) as s:
        s.add(ApiKey(
            user_id=user_id,
            key_hash=uuid.uuid4().bytes * 2,
            key_prefix="newkey00",
            monthly_token_budget=20000,
        ))

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        assert caller is not None
        assert caller.key_prefix == "newkey00"
        assert caller.monthly_token_budget == 20000
