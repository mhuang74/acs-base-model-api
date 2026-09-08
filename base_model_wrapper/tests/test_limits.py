"""Tests for the item-9 limits backend.

Pure unit tests cover the new ``effective_remaining`` logic + the
multi-dimensional clamp in ``run_completion_nonstream``. DB-gated tests
exercise per-key daily ceilings, user-aggregate enforcement across two keys,
the I/O split, the ``usage_daily`` write, and full backward compat for
legacy (all-NULL) keys.

Same gating pattern as ``tests/test_web_auth.py``: set ``TEST_DATABASE_URL``
to a migrated Postgres to run the DB tests. The migration must include
``0003_add_limits_backend``.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from wrapper.auth import (
    AuthedCaller,
    _current_day_start,
    _current_period_start,
    authenticate,
    commit_usage,
    primary_authed_caller,
)
from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import ApiKey, UsageDaily, UsageMonthly, User
from wrapper.web_auth import hash_password


# =============================================================================
# Pure unit tests: AuthedCaller.effective_remaining()
# =============================================================================


def _caller(**overrides):
    """Build an AuthedCaller with sensible defaults; override what each test needs."""
    base = dict(
        key_id=uuid.uuid4(),
        key_prefix="testpfx0",
        user_email="t@example.local",
        monthly_token_budget=0,  # legacy: 0 = unlimited
        tokens_used_this_month=0,
    )
    base.update(overrides)
    return AuthedCaller(**base)


def test_effective_remaining_all_null_is_fully_unlimited():
    """Legacy key with monthly_token_budget=0 + every new column NULL → every
    dimension is None. This is the backward-compat path that must keep the
    benchmark key (and every existing key) working unchanged."""
    c = _caller()
    rem = c.effective_remaining()
    assert rem == {
        "key_monthly": None,
        "user_monthly": None,
        "daily": None,
        "input": None,
        "output": None,
    }
    assert c.is_unlimited() is True


def test_effective_remaining_legacy_monthly_only():
    """Pre-item-9 key with a monthly cap and no other limits."""
    c = _caller(monthly_token_budget=10_000, tokens_used_this_month=2_500)
    rem = c.effective_remaining()
    assert rem["key_monthly"] == 7_500
    assert rem["user_monthly"] is None
    assert rem["daily"] is None
    assert rem["input"] is None
    assert rem["output"] is None
    assert c.is_unlimited() is False


def test_effective_remaining_daily_only():
    c = _caller(daily_token_budget=1_000, tokens_used_today=300)
    assert c.effective_remaining()["daily"] == 700
    # monthly unlimited (legacy 0) still reads as None.
    assert c.effective_remaining()["key_monthly"] is None


def test_effective_remaining_user_aggregate():
    c = _caller(
        monthly_token_budget_total=100_000,
        user_tokens_used_this_month=40_000,
    )
    assert c.effective_remaining()["user_monthly"] == 60_000


def test_effective_remaining_input_output_split():
    c = _caller(
        monthly_input_token_budget=5_000,
        monthly_output_token_budget=2_000,
        input_tokens_used_this_month=1_000,
        output_tokens_used_this_month=500,
    )
    rem = c.effective_remaining()
    assert rem["input"] == 4_000
    assert rem["output"] == 1_500


def test_effective_remaining_clamps_at_zero_never_negative():
    """Once usage exceeds a cap (which can happen if a single request blows
    through the budget), remaining must read 0 rather than a negative — the
    hot-path comparison ``rem <= 0`` then short-circuits to 429."""
    c = _caller(monthly_token_budget=100, tokens_used_this_month=500)
    assert c.effective_remaining()["key_monthly"] == 0


def test_legacy_remaining_budget_property_still_works():
    """The old ``caller.remaining_budget`` property is kept as a back-compat
    alias for the per-key monthly dimension. Removing it would break older
    test fixtures + any inline callers."""
    c = _caller(monthly_token_budget=1_000, tokens_used_this_month=400)
    assert c.remaining_budget == 600
    c2 = _caller()  # legacy unlimited
    assert c2.remaining_budget is None


# =============================================================================
# Pure unit tests: multi-dimensional clamp in run_completion_nonstream
# =============================================================================


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


async def _run(caller, body, fake_settings, monkeypatch, *, upstream_payload=None):
    """Stub upstream/tokenizer/usage-commit and call run_completion_nonstream."""
    from wrapper import main as mainmod

    if upstream_payload is None:
        upstream_payload = {
            "id": "cmpl-test",
            "choices": [{"text": " ok"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
    monkeypatch.setattr(
        mainmod.proxymod, "post_nonstream",
        AsyncMock(return_value=(200, upstream_payload, 100)),
    )
    counter = MagicMock()
    # Default: 1 token per whitespace-split word; tests can monkeypatch
    # further if they want something specific.
    counter.count = lambda txt: len(str(txt).split())
    monkeypatch.setattr(mainmod, "get_token_counter", lambda *a, **kw: counter)
    monkeypatch.setattr(mainmod.authmod, "commit_usage", AsyncMock())
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    return await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=caller, body=body, ip=None, endpoint="/v1/completions",
    )


async def test_daily_exhausted_returns_429(monkeypatch, fake_settings):
    """Daily budget at 0 → 429 even if monthly has headroom."""
    caller = _caller(
        monthly_token_budget=10_000, tokens_used_this_month=100,
        daily_token_budget=1_000, tokens_used_today=1_000,
    )
    status_code, payload, _ = await _run(
        caller, {"prompt": "hello world", "max_tokens": 50}, fake_settings, monkeypatch
    )
    assert status_code == 429
    assert payload["error"]["code"] == "budget_exceeded"
    assert "daily" in payload["error"]["message"].lower()


async def test_user_aggregate_exhausted_returns_429(monkeypatch, fake_settings):
    caller = _caller(
        monthly_token_budget=0,  # per-key unlimited
        monthly_token_budget_total=5_000,
        user_tokens_used_this_month=5_000,
    )
    status_code, payload, _ = await _run(
        caller, {"prompt": "hello", "max_tokens": 10}, fake_settings, monkeypatch
    )
    assert status_code == 429
    assert payload["error"]["code"] == "budget_exceeded"
    assert "user" in payload["error"]["message"].lower()


async def test_output_budget_exhausted_returns_429(monkeypatch, fake_settings):
    caller = _caller(
        monthly_output_token_budget=100, output_tokens_used_this_month=100,
    )
    status_code, payload, _ = await _run(
        caller, {"prompt": "hi", "max_tokens": 10}, fake_settings, monkeypatch
    )
    assert status_code == 429
    assert "output" in payload["error"]["message"].lower()


async def test_input_budget_rejects_when_prompt_alone_too_big(monkeypatch, fake_settings):
    """The input-budget dimension is special: prompt tokens count directly
    against it (no headroom subtraction). If the prompt alone is bigger than
    remaining input budget, reject — there's no way to clamp ``max_tokens``
    that helps."""
    caller = _caller(
        monthly_input_token_budget=5, input_tokens_used_this_month=0,
    )
    status_code, payload, _ = await _run(
        caller,
        {"prompt": "this prompt has more than five tokens for sure", "max_tokens": 10},
        fake_settings, monkeypatch,
    )
    assert status_code == 429
    assert payload["error"]["code"] == "budget_exceeded"


async def test_clamp_uses_minimum_of_all_dimensions(monkeypatch, fake_settings):
    """Final ``max_tokens`` clamp is min of every output-side headroom.

    Per-key monthly: 100 remaining − 2 prompt = 98 headroom.
    Daily: 50 remaining − 2 prompt = 48 headroom.
    Output: 30 remaining (prompt doesn't count) = 30 headroom.
    Asked for 200. → clamped to 30.
    """
    from wrapper import main as mainmod

    caller = _caller(
        monthly_token_budget=100, tokens_used_this_month=0,
        daily_token_budget=50, tokens_used_today=0,
        monthly_output_token_budget=30, output_tokens_used_this_month=0,
    )

    seen_body: dict = {}

    async def fake_post(client, url, key, body, timeout, *, ctx=None):
        seen_body.update(body)
        return 200, {"choices": [{"text": "x"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}, 50

    monkeypatch.setattr(mainmod.proxymod, "post_nonstream", fake_post)
    monkeypatch.setattr(mainmod, "get_token_counter",
                        lambda *a, **kw: MagicMock(count=lambda t: 2))
    monkeypatch.setattr(mainmod.authmod, "commit_usage", AsyncMock())
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status_code, _payload, extras = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=caller,
        body={"prompt": "two words", "max_tokens": 200},
        ip=None, endpoint="/v1/completions",
    )
    assert status_code == 200
    assert seen_body["max_tokens"] == 30  # output cap was the tightest
    # The clamp must surface in extras so the public handler can set the
    # X-Acs-Max-Tokens-Clamped response header.
    assert extras["max_tokens_clamped"] == {
        "requested": 200,
        "applied": 30,
        "reason": "budget",
    }


async def test_unlimited_caller_skips_tokenizer_completely(monkeypatch, fake_settings):
    """Backward compat: a caller with every dimension None still skips the
    tokenizer call entirely (matches the existing
    ``test_run_completion_nonstream_unlimited_budget_skips_clamp``)."""
    from wrapper import main as mainmod

    caller = _caller()  # all defaults: nothing capped
    calls: list = []

    def fake_counter(*a, **kw):
        calls.append((a, kw))
        return MagicMock(count=lambda t: 1)

    monkeypatch.setattr(mainmod, "get_token_counter", fake_counter)
    monkeypatch.setattr(mainmod.proxymod, "post_nonstream", AsyncMock(
        return_value=(200, {"choices": [{"text": "x"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, 10),
    ))
    monkeypatch.setattr(mainmod.authmod, "commit_usage", AsyncMock())
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status, _payload, _extras = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=caller, body={"prompt": "hi"}, ip=None, endpoint="/v1/completions",
    )
    assert status == 200
    assert calls == []  # tokenizer never called


# =============================================================================
# DB-gated tests: real Postgres with the 0003 migration applied.
# =============================================================================

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
    # (pytest-asyncio has already closed the per-test loop here). Spin up a
    # one-shot loop to dispose the engine cleanly (same pattern as
    # test_web_auth.py).
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(eng.dispose())
    finally:
        loop.close()


def _fake_key_hash() -> bytes:
    """Deterministic 32-byte key_hash placeholder unique per test row."""
    return uuid.uuid4().bytes * 2  # 32 bytes


@dbtest
async def test_db_usage_daily_written_on_commit(engine):
    """Every successful ``commit_usage`` must land a row in usage_daily as
    well as usage_monthly. Without this the daily ceiling enforcement reads
    zero forever."""
    factory = make_session_factory(engine)
    email = f"daily-write-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(u)
        await s.flush()
        ak = ApiKey(
            user_id=u.id,
            key_hash=_fake_key_hash(),
            key_prefix="daily000",
            monthly_token_budget=0,
        )
        s.add(ak)
        await s.flush()
        key_id = ak.id

    async with session_scope(factory) as s:
        await commit_usage(s, key_id, prompt_tokens=10, completion_tokens=20)

    async with session_scope(factory) as s:
        daily = (
            await s.execute(
                select(UsageDaily).where(
                    UsageDaily.key_id == key_id,
                    UsageDaily.period_start == _current_day_start(),
                )
            )
        ).scalar_one()
        assert daily.tokens_prompt == 10
        assert daily.tokens_completion == 20
        assert daily.request_count == 1

        monthly = (
            await s.execute(
                select(UsageMonthly).where(
                    UsageMonthly.key_id == key_id,
                    UsageMonthly.period_start == _current_period_start(),
                )
            )
        ).scalar_one()
        assert monthly.tokens_prompt == 10
        assert monthly.tokens_completion == 20

    # And UPSERT: a second commit increments both rows atomically.
    async with session_scope(factory) as s:
        await commit_usage(s, key_id, prompt_tokens=5, completion_tokens=7)

    async with session_scope(factory) as s:
        daily = (
            await s.execute(
                select(UsageDaily).where(UsageDaily.key_id == key_id)
            )
        ).scalar_one()
        assert daily.tokens_prompt == 15
        assert daily.tokens_completion == 27
        assert daily.request_count == 2


@dbtest
async def test_db_backward_compat_legacy_keys_unchanged(engine):
    """A key created with all new columns at their defaults (NULL / 0)
    presents an AuthedCaller whose every new dimension is unlimited — exactly
    matching pre-item-9 behaviour. This is the critical compat path."""
    factory = make_session_factory(engine)
    email = f"legacy-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(u)
        await s.flush()
        s.add(ApiKey(
            user_id=u.id,
            key_hash=_fake_key_hash(),
            key_prefix="legacy00",
            monthly_token_budget=0,  # legacy unlimited
        ))
        await s.flush()
        user_id = u.id

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        assert caller is not None
        # Every new dimension reads as unlimited.
        rem = caller.effective_remaining()
        assert rem == {
            "key_monthly": None,
            "user_monthly": None,
            "daily": None,
            "input": None,
            "output": None,
        }
        assert caller.is_unlimited() is True


@dbtest
async def test_db_daily_ceiling_clamps_max_tokens(engine, monkeypatch):
    """A per-key daily ceiling shows up as a ``daily`` headroom in
    ``effective_remaining()``, and the multi-dim clamp picks the daily
    headroom when it's the tightest."""
    from wrapper import main as mainmod

    factory = make_session_factory(engine)
    email = f"daily-cap-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(u)
        await s.flush()
        s.add(ApiKey(
            user_id=u.id,
            key_hash=_fake_key_hash(),
            key_prefix="dailycap",
            monthly_token_budget=1_000_000,  # big monthly
            daily_token_budget=100,           # tight daily
        ))
        await s.flush()
        user_id = u.id

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        assert caller is not None
        rem = caller.effective_remaining()
        assert rem["daily"] == 100
        assert rem["key_monthly"] == 1_000_000

    # Verify the clamp picks the daily headroom when running a request.
    settings = MagicMock()
    settings.modal_base_url = "https://upstream.example/"
    settings.vllm_api_key = "vllm-test"
    settings.upstream_timeout_s = 30.0
    settings.served_model_name = "test-model"
    settings.hf_token = None
    settings.log_ip = False

    seen: dict = {}

    async def fake_post(client, url, key, body, timeout, *, ctx=None):
        seen.update(body)
        return 200, {"choices": [{"text": "x"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}, 10

    monkeypatch.setattr(mainmod.proxymod, "post_nonstream", fake_post)
    monkeypatch.setattr(mainmod, "get_token_counter",
                        lambda *a, **kw: MagicMock(count=lambda t: 2))
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        status, _payload, _extras = await mainmod.run_completion_nonstream(
            session=s, http=MagicMock(), settings=settings, caller=caller,
            body={"prompt": "two words", "max_tokens": 500},
            ip=None, endpoint="/v1/completions",
        )
    assert status == 200
    # daily headroom = 100 − 2 prompt tokens = 98 → clamped from 500 to 98.
    assert seen["max_tokens"] == 98


@dbtest
async def test_db_user_aggregate_counts_across_keys(engine):
    """User-aggregate budget spans every key the user owns: usage on key A is
    visible to a caller built from key B, and vice-versa.

    Note: ``primary_authed_caller`` picks the newest key by ``created_at``;
    to make that deterministic we insert the two keys in separate transactions
    with a short sleep, matching ``test_primary_authed_caller_picks_newest_active``
    in ``tests/test_chat_helpers.py``.
    """
    import time as _time

    factory = make_session_factory(engine)
    email = f"agg-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(
            email=email,
            password_hash=hash_password("hunter22-pw"),
            monthly_token_budget_total=10_000,
        )
        s.add(u)
        await s.flush()
        ak_a = ApiKey(
            user_id=u.id,
            key_hash=_fake_key_hash(),
            key_prefix="aggkeyA0",
            monthly_token_budget=0,
        )
        s.add(ak_a)
        await s.flush()
        key_a_id, user_id = ak_a.id, u.id

    _time.sleep(0.05)
    async with session_scope(factory) as s:
        ak_b = ApiKey(
            user_id=user_id,
            key_hash=_fake_key_hash(),
            key_prefix="aggkeyB0",
            monthly_token_budget=0,
        )
        s.add(ak_b)
        await s.flush()
        key_b_id = ak_b.id

    # Spend on key A.
    async with session_scope(factory) as s:
        await commit_usage(s, key_a_id, prompt_tokens=3_000, completion_tokens=2_000)

    # primary_authed_caller picks the newest active key, which is B. Its
    # user-aggregate must see A's spend.
    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        assert caller is not None
        assert caller.key_id == key_b_id  # newest
        rem = caller.effective_remaining()
        assert rem["user_monthly"] == 5_000  # 10_000 − 5_000 spent on A

    # Spend more on key B; the aggregate ticks down.
    async with session_scope(factory) as s:
        await commit_usage(s, key_b_id, prompt_tokens=1_000, completion_tokens=1_000)

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        assert caller is not None
        assert caller.effective_remaining()["user_monthly"] == 3_000


@dbtest
async def test_db_io_split_blocks_when_output_exhausted(engine, monkeypatch):
    """Output-token budget hits 0 → next call gets 429, even if the input
    budget still has plenty of headroom."""
    from wrapper import main as mainmod

    factory = make_session_factory(engine)
    email = f"io-{uuid.uuid4().hex[:8]}@example.local"
    async with session_scope(factory) as s:
        u = User(email=email, password_hash=hash_password("hunter22-pw"))
        s.add(u)
        await s.flush()
        s.add(ApiKey(
            user_id=u.id,
            key_hash=_fake_key_hash(),
            key_prefix="iosplit0",
            monthly_token_budget=0,  # no total cap
            monthly_input_token_budget=10_000,    # roomy input
            monthly_output_token_budget=100,      # tight output
        ))
        await s.flush()
        user_id = u.id

    # Pre-load usage: 0 input used, 100 output used → output exhausted.
    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        await commit_usage(s, caller.key_id, prompt_tokens=0, completion_tokens=100)

    settings = MagicMock()
    settings.modal_base_url = "https://upstream.example/"
    settings.vllm_api_key = "vllm-test"
    settings.upstream_timeout_s = 30.0
    settings.served_model_name = "test-model"
    settings.hf_token = None
    settings.log_ip = False

    upstream_mock = AsyncMock()
    monkeypatch.setattr(mainmod.proxymod, "post_nonstream", upstream_mock)
    monkeypatch.setattr(mainmod, "get_token_counter",
                        lambda *a, **kw: MagicMock(count=lambda t: 5))
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    async with session_scope(factory) as s:
        caller = await primary_authed_caller(s, user_id)
        # Sanity: input has lots left, output is 0.
        rem = caller.effective_remaining()
        assert rem["input"] == 10_000
        assert rem["output"] == 0

        status, payload, _extras = await mainmod.run_completion_nonstream(
            session=s, http=MagicMock(), settings=settings, caller=caller,
            body={"prompt": "five words two three four", "max_tokens": 50},
            ip=None, endpoint="/v1/completions",
        )
    assert status == 429
    assert payload["error"]["code"] == "budget_exceeded"
    assert "output" in payload["error"]["message"].lower()
    upstream_mock.assert_not_called()


@dbtest
async def test_db_authenticate_loads_all_limit_columns(engine):
    """authenticate() must populate every new dimension on AuthedCaller —
    not just monthly_token_budget. This is how the bearer-keyed
    /v1/completions path picks up the new limits."""
    from wrapper.keys import generate as generate_key
    from fastapi import Request

    factory = make_session_factory(engine)
    gk = generate_key()
    email = f"auth-limits-{gk.prefix}@example.local"

    async with session_scope(factory) as s:
        u = User(
            email=email,
            password_hash=hash_password("hunter22-pw"),
            monthly_token_budget_total=99_999,
        )
        s.add(u)
        await s.flush()
        s.add(ApiKey(
            user_id=u.id,
            key_hash=gk.hash_,
            key_prefix=gk.prefix,
            monthly_token_budget=88_888,
            daily_token_budget=7_777,
            monthly_input_token_budget=6_666,
            monthly_output_token_budget=5_555,
        ))

    # Build a minimal Request stub carrying the bearer header.
    scope = {
        "type": "http",
        "headers": [(b"authorization", f"Bearer {gk.plaintext}".encode())],
    }
    req = Request(scope)

    async with session_scope(factory) as s:
        caller = await authenticate(req, s)
        assert caller.monthly_token_budget == 88_888
        assert caller.daily_token_budget == 7_777
        assert caller.monthly_input_token_budget == 6_666
        assert caller.monthly_output_token_budget == 5_555
        assert caller.monthly_token_budget_total == 99_999
        assert caller.tokens_used_today == 0
        assert caller.user_tokens_used_this_month == 0
