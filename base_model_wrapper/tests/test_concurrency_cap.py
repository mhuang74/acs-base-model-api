"""Lock in the per-key concurrency cap on /v1/completions.

The full handler path needs Postgres + a fake upstream, so we test the
concurrency-gate mechanics directly: the active cap, bounded waiter queue,
per-key cache, and slot release behavior.

If `_get_key_semaphore`, `MAX_INFLIGHT_PER_KEY`, or the dict-on-app.state
plumbing changes shape, this test fails fast — exactly the regressions
the streaming-slot lifetime in `_serve_stream` depends on.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from wrapper.main import (
    COMPLETIONS_RATE_LIMIT,
    COMPLETIONS_RATE_LIMIT_PER_MINUTE,
    MAX_INFLIGHT_PER_KEY,
    MAX_QUEUED_PER_KEY,
    _get_key_semaphore,
)
from wrapper.routes import api as api_routes


def _fake_app_state() -> SimpleNamespace:
    return SimpleNamespace(key_semaphores={})


def test_max_inflight_per_key_is_sensible():
    # Upstream vLLM is --max-num-seqs 128. Per-key cap should leave room for
    # many concurrent keys; if anyone bumps it above ~32, that's a deliberate
    # capacity-planning change and this test should be updated alongside.
    assert 1 <= MAX_INFLIGHT_PER_KEY <= 32
    assert MAX_QUEUED_PER_KEY >= MAX_INFLIGHT_PER_KEY
    assert COMPLETIONS_RATE_LIMIT == f"{COMPLETIONS_RATE_LIMIT_PER_MINUTE}/minute"
    assert COMPLETIONS_RATE_LIMIT_PER_MINUTE == 600


def test_same_key_returns_same_semaphore():
    state = _fake_app_state()
    key_id = uuid.uuid4()
    sem1 = _get_key_semaphore(state, key_id)
    sem2 = _get_key_semaphore(state, key_id)
    assert sem1 is sem2


def test_distinct_keys_get_distinct_semaphores():
    state = _fake_app_state()
    sem_a = _get_key_semaphore(state, uuid.uuid4())
    sem_b = _get_key_semaphore(state, uuid.uuid4())
    assert sem_a is not sem_b


@pytest.mark.asyncio
async def test_cap_admits_n_blocks_n_plus_one():
    state = _fake_app_state()
    key_id = uuid.uuid4()
    sem = _get_key_semaphore(state, key_id)

    # First MAX_INFLIGHT acquires must succeed without waiting.
    for _ in range(MAX_INFLIGHT_PER_KEY):
        await asyncio.wait_for(sem.acquire(), timeout=0.1)

    # The next acquire must block — wait_for(timeout) should raise.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(sem.acquire(), timeout=0.05)

    # Release one and confirm the next acquire now succeeds promptly.
    sem.release()
    await asyncio.wait_for(sem.acquire(), timeout=0.1)


@pytest.mark.asyncio
async def test_cap_is_per_key_not_global():
    """A second key must not be throttled by the first key saturating its cap."""
    state = _fake_app_state()
    sem_a = _get_key_semaphore(state, uuid.uuid4())
    sem_b = _get_key_semaphore(state, uuid.uuid4())

    # Saturate key A.
    for _ in range(MAX_INFLIGHT_PER_KEY):
        await sem_a.acquire()

    # Key B should still admit MAX_INFLIGHT acquires with no waiting.
    for _ in range(MAX_INFLIGHT_PER_KEY):
        await asyncio.wait_for(sem_b.acquire(), timeout=0.1)


@pytest.mark.asyncio
async def test_burst_of_2x_cap_all_eventually_succeed():
    """An ordinary batch larger than the active cap still queues and drains."""
    state = _fake_app_state()
    sem = _get_key_semaphore(state, uuid.uuid4())

    completed = []

    async def worker(i: int) -> None:
        async with sem:
            # Tiny "work" — the real handler awaits the upstream.
            await asyncio.sleep(0.01)
            completed.append(i)

    n = MAX_INFLIGHT_PER_KEY * 2
    await asyncio.gather(*[worker(i) for i in range(n)])
    assert sorted(completed) == list(range(n))


@pytest.mark.asyncio
async def test_waiter_queue_is_bounded():
    state = _fake_app_state()
    gate = _get_key_semaphore(state, uuid.uuid4())

    for _ in range(MAX_INFLIGHT_PER_KEY):
        assert await gate.acquire()

    async def queued_worker() -> bool:
        admitted = await gate.acquire()
        if admitted:
            gate.release()
        return admitted

    waiters = [asyncio.create_task(queued_worker()) for _ in range(MAX_QUEUED_PER_KEY)]
    for _ in range(100):
        if gate.queued == MAX_QUEUED_PER_KEY:
            break
        await asyncio.sleep(0)
    assert gate.queued == MAX_QUEUED_PER_KEY
    assert await gate.acquire() is False

    for _ in range(MAX_INFLIGHT_PER_KEY):
        gate.release()
    assert all(await asyncio.gather(*waiters))
    assert gate.queued == 0


@pytest.mark.asyncio
async def test_auth_transaction_is_committed_before_waiting_for_gate(monkeypatch):
    """Queued requests must not hold scarce Postgres pool connections."""
    session = SimpleNamespace(commit=AsyncMock())
    caller = SimpleNamespace(key_id=uuid.uuid4())
    request = SimpleNamespace(
        state=SimpleNamespace(request_id="req-test"),
        client=None,
        app=SimpleNamespace(state=_fake_app_state()),
    )
    settings = SimpleNamespace(last_used_throttle_s=300, log_ip=False)

    class FullGate:
        async def acquire(self) -> bool:
            assert session.commit.await_count == 1
            return False

    response = SimpleNamespace(status_code=429, headers={})
    monkeypatch.setattr(
        api_routes.authmod,
        "authenticate",
        AsyncMock(return_value=caller),
    )
    monkeypatch.setattr(api_routes, "_get_key_semaphore", lambda *_: FullGate())
    monkeypatch.setattr(api_routes, "_error", AsyncMock(return_value=response))

    result = await api_routes.completions.__wrapped__(
        request=request,
        settings=settings,
        session=session,
        http=MagicMock(),
    )

    assert result is response
    assert result.headers["Retry-After"] == "5"
    session.commit.assert_awaited_once()
