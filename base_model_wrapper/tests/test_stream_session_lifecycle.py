"""Unit tests for the streaming-handler session-lifecycle fix.

``_serve_stream`` and ``chat_stream`` must ``await session.commit()`` before
returning ``StreamingResponse``. Without this, the auth-time tx (SELECT +
optional ``UPDATE api_keys.last_used_at``) sits open for the entire stream
duration — which during cold-boot retries can be many minutes — leaving the
Postgres connection in "idle in transaction" state, blocking vacuum on
``api_keys`` and starving the connection pool under load.

These tests stub the upstream stream and the DB session at the function
boundary so we can directly observe whether ``commit()`` was called between
auth and stream construction.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

import pytest

from wrapper import main as main_module
from wrapper.auth import AuthedCaller


class _RecordingSession:
    """Minimal AsyncSession stand-in that records commit / rollback calls.

    ``_serve_stream`` only awaits ``session.commit()`` on the happy path (and
    ``session.execute`` indirectly via ``_record_request`` on the cold-boot
    503 fallback). We don't exercise either DB write here — we just need the
    commit observable. Anything else the function reaches on this object will
    surface as an AttributeError and fail the test loudly, which is the right
    signal: the lifecycle invariant only holds if the function reaches commit
    before doing anything else.
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    async def commit(self) -> None:
        self.events.append("commit")

    async def rollback(self) -> None:
        self.events.append("rollback")


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
def warm_upstream(monkeypatch):
    """Patch ``proxymod.stream_post`` to yield one chunk + a final status.

    Keeps ``_serve_stream``'s happy-path prefetch synchronous (one chunk, no
    cold-boot retry, no error) so the function reaches the
    ``await session.commit()`` line we care about.
    """

    async def fake_stream(client, url, api_key, body, timeout_s, *, ctx=None):
        # First yield: (chunk_bytes, usage_or_None, status_code_int).
        yield b"data: {\"choices\":[{\"text\":\"hi\"}]}\n\n", None, 200
        # Final usage chunk; status code None after the first yield.
        yield b"data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n", \
            {"prompt_tokens": 1, "completion_tokens": 1}, None

    monkeypatch.setattr(main_module.proxymod, "stream_post", fake_stream)


async def test_serve_stream_commits_auth_tx_before_returning(
    monkeypatch, fake_caller, warm_upstream
):
    """The auth-time transaction must be closed before the StreamingResponse
    starts — otherwise the request-scoped connection sits "idle in transaction"
    for the entire stream lifetime.
    """
    session = _RecordingSession()

    response = await main_module._serve_stream(
        request_id="req_test",
        caller=fake_caller,
        ip=None,
        upstream_url="https://upstream.example/v1/completions",
        body={"model": "gpt2", "prompt": "hello", "stream": True},
        settings=main_module.get_settings.__wrapped__()
        if hasattr(main_module.get_settings, "__wrapped__")
        else _stub_settings(),
        session=session,  # type: ignore[arg-type]
        http=None,  # not used by the patched stream_post
        t0=0.0,
        model_name="gpt2",
        request=None,
        key_semaphore=None,
    )

    # Two invariants:
    # 1. commit() was called (i.e. the tx is closed before the stream starts).
    # 2. It was called exactly once at handoff time, not zero (silent skip) and
    #    not many (would mean a refactor accidentally added duplicate commits).
    assert "commit" in session.events, \
        "session.commit() was not awaited before StreamingResponse — auth tx leaks"
    assert session.events.count("commit") == 1
    # And the response is the streaming one, not the 503 cold-boot fallback.
    from starlette.responses import StreamingResponse
    assert isinstance(response, StreamingResponse)


def _stub_settings():
    """Bare-minimum Settings for the streaming code paths that don't touch
    HF / Modal / cron config."""
    from wrapper.settings import Settings

    return Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        upstream_timeout_s=10.0,
    )
