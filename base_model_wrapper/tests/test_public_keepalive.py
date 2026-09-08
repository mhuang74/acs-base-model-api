from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.responses import JSONResponse, StreamingResponse

from wrapper import proxy as proxymod
from wrapper.routes import api as api_routes


class _Session:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


def _caller():
    return SimpleNamespace(key_id=uuid.uuid4())


def _settings():
    return SimpleNamespace(vllm_api_key="upstream-key", upstream_timeout_s=1200.0)


def _request(*, disconnected: bool = False):
    return SimpleNamespace(
        is_disconnected=AsyncMock(return_value=disconnected),
        state=SimpleNamespace(),
        app=SimpleNamespace(state=SimpleNamespace(last_completion_at={})),
    )


async def test_nonstream_keepalive_prefix_plus_result_is_valid_json(monkeypatch):
    release = asyncio.Event()

    async def slow_result(**_kwargs):
        await release.wait()
        return 200, {"choices": [{"text": "Paris"}], "usage": {}}, {}

    monkeypatch.setattr(api_routes, "run_completion_nonstream", slow_result)
    monkeypatch.setattr(api_routes, "PUBLIC_RESPONSE_GRACE_S", 0.0)
    monkeypatch.setattr(api_routes, "PUBLIC_KEEPALIVE_INTERVAL_S", 0.01)

    session = _Session()
    semaphore = asyncio.Semaphore(0)
    response = await api_routes._serve_nonstream_with_keepalive(
        request_id="req_keepalive",
        caller=_caller(),
        ip=None,
        upstream_url="https://up/v1/completions",
        body={"prompt": "x"},
        settings=_settings(),
        session=session,
        http=SimpleNamespace(),
        model_id="llama-405b",
        request=_request(),
        key_semaphore=semaphore,
        backend_ctx=proxymod.BackendContext(model_id="llama-405b", cold_hint=True),
        breakers=SimpleNamespace(),
        effective_timeout=1200.0,
        tokenizer_repo="gpt2",
    )

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "application/json"
    first = await response.body_iterator.__anext__()
    assert first == api_routes.JSON_KEEPALIVE_CHUNK
    release.set()
    rest = [chunk async for chunk in response.body_iterator]
    payload = json.loads((first + b"".join(rest)).strip())
    assert payload["choices"][0]["text"] == "Paris"
    assert session.commits == 1
    assert semaphore._value == 1


async def test_nonstream_quick_error_preserves_http_status(monkeypatch):
    async def quick_error(**_kwargs):
        return 503, {
            "error": {"code": "circuit_open", "retry_after_seconds": 12}
        }, {}

    monkeypatch.setattr(api_routes, "run_completion_nonstream", quick_error)
    monkeypatch.setattr(api_routes, "PUBLIC_RESPONSE_GRACE_S", 0.1)

    response = await api_routes._serve_nonstream_with_keepalive(
        request_id="req_quick",
        caller=_caller(),
        ip=None,
        upstream_url="https://up/v1/completions",
        body={"prompt": "x"},
        settings=_settings(),
        session=_Session(),
        http=SimpleNamespace(),
        model_id="llama-405b",
        request=_request(),
        key_semaphore=asyncio.Semaphore(0),
        backend_ctx=proxymod.BackendContext(model_id="llama-405b"),
        breakers=SimpleNamespace(),
        effective_timeout=1200.0,
        tokenizer_repo="gpt2",
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "12"


async def test_nonstream_disconnect_cancels_upstream_and_releases_slot(monkeypatch):
    cancelled = asyncio.Event()

    async def never_returns(**_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(api_routes, "run_completion_nonstream", never_returns)
    monkeypatch.setattr(api_routes, "PUBLIC_RESPONSE_GRACE_S", 0.0)
    monkeypatch.setattr(api_routes, "PUBLIC_KEEPALIVE_INTERVAL_S", 0.0)
    semaphore = asyncio.Semaphore(0)

    session = _Session()
    response = await api_routes._serve_nonstream_with_keepalive(
        request_id="req_disconnect",
        caller=_caller(),
        ip=None,
        upstream_url="https://up/v1/completions",
        body={"prompt": "x"},
        settings=_settings(),
        session=session,
        http=SimpleNamespace(),
        model_id="llama-405b",
        request=_request(disconnected=True),
        key_semaphore=semaphore,
        backend_ctx=proxymod.BackendContext(model_id="llama-405b", cold_hint=True),
        breakers=SimpleNamespace(),
        effective_timeout=1200.0,
        tokenizer_repo="gpt2",
    )

    assert await response.body_iterator.__anext__() == api_routes.JSON_KEEPALIVE_CHUNK
    try:
        await response.body_iterator.__anext__()
    except StopAsyncIteration:
        pass
    else:
        raise AssertionError("disconnect should end the response stream")
    assert cancelled.is_set()
    assert semaphore._value == 1
    assert session.rollbacks == 1


async def test_nonstream_grace_cancellation_cleans_child_and_rolls_back(monkeypatch):
    entered = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def never_returns(**_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            child_cancelled.set()

    monkeypatch.setattr(api_routes, "run_completion_nonstream", never_returns)
    monkeypatch.setattr(api_routes, "PUBLIC_RESPONSE_GRACE_S", 30.0)
    session = _Session()
    handler = asyncio.create_task(
        api_routes._serve_nonstream_with_keepalive(
            request_id="req_grace_cancel",
            caller=_caller(),
            ip=None,
            upstream_url="https://up/v1/completions",
            body={"prompt": "x"},
            settings=_settings(),
            session=session,
            http=SimpleNamespace(),
            model_id="llama-405b",
            request=_request(),
            key_semaphore=asyncio.Semaphore(0),
            backend_ctx=proxymod.BackendContext(model_id="llama-405b", cold_hint=True),
            breakers=SimpleNamespace(),
            effective_timeout=1200.0,
            tokenizer_repo="gpt2",
        )
    )
    await entered.wait()
    handler.cancel()
    try:
        await handler
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancelled handler should propagate cancellation")
    assert child_cancelled.is_set()
    assert session.rollbacks == 1


async def test_nonstream_slow_path_keeps_preflight_headers(monkeypatch):
    release = asyncio.Event()

    async def slow_result(**kwargs):
        kwargs["transport_extras"]["max_tokens_clamped"] = {
            "requested": 100,
            "applied": 8,
            "reason": "budget",
        }
        await release.wait()
        return 200, {"choices": [{"text": "ok"}], "usage": {}}, kwargs["transport_extras"]

    monkeypatch.setattr(api_routes, "run_completion_nonstream", slow_result)
    monkeypatch.setattr(api_routes, "PUBLIC_RESPONSE_GRACE_S", 0.01)
    response = await api_routes._serve_nonstream_with_keepalive(
        request_id="req_headers",
        caller=_caller(),
        ip=None,
        upstream_url="https://up/v1/completions",
        body={"prompt": "x"},
        settings=_settings(),
        session=_Session(),
        http=SimpleNamespace(),
        model_id="llama-405b",
        request=_request(),
        key_semaphore=asyncio.Semaphore(0),
        backend_ctx=proxymod.BackendContext(model_id="llama-405b", cold_hint=True),
        breakers=SimpleNamespace(),
        effective_timeout=1200.0,
        tokenizer_repo="gpt2",
    )
    assert response.headers["x-acs-max-tokens-clamped"] == (
        "requested=100,applied=8,reason=budget"
    )
    release.set()
    _ = [chunk async for chunk in response.body_iterator]


async def test_public_sse_emits_comment_while_upstream_open_is_pending(monkeypatch):
    release = asyncio.Event()

    async def slow_stream(*_args, **_kwargs):
        await release.wait()
        yield b'data: {"choices":[{"text":"Paris"}]}\n\n', None, 200

    monkeypatch.setattr(api_routes.proxymod, "stream_post", slow_stream)
    monkeypatch.setattr(api_routes, "PUBLIC_KEEPALIVE_INTERVAL_S", 0.01)
    monkeypatch.setattr(api_routes, "_record_request", AsyncMock())
    session = _Session()
    semaphore = asyncio.Semaphore(0)

    response = await api_routes._serve_stream(
        request_id="req_sse",
        caller=_caller(),
        ip=None,
        upstream_url="https://up/v1/completions",
        body={"prompt": "x", "stream": True},
        settings=_settings(),
        session=session,
        http=SimpleNamespace(),
        t0=0.0,
        model_name="llama-405b",
        request=None,
        key_semaphore=semaphore,
        backend_ctx=proxymod.BackendContext(model_id="llama-405b", cold_hint=True),
        breakers=None,
    )

    assert isinstance(response, StreamingResponse)
    assert await response.body_iterator.__anext__() == b": keepalive\n\n"
    release.set()
    data = await response.body_iterator.__anext__()
    assert b"Paris" in data
    await response.body_iterator.aclose()
    assert session.commits == 1
    assert semaphore._value == 1


async def test_public_sse_turns_late_cold_failure_into_data_error(monkeypatch):
    async def late_error(*_args, **_kwargs):
        await asyncio.sleep(0.02)
        raise proxymod.ColdBootError(303, 3)
        yield  # pragma: no cover

    monkeypatch.setattr(api_routes.proxymod, "stream_post", late_error)
    monkeypatch.setattr(api_routes, "PUBLIC_KEEPALIVE_INTERVAL_S", 0.005)
    monkeypatch.setattr(api_routes, "_record_request", AsyncMock())

    breakers = SimpleNamespace(
        allow=AsyncMock(return_value=True),
        record_failure=AsyncMock(),
        record_success=AsyncMock(),
    )
    response = await api_routes._serve_stream(
        request_id="req_sse_error",
        caller=_caller(),
        ip=None,
        upstream_url="https://up/v1/completions",
        body={"prompt": "x", "stream": True},
        settings=_settings(),
        session=_Session(),
        http=SimpleNamespace(),
        t0=0.0,
        model_name="llama-405b",
        request=None,
        key_semaphore=asyncio.Semaphore(0),
        backend_ctx=proxymod.BackendContext(model_id="llama-405b", cold_hint=True),
        breakers=breakers,
    )

    chunks = [chunk async for chunk in response.body_iterator]
    assert any(chunk == b": keepalive\n\n" for chunk in chunks)
    error_frames = [chunk for chunk in chunks if chunk.startswith(b"data: {")]
    assert len(error_frames) == 1
    error = json.loads(error_frames[0][len(b"data: ") :])
    assert error["error"]["code"] == "modal_cold_boot"
    breakers.record_failure.assert_not_awaited()
