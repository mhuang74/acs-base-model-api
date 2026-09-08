from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from wrapper import proxy as proxymod
from wrapper.proxy import _extract_usage, clamp_max_tokens


def test_clamp_no_budget_is_noop():
    body = {"max_tokens": 100}
    orig, clamped = clamp_max_tokens(body, None)
    assert body["max_tokens"] == 100
    assert orig == 100 and clamped == 100


def test_clamp_when_unset_fills_with_remaining():
    body = {}
    orig, clamped = clamp_max_tokens(body, 50)
    assert body["max_tokens"] == 50
    assert orig is None and clamped == 50


def test_clamp_when_requested_exceeds_remaining():
    body = {"max_tokens": 32768}
    orig, clamped = clamp_max_tokens(body, 200)
    assert body["max_tokens"] == 200
    assert orig == 32768 and clamped == 200


def test_clamp_when_requested_under_remaining():
    body = {"max_tokens": 80}
    orig, clamped = clamp_max_tokens(body, 200)
    assert body["max_tokens"] == 80
    assert orig == 80 and clamped == 80


def test_extract_usage_finds_block():
    chunk = (
        b'data: {"id":"x","object":"text_completion","choices":[],'
        b'"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n'
        b"data: [DONE]\n\n"
    )
    u = _extract_usage(chunk)
    assert u == {"prompt_tokens": 7, "completion_tokens": 3}


def test_extract_usage_returns_none_when_absent():
    chunk = b'data: {"id":"x","choices":[{"text":"hi"}]}\n\n'
    assert _extract_usage(chunk) is None


def test_extract_usage_ignores_done_line():
    chunk = b"data: [DONE]\n\n"
    assert _extract_usage(chunk) is None


# --- Cold-boot 303 handling ---------------------------------------------------
#
# Modal returns HTTP 303 when a long-running Web Function reaches one HTTP
# hop's time limit. Its Location points at the result for the same invocation:
# the proxy must follow it as GET, not submit another POST and start over.


@pytest.fixture(autouse=True)
def _fast_coldboot_backoff(monkeypatch):
    """Patch the 30 s sleep down to ~0 so the retry tests stay quick."""
    monkeypatch.setattr(proxymod, "COLD_BOOT_BACKOFF_S", 0.0)


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._body = body
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise json.JSONDecodeError("no body", "", 0)
        return self._body


class _FakeStreamResponse:
    """Async-context-manager mimic of ``client.stream(...)``'s return value."""

    def __init__(
        self,
        status_code: int,
        chunks: list[bytes],
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._chunks = chunks
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return None

    async def aread(self) -> bytes:
        return b"".join(self._chunks)

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aiter_bytes(self):
        # The fakes carry pre-decoded chunks, so decoded iteration == raw.
        for chunk in self._chunks:
            yield chunk


class _FakeRequestClient:
    """Returns scripted ``_FakeResponse`` values from ``.request(...)``."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def request(self, method: str, url: str, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        return self._responses.pop(0)


class _FakeStreamClient:
    def __init__(self, responses: list[_FakeStreamResponse]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        return self._responses.pop(0)


async def _collect(stream):
    out = []
    async for item in stream:
        out.append(item)
    return out


async def test_post_nonstream_disables_redirect_following():
    client = _FakeRequestClient([_FakeResponse(200, body={"choices": [], "usage": {}})])
    await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 5.0)
    # Critical: the global httpx client has follow_redirects=True; the proxy
    # must override it so cold-boot 303s aren't silently followed.
    assert client.calls[0]["follow_redirects"] is False


async def test_post_nonstream_303_then_success_returns_payload():
    body = {"prompt": "x"}
    client = _FakeRequestClient([
        _FakeResponse(303, headers={"location": "/continuations/first"}),
        _FakeResponse(303, headers={"location": "second"}),
        _FakeResponse(200, body={"choices": [{"text": "hi"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
    ])
    status, payload, elapsed_ms = await proxymod.post_nonstream(
        client, "https://up/v1/completions", "k", body, 5.0
    )
    assert status == 200
    assert payload["choices"][0]["text"] == "hi"
    assert len(client.calls) == 3
    assert client.calls[0]["method"] == "POST"
    assert client.calls[0]["url"] == "https://up/v1/completions"
    assert client.calls[0]["json"] == body
    assert [call["method"] for call in client.calls[1:]] == ["GET", "GET"]
    assert client.calls[1]["url"] == "https://up/continuations/first"
    assert client.calls[2]["url"] == "https://up/continuations/second"
    assert all("json" not in call for call in client.calls[1:])


async def test_post_nonstream_303_exhausts_continuation_hops_raises_coldboot():
    responses = [
        _FakeResponse(303, headers={"location": f"/continuations/{i}"})
        for i in range(proxymod.MODAL_CONTINUATION_MAX_REDIRECTS + 1)
    ]
    client = _FakeRequestClient(responses)
    with pytest.raises(proxymod.ColdBootError) as exc_info:
        await proxymod.post_nonstream(client, "https://up/v1/completions", "k", {"prompt": "x"}, 5.0)
    assert exc_info.value.upstream_status == 303
    assert exc_info.value.attempts == proxymod.MODAL_CONTINUATION_MAX_REDIRECTS + 1
    assert len(client.calls) == proxymod.MODAL_CONTINUATION_MAX_REDIRECTS + 1
    assert client.calls[0]["method"] == "POST"
    assert all(call["method"] == "GET" for call in client.calls[1:])


async def test_post_nonstream_200_with_bad_shape_is_upstream_error():
    bad = _FakeResponse(200, body={"detail": "Starting container..."})
    client = _FakeRequestClient([bad])
    with pytest.raises(proxymod.UpstreamUnreachable):
        await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 5.0)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("location", "reason"),
    [
        (None, "omitted Location"),
        ("https://[invalid", "invalid Location"),
        ("javascript:alert(1)", "different origin"),
        ("https://elsewhere.example/result", "different origin"),
    ],
)
async def test_post_nonstream_rejects_invalid_continuation_location(location, reason):
    headers = {} if location is None else {"location": location}
    client = _FakeRequestClient([_FakeResponse(303, headers=headers)])
    with pytest.raises(proxymod.UpstreamUnreachable, match=reason):
        await proxymod.post_nonstream(
            client, "https://up/v1/completions", "k", {"prompt": "x"}, 5.0
        )
    assert len(client.calls) == 1


async def test_post_nonstream_caps_public_timeout_at_fourteen_minutes():
    client = _FakeRequestClient([_FakeResponse(200, body={"choices": [], "usage": {}})])
    await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 1_200.0)
    timeout = client.calls[0]["timeout"]
    assert timeout.read == pytest.approx(proxymod.PUBLIC_REQUEST_TIMEOUT_S, abs=0.1)
    assert timeout.read == pytest.approx(14 * 60, abs=0.1)


async def test_stream_post_disables_redirect_following():
    client = _FakeStreamClient([
        _FakeStreamResponse(200, [b'data: {"choices":[{"text":"hi"}]}\n\n', b"data: [DONE]\n\n"]),
    ])
    body = {"prompt": "x", "stream": True}
    await _collect(proxymod.stream_post(client, "https://up", "k", body, 5.0))
    assert client.calls[0]["follow_redirects"] is False


async def test_stream_post_303_then_success_yields_chunks():
    client = _FakeStreamClient([
        _FakeStreamResponse(303, [], headers={"location": "/continuations/result-1"}),
        _FakeStreamResponse(200, [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"]),
    ])
    body = {"prompt": "x", "stream": True}
    out = await _collect(
        proxymod.stream_post(client, "https://up/v1/completions", "k", body, 5.0)
    )
    # First yield carries the status code so the caller can short-circuit.
    assert out[0][2] == 200
    assert b"ok" in out[0][0]
    assert len(client.calls) == 2
    assert client.calls[0]["method"] == "POST"
    assert client.calls[0]["json"] == body
    assert client.calls[1]["method"] == "GET"
    assert client.calls[1]["url"] == "https://up/continuations/result-1"
    assert "json" not in client.calls[1]


async def test_stream_post_303_exhausts_continuation_hops_raises_before_yield():
    client = _FakeStreamClient([
        _FakeStreamResponse(303, [], headers={"location": f"/continuations/{i}"})
        for i in range(proxymod.MODAL_CONTINUATION_MAX_REDIRECTS + 1)
    ])
    gen = proxymod.stream_post(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    with pytest.raises(proxymod.ColdBootError) as exc_info:
        await gen.__anext__()
    assert exc_info.value.upstream_status == 303
    assert exc_info.value.attempts == proxymod.MODAL_CONTINUATION_MAX_REDIRECTS + 1
    assert client.calls[0]["method"] == "POST"
    assert all(call["method"] == "GET" for call in client.calls[1:])
    # If ColdBootError surfaced *after* a chunk was yielded, the caller would
    # already have committed to a 200 StreamingResponse and couldn't fall back
    # to a 503. Verify the raise happens before the first yield.


def test_cold_boot_error_payload_contract():
    payload = proxymod.cold_boot_error_payload(303)
    err = payload["error"]
    # Stable contract — the workbench JS keys off this exact code.
    assert err["code"] == "modal_cold_boot"
    assert err["retryable"] is True
    assert err["upstream_status"] == 303
    assert err["type"] == "upstream_not_ready"
    assert "message" in err and err["message"]


# --- stream_post_with_status (workbench SSE path) ----------------------------
#
# The workbench streaming endpoint commits to ``StreamingResponse(200, SSE)`` from
# the moment the client connects — a cold-boot wait must surface *inside* the
# stream as discriminated events rather than as a 503 JSONResponse.


async def test_stream_status_emits_chunks_on_immediate_200():
    client = _FakeStreamClient([
        _FakeStreamResponse(200, [
            b'data: {"choices":[{"text":"ok"}]}\n\n',
            b"data: [DONE]\n\n",
        ]),
    ])
    out = await _collect(
        proxymod.stream_post_with_status(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    )
    kinds = [ev["kind"] for ev in out]
    assert kinds[0] == "chunk"
    assert all(k == "chunk" for k in kinds)
    assert b"ok" in out[0]["data"]
    # First chunk carries the upstream status code so the caller knows it's 200.
    assert out[0]["status"] == 200


async def test_stream_status_emits_status_then_chunks_on_303_then_200():
    client = _FakeStreamClient([
        _FakeStreamResponse(303, []),
        _FakeStreamResponse(200, [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"]),
    ])
    out = await _collect(
        proxymod.stream_post_with_status(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    )
    # At least one cold_boot status frame must arrive before the first chunk.
    first_chunk_idx = next(i for i, ev in enumerate(out) if ev["kind"] == "chunk")
    assert first_chunk_idx > 0
    status_events = out[:first_chunk_idx]
    assert all(ev["kind"] == "status" for ev in status_events)
    assert all(ev["phase"] == "cold_boot" for ev in status_events)
    # Status carries attempt + elapsed_ms so the frontend can show a server-
    # authoritative timer.
    assert status_events[0]["attempt"] == 1
    assert "elapsed_ms" in status_events[0]


async def test_stream_status_keeps_retrying_indefinitely_no_failed_phase(monkeypatch):
    # Workbench path now retries cold-boot effectively forever (Modal H200
    # allocation can legitimately take 30 min – 1 h on a constrained day).
    # Pin the workbench budget down to a small number for the test, drive
    # that-many 303s followed by a 200, and assert: (a) every status frame
    # is ``phase: cold_boot``; (b) no ``phase: failed`` ever appears;
    # (c) a chunk eventually arrives.
    monkeypatch.setattr(proxymod, "COLD_BOOT_MAX_RETRIES_WORKBENCH", 3)
    client = _FakeStreamClient(
        [_FakeStreamResponse(303, []) for _ in range(3)]
        + [_FakeStreamResponse(200, [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"])]
    )
    out = await _collect(
        proxymod.stream_post_with_status(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    )
    statuses = [ev for ev in out if ev["kind"] == "status"]
    chunks = [ev for ev in out if ev["kind"] == "chunk"]
    assert all(ev["phase"] == "cold_boot" for ev in statuses)
    assert not any(ev.get("phase") == "failed" for ev in out)
    assert len(chunks) >= 1


async def test_stream_status_workbench_budget_is_effectively_unlimited():
    # Documented contract: the workbench path's retry budget is large enough
    # that exhaustion is operationally impossible (1M retries × 30 s ≈
    # 30,000 hours). Public paths instead keep one Modal invocation alive,
    # bounded by a 14-minute deadline below Railway's 15-minute hard ceiling.
    assert proxymod.COLD_BOOT_MAX_RETRIES_WORKBENCH >= 100_000
    assert proxymod.PUBLIC_REQUEST_TIMEOUT_S == 14 * 60


async def test_stream_status_elapsed_ms_monotonically_nondecreasing(monkeypatch):
    # Use a small positive backoff so the cadence loop actually advances
    # clock time between status emissions. The autouse fixture pins it to 0.0;
    # override locally so the cadence test is meaningful.
    monkeypatch.setattr(proxymod, "COLD_BOOT_BACKOFF_S", 0.05)
    monkeypatch.setattr(proxymod, "COLD_BOOT_STATUS_INTERVAL_S", 0.02)
    client = _FakeStreamClient([
        _FakeStreamResponse(303, []),
        _FakeStreamResponse(200, [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"]),
    ])
    out = await _collect(
        proxymod.stream_post_with_status(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    )
    status_elapsed = [ev["elapsed_ms"] for ev in out if ev["kind"] == "status"]
    assert len(status_elapsed) >= 2  # multiple ticks within one 50 ms backoff
    assert status_elapsed == sorted(status_elapsed)  # monotonically non-decreasing


async def test_stream_status_non_coldboot_error_yields_error_event():
    # A 500 (or any non-3xx 4xx/5xx) is not a cold boot; we don't want to
    # retry, just forward the error body and stop.
    client = _FakeStreamClient([
        _FakeStreamResponse(500, [b'{"error":"boom"}']),
    ])
    out = await _collect(
        proxymod.stream_post_with_status(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    )
    assert len(out) == 1
    assert out[0]["kind"] == "error"
    assert out[0]["status"] == 500


async def test_stream_status_cancel_event_unwinds_cold_boot_sleep(monkeypatch):
    """ACS-132 regression: a cancel_event must short-circuit the cold-boot
    backoff sleep within the slice window, not wait it out.

    Sets a 5 s slice + 5 s backoff and fires cancel_event 50 ms after the
    first status frame. Without the fix the generator would sleep the full
    slice — the asyncio.wait_for(timeout=1.0) below would trip first.
    """
    import asyncio
    monkeypatch.setattr(proxymod, "COLD_BOOT_BACKOFF_S", 5.0)
    monkeypatch.setattr(proxymod, "COLD_BOOT_STATUS_INTERVAL_S", 5.0)
    client = _FakeStreamClient(
        [_FakeStreamResponse(303, []) for _ in range(10)]
    )
    cancel_event = asyncio.Event()

    async def run() -> list[dict]:
        out: list[dict] = []
        gen = proxymod.stream_post_with_status(
            client,
            "https://up",
            "k",
            {"prompt": "x", "stream": True},
            5.0,
            cancel_event=cancel_event,
        )
        async for ev in gen:
            out.append(ev)
            if len(out) == 1:
                asyncio.get_running_loop().call_later(0.05, cancel_event.set)
        return out

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    out = await asyncio.wait_for(run(), timeout=1.0)
    elapsed = loop.time() - t0
    assert elapsed < 0.5, f"cancel_event did not interrupt sleep: elapsed={elapsed}"
    assert out[0]["kind"] == "status"
    assert out[0]["phase"] == "cold_boot"


async def test_stream_status_no_cancel_event_keeps_legacy_sleep(monkeypatch):
    """When cancel_event is None the generator must still use asyncio.sleep
    (i.e. the cold-boot cadence loop is unchanged for callers that don't
    care about cancellation — the public API path).
    """
    monkeypatch.setattr(proxymod, "COLD_BOOT_BACKOFF_S", 0.05)
    monkeypatch.setattr(proxymod, "COLD_BOOT_STATUS_INTERVAL_S", 0.02)
    client = _FakeStreamClient([
        _FakeStreamResponse(303, []),
        _FakeStreamResponse(200, [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"]),
    ])
    out = await _collect(
        proxymod.stream_post_with_status(
            client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0
        )
    )
    # Same shape as the legacy test above — backoff still ticked and a chunk arrived.
    assert any(ev["kind"] == "status" and ev["phase"] == "cold_boot" for ev in out)
    assert any(ev["kind"] == "chunk" for ev in out)


async def test_stream_status_upstream_503_is_error_not_coldboot():
    # An upstream 503 (e.g. vLLM overload) is NOT a cold boot. ``_is_cold_boot_status``
    # only matches 3xx, so 503 must surface as an ``error`` event without
    # retries. The caller uses this to distinguish real upstream errors from
    # cold-boot exhaustion (which yields a ``status: failed`` event instead).
    client = _FakeStreamClient([
        _FakeStreamResponse(503, [b'{"error":"backend_overloaded"}']),
    ])
    out = await _collect(
        proxymod.stream_post_with_status(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    )
    assert len(out) == 1
    assert out[0]["kind"] == "error"
    assert out[0]["status"] == 503
    # And exactly one upstream call — no retry budget was spent.
    assert len(client.calls) == 1


async def test_stream_post_rejects_cross_origin_continuation_before_yield():
    client = _FakeStreamClient(
        [_FakeStreamResponse(303, [], headers={"location": "https://evil.example/result"})]
    )
    gen = proxymod.stream_post(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    with pytest.raises(proxymod.UpstreamUnreachable, match="different origin"):
        await gen.__anext__()
    assert len(client.calls) == 1


# --- passthrough_post (full-vocab non-streaming pass-through, ACS-198) --------
#
# The pass-through must speak the same Modal continuation protocol and error
# taxonomy as post_nonstream, but yield the body as bytes without parsing it —
# that unparsed flow is what lifted the full-vocab prompt-length RAM cap.

_COMPLETION_HEAD = b'{"id":"cmpl-1","object":"text_completion","choices":[{"text":"'


def _completion_chunks() -> list[bytes]:
    return [
        _COMPLETION_HEAD,
        b'ok"}],',
        b'"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}',
    ]


async def test_passthrough_disables_redirect_following():
    client = _FakeStreamClient([_FakeStreamResponse(200, _completion_chunks())])
    await _collect(proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0))
    assert client.calls[0]["follow_redirects"] is False


async def test_passthrough_303_then_200_yields_bytes_with_status_once():
    client = _FakeStreamClient([
        _FakeStreamResponse(303, [], headers={"location": "/continuations/result-1"}),
        _FakeStreamResponse(200, _completion_chunks()),
    ])
    out = await _collect(
        proxymod.passthrough_post(client, "https://up/v1/completions", "k", {"prompt": "x"}, 5.0)
    )
    # Status rides only the first tuple; the body round-trips byte-identical.
    assert out[0][1] == 200
    assert all(status is None for _chunk, status in out[1:])
    assert b"".join(chunk for chunk, _ in out) == b"".join(_completion_chunks())
    assert client.calls[0]["method"] == "POST"
    assert client.calls[1]["method"] == "GET"
    assert client.calls[1]["url"] == "https://up/continuations/result-1"
    assert "json" not in client.calls[1]


async def test_passthrough_head_buffered_across_tiny_chunks():
    # The completion marker may arrive split over several small chunks; the
    # head check must buffer rather than reject on the first fragment.
    chunks = [bytes([b]) for b in b'{"id":"x","choices"'] + [b':[{"text":"y"}],"usage":{}}']
    client = _FakeStreamClient([_FakeStreamResponse(200, chunks)])
    out = await _collect(
        proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0)
    )
    assert out[0][1] == 200
    assert b"".join(chunk for chunk, _ in out) == b"".join(chunks)


async def test_passthrough_200_with_non_completion_body_raises_before_yield():
    client = _FakeStreamClient([_FakeStreamResponse(200, [b"<html>not ready</html>"])])
    gen = proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0)
    with pytest.raises(proxymod.UpstreamUnreachable, match="non-completion body"):
        await gen.__anext__()


async def test_passthrough_4xx_yields_single_full_body():
    err_body = b'{"error":{"message":"bad params"}}'
    client = _FakeStreamClient([_FakeStreamResponse(400, [err_body[:10], err_body[10:]])])
    out = await _collect(
        proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0)
    )
    # 4xx bodies are small: read whole and yielded as one chunk for reshaping.
    assert out == [(err_body, 400)]


async def test_passthrough_5xx_retries_then_raises(monkeypatch):
    monkeypatch.setattr(proxymod, "_exp_backoff", lambda attempt: 0.0)
    client = _FakeStreamClient([
        _FakeStreamResponse(503, [b"overloaded"])
        for _ in range(proxymod.RETRY_5XX_MAX_RETRIES + 1)
    ])
    gen = proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0)
    with pytest.raises(proxymod.UpstreamServerError) as exc_info:
        await gen.__anext__()
    assert exc_info.value.upstream_status == 503
    assert exc_info.value.attempts == proxymod.RETRY_5XX_MAX_RETRIES + 1
    assert len(client.calls) == proxymod.RETRY_5XX_MAX_RETRIES + 1


async def test_passthrough_connect_error_retries_then_raises(monkeypatch):
    # Parity with the buffered path's _request_with_5xx_retry: pre-body network
    # failures are retried with backoff, not surfaced as an instant 502.
    monkeypatch.setattr(proxymod, "_exp_backoff", lambda attempt: 0.0)

    class _RaisingCM:
        def __init__(self, exc):
            self._exc = exc

        async def __aenter__(self):
            raise self._exc

        async def __aexit__(self, *_a):
            return None

    class _ConnectErrorClient:
        def __init__(self):
            self.calls = 0

        def stream(self, method, url, **kw):
            # Real httpx does no I/O in .stream(); errors raise at __aenter__.
            self.calls += 1
            return _RaisingCM(httpx.ConnectError("connection refused"))

    client = _ConnectErrorClient()
    gen = proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0)
    with pytest.raises(proxymod.UpstreamUnreachable) as exc_info:
        await gen.__anext__()
    assert exc_info.value.attempts == proxymod.RETRY_5XX_MAX_RETRIES + 1
    assert client.calls == proxymod.RETRY_5XX_MAX_RETRIES + 1


async def test_passthrough_connect_blip_then_200_recovers(monkeypatch):
    monkeypatch.setattr(proxymod, "_exp_backoff", lambda attempt: 0.0)

    class _RaisingCM:
        def __init__(self, exc):
            self._exc = exc

        async def __aenter__(self):
            raise self._exc

        async def __aexit__(self, *_a):
            return None

    class _BlipClient:
        def __init__(self, responses):
            self._responses = responses
            self.calls = 0

        def stream(self, method, url, **kw):
            self.calls += 1
            if self.calls == 1:
                return _RaisingCM(httpx.ReadError("peer reset"))
            return self._responses.pop(0)

    client = _BlipClient([_FakeStreamResponse(200, _completion_chunks())])
    out = await _collect(
        proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0)
    )
    assert out[0][1] == 200
    assert b"".join(chunk for chunk, _ in out) == b"".join(_completion_chunks())


async def test_passthrough_5xx_then_200_recovers(monkeypatch):
    monkeypatch.setattr(proxymod, "_exp_backoff", lambda attempt: 0.0)
    client = _FakeStreamClient([
        _FakeStreamResponse(502, [b"blip"]),
        _FakeStreamResponse(200, _completion_chunks()),
    ])
    out = await _collect(
        proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0)
    )
    assert out[0][1] == 200
    assert b"".join(chunk for chunk, _ in out) == b"".join(_completion_chunks())


async def test_passthrough_303_exhausts_continuation_hops_raises_coldboot():
    client = _FakeStreamClient([
        _FakeStreamResponse(303, [], headers={"location": f"/continuations/{i}"})
        for i in range(proxymod.MODAL_CONTINUATION_MAX_REDIRECTS + 1)
    ])
    gen = proxymod.passthrough_post(client, "https://up", "k", {"prompt": "x"}, 5.0)
    with pytest.raises(proxymod.ColdBootError):
        await gen.__anext__()


# --- extract_usage_from_json_tail ---------------------------------------------


def test_tail_usage_extracts_counts():
    tail = b'...,"text":"x"}],"usage":{"prompt_tokens":123,"completion_tokens":4,"total_tokens":127}}'
    assert proxymod.extract_usage_from_json_tail(tail) == {
        "prompt_tokens": 123,
        "completion_tokens": 4,
    }


def test_tail_usage_takes_last_occurrence():
    # A completion whose *text* mentions "usage" must not shadow the real block
    # — the top-level usage key serializes after choices, so last wins.
    tail = (
        b'{"text":"talk about \\"usage\\":{} here"}],'
        b'"usage":{"prompt_tokens":9,"completion_tokens":0,"total_tokens":9}}'
    )
    assert proxymod.extract_usage_from_json_tail(tail) == {
        "prompt_tokens": 9,
        "completion_tokens": 0,
    }


def test_tail_usage_handles_nested_detail_objects():
    tail = (
        b'"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,'
        b'"prompt_tokens_details":{"cached_tokens":0}}}'
    )
    assert proxymod.extract_usage_from_json_tail(tail) == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
    }


def test_tail_usage_returns_none_when_absent_or_garbled():
    assert proxymod.extract_usage_from_json_tail(b'{"choices":[]}') is None
    assert proxymod.extract_usage_from_json_tail(b'"usage":{"prompt_tokens":') is None
    assert proxymod.extract_usage_from_json_tail(b"") is None


async def test_stream_status_cold_boot_carries_boot_stage_fields():
    # ACS-272: a boot_stage_provider's fields ride along on every cold_boot
    # status event so the workbench banner can name the real container stage.
    client = _FakeStreamClient([
        _FakeStreamResponse(303, []),
        _FakeStreamResponse(200, [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"]),
    ])

    async def provider():
        return {"stage": "weights_loading", "stage_label": "Loading model weights"}

    out = await _collect(
        proxymod.stream_post_with_status(
            client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0,
            boot_stage_provider=provider,
        )
    )
    statuses = [ev for ev in out if ev["kind"] == "status"]
    assert statuses
    assert all(ev["stage"] == "weights_loading" for ev in statuses)
    assert all(ev["stage_label"] == "Loading model weights" for ev in statuses)
    assert any(ev["kind"] == "chunk" for ev in out)


async def test_stream_status_boot_stage_provider_failure_degrades_to_plain():
    # A broken provider must never break the stream — events just lose the
    # stage fields (pre-ACS-272 behaviour).
    client = _FakeStreamClient([
        _FakeStreamResponse(303, []),
        _FakeStreamResponse(200, [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"]),
    ])

    async def provider():
        raise RuntimeError("modal down")

    out = await _collect(
        proxymod.stream_post_with_status(
            client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0,
            boot_stage_provider=provider,
        )
    )
    statuses = [ev for ev in out if ev["kind"] == "status"]
    assert statuses
    assert all("stage" not in ev for ev in statuses)
    assert any(ev["kind"] == "chunk" for ev in out)


# --- ACS-277: silent stream-death fixes --------------------------------------
#
# A streamed cold-boot wait died with a clean EOF (no error frame) when the
# upstream leg ended without delivering a byte. These pin the new contract:
# empty-ended or mid-body-failed upstreams surface as TYPED errors, and the
# one safe retry (re-GET of a continuation result URL) actually happens.


class _MidBodyExplodingResponse(_FakeStreamResponse):
    """200 whose body raises after yielding its scripted chunks."""

    def __init__(self, status_code, chunks, exc):
        super().__init__(status_code, chunks)
        self._exc = exc

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk
        raise self._exc


async def test_stream_post_empty_continuation_body_retries_once_then_succeeds():
    client = _FakeStreamClient([
        _FakeStreamResponse(303, [], headers={"location": "/continuations/inv1"}),
        _FakeStreamResponse(200, []),  # continuation hop ends empty → one re-GET
        _FakeStreamResponse(200, [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"]),
    ])
    out = await _collect(
        proxymod.stream_post(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
    )
    assert out and b"ok" in out[0][0]
    # The retry must re-GET the result URL — never re-POST (second input).
    assert [c["method"] for c in client.calls] == ["POST", "GET", "GET"]


async def test_stream_post_empty_continuation_body_twice_raises_typed_error():
    client = _FakeStreamClient([
        _FakeStreamResponse(303, [], headers={"location": "/continuations/inv1"}),
        _FakeStreamResponse(200, []),
        _FakeStreamResponse(200, []),
    ])
    with pytest.raises(proxymod.UpstreamUnreachable, match="ended before any bytes"):
        await _collect(
            proxymod.stream_post(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
        )


async def test_stream_post_empty_body_on_original_post_never_reposts():
    client = _FakeStreamClient([_FakeStreamResponse(200, [])])
    with pytest.raises(proxymod.UpstreamUnreachable, match="ended before any bytes"):
        await _collect(
            proxymod.stream_post(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
        )
    assert [c["method"] for c in client.calls] == ["POST"]


async def test_stream_post_midbody_error_prefirstchunk_cold_hint_is_cold_boot():
    client = _FakeStreamClient([
        _MidBodyExplodingResponse(200, [], httpx.ReadTimeout("stalled mid-body")),
    ])
    ctx = proxymod.BackendContext(model_id="llama-405b", cold_hint=True)
    with pytest.raises(proxymod.ColdBootError):
        await _collect(
            proxymod.stream_post(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0, ctx=ctx)
        )


async def test_stream_post_midbody_error_after_first_chunk_is_typed_not_silent():
    client = _FakeStreamClient([
        _MidBodyExplodingResponse(
            200, [b'data: {"choices":[{"text":"hi"}]}\n\n'], httpx.ReadError("conn reset")
        ),
    ])
    got: list = []
    with pytest.raises(proxymod.UpstreamUnreachable, match="while reading upstream body"):
        async for item in proxymod.stream_post(
            client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0
        ):
            got.append(item)
    assert len(got) == 1  # the delivered chunk still reached the caller first


def test_server_restarting_payload_contract():
    p = proxymod.server_restarting_payload()
    assert p["error"]["code"] == "server_restarting"
    assert p["error"]["retryable"] is True
    assert p["error"]["type"] == "server_error"


async def test_stream_post_empty_body_then_midbody_error_on_retry_is_typed():
    # Review nit on #283: the retry GET's body-phase failure must also surface
    # as a typed error, not silence.
    client = _FakeStreamClient([
        _FakeStreamResponse(303, [], headers={"location": "/continuations/inv1"}),
        _FakeStreamResponse(200, []),  # empty hop -> one re-GET
        _MidBodyExplodingResponse(200, [], httpx.ReadError("reset on retry")),
    ])
    # Pre-first-byte failure on a continuation hop classifies as a cold boot
    # (the model is mid-boot; gen() turns this into the retryable
    # modal_cold_boot frame) — typed either way, never silence.
    with pytest.raises(proxymod.ColdBootError):
        await _collect(
            proxymod.stream_post(client, "https://up", "k", {"prompt": "x", "stream": True}, 5.0)
        )
    assert [c["method"] for c in client.calls] == ["POST", "GET", "GET"]
