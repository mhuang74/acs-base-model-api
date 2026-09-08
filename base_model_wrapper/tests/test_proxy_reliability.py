"""Reliability-hardening tests for wrapper/proxy.py.

Covers, by item from the audit:
  1. ``UpstreamUnreachable`` raised on httpx ConnectError / TimeoutException
     past the retry budget; payload includes backend identity.
  2. Cold boots receive one long read budget, capped only by the public request
     deadline; the removed 25/60-second clamps must not return.
  3. 5xx retry: succeeds after N failures; exhausts to ``UpstreamServerError``
     past budget. Exponential backoff helper computes the right curve.
  6. ``classify_upstream_error_body`` labels OOM / context-length / engine-dead
     correctly.
  7. ``cold_boot_error_payload`` carries ``retry_after_seconds``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")

from wrapper import proxy as proxymod


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Pin every backoff to ~0 so retry tests are fast."""
    monkeypatch.setattr(proxymod, "COLD_BOOT_BACKOFF_S", 0.0)
    monkeypatch.setattr(proxymod, "RETRY_5XX_BASE_BACKOFF_S", 0.0)
    monkeypatch.setattr(proxymod, "RETRY_5XX_JITTER_S", 0.0)


# --- Fake httpx primitives ------------------------------------------------


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
        self.headers = httpx.Headers(headers)

    def json(self):
        import json
        if self._body is None:
            raise json.JSONDecodeError("no body", "", 0)
        return self._body


class _ScriptedPostClient:
    """``request`` returns scripted responses; if an entry is an Exception
    instance, raise it instead. Lets tests interleave network errors and
    HTTP responses arbitrarily."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def request(self, method: str, url: str, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeStreamResponse:
    """Async-context-manager mimic of ``client.stream(...)``'s return value.

    If ``raise_on_enter`` is set, ``__aenter__`` raises it — simulating Modal
    stalling the socket open (ReadTimeout) during an 8×H200 cold boot.
    """

    def __init__(self, status_code: int = 200, chunks: list[bytes] | None = None,
                 raise_on_enter: BaseException | None = None):
        self.status_code = status_code
        self._chunks = chunks or []
        self._raise = raise_on_enter

    async def __aenter__(self):
        if self._raise is not None:
            raise self._raise
        return self

    async def __aexit__(self, *_a):
        return None

    async def aread(self) -> bytes:
        return b"".join(self._chunks)

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk


class _ScriptedStreamClient:
    """``stream`` returns scripted ``_FakeStreamResponse`` objects in order."""

    def __init__(self, responses: list):
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


# --- UpstreamUnreachable on persistent network failure --------------------


async def test_upstream_unreachable_on_connect_error_past_budget():
    client = _ScriptedPostClient(
        [httpx.ConnectError("DNS failed") for _ in range(proxymod.RETRY_5XX_MAX_RETRIES + 1)]
    )
    ctx = proxymod.BackendContext(model_id="llama-405b", gpu_shape="8×H200")
    with pytest.raises(proxymod.UpstreamUnreachable) as exc_info:
        await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 5.0, ctx=ctx)
    exc = exc_info.value
    assert exc.attempts == proxymod.RETRY_5XX_MAX_RETRIES + 1
    assert "ConnectError" in exc.reason
    assert exc.ctx.model_id == "llama-405b"
    # The error payload helper includes the backend identity.
    payload = proxymod.upstream_unreachable_payload(exc)
    assert payload["error"]["code"] == "upstream_unreachable"
    assert payload["error"]["model_id"] == "llama-405b"
    assert payload["error"]["gpu_shape"] == "8×H200"


async def test_network_error_recovers_within_budget():
    # First two calls fail, third succeeds.
    client = _ScriptedPostClient([
        httpx.ConnectError("transient"),
        httpx.ReadError("transient"),
        _FakeResponse(200, body={"choices": [{"text": "ok"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
    ])
    status, payload, _ms = await proxymod.post_nonstream(
        client, "https://up", "k", {"prompt": "x"}, 5.0,
    )
    assert status == 200
    assert payload["choices"][0]["text"] == "ok"
    assert len(client.calls) == 3


# --- 5xx retry / UpstreamServerError --------------------------------------


async def test_5xx_recovers_within_budget():
    client = _ScriptedPostClient([
        _FakeResponse(503, text="overloaded"),
        _FakeResponse(502, text="bad gateway"),
        _FakeResponse(200, body={"choices": [{"text": "yay"}], "usage": {}}),
    ])
    status, payload, _ms = await proxymod.post_nonstream(
        client, "https://up", "k", {"prompt": "x"}, 5.0,
    )
    assert status == 200
    assert len(client.calls) == 3


async def test_5xx_exhausts_to_upstream_server_error():
    client = _ScriptedPostClient(
        [_FakeResponse(500, text="boom") for _ in range(proxymod.RETRY_5XX_MAX_RETRIES + 1)]
    )
    ctx = proxymod.BackendContext(model_id="trinity-truebase")
    with pytest.raises(proxymod.UpstreamServerError) as exc_info:
        await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 5.0, ctx=ctx)
    exc = exc_info.value
    assert exc.upstream_status == 500
    assert exc.attempts == proxymod.RETRY_5XX_MAX_RETRIES + 1
    assert exc.ctx.model_id == "trinity-truebase"
    # Payload contract.
    payload = proxymod.upstream_server_error_payload(exc)
    assert payload["error"]["upstream_status"] == 500
    assert payload["error"]["model_id"] == "trinity-truebase"


async def test_5xx_then_network_error_then_success():
    """Heterogeneous errors should still be retried up to budget."""
    client = _ScriptedPostClient([
        _FakeResponse(502, text="bad"),
        httpx.ConnectError("blip"),
        _FakeResponse(200, body={"choices": [], "usage": {}}),
    ])
    status, _payload, _ms = await proxymod.post_nonstream(
        client, "https://up", "k", {"prompt": "x"}, 5.0,
    )
    assert status == 200
    assert len(client.calls) == 3


async def test_4xx_is_not_retried():
    """Client errors propagate immediately."""
    client = _ScriptedPostClient([
        _FakeResponse(400, body={"error": {"message": "bad"}}),
    ])
    status, payload, _ms = await proxymod.post_nonstream(
        client, "https://up", "k", {"prompt": "x"}, 5.0,
    )
    assert status == 400
    assert len(client.calls) == 1  # no retries on 4xx


async def test_client_ish_5xx_body_is_not_retried():
    """A 5xx whose body is a DETERMINISTIC client error (out-of-range steering
    layer_index reflected by vLLM-Lens as a 500) must raise immediately without
    burning the retry budget — retrying can never succeed and just adds ~10s
    latency (ACS-322). Contrast test_5xx_exhausts_to_upstream_server_error,
    where a generic 500 body still retries to exhaustion."""
    client = _ScriptedPostClient([
        _FakeResponse(500, text="ValueError: layer_index 32 out of range [0, 32)"),
    ])
    with pytest.raises(proxymod.UpstreamServerError) as exc_info:
        await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 5.0)
    exc = exc_info.value
    assert exc.attempts == 1  # first attempt, no retry
    assert exc.upstream_kind == "vllm_invalid_request"
    assert len(client.calls) == 1  # exactly one upstream call


# --- One long request budget (no short cold-attempt clamp) ----------------


async def test_request_read_timeout_uses_public_deadline_without_short_clamp():
    """The upstream read may span Modal cold boot, up to the public deadline."""
    client = _ScriptedPostClient([_FakeResponse(200, body={"choices": [], "usage": {}})])
    await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 1200.0)
    timeout = client.calls[0]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read <= proxymod.PUBLIC_REQUEST_TIMEOUT_S
    assert timeout.read > 60.0  # regression: no former 25/60-second clamp


# --- Overall cold-boot deadline classification -----------------------------
#
# The read timeout covers the whole public request budget. In production it
# therefore fires only at that deadline. Unit fakes raise immediately, but the
# resulting classification is the same: cold-hinted requests become ColdBoot.


async def test_cold_hint_readtimeout_is_coldboot_nonstream():
    client = _ScriptedPostClient([httpx.ReadTimeout("modal stalled the socket")])
    ctx = proxymod.BackendContext(model_id="trinity-truebase", gpu_shape="8×H200", cold_hint=True)
    with pytest.raises(proxymod.ColdBootError) as exc_info:
        await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 1200.0, ctx=ctx)
    # No status code seen (stall, not a 3xx) → 0; surfaces in the 503 payload.
    assert exc_info.value.upstream_status == 0
    # ReadTimeout means the single overall read budget was exhausted; retrying
    # would create a second model invocation.
    assert len(client.calls) == 1
    timeout = client.calls[0]["timeout"]
    assert timeout.read <= proxymod.PUBLIC_REQUEST_TIMEOUT_S
    assert timeout.read > 60.0
    # The late failure payload still carries modal_cold_boot + retry_after.
    # (Use an explicit cadence here: the autouse _fast_backoff fixture pins
    # COLD_BOOT_BACKOFF_S to 0 for speed, which would zero the retry_after.)
    payload = proxymod.cold_boot_error_payload(exc_info.value.upstream_status, retry_after_s=30.0)
    assert payload["error"]["code"] == "modal_cold_boot"
    assert payload["error"]["retry_after_seconds"] == 30


async def test_cold_hint_connecttimeout_still_unreachable_nonstream():
    client = _ScriptedPostClient(
        [httpx.ConnectTimeout("router slow") for _ in range(proxymod.RETRY_5XX_MAX_RETRIES + 1)]
    )
    ctx = proxymod.BackendContext(model_id="llama-405b", cold_hint=True)
    with pytest.raises(proxymod.UpstreamUnreachable) as exc_info:
        await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 1200.0, ctx=ctx)
    assert exc_info.value.attempts == proxymod.RETRY_5XX_MAX_RETRIES + 1
    assert len(client.calls) == proxymod.RETRY_5XX_MAX_RETRIES + 1


async def test_warm_model_readtimeout_still_unreachable_502():
    """Expiry of the overall read budget on a warm model surfaces as 502."""
    client = _ScriptedPostClient([httpx.ReadTimeout("dead socket")])
    ctx = proxymod.BackendContext(model_id="trinity-truebase", cold_hint=False)
    with pytest.raises(proxymod.UpstreamUnreachable) as exc_info:
        await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 5.0, ctx=ctx)
    assert exc_info.value.attempts == 1
    assert len(client.calls) == 1


async def test_cold_hint_hard_connecterror_still_unreachable_502():
    """A hard ConnectError (DNS / refused) is a real outage even on a cold
    model — Modal's router stays reachable while a container boots, so this
    must NOT be masked as a cold boot."""
    client = _ScriptedPostClient(
        [httpx.ConnectError("DNS failed") for _ in range(proxymod.RETRY_5XX_MAX_RETRIES + 1)]
    )
    ctx = proxymod.BackendContext(model_id="trinity-truebase", cold_hint=True)
    with pytest.raises(proxymod.UpstreamUnreachable):
        await proxymod.post_nonstream(client, "https://up", "k", {"prompt": "x"}, 5.0, ctx=ctx)
    assert len(client.calls) == proxymod.RETRY_5XX_MAX_RETRIES + 1


async def test_modal_303_follows_same_invocation_as_get():
    """A Modal continuation is followed as GET instead of re-POSTing."""
    client = _ScriptedPostClient([
        _FakeResponse(303, headers={"location": "/continuation/abc"}),
        _FakeResponse(200, body={"choices": [{"text": "done"}], "usage": {}}),
    ])
    ctx = proxymod.BackendContext(model_id="trinity-truebase", cold_hint=True)
    status, payload, _elapsed_ms = await proxymod.post_nonstream(
        client, "https://up/v1/completions", "k", {"prompt": "x"}, 1200.0, ctx=ctx
    )
    assert status == 200
    assert payload["choices"][0]["text"] == "done"
    assert [call["method"] for call in client.calls] == ["POST", "GET"]
    assert "json" in client.calls[0]
    assert "json" not in client.calls[1]
    assert client.calls[1]["url"] == "https://up/continuation/abc"
    assert all(call["timeout"].read <= proxymod.PUBLIC_REQUEST_TIMEOUT_S for call in client.calls)
    assert all(call["timeout"].read > 60.0 for call in client.calls)


async def test_cold_hint_readtimeout_is_coldboot_stream():
    """Streaming variant classifies expiry of its one read budget as cold boot."""
    client = _ScriptedStreamClient([
        _FakeStreamResponse(raise_on_enter=httpx.ReadTimeout("modal stalled")),
    ])
    ctx = proxymod.BackendContext(model_id="trinity-truebase", cold_hint=True)
    gen = proxymod.stream_post(client, "https://up", "k", {"prompt": "x"}, 1200.0, ctx=ctx)
    with pytest.raises(proxymod.ColdBootError):
        await _collect(gen)
    # Single open attempt with the same long public-deadline read budget.
    assert len(client.calls) == 1
    timeout = client.calls[0]["timeout"]
    assert timeout.read <= proxymod.PUBLIC_REQUEST_TIMEOUT_S
    assert timeout.read > 60.0


async def test_warm_model_readtimeout_still_unreachable_stream():
    client = _ScriptedStreamClient([
        _FakeStreamResponse(raise_on_enter=httpx.ReadTimeout("dead")),
    ])
    ctx = proxymod.BackendContext(model_id="trinity-truebase", cold_hint=False)
    gen = proxymod.stream_post(client, "https://up", "k", {"prompt": "x"}, 5.0, ctx=ctx)
    with pytest.raises(proxymod.UpstreamUnreachable):
        await _collect(gen)
    assert len(client.calls) == 1


async def test_stream_5xx_is_single_attempt():
    client = _ScriptedStreamClient([
        _FakeStreamResponse(status_code=503, chunks=[b"overloaded"]),
    ])
    gen = proxymod.stream_post(client, "https://up", "k", {"prompt": "x"}, 1200.0)
    with pytest.raises(proxymod.UpstreamServerError) as exc_info:
        await _collect(gen)
    assert exc_info.value.upstream_status == 503
    assert exc_info.value.attempts == 1
    assert len(client.calls) == 1


# --- classify_upstream_error_body -----------------------------------------


def test_classify_recognises_oom():
    assert proxymod.classify_upstream_error_body("CUDA out of memory") == "vllm_oom"
    assert proxymod.classify_upstream_error_body(b"OOM error") == "vllm_oom"
    assert proxymod.classify_upstream_error_body({"detail": "cuda_oom"}) == "vllm_oom"


def test_classify_labels_out_of_range_layer_index_as_invalid_request():
    # vLLM-Lens raises ValueError -> 500 for an out-of-range steering
    # layer_index; recognise it as a client-ish invalid request so the proxy
    # stops retrying and the /v1 handler returns 400 (ACS-322).
    assert (
        proxymod.classify_upstream_error_body(
            "ValueError: layer_index 32 out of range [0, 32)"
        )
        == "vllm_invalid_request"
    )
    assert proxymod.classify_upstream_error_body("layer_index 999 out of range [0, 126)") == (
        "vllm_invalid_request"
    )


def test_classify_recognises_context_length():
    assert proxymod.classify_upstream_error_body(
        "maximum context length 8192"
    ) == "vllm_context_length"
    assert proxymod.classify_upstream_error_body(
        "Request exceeds max_model_len"
    ) == "vllm_context_length"


def test_classify_recognises_engine_dead():
    assert proxymod.classify_upstream_error_body(
        "engine has crashed"
    ) == "vllm_engine_dead"


def test_classify_returns_none_on_unknown():
    assert proxymod.classify_upstream_error_body("some random text") is None
    assert proxymod.classify_upstream_error_body(None) is None
    assert proxymod.classify_upstream_error_body("") is None


# --- Cold-boot retry_after ------------------------------------------------


def test_cold_boot_payload_includes_retry_after(monkeypatch):
    # Restore a non-zero default cadence (the autouse fixture pins it to 0 for
    # other retry tests).
    monkeypatch.setattr(proxymod, "COLD_BOOT_BACKOFF_S", 30.0)
    p = proxymod.cold_boot_error_payload(303)
    err = p["error"]
    assert err["retry_after_seconds"] == 30
    assert err["code"] == "modal_cold_boot"
    assert "warming up" in err["message"]
    assert err["retryable"] is True


def test_cold_boot_payload_honours_explicit_retry_after():
    p = proxymod.cold_boot_error_payload(303, retry_after_s=5.0)
    assert p["error"]["retry_after_seconds"] == 5


# --- Exponential backoff curve --------------------------------------------


def test_exp_backoff_grows_then_caps():
    # With pinned base + zero jitter, the values are deterministic.
    proxymod.RETRY_5XX_BASE_BACKOFF_S = 1.0
    proxymod.RETRY_5XX_JITTER_S = 0.0
    values = [proxymod._exp_backoff(i) for i in range(6)]
    # 1, 2, 4, 8, 16, 16 — caps at 2^4 = 16
    assert values == [1.0, 2.0, 4.0, 8.0, 16.0, 16.0]


# --- BackendContext -------------------------------------------------------


def test_backend_context_log_fields():
    ctx = proxymod.BackendContext(model_id="m", gpu_shape="g", modal_app_name="a")
    assert ctx.as_log_fields() == {
        "upstream_model": "m",
        "upstream_gpu": "g",
        "upstream_app": "a",
    }
