"""Full-vocab logprobs (ACS-191): response gzip + the prompt-length guard.

Two runtime surfaces the schema tests don't reach:
  - ``_render_completion_json`` actually gzips a large completion body (and only
    when the client asked and the body is worth it), producing the real
    Starlette ``Response`` bytes that go on the wire.
  - ``check_full_vocab_prompt_logprobs`` rejects an over-long ``prompt_logprobs=-1``
    prompt using the real tokenizer, before it can OOM the model server.
"""

from __future__ import annotations

import gzip
import json
import os
import uuid

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fastapi.responses import JSONResponse, StreamingResponse

from wrapper import proxy as proxymod
from wrapper.auth import AuthedCaller
from wrapper.routes import api as api_routes
from wrapper.schemas import FULL_VOCAB_MAX_PROMPT_TOKENS, CompletionsRequest
from wrapper.services import completions as completion_svc


# --- gzip rendering ---------------------------------------------------------


def _big_logprobs_payload(n: int = 4000) -> dict:
    """A completion body shaped like a full-vocab prompt_logprobs response."""
    dist = {f"tok{i}": -round(i * 0.001, 6) for i in range(n)}
    return {
        "id": "cmpl-x",
        "object": "text_completion",
        "choices": [{"text": "", "index": 0, "logprobs": {"top_logprobs": [dist]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 0},
    }


async def test_render_gzips_large_body_when_client_accepts():
    payload = _big_logprobs_payload()
    resp = await api_routes._render_completion_json(200, payload, accept_gzip=True)
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.headers["vary"] == "Accept-Encoding"
    # Body is really gzip and round-trips to the exact same JSON.
    restored = json.loads(gzip.decompress(resp.body).decode("utf-8"))
    assert restored == payload
    # And it actually saved space (repetitive floats compress).
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(resp.body) < len(raw)


async def test_render_skips_gzip_when_client_does_not_accept():
    resp = await api_routes._render_completion_json(
        200, _big_logprobs_payload(), accept_gzip=False
    )
    assert "content-encoding" not in resp.headers
    assert resp.headers["vary"] == "Accept-Encoding"
    assert isinstance(resp, JSONResponse)


async def test_render_skips_gzip_below_min_size():
    # A tiny body: gzip overhead isn't worth it even if the client accepts it.
    payload = {"ok": True}
    resp = await api_routes._render_completion_json(200, payload, accept_gzip=True)
    assert "content-encoding" not in resp.headers
    assert resp.headers["vary"] == "Accept-Encoding"
    # Uncompressed bytes are the exact JSON (no second serialize needed).
    assert json.loads(resp.body) == payload
    assert resp.media_type == "application/json"


def test_accepts_gzip_header_parsing():
    def _req(accept):
        return SimpleNamespace(headers={"accept-encoding": accept})

    assert api_routes._accepts_gzip(_req("gzip, deflate, br")) is True
    assert api_routes._accepts_gzip(_req("GZIP")) is True
    assert api_routes._accepts_gzip(_req("identity")) is False
    assert api_routes._accepts_gzip(SimpleNamespace(headers={})) is False
    assert api_routes._accepts_gzip(None) is False


# --- full-vocab prompt-length guard -----------------------------------------


def _caller() -> AuthedCaller:
    return AuthedCaller(
        key_id=uuid.uuid4(),
        key_prefix="fvtest00",
        user_email="fv@example.local",
        monthly_token_budget=0,
        tokens_used_this_month=0,
    )


class _RecordingError:
    """Stand-in for the route's ``_error`` factory; captures the 400 it builds."""

    def __init__(self):
        self.calls = []

    async def __call__(self, session, request_id, caller, ip, endpoint, t0, code, kind, msg):
        self.calls.append({"code": code, "kind": kind, "msg": msg})
        return JSONResponse(status_code=code, content={"error": {"code": kind, "message": msg}})


async def _run_guard(parsed: CompletionsRequest):
    entry = SimpleNamespace(tokenizer_repo="gpt2", max_model_len=None)
    settings = SimpleNamespace(hf_token=None, served_model_name="gpt2")
    err = _RecordingError()
    resp = await completion_svc.check_full_vocab_prompt_logprobs(
        session=SimpleNamespace(),
        request_id="req_fv",
        caller=_caller(),
        ip=None,
        t0=0.0,
        entry=entry,
        parsed=parsed,
        settings=settings,
        error_response=err,
    )
    return resp, err


async def test_guard_rejects_long_full_vocab_prompt():
    # 300 repeats ≈ 1200 gpt2 tokens — over the 1024-token cap (ACS-198).
    long_prompt = "the quick brown fox " * 300
    parsed = CompletionsRequest.model_validate({"prompt": long_prompt, "prompt_logprobs": -1})
    resp, err = await _run_guard(parsed)
    assert resp is not None and resp.status_code == 400
    assert err.calls and err.calls[0]["kind"] == "full_vocab_prompt_too_long"
    assert str(FULL_VOCAB_MAX_PROMPT_TOKENS) in err.calls[0]["msg"]


async def test_guard_allows_short_full_vocab_prompt():
    parsed = CompletionsRequest.model_validate(
        {"prompt": "Paris is the capital of", "prompt_logprobs": -1}
    )
    resp, err = await _run_guard(parsed)
    assert resp is None
    assert err.calls == []


async def test_guard_keeps_tight_cap_for_activation_combo():
    # Full-vocab + activation params is served by the BUFFERED activation path
    # (the ACS-198 pass-through only handles plain completions), so it keeps
    # the old wrapper-RAM-sized 16-token bound. ~200 gpt2 tokens: passes the
    # 1024 pass-through cap but must fail the buffered cap.
    from wrapper.schemas import FULL_VOCAB_MAX_PROMPT_TOKENS_BUFFERED

    steering_vector = {
        "activations": {
            "data": "AAAA",
            "dtype": "int16",
            "original_dtype": "torch.bfloat16",
            "shape": [1, 4096],
            "compression": "zstd",
        },
        "layer_indices": [16],
        "scale": 1.0,
    }
    parsed = CompletionsRequest.model_validate(
        {
            "prompt": "the quick brown fox " * 40,
            "prompt_logprobs": -1,
            "apply_steering_vectors": [steering_vector],
        }
    )
    resp, err = await _run_guard(parsed)
    assert resp is not None and resp.status_code == 400
    assert err.calls and err.calls[0]["kind"] == "full_vocab_prompt_too_long"
    assert str(FULL_VOCAB_MAX_PROMPT_TOKENS_BUFFERED) in err.calls[0]["msg"]
    assert "activation" in err.calls[0]["msg"]


async def test_guard_noop_for_positive_prompt_logprobs():
    # A long prompt with a *positive* top-k is not a full-vocab request → no gate.
    parsed = CompletionsRequest.model_validate(
        {"prompt": "the quick brown fox " * 40, "prompt_logprobs": 20}
    )
    resp, err = await _run_guard(parsed)
    assert resp is None
    assert err.calls == []


# --- streaming pass-through serve path (ACS-198) -----------------------------
#
# The wrapper must forward a full-vocab body as bytes (never r.json() it),
# gzip on the fly when the client accepts it, and still commit usage plucked
# from the tail of the stream. These tests drive _serve_fullvocab_passthrough
# with a scripted proxy iterator — the proxy protocol itself is covered in
# test_proxy.py.

_BODY_CHUNKS = [
    b'{"id":"cmpl-1","object":"text_completion","choices":[{"text":"',
    b'ok"}],',
    b'"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}',
]


class _FakeSemaphore:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


async def _drain(resp: StreamingResponse) -> bytes:
    out = b""
    async for chunk in resp.body_iterator:
        out += chunk
    return out


def _serve_kwargs(monkeypatch, upstream_gen, *, accept_gzip: bool, cold_hint: bool = False):
    recorded = AsyncMock()
    usage_commits = AsyncMock()
    monkeypatch.setattr(api_routes, "_record_request", recorded)
    monkeypatch.setattr(api_routes.authmod, "commit_usage", usage_commits)
    monkeypatch.setattr(api_routes, "_mark_model_warm", lambda *_a: None)
    monkeypatch.setattr(
        api_routes.proxymod, "passthrough_post", lambda *a, **kw: upstream_gen
    )
    session = AsyncMock()
    request = AsyncMock()
    request.is_disconnected = AsyncMock(return_value=False)
    sem = _FakeSemaphore()
    kwargs = dict(
        request_id="req_fv_pt",
        caller=_caller(),
        ip=None,
        upstream_url="https://up/v1/completions",
        body={"prompt": "Paris is", "prompt_logprobs": -1},
        settings=SimpleNamespace(
            vllm_api_key="k", served_model_name="gpt2", hf_token=None
        ),
        session=session,
        http=object(),
        t0=0.0,
        model_id="m1",
        request=request,
        key_semaphore=sem,
        backend_ctx=proxymod.BackendContext(model_id="m1", cold_hint=cold_hint),
        breakers=None,
        effective_timeout=5.0,
        tokenizer_repo="gpt2",
        telemetry=None,
        accept_gzip=accept_gzip,
        breaker_key="m1",
    )
    return kwargs, recorded, usage_commits, sem


async def _scripted_iter(events):
    """Async generator yielding scripted (chunk, status) tuples / exceptions."""
    for ev in events:
        if isinstance(ev, float):
            await asyncio.sleep(ev)
            continue
        if isinstance(ev, Exception):
            raise ev
        yield ev


async def test_passthrough_serve_streams_gzip_and_commits_usage(monkeypatch):
    gen = _scripted_iter([(_BODY_CHUNKS[0], 200), (_BODY_CHUNKS[1], None), (_BODY_CHUNKS[2], None)])
    kwargs, recorded, usage_commits, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=True)
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert isinstance(resp, StreamingResponse)
    assert resp.status_code == 200
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.headers["vary"] == "Accept-Encoding"
    body = await _drain(resp)
    assert gzip.decompress(body) == b"".join(_BODY_CHUNKS)
    # Usage was plucked from the tail and committed; request recorded as 200.
    usage_commits.assert_awaited_once()
    assert usage_commits.await_args.args[2:] == (7, 1)
    assert recorded.await_args.kwargs["status_code"] == 200
    assert recorded.await_args.kwargs["n_prompt"] == 7
    assert recorded.await_args.kwargs["n_completion"] == 1
    # The generator owns and released the per-key slot exactly once.
    assert sem.released == 1


def test_usage_extraction_finds_head_usage_but_not_trailing_blob():
    # ACS-250: extract_usage scans for the last `"usage"` in the given window, so
    # it works on a HEAD buffer (activation shape, usage before the big blob) and
    # returns None on the trailing `activations` blob — exactly why activation
    # needs usage_at_head rather than the full-vocab tail window.
    head = (
        b'{"id":"cmpl-a","choices":[{"text":"ok"}],'
        b'"usage":{"prompt_tokens":7,"completion_tokens":1},"activations":"'
    )
    assert proxymod.extract_usage_from_json_tail(head) == {
        "prompt_tokens": 7,
        "completion_tokens": 1,
    }
    assert proxymod.extract_usage_from_json_tail(b"A" * 40000 + b'"}') is None


async def test_passthrough_serve_activation_commits_usage_from_head(monkeypatch):
    # ACS-250: an activation capture body carries `usage` in the HEAD, before a
    # multi-KB `activations` blob, so it is NOT in the last PASSTHROUGH_USAGE_TAIL
    # bytes. usage_at_head=True must retain the head and still commit token counts.
    head = (
        b'{"id":"cmpl-a","object":"text_completion","choices":[{"text":"ok"}],'
        b'"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8},'
        b'"kv_transfer_params":null,"activations":"'
    )
    blob = b"A" * 40000  # > PASSTHROUGH_USAGE_TAIL_BYTES → usage absent from the tail
    close = b'"}'
    gen = _scripted_iter([(head, 200), (blob, None), (close, None)])
    kwargs, recorded, usage_commits, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=False)
    kwargs["body"] = {"prompt": "Paris is", "vllm_xargs": {"output_residual_stream": True}}
    kwargs["usage_at_head"] = True
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert isinstance(resp, StreamingResponse)
    assert resp.status_code == 200
    body = await _drain(resp)
    assert body == head + blob + close  # streamed through unbuffered
    # Usage plucked from the HEAD (absent from the last 16 KiB) and committed.
    usage_commits.assert_awaited_once()
    assert usage_commits.await_args.args[2:] == (7, 1)
    assert recorded.await_args.kwargs["status_code"] == 200
    assert recorded.await_args.kwargs["n_prompt"] == 7
    assert sem.released == 1


_FV = api_routes.FULL_VOCAB_SENTINEL


@pytest.mark.parametrize(
    "prompt_logprobs, capture, has_activation, expected",
    [
        (None, True, True, "head"),   # capture-only → head window
        (None, False, True, None),    # steering-only → buffered (usage in tail, no blob)
        (_FV, False, False, "tail"),  # full-vocab-only → tail window (unchanged)
        (_FV, True, True, None),      # full-vocab + capture combo → buffered
        (_FV, False, True, None),     # full-vocab + steering → buffered
        (None, False, False, None),   # plain → buffered
    ],
)
def test_passthrough_usage_mode_routes_only_capture_to_head(
    prompt_logprobs, capture, has_activation, expected
):
    # ACS-250 regression guard: the head window must key on actual capture
    # (output_residual_stream), NOT has_activation_params — else steering-only
    # (usage in the tail, no activations blob) would take the head window and
    # under-bill a long steered generation.
    parsed = SimpleNamespace(
        prompt_logprobs=prompt_logprobs,
        output_residual_stream=capture,
        has_activation_params=has_activation,
    )
    assert api_routes._passthrough_usage_mode(parsed) == expected


async def test_passthrough_serve_identity_when_client_declines_gzip(monkeypatch):
    gen = _scripted_iter([(b"".join(_BODY_CHUNKS), 200)])
    kwargs, recorded, _usage, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=False)
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert isinstance(resp, StreamingResponse)
    assert "content-encoding" not in resp.headers
    body = await _drain(resp)
    assert body == b"".join(_BODY_CHUNKS)
    assert json.loads(body)["usage"]["prompt_tokens"] == 7
    assert sem.released == 1


async def test_passthrough_serve_cold_boot_maps_to_503(monkeypatch):
    gen = _scripted_iter([proxymod.ColdBootError(303, 2)])
    kwargs, recorded, _usage, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=False)
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert not isinstance(resp, StreamingResponse)
    assert resp.status_code == 503
    payload = json.loads(bytes(resp.body))
    assert payload["error"]["code"] == "modal_cold_boot"
    assert recorded.await_args.kwargs["status_code"] == 503
    # Error path: the route handler's finally owns the slot, not the generator.
    assert sem.released == 0


async def test_passthrough_serve_forwards_upstream_4xx_status(monkeypatch):
    err_body = b'{"error":{"message":"bad params","code":"invalid_request"}}'
    gen = _scripted_iter([(err_body, 400)])
    kwargs, recorded, _usage, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=False)
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert not isinstance(resp, StreamingResponse)
    assert resp.status_code == 400
    assert json.loads(bytes(resp.body))["error"]["code"] == "invalid_request"
    assert recorded.await_args.kwargs["error_kind"] == "upstream_4xx"


async def test_passthrough_serve_slow_upstream_gets_keepalives_then_body(monkeypatch):
    # First upstream byte arrives after the grace window → the response commits
    # to 200 and emits JSON-whitespace keepalives, then the body. Leading
    # whitespace is legal before a JSON value (RFC 8259).
    monkeypatch.setattr(api_routes, "PUBLIC_RESPONSE_GRACE_S", 0.01)
    monkeypatch.setattr(api_routes, "PUBLIC_KEEPALIVE_INTERVAL_S", 0.01)
    gen = _scripted_iter([0.05, (b"".join(_BODY_CHUNKS), 200)])
    kwargs, recorded, usage_commits, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=False)
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert isinstance(resp, StreamingResponse)
    assert resp.status_code == 200
    body = await _drain(resp)
    assert body.startswith(b" ")
    assert json.loads(body)["usage"]["prompt_tokens"] == 7
    usage_commits.assert_awaited_once()
    assert sem.released == 1


async def test_passthrough_serve_slow_cold_boot_error_inside_committed_200(monkeypatch):
    # Late failure after the grace window: status is committed to 200; the JSON
    # error body carries the real error while api_requests records 503 — the
    # same contract as _serve_nonstream_with_keepalive.
    monkeypatch.setattr(api_routes, "PUBLIC_RESPONSE_GRACE_S", 0.01)
    monkeypatch.setattr(api_routes, "PUBLIC_KEEPALIVE_INTERVAL_S", 0.01)
    gen = _scripted_iter([0.05, proxymod.ColdBootError(303, 5)])
    kwargs, recorded, _usage, sem = _serve_kwargs(
        monkeypatch, gen, accept_gzip=False, cold_hint=True
    )
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert isinstance(resp, StreamingResponse)
    assert resp.status_code == 200
    body = await _drain(resp)
    assert json.loads(body)["error"]["code"] == "modal_cold_boot"
    assert recorded.await_args.kwargs["status_code"] == 503
    assert sem.released == 1


async def test_passthrough_serve_records_mid_stream_abort(monkeypatch):
    # An upstream failure after bytes have flowed can't change the status any
    # more, but it must still leave an api_requests row (the buffered path
    # recorded every upstream error).
    gen = _scripted_iter([(_BODY_CHUNKS[0], 200), RuntimeError("upstream died")])
    kwargs, recorded, usage_commits, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=False)
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert isinstance(resp, StreamingResponse)
    try:
        await _drain(resp)
    except RuntimeError:
        pass
    assert recorded.await_args.kwargs["error_kind"] == "passthrough_aborted"
    assert recorded.await_args.kwargs["status_code"] == 502
    usage_commits.assert_not_awaited()
    assert sem.released == 1


async def test_passthrough_serve_usage_fallback_counts_prompt_wrapper_side(monkeypatch):
    # Body whose tail has no parseable usage block: the wrapper must floor the
    # commit with its own prompt-token count instead of metering zero.
    body_without_usage = b'{"id":"cmpl-1","choices":[{"text":"ok"}]}'
    gen = _scripted_iter([(body_without_usage, 200)])
    kwargs, recorded, usage_commits, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=False)

    class _Counter:
        def encode(self, text):
            return text.split()

    monkeypatch.setattr(api_routes, "get_token_counter", lambda *a, **kw: _Counter())
    monkeypatch.setattr(
        api_routes.completion_svc, "count_prompt_tokens", lambda prompt, c: 42
    )
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    body = await _drain(resp)
    assert body == body_without_usage
    usage_commits.assert_awaited_once()
    assert usage_commits.await_args.args[2:] == (42, 0)
    assert recorded.await_args.kwargs["n_prompt"] == 42
    assert sem.released == 1


async def test_passthrough_serve_gzip_keepalive_stream_decodes(monkeypatch):
    # gzip + keepalive mode: whitespace ticks and body all live inside one
    # gzip container (sync-flushed so bytes still hit the wire per tick).
    monkeypatch.setattr(api_routes, "PUBLIC_RESPONSE_GRACE_S", 0.01)
    monkeypatch.setattr(api_routes, "PUBLIC_KEEPALIVE_INTERVAL_S", 0.01)
    gen = _scripted_iter([0.05, (b"".join(_BODY_CHUNKS), 200)])
    kwargs, _recorded, _usage, sem = _serve_kwargs(monkeypatch, gen, accept_gzip=True)
    resp = await api_routes._serve_fullvocab_passthrough(**kwargs)
    assert isinstance(resp, StreamingResponse)
    assert resp.headers["content-encoding"] == "gzip"
    body = gzip.decompress(await _drain(resp))
    assert json.loads(body)["usage"]["prompt_tokens"] == 7
    assert sem.released == 1
