"""End-to-end routing tests for activation harvesting / steering (ACS-199 PR2).

The wrapper dispatches a ``/v1/completions`` request carrying activation params
(``output_residual_stream`` / ``apply_steering_vectors``) to the model's separate
activation upstream (``acs-<id>-activation``), nesting the controls under
``vllm_xargs``; a plain completion goes to the normal upstream untouched; and an
activation request against a model with no activation upstream is rejected.

We capture at the ``proxy.post_nonstream`` boundary — the exact point the routing
decision (which URL + what body) has been made — so the assertions pin the
dispatch, not vLLM behavior. DB-gated like ``test_multi_model.py`` (needs a
migrated Postgres via ``TEST_DATABASE_URL``).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run activation-routing tests",
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-activation-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ["MODELS_REGISTRY_JSON"] = json.dumps(
        {
            # Activation-capable model: has a distinct activation upstream.
            "act": {
                "upstream_url": "https://act-workbench.example",
                "activation_upstream_url": "https://act-activation.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "test",
                "status": "live",
                "n_layers": 32,  # enables steering layer_index range validation (ACS-322)
            },
            # Plain model: no activation upstream → activation requests rejected.
            "plain": {
                "upstream_url": "https://plain-workbench.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "test",
                "status": "live",
            },
        }
    )
    os.environ["DEFAULT_MODEL_ID"] = "act"
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True
    os.environ.pop("MODELS_REGISTRY_JSON", None)
    os.environ.pop("DEFAULT_MODEL_ID", None)


def _make_key(activation_budget: int = 0) -> str:
    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.keys import generate as generate_key
    from wrapper.models import ApiKey, User

    async def _setup() -> str:
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                u = User(email=f"act-{uuid.uuid4().hex[:6]}@example.local")
                s.add(u)
                await s.flush()
                gk = generate_key()
                s.add(
                    ApiKey(
                        user_id=u.id,
                        key_hash=gk.hash_,
                        key_prefix=gk.prefix,
                        monthly_token_budget=0,  # unlimited → skips tokenizer path
                        monthly_activation_budget=activation_budget,  # 0 = unlimited
                    )
                )
                return gk.plaintext
        finally:
            await engine.dispose()

    return asyncio.run(_setup())


def _latest_request_row():
    """Most recent api_requests row (single test DB, requests are sequential)."""
    from sqlalchemy import desc, select

    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.models import ApiRequest

    async def _fetch():
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                return (
                    await s.execute(select(ApiRequest).order_by(desc(ApiRequest.id)).limit(1))
                ).scalar_one()
        finally:
            await engine.dispose()

    return asyncio.run(_fetch())


_CANNED = {
    "id": "cmpl-test",
    "object": "text_completion",
    "choices": [{"text": "ok", "index": 0, "finish_reason": "stop", "logprobs": None}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def captured(monkeypatch):
    """Capture the (url, body) handed to the upstream, returning a canned reply.

    Patches BOTH proxy seams a ``/v1/completions`` request can take: the buffered
    ``post_nonstream`` (plain + steering requests) and the streamed
    ``passthrough_post`` (activation *capture* requests route here since ACS-250,
    5cb92cf). Without the second patch, ``output_residual_stream`` requests make a
    real network call and the fixture records nothing (ACS-309).
    """
    calls: list[tuple[str, dict]] = []

    async def fake_post_nonstream(http, url, api_key, body, timeout, ctx=None):
        calls.append((url, body))
        return 200, dict(_CANNED), 5.0

    async def fake_passthrough_post(http, url, api_key, body, timeout, *, ctx=None):
        calls.append((url, body))
        # passthrough_post yields (chunk, status); status is set on the first tuple.
        yield json.dumps(_CANNED).encode(), 200

    monkeypatch.setattr("wrapper.proxy.post_nonstream", fake_post_nonstream)
    monkeypatch.setattr("wrapper.proxy.passthrough_post", fake_passthrough_post)
    return calls


@dbtest
def test_plain_completion_hits_workbench_upstream(client, captured):
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4},
    )
    assert r.status_code == 200, r.text
    assert len(captured) == 1
    url, body = captured[0]
    assert url == "https://act-workbench.example/v1/completions"
    assert "vllm_xargs" not in body
    assert "output_residual_stream" not in body


@dbtest
def test_capture_request_routes_to_activation_upstream(client, captured):
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4, "output_residual_stream": True},
    )
    assert r.status_code == 200, r.text
    assert len(captured) == 1
    url, body = captured[0]
    # Routed to the activation engine, not the workbench.
    assert url == "https://act-activation.example/v1/completions"
    # The engine takes the bare bool `true` (a list 400s); captures all layers.
    assert body["vllm_xargs"] == {"output_residual_stream": True}
    # The control never leaks as a top-level vLLM field.
    assert "output_residual_stream" not in body


@dbtest
def test_capture_short_prompt_large_max_tokens_rejected_by_cap(client, captured):
    """ACS-255: the activation cap counts prompt + generated − 1, so a short
    prompt with a large max_tokens is rejected pre-flight (the buffered response
    would otherwise be enormous) — exercising the full route wiring of the
    resolved max_tokens into the gate. `act` has no per-model cap → default 64."""
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 1000, "output_residual_stream": True},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "activation_prompt_too_long"
    # Rejected pre-flight: never reached the engine.
    assert captured == []


@dbtest
def test_capture_batched_prompt_rejected_pre_flight(client, captured):
    """ACS-255: capture is single-sequence only; a batched prompt is rejected at
    request validation (the app maps the pydantic model_validator error to a 400
    invalid_request) and never reaches the engine."""
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "act",
            "prompt": ["one", "two"],
            "max_tokens": 4,
            "output_residual_stream": True,
        },
    )
    assert r.status_code == 400, r.text
    body = r.json()["error"]
    assert body["code"] == "invalid_request"
    assert "batched prompt" in body["message"]
    assert captured == []


@dbtest
def test_steering_request_routes_to_activation_upstream(client, captured):
    import json as _json

    key = _make_key()
    sv = {
        "activations": {
            "data": "AAAA", "dtype": "int16", "original_dtype": "torch.bfloat16",
            "shape": [1, 4096], "compression": "zstd",
        },
        "layer_indices": [3],
        "scale": 2.0,
    }
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "act",
            "prompt": "hi",
            "max_tokens": 4,
            "apply_steering_vectors": [sv],
        },
    )
    assert r.status_code == 200, r.text
    url, body = captured[0]
    assert url == "https://act-activation.example/v1/completions"
    # vLLM-Lens wants a json.dumps'd STRING of the vector list.
    raw = body["vllm_xargs"]["apply_steering_vectors"]
    assert isinstance(raw, str)
    assert _json.loads(raw)[0]["layer_indices"] == [3]


@dbtest
def test_span_steering_expands_and_tiles_through_real_app(client, captured):
    """End-to-end through the real ASGI app: a 2-D vector + position_spans
    reaches the activation upstream as the expanded explicit-indices path — real
    position_indices + a tiled 3-D tensor, and NO position_spans key leaking to
    the engine."""
    import base64 as _b64
    import json as _json

    import zstandard as _zstd

    key = _make_key()
    # real 1-layer 2-D codec so the wrapper can tile it
    raw = bytes([7]) * (4096 * 2)
    codec = {
        "data": _b64.b64encode(_zstd.ZstdCompressor(level=1).compress(raw)).decode(),
        "dtype": "int16", "original_dtype": "torch.bfloat16",
        "shape": [1, 4096], "compression": "zstd",
    }
    sv = {"activations": codec, "layer_indices": [3], "scale": 2.0,
          "position_spans": [[5, 8]]}  # 3 positions
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4,
              "apply_steering_vectors": [sv]},
    )
    assert r.status_code == 200, r.text
    _url, body = captured[0]
    forwarded = _json.loads(body["vllm_xargs"]["apply_steering_vectors"])[0]
    assert "position_spans" not in forwarded  # never reaches the engine
    assert forwarded["position_indices"] == [5, 6, 7]
    assert forwarded["activations"]["shape"] == [1, 3, 4096]


_STEER_VEC = {
    "activations": {
        "data": "AAAA", "dtype": "int16", "original_dtype": "torch.bfloat16",
        "shape": [1, 4096], "compression": "zstd",
    },
    "scale": 1.0,
}


@dbtest
def test_steering_layer_index_out_of_range_is_rejected_preflight(client, captured):
    """Fix #1 (ACS-322): an out-of-range steering ``layer_index`` is rejected at
    the wrapper boundary with a 400 BEFORE the upstream is called — not
    forwarded to vLLM-Lens (which would raise ValueError -> 500 -> a mislabelled
    upstream 5xx). ``act`` has ``n_layers=32`` so valid indices are 0-31."""
    key = _make_key()
    sv = {**_STEER_VEC, "layer_indices": [32]}
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4, "apply_steering_vectors": [sv]},
    )
    assert r.status_code == 400, r.text
    assert "layer_index" in r.text and "out of range" in r.text
    assert captured == []  # upstream never called — rejected pre-flight
    row = _latest_request_row()
    assert row.status == 400
    assert row.error_kind == "invalid_request"


@dbtest
def test_capture_layer_index_out_of_range_is_rejected_preflight(client, captured):
    """ACS-342: an out-of-range activation-*capture* layer index
    (``output_residual_stream`` as a subset list) is rejected at the wrapper
    boundary with a clean 400 BEFORE the activation upstream is called. Left
    unchecked, capture routes through the streamed ``passthrough_post`` (ACS-250),
    so the 200 status line is already committed when vLLM-Lens raises ->
    ``layer_index out of range`` — the caller sees HTTP 200 with no ``choices``.
    ``act`` has ``n_layers=32`` so valid indices are 0-31; test the boundary
    ([32]) and a way-out value ([999])."""
    key = _make_key()
    for bad in ([32], [999]):
        r = client.post(
            "/v1/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "act",
                "prompt": "hi",
                "max_tokens": 4,
                "output_residual_stream": bad,
            },
        )
        assert r.status_code == 400, (bad, r.text)
        body = r.json()["error"]
        assert body["code"] == "invalid_request"
        assert "output_residual_stream" in body["message"]
        assert "out of range" in body["message"]
        assert captured == []  # upstream never called — rejected pre-flight


@dbtest
def test_capture_negative_layer_index_rejected_at_schema(client, captured):
    """A negative capture index ([-1]) is rejected too — at schema parse
    (schemas.py normalizer) rather than the model-range check, but still a clean
    400 that never touches the upstream. Pins the whole [32]/[999]/[-1] set the
    ACS-342 report calls out as returning a bad 200."""
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4, "output_residual_stream": [-1]},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_request"
    assert captured == []  # upstream never called


@dbtest
def test_capture_in_range_layer_index_reaches_upstream(client, captured):
    """A valid capture subset ([31] on a 32-layer model) passes pre-flight and
    routes to the activation upstream — the guard rejects only out-of-range
    indices, it doesn't block legitimate subset capture."""
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4, "output_residual_stream": [31]},
    )
    assert r.status_code == 200, r.text
    assert len(captured) == 1
    url, _ = captured[0]
    assert url == "https://act-activation.example/v1/completions"


@dbtest
def test_client_ish_upstream_500_returns_400_and_spares_breaker(client, monkeypatch):
    """Fix #2 (ACS-322): if a bad steering request still reaches the engine
    (e.g. n_layers unknown) and returns a client-ish 500, the wrapper returns
    400 (not 502) and does NOT record a breaker failure — so repeated bad
    requests can't open the circuit and 503 every caller on the model. Uses a
    VALID layer_index so it passes pre-flight and exercises the upstream-error
    handler."""
    from wrapper import breaker as breakermod
    from wrapper import proxy as proxymod

    async def client_ish_500(http, url, api_key, body, timeout, ctx=None):
        raise proxymod.UpstreamServerError(
            upstream_status=500,
            attempts=1,
            body_excerpt="ValueError: layer_index 32 out of range [0, 32)",
            upstream_kind="vllm_invalid_request",
            ctx=ctx,
        )

    monkeypatch.setattr("wrapper.proxy.post_nonstream", client_ish_500)
    key = _make_key()
    sv = {**_STEER_VEC, "layer_indices": [3]}
    for _ in range(breakermod.FAILURE_THRESHOLD + 1):
        r = client.post(
            "/v1/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "act", "prompt": "hi", "max_tokens": 4, "apply_steering_vectors": [sv]},
        )
        assert r.status_code == 400, r.text

    breakers = client.app.state.breakers
    assert breakers.snapshot("act::activation").state == "closed"


@dbtest
def test_activation_failures_do_not_trip_the_workbench_breaker(client, monkeypatch):
    """Isolation: a failing activation engine opens ONLY the ``<id>::activation``
    breaker, never the workbench breaker that gates production completions."""
    from wrapper import breaker as breakermod
    from wrapper import proxy as proxymod

    async def always_unreachable(http, url, api_key, body, timeout, ctx=None):
        raise proxymod.UpstreamUnreachable(reason="test", attempts=1, ctx=ctx)

    async def always_unreachable_passthrough(http, url, api_key, body, timeout, *, ctx=None):
        # Capture requests (output_residual_stream) route through passthrough_post
        # (ACS-250), not post_nonstream. A bare-raise async generator: the raise
        # fires on the first __anext__, before any byte is yielded.
        raise proxymod.UpstreamUnreachable(reason="test", attempts=1, ctx=ctx)
        yield  # unreachable; only here to make this a coroutine-async-generator

    monkeypatch.setattr("wrapper.proxy.post_nonstream", always_unreachable)
    monkeypatch.setattr("wrapper.proxy.passthrough_post", always_unreachable_passthrough)
    key = _make_key()
    # Drive FAILURE_THRESHOLD activation failures on model "act".
    for _ in range(breakermod.FAILURE_THRESHOLD):
        r = client.post(
            "/v1/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "act", "prompt": "hi", "max_tokens": 4, "output_residual_stream": True},
        )
        assert r.status_code in (502, 503), r.text

    breakers = client.app.state.breakers
    assert breakers.snapshot("act::activation").state == "open"
    # The workbench breaker for the same model must be untouched.
    assert breakers.snapshot("act").state == "closed"


@dbtest
def test_workbench_failures_do_not_trip_the_activation_breaker(client, monkeypatch):
    """Converse isolation: failing plain completions must not open the
    activation breaker."""
    from wrapper import breaker as breakermod
    from wrapper import proxy as proxymod

    async def always_unreachable(http, url, api_key, body, timeout, ctx=None):
        raise proxymod.UpstreamUnreachable(reason="test", attempts=1, ctx=ctx)

    monkeypatch.setattr("wrapper.proxy.post_nonstream", always_unreachable)
    key = _make_key()
    for _ in range(breakermod.FAILURE_THRESHOLD):
        r = client.post(
            "/v1/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "act", "prompt": "hi", "max_tokens": 4},
        )
        assert r.status_code in (502, 503), r.text

    breakers = client.app.state.breakers
    assert breakers.snapshot("act").state == "open"
    assert breakers.snapshot("act::activation").state == "closed"


@dbtest
def test_capture_request_records_activation_telemetry(client, captured):
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4, "output_residual_stream": True},
    )
    assert r.status_code == 200, r.text
    row = _latest_request_row()
    assert row.activation is True
    assert row.activation_layers == -1  # ALL_LAYERS_SENTINEL (output_residual_stream: true)
    assert row.activation_steering_vectors is None


@dbtest
def test_capture_subset_records_layer_count(client, captured):
    # ACS-266: a layer subset logs the (deduped) COUNT, not the ALL sentinel.
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4, "output_residual_stream": [3, 7, 3]},
    )
    assert r.status_code == 200, r.text
    row = _latest_request_row()
    assert row.activation is True
    assert row.activation_layers == 2  # two distinct layers (3, 7)
    assert row.activation_steering_vectors is None


@dbtest
def test_plain_request_records_no_activation_telemetry(client, captured):
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4},
    )
    assert r.status_code == 200, r.text
    row = _latest_request_row()
    assert row.activation is False
    assert row.activation_layers is None


@dbtest
def test_activation_quota_enforced_separately_from_completions(client, captured):
    # Budget of 2 activation requests; the 3rd is rejected, but plain
    # completions remain unaffected (activation quota is a separate dimension).
    key = _make_key(activation_budget=2)
    for _ in range(2):
        r = client.post(
            "/v1/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "act", "prompt": "hi", "max_tokens": 4, "output_residual_stream": True},
        )
        assert r.status_code == 200, r.text

    # 3rd activation request over quota → 429.
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4, "output_residual_stream": True},
    )
    assert r.status_code == 429, r.text
    assert "activation" in r.text.lower()

    # A plain completion on the same key still works — not counted against the
    # activation quota.
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "act", "prompt": "hi", "max_tokens": 4},
    )
    assert r.status_code == 200, r.text


@dbtest
def test_activation_request_on_plain_model_is_rejected(client, captured):
    key = _make_key()
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "plain", "prompt": "hi", "max_tokens": 4, "output_residual_stream": True},
    )
    assert r.status_code == 400, r.text
    # Never reached the proxy — rejected at the routing boundary.
    assert captured == []
    assert "does not support activation" in r.text
