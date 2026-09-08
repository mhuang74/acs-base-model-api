"""Integration tests for the reliability surfaces in main.py.

Covers the wiring rather than the unit logic (which is exercised in
test_breaker.py and test_proxy_reliability.py):

- ``_apply_backend_headers`` sets X-Acs-Upstream-Model / X-Acs-Upstream-Gpu
- ``_apply_extras_headers`` sets X-Acs-Upstream-Error-Kind from extras
- The handler emits the structured cold-boot payload on late startup failures
- ``run_completion_nonstream`` short-circuits when the breaker is open
- ``run_completion_nonstream`` translates UpstreamUnreachable to 502
- ``run_completion_nonstream`` translates UpstreamServerError to 502
- Cold-boot does NOT trip the breaker
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")

import datetime as dt
from types import SimpleNamespace

from fastapi.responses import JSONResponse

from wrapper import breaker as breakermod
from wrapper import main as mainmod
from wrapper import proxy as proxymod
from wrapper.auth import AuthedCaller
from wrapper.routes import api as api_routes
from wrapper.settings import ModelEntry
from wrapper.warm_state import MODAL_SCALEDOWN_WINDOW


def _caller() -> AuthedCaller:
    return AuthedCaller(
        key_id=uuid.uuid4(),
        key_prefix="reltest0",
        user_email="r@example.local",
        monthly_token_budget=0,  # unlimited → fast path, no tokenizer call
        tokens_used_this_month=0,
    )


@pytest.fixture
def fake_settings():
    s = MagicMock()
    s.modal_base_url = "https://upstream.example/"
    s.vllm_api_key = "vllm-test"
    s.upstream_timeout_s = 30.0
    s.served_model_name = "test-model"
    s.hf_token = None
    s.log_ip = False
    return s


# --- _model_is_cold heuristic (RPC-free warm-state guess) ------------------


def _state_with(last_completion_at: dict, models: dict | None = None):
    return SimpleNamespace(last_completion_at=last_completion_at, models=models or {})


def test_model_is_cold_never_seen_completion():
    # No completion this process lifetime → assume cold.
    assert api_routes._model_is_cold(_state_with({}), "trinity-truebase") is True


def test_model_is_cold_recent_completion_is_warm():
    now = dt.datetime.now(tz=dt.UTC)
    recent = now - (MODAL_SCALEDOWN_WINDOW / 2)
    assert api_routes._model_is_cold(_state_with({"trinity-truebase": recent}), "trinity-truebase") is False


def test_model_is_cold_stale_completion_is_cold():
    now = dt.datetime.now(tz=dt.UTC)
    stale = now - (MODAL_SCALEDOWN_WINDOW + dt.timedelta(minutes=1))
    assert api_routes._model_is_cold(_state_with({"trinity-truebase": stale}), "trinity-truebase") is True


def test_model_is_cold_none_model_id():
    assert api_routes._model_is_cold(_state_with({}), None) is False


# --- ACS-226: per-breaker-key scaledown window -----------------------------
#
# A serving engine (30-min window) and its activation side-car (10-min window)
# scale down on different schedules. The cold-hint must use the right one per
# key, or an activation request 10–30 min after the last one is judged warm and
# skips the keepalive path during a real cold boot.


def _entry_with_windows(model_id: str, *, serving_s: int, activation_s: int):
    return ModelEntry(
        model_id=model_id,
        upstream_url="https://upstream.example/",
        served_model_name="test-model",
        tokenizer_repo="gpt2",
        gpu_shape_label="8xH200",
        scaledown_window_s=serving_s,
        activation_scaledown_window_s=activation_s,
    )


def test_model_is_cold_activation_key_uses_shorter_window():
    # 15 min since the last activation completion: past the 10-min activation
    # teardown but inside the 30-min serving window. The activation key must
    # read cold; the serving key (same elapsed) must read warm.
    now = dt.datetime.now(tz=dt.UTC)
    fifteen_min_ago = now - dt.timedelta(minutes=15)
    entry = _entry_with_windows("llama-405b", serving_s=30 * 60, activation_s=10 * 60)
    state = _state_with(
        {
            "llama-405b": fifteen_min_ago,
            "llama-405b::activation": fifteen_min_ago,
        },
        models={"llama-405b": entry},
    )
    assert api_routes._model_is_cold(state, "llama-405b::activation") is True
    assert api_routes._model_is_cold(state, "llama-405b") is False


def test_model_is_cold_activation_key_warm_inside_activation_window():
    # 5 min since the last activation completion: inside the 10-min activation
    # window → still warm, keepalive path not forced.
    now = dt.datetime.now(tz=dt.UTC)
    five_min_ago = now - dt.timedelta(minutes=5)
    entry = _entry_with_windows("llama-405b", serving_s=30 * 60, activation_s=10 * 60)
    state = _state_with(
        {"llama-405b::activation": five_min_ago},
        models={"llama-405b": entry},
    )
    assert api_routes._model_is_cold(state, "llama-405b::activation") is False


def test_model_is_cold_unknown_key_falls_back_to_global_window():
    # Aliased id / legacy fallback with no registry entry → use the global
    # 30-min window rather than raising on the hot path.
    now = dt.datetime.now(tz=dt.UTC)
    recent = now - (MODAL_SCALEDOWN_WINDOW / 2)
    stale = now - (MODAL_SCALEDOWN_WINDOW + dt.timedelta(minutes=1))
    assert api_routes._model_is_cold(_state_with({"aliased": recent}), "aliased") is False
    assert api_routes._model_is_cold(_state_with({"aliased": stale}), "aliased") is True


# --- header helpers --------------------------------------------------------


def test_backend_headers_set_when_ctx_populated():
    resp = JSONResponse(content={"ok": True})
    ctx = proxymod.BackendContext(model_id="llama-405b", gpu_shape="8×H200")
    mainmod._apply_backend_headers(resp, ctx)
    assert resp.headers["X-Acs-Upstream-Model"] == "llama-405b"
    # The ``×`` (U+00D7) in the GPU label must be ASCII-folded to ``x`` so the
    # header value is RFC 7230-legal on the wire (bare 0xD7 is rejected /
    # corrupted by strict clients and proxies). JSON bodies keep ``×``.
    assert resp.headers["X-Acs-Upstream-Gpu"] == "8xH200"


def test_backend_headers_are_ascii_only():
    """No non-ASCII byte may reach an HTTP header value (RFC 7230)."""
    resp = JSONResponse(content={"ok": True})
    ctx = proxymod.BackendContext(model_id="llama-405b", gpu_shape="1×L40S")
    mainmod._apply_backend_headers(resp, ctx)
    for name in ("X-Acs-Upstream-Gpu", "X-Acs-Upstream-Model"):
        value = resp.headers[name]
        # Encodes cleanly as ASCII → no illegal bytes.
        value.encode("ascii")
        assert value.isascii()
    assert resp.headers["X-Acs-Upstream-Gpu"] == "1xL40S"


def test_backend_headers_empty_ctx_sets_nothing():
    resp = JSONResponse(content={"ok": True})
    mainmod._apply_backend_headers(resp, proxymod.BackendContext())
    assert "X-Acs-Upstream-Model" not in resp.headers
    assert "X-Acs-Upstream-Gpu" not in resp.headers


def test_extras_headers_surface_upstream_kind():
    resp = JSONResponse(content={"error": {"code": "vllm_oom"}})
    mainmod._apply_extras_headers(resp, {"upstream_error_kind": "vllm_oom"})
    assert resp.headers["X-Acs-Upstream-Error-Kind"] == "vllm_oom"


# --- run_completion_nonstream upstream-error handling ---------------------


async def test_breaker_short_circuits_request(monkeypatch, fake_settings):
    breakers = breakermod.BackendBreakers()
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)
    await breakers.record_failure("llama-405b", "upstream_unreachable")
    # Now OPEN. allow() should refuse without ever calling the upstream.
    upstream_mock = AsyncMock()
    monkeypatch.setattr(proxymod, "post_nonstream", upstream_mock)
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status, payload, extras = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=_caller(), body={"prompt": "x"}, ip=None,
        endpoint="/v1/completions",
        model_id="llama-405b",
        backend_ctx=proxymod.BackendContext(model_id="llama-405b"),
        breakers=breakers,
    )
    assert status == 503
    assert payload["error"]["code"] == "circuit_open"
    assert payload["error"]["model_id"] == "llama-405b"
    upstream_mock.assert_not_called()
    assert extras["upstream_error_kind"] == "circuit_open"


async def test_upstream_unreachable_returns_502_and_trips_breaker(monkeypatch, fake_settings):
    breakers = breakermod.BackendBreakers()
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)
    ctx = proxymod.BackendContext(model_id="llama-8b", gpu_shape="1×H100")

    async def boom(*a, **kw):
        raise proxymod.UpstreamUnreachable(reason="DNS", attempts=4, ctx=ctx)
    monkeypatch.setattr(proxymod, "post_nonstream", boom)
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status, payload, extras = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=_caller(), body={"prompt": "x"}, ip=None,
        endpoint="/v1/completions",
        model_id="llama-8b", backend_ctx=ctx, breakers=breakers,
    )
    assert status == 502
    assert payload["error"]["code"] == "upstream_unreachable"
    assert payload["error"]["model_id"] == "llama-8b"
    assert extras["upstream_error_kind"] == "upstream_unreachable"
    # Single failure was enough to trip with threshold=1.
    assert breakers.snapshot("llama-8b").state == "open"


async def test_upstream_server_error_returns_502_with_vllm_kind(monkeypatch, fake_settings):
    breakers = breakermod.BackendBreakers()
    ctx = proxymod.BackendContext(model_id="trinity-truebase")

    async def boom(*a, **kw):
        raise proxymod.UpstreamServerError(
            upstream_status=500, attempts=4, body_excerpt="CUDA out of memory",
            upstream_kind="vllm_oom", ctx=ctx,
        )
    monkeypatch.setattr(proxymod, "post_nonstream", boom)
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status, payload, extras = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=_caller(), body={"prompt": "x"}, ip=None,
        endpoint="/v1/completions",
        model_id="trinity-truebase", backend_ctx=ctx, breakers=breakers,
    )
    assert status == 502
    assert payload["error"]["code"] == "vllm_oom"
    assert payload["error"]["upstream_status"] == 500
    assert extras["upstream_error_kind"] == "vllm_oom"
    assert breakers.snapshot("trinity-truebase").total_failures == 1


async def test_cold_boot_does_not_trip_breaker(monkeypatch, fake_settings):
    breakers = breakermod.BackendBreakers()
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)

    async def boom(*a, **kw):
        raise proxymod.ColdBootError(303, 6)
    monkeypatch.setattr(proxymod, "post_nonstream", boom)
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status, payload, _ = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=_caller(), body={"prompt": "x"}, ip=None,
        endpoint="/v1/completions",
        model_id="llama-405b",
        backend_ctx=proxymod.BackendContext(model_id="llama-405b"),
        breakers=breakers,
    )
    assert status == 503
    assert payload["error"]["code"] == "modal_cold_boot"
    # Breaker stays closed — cold-boot is expected, not a failure.
    assert breakers.snapshot("llama-405b").state == "closed"
    assert breakers.snapshot("llama-405b").total_failures == 0


async def test_stalled_cold_boot_returns_503_and_spares_breaker(monkeypatch, fake_settings):
    """End-to-end of the prod #1 blocker fix: when ``post_nonstream`` raises
    ColdBootError (because a stalled socket on a cold model was reclassified),
    the route emits the late cold-boot payload and the breaker is NOT tripped."""
    breakers = breakermod.BackendBreakers()
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)

    async def stalled(*a, **kw):
        # upstream_status 0 = stall, no HTTP status seen (vs. 303 fast path).
        raise proxymod.ColdBootError(0, 1)
    monkeypatch.setattr(proxymod, "post_nonstream", stalled)
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status, payload, _ = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=_caller(), body={"prompt": "x"}, ip=None,
        endpoint="/v1/completions",
        model_id="trinity-truebase",
        backend_ctx=proxymod.BackendContext(model_id="trinity-truebase", cold_hint=True),
        breakers=breakers,
    )
    assert status == 503
    assert payload["error"]["code"] == "modal_cold_boot"
    assert payload["error"]["retry_after_seconds"] >= 1
    # Cold-boot (even the stalled kind) must never count against the breaker.
    assert breakers.snapshot("trinity-truebase").state == "closed"
    assert breakers.snapshot("trinity-truebase").total_failures == 0


async def test_success_records_breaker_success(monkeypatch, fake_settings):
    breakers = breakermod.BackendBreakers()
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 3)
    # Pre-stage one failure on the breaker so we can verify success resets.
    await breakers.record_failure("llama-8b", "upstream_5xx")

    async def ok(*a, **kw):
        return 200, {"choices": [{"text": "hi"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, 50
    monkeypatch.setattr(proxymod, "post_nonstream", ok)
    monkeypatch.setattr(mainmod.authmod, "commit_usage", AsyncMock())
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    status, _payload, _extras = await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=_caller(), body={"prompt": "x"}, ip=None,
        endpoint="/v1/completions",
        model_id="llama-8b",
        backend_ctx=proxymod.BackendContext(model_id="llama-8b"),
        breakers=breakers,
    )
    assert status == 200
    snap = breakers.snapshot("llama-8b")
    assert snap.consecutive_failures == 0  # success cleared the counter
    assert snap.total_successes == 1


# --- per-model timeout override -------------------------------------------


async def test_per_model_timeout_override(monkeypatch, fake_settings):
    """run_completion_nonstream uses ``timeout_s`` when provided, not the
    global setting."""
    seen_timeout = {}

    async def fake_post(client, url, key, body, timeout, *, ctx=None):
        seen_timeout["t"] = timeout
        return 200, {"choices": [], "usage": {}}, 10

    monkeypatch.setattr(proxymod, "post_nonstream", fake_post)
    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())

    fake_settings.upstream_timeout_s = 1200.0
    await mainmod.run_completion_nonstream(
        session=MagicMock(), http=MagicMock(), settings=fake_settings,
        caller=_caller(), body={"prompt": "x"}, ip=None,
        endpoint="/v1/completions",
        model_id="llama-8b",
        timeout_s=60.0,  # per-model override
    )
    assert seen_timeout["t"] == 60.0
