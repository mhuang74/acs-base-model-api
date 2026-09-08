"""Unit tests for the per-backend circuit breaker (wrapper/breaker.py).

Pure async — no DB needed. Covers state transitions (closed → open →
half-open → closed/open), the failure threshold, the open-duration timer
(via monkeypatched ``time.monotonic``), success resets counters, manual
reset, and the structured payload.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")

from wrapper import breaker as breakermod


@pytest.fixture
def br():
    return breakermod.BackendBreakers()


# --- Default state + happy path --------------------------------------------

async def test_default_state_is_closed(br):
    assert await br.allow("any-model") is True
    snap = br.snapshot("any-model")
    assert snap.state == "closed"
    assert snap.consecutive_failures == 0
    assert snap.total_failures == 0


async def test_success_keeps_state_closed(br):
    for _ in range(10):
        await br.record_success("m")
    assert (await br.allow("m")) is True
    snap = br.snapshot("m")
    assert snap.state == "closed"
    assert snap.total_successes == 10


# --- Tripping the breaker --------------------------------------------------

async def test_trips_after_threshold_consecutive_failures(br, monkeypatch):
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 3)
    for _ in range(2):
        await br.record_failure("m", "upstream_unreachable")
    # Two failures: still closed.
    assert (await br.allow("m")) is True
    await br.record_failure("m", "upstream_unreachable")
    # Third failure hit the threshold → open.
    snap = br.snapshot("m")
    assert snap.state == "open"
    assert snap.total_trips == 1
    # And subsequent requests are denied (until the open window elapses).
    assert (await br.allow("m")) is False


async def test_success_resets_consecutive_failures(br, monkeypatch):
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 3)
    await br.record_failure("m", "upstream_5xx")
    await br.record_failure("m", "upstream_5xx")
    await br.record_success("m")  # resets counter
    await br.record_failure("m", "upstream_5xx")
    # Counter is at 1 again (post-reset), not at 3 — still closed.
    assert (await br.allow("m")) is True
    assert br.snapshot("m").state == "closed"


# --- Half-open recovery ----------------------------------------------------

async def test_open_to_half_open_after_duration(br, monkeypatch):
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(breakermod, "OPEN_DURATION_S", 0.05)
    await br.record_failure("m", "upstream_unreachable")
    assert (await br.allow("m")) is False  # immediately blocked

    import asyncio
    await asyncio.sleep(0.06)
    # Window elapsed → next allow() transitions open → half-open.
    assert (await br.allow("m")) is True
    assert br.snapshot("m").state == "half_open"


async def test_half_open_success_closes(br, monkeypatch):
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(breakermod, "OPEN_DURATION_S", 0.01)
    await br.record_failure("m", "upstream_5xx")

    import asyncio
    await asyncio.sleep(0.02)
    await br.allow("m")  # transitions to half-open
    await br.record_success("m")
    assert br.snapshot("m").state == "closed"


async def test_half_open_failure_reopens(br, monkeypatch):
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(breakermod, "OPEN_DURATION_S", 0.01)
    await br.record_failure("m", "upstream_5xx")

    import asyncio
    await asyncio.sleep(0.02)
    await br.allow("m")  # → half_open
    await br.record_failure("m", "upstream_unreachable")
    snap = br.snapshot("m")
    assert snap.state == "open"
    assert snap.total_trips == 2  # original trip + half-open re-trip


# --- Independence per model -----------------------------------------------

async def test_independent_per_model(br, monkeypatch):
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 2)
    await br.record_failure("model-a", "x")
    await br.record_failure("model-a", "x")
    assert br.snapshot("model-a").state == "open"
    # model-b is untouched.
    assert br.snapshot("model-b").state == "closed"
    assert (await br.allow("model-b")) is True


# --- Admin reset ----------------------------------------------------------

async def test_reset_force_closes_and_preserves_totals(br, monkeypatch):
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)
    await br.record_failure("m", "x")
    assert br.snapshot("m").state == "open"
    await br.reset("m")
    snap = br.snapshot("m")
    assert snap.state == "closed"
    assert snap.consecutive_failures == 0
    # Audit counters preserved.
    assert snap.total_failures == 1
    assert snap.total_trips == 1


# --- circuit_open_payload contract -----------------------------------------

def test_circuit_open_payload_includes_retry_after_and_model(br):
    # Force-open via private state for synchronous test.
    import time as _time
    st = br._get("llama-405b")
    st.state = "open"
    st.opened_at = _time.monotonic()
    st.consecutive_failures = 7
    st.last_failure_kind = "vllm_oom"
    snap = br.snapshot("llama-405b")
    payload = breakermod.circuit_open_payload(snap)
    err = payload["error"]
    assert err["code"] == "circuit_open"
    assert err["model_id"] == "llama-405b"
    assert err["consecutive_failures"] == 7
    assert err["last_failure_kind"] == "vllm_oom"
    # Retry-after is roughly OPEN_DURATION_S; allow a bit of slack.
    assert err["retry_after_seconds"] >= 1
    assert "llama-405b" in err["message"]


# --- breaker-open alert (Sentry/Discord degraded-model page) ----------------

async def test_alert_fires_once_when_breaker_trips(br, monkeypatch):
    """Tripping closed→open pages exactly once, with the model id + count.

    /health returns 200 for degraded, so this is the only signal that a model
    went bad — it must fire (and only on the actual trip, not every failure)."""
    calls = []
    monkeypatch.setattr(breakermod, "_alert_breaker_open", lambda *a: calls.append(a))
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 3)

    await br.record_failure("m", "upstream_5xx")
    await br.record_failure("m", "upstream_5xx")
    assert calls == []  # below threshold → no alert yet
    await br.record_failure("m", "upstream_5xx")  # trips
    assert len(calls) == 1
    assert calls[0] == ("m", "upstream_5xx", 3)


async def test_alert_does_not_fire_on_further_failures_while_open(br, monkeypatch):
    """Once open, additional failures don't re-page (grouping aside, the breaker
    shouldn't even call the alert again until a half-open re-open)."""
    calls = []
    monkeypatch.setattr(breakermod, "_alert_breaker_open", lambda *a: calls.append(a))
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)

    await br.record_failure("m", "x")  # trips immediately
    await br.record_failure("m", "x")  # already open → no new alert
    assert len(calls) == 1


async def test_alert_fires_again_on_half_open_reopen(br, monkeypatch):
    """A failed half-open probe re-opens → pages again (keeps the issue active)."""
    calls = []
    monkeypatch.setattr(breakermod, "_alert_breaker_open", lambda *a: calls.append(a))
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(breakermod, "OPEN_DURATION_S", 0.01)

    import asyncio as _asyncio

    await br.record_failure("m", "x")  # → open (alert 1)
    await _asyncio.sleep(0.02)
    assert await br.allow("m") is True  # → half_open
    await br.record_failure("m", "x")  # half-open probe fails → re-open (alert 2)
    assert len(calls) == 2


async def test_alert_helper_is_safe_without_sentry(monkeypatch):
    """_alert_breaker_open must never raise (alerting can't break requests),
    even with no Sentry DSN configured — capture_message is a no-op then."""
    # Should not raise regardless of Sentry init state.
    breakermod._alert_breaker_open("m", "upstream_unreachable", 5)
