"""Tests for the aggregated /health endpoint.

Covers:
- ``_backend_health_entry`` derives ``warm`` vs ``cold`` from
  ``last_completion_at`` against the model's per-model ``scaledown_window_s``.
- ``aggregate_health`` returns 200 ``ok`` with all backends closed.
- ``aggregate_health`` returns 200 ``degraded`` when any breaker is open,
  with the open backends listed in ``degraded_backends``.
- ``aggregate_health`` returns 503 ``down`` when the DB ping fails.
- Disabled registry entries are excluded from the backends list.
- Boot-time + uptime survive a missing ``boot_time`` attribute (defensive
  fallback).

No DB needed: the engine is stubbed; the rest is in-memory.
"""

from __future__ import annotations

import datetime as dt
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")

from wrapper import breaker as breakermod
from wrapper import main as mainmod
from wrapper.settings import ModelEntry


def _entry(model_id: str, *, status: str = "live", gpu: str = "1×H100") -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        upstream_url=f"https://{model_id}.example",
        served_model_name=f"served/{model_id}",
        tokenizer_repo="gpt2",
        gpu_shape_label=gpu,
        status=status,
    )


def _state(registry, breakers, last_completion_at=None, boot_time=None,
           db_ok=True, db_latency_ms=3) -> SimpleNamespace:
    """Build a minimal ``app.state`` shim. The aggregator only reads
    ``models``, ``breakers``, ``last_completion_at``, ``engine``, and
    optionally ``boot_time``."""

    async def fake_ping(_engine):
        return db_ok, db_latency_ms

    state = SimpleNamespace(
        models=registry,
        breakers=breakers,
        last_completion_at=last_completion_at or {},
        engine=MagicMock(),
    )
    if boot_time is not None:
        state.boot_time = boot_time
    return state, fake_ping


# --- _backend_health_entry ------------------------------------------------


def test_backend_entry_warm_within_scaledown_window():
    """A completion 5 minutes ago → warm (MODAL_SCALEDOWN_WINDOW is 30 min)."""
    now = dt.datetime(2026, 6, 3, 12, 0, tzinfo=dt.UTC)
    last_at = now - dt.timedelta(minutes=5)
    br = breakermod.BackendBreakers()
    snap = br.snapshot("m")
    entry = _entry("m")
    out = mainmod._backend_health_entry(entry, snap, last_at, now)
    assert out["warm_estimate"] == "warm"
    assert out["last_completion_seconds_ago"] == 300
    assert out["last_completion_at"] == last_at.isoformat()


def test_backend_entry_cold_past_scaledown_window():
    """A completion 31 minutes ago → cold."""
    now = dt.datetime(2026, 6, 3, 12, 0, tzinfo=dt.UTC)
    last_at = now - dt.timedelta(minutes=31)
    br = breakermod.BackendBreakers()
    out = mainmod._backend_health_entry(_entry("m"), br.snapshot("m"), last_at, now)
    assert out["warm_estimate"] == "cold"
    assert out["last_completion_seconds_ago"] == 31 * 60


def test_backend_entry_cold_when_never_seen():
    """No prior completion → cold + null timestamps."""
    now = dt.datetime(2026, 6, 3, 12, 0, tzinfo=dt.UTC)
    br = breakermod.BackendBreakers()
    out = mainmod._backend_health_entry(_entry("m"), br.snapshot("m"), None, now)
    assert out["warm_estimate"] == "cold"
    assert out["last_completion_at"] is None
    assert out["last_completion_seconds_ago"] is None


def test_backend_entry_uses_per_model_scaledown_window():
    """A model with a short serving window reads cold before the 30-min global.

    ACS-226: ``warm_estimate`` uses the entry's ``scaledown_window_s`` (e.g. the
    2-min ``llama-8b-snapprod`` staging engine), not the global 30-min window —
    so a completion 5 min ago is cold, not warm.
    """
    now = dt.datetime(2026, 6, 3, 12, 0, tzinfo=dt.UTC)
    last_at = now - dt.timedelta(minutes=5)
    entry = ModelEntry(
        model_id="snapprod",
        upstream_url="https://snapprod.example",
        served_model_name="served/snapprod",
        tokenizer_repo="gpt2",
        gpu_shape_label="1×L40S",
        scaledown_window_s=2 * 60,
    )
    br = breakermod.BackendBreakers()
    out = mainmod._backend_health_entry(entry, br.snapshot("snapprod"), last_at, now)
    assert out["warm_estimate"] == "cold"


def test_backend_entry_surfaces_breaker_state():
    now = dt.datetime(2026, 6, 3, 12, 0, tzinfo=dt.UTC)
    br = breakermod.BackendBreakers()
    # Force open via private state (sync test).
    import time as _time
    st = br._get("m")
    st.state = "open"
    st.opened_at = _time.monotonic()
    st.consecutive_failures = 5
    st.last_failure_kind = "vllm_oom"
    out = mainmod._backend_health_entry(_entry("m"), br.snapshot("m"), None, now)
    assert out["breaker_state"] == "open"
    assert out["breaker_consecutive_failures"] == 5
    assert out["breaker_last_failure_kind"] == "vllm_oom"


def test_backend_entry_no_last_failure_kind_is_null():
    """Empty string → null in the response (cleaner JSON)."""
    now = dt.datetime(2026, 6, 3, 12, 0, tzinfo=dt.UTC)
    br = breakermod.BackendBreakers()
    out = mainmod._backend_health_entry(_entry("m"), br.snapshot("m"), None, now)
    assert out["breaker_last_failure_kind"] is None


# --- aggregate_health ------------------------------------------------------


async def test_aggregate_all_closed_returns_ok(monkeypatch):
    boot = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=2)
    registry = {"llama-8b": _entry("llama-8b"), "trinity-truebase": _entry("trinity-truebase", gpu="8×H200")}
    br = breakermod.BackendBreakers()
    state, fake_ping = _state(registry, br, boot_time=boot)
    monkeypatch.setattr(mainmod, "_ping_database", fake_ping)

    status, body = await mainmod.aggregate_health(state)
    assert status == 200
    assert body["status"] == "ok"
    assert body["wrapper"]["database"]["ok"] is True
    assert body["wrapper"]["uptime_seconds"] >= 7000
    assert len(body["backends"]) == 2
    assert body["degraded_backends"] == []
    # Backends sorted by id for deterministic output.
    ids = [b["model_id"] for b in body["backends"]]
    assert ids == sorted(ids)


async def test_aggregate_open_breaker_returns_degraded(monkeypatch):
    """A tripped breaker downgrades overall to 'degraded' but stays HTTP 200 —
    one bad model isn't a wrapper failure; we don't want Railway to restart."""
    monkeypatch.setattr(breakermod, "FAILURE_THRESHOLD", 1)
    registry = {"llama-8b": _entry("llama-8b"), "llama-405b": _entry("llama-405b", gpu="8×H200")}
    br = breakermod.BackendBreakers()
    await br.record_failure("llama-405b", "upstream_unreachable")  # trips with threshold=1

    state, fake_ping = _state(registry, br)
    monkeypatch.setattr(mainmod, "_ping_database", fake_ping)

    status, body = await mainmod.aggregate_health(state)
    assert status == 200
    assert body["status"] == "degraded"
    assert body["degraded_backends"] == ["llama-405b"]
    bad = next(b for b in body["backends"] if b["model_id"] == "llama-405b")
    assert bad["breaker_state"] == "open"
    assert bad["breaker_last_failure_kind"] == "upstream_unreachable"


async def test_aggregate_db_down_returns_503(monkeypatch):
    """Wrapper-self failure → 503 status. Railway healthcheck reads the
    status code and restarts on 503."""
    registry = {"llama-8b": _entry("llama-8b")}
    br = breakermod.BackendBreakers()
    state, fake_ping = _state(registry, br, db_ok=False, db_latency_ms=502)
    monkeypatch.setattr(mainmod, "_ping_database", fake_ping)

    status, body = await mainmod.aggregate_health(state)
    assert status == 503
    assert body["status"] == "down"
    assert body["wrapper"]["database"]["ok"] is False
    # Backends are still reported — useful in the 503 body for triage.
    assert len(body["backends"]) == 1


async def test_aggregate_excludes_disabled_models(monkeypatch):
    """Disabled entries shouldn't appear in /health — they're not part of
    the active surface, and a disabled model's breaker state is irrelevant."""
    registry = {
        "live-model": _entry("live-model"),
        "old-model": _entry("old-model", status="disabled"),
    }
    br = breakermod.BackendBreakers()
    # Even a tripped breaker on the disabled model shouldn't mark us degraded.
    import time as _time
    st = br._get("old-model")
    st.state = "open"
    st.opened_at = _time.monotonic()
    state, fake_ping = _state(registry, br)
    monkeypatch.setattr(mainmod, "_ping_database", fake_ping)

    status, body = await mainmod.aggregate_health(state)
    assert status == 200
    assert body["status"] == "ok"  # disabled model's open breaker ignored
    assert [b["model_id"] for b in body["backends"]] == ["live-model"]
    assert body["degraded_backends"] == []


async def test_aggregate_uses_last_completion_for_warm(monkeypatch):
    """End-to-end: warm/cold derivation flows from the per-model last-completion map."""
    now = dt.datetime.now(tz=dt.UTC)
    registry = {"m": _entry("m")}
    br = breakermod.BackendBreakers()
    state, fake_ping = _state(
        registry, br,
        last_completion_at={"m": now - dt.timedelta(minutes=2)},
    )
    monkeypatch.setattr(mainmod, "_ping_database", fake_ping)

    _, body = await mainmod.aggregate_health(state)
    assert body["backends"][0]["warm_estimate"] == "warm"


async def test_aggregate_missing_boot_time_does_not_crash(monkeypatch):
    """Defensive: if lifespan didn't run (e.g. partial init / test harness),
    /health must still respond rather than 500."""
    registry = {"m": _entry("m")}
    br = breakermod.BackendBreakers()
    state, fake_ping = _state(registry, br)  # boot_time omitted
    monkeypatch.setattr(mainmod, "_ping_database", fake_ping)

    status, body = await mainmod.aggregate_health(state)
    assert status == 200
    # uptime falls back to ~0 since boot_time defaults to "now" in the helper.
    assert body["wrapper"]["uptime_seconds"] >= 0
