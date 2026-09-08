"""Tests for the GPU cost monitor (sampler math, rate parsing, spike alert).

The math + settings-parse tests need no DB. The sampler/alert tests are
DB-gated (TEST_DATABASE_URL → migrated Postgres), mirroring test_invites.py.
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from wrapper import cost_monitor
from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import GpuCostSample
from wrapper.settings import Settings

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run cost-monitor DB tests",
)


def _settings(**overrides):
    base = dict(
        database_url="sqlite+aiosqlite:///:memory:",
        modal_base_url="https://x/",
        vllm_api_key="x",
        admin_token="x",
    )
    base.update(overrides)
    return Settings(**base)


# ---- pure helpers (no DB) ---------------------------------------------------

def test_estimate_usd_math():
    # 1 container of 8×H200 at $4.54/GPU-hr = $36.32/hr for a full hour.
    assert cost_monitor.estimate_usd(1, 36.32, 3600) == pytest.approx(36.32)
    # half an hour → half cost; 2 containers → double.
    assert cost_monitor.estimate_usd(1, 36.32, 1800) == pytest.approx(18.16)
    assert cost_monitor.estimate_usd(2, 36.32, 3600) == pytest.approx(72.64)
    # nothing running → no cost.
    assert cost_monitor.estimate_usd(0, 36.32, 3600) == 0.0


def test_parsed_gpu_hourly_rates_default_and_malformed():
    rates = _settings().parsed_gpu_hourly_rates()
    assert rates["H200"] == pytest.approx(4.54)
    assert rates["L40S"] == pytest.approx(1.95)
    # malformed JSON → empty dict (never crashes the job).
    assert _settings(gpu_hourly_usd_by_type_json="{not json").parsed_gpu_hourly_rates() == {}
    assert _settings(gpu_hourly_usd_by_type_json="[]").parsed_gpu_hourly_rates() == {}


def test_rate_config_problem():
    # Valid config (default) → no problem.
    assert cost_monitor.rate_config_problem(_settings()) is None
    assert cost_monitor.rate_config_problem(
        _settings(gpu_hourly_usd_by_type_json='{"H200": 4.54}')
    ) is None
    # Intentionally-empty config → not an error (rates deliberately unset).
    assert cost_monitor.rate_config_problem(
        _settings(gpu_hourly_usd_by_type_json="")
    ) is None
    assert cost_monitor.rate_config_problem(
        _settings(gpu_hourly_usd_by_type_json="   ")
    ) is None
    # Set-but-broken: non-empty raw that parses to no usable rates → warning.
    for bad in ("{not json", "[]", '{"H200": "abc"}', "null", "42"):
        problem = cost_monitor.rate_config_problem(
            _settings(gpu_hourly_usd_by_type_json=bad)
        )
        assert problem is not None, f"expected a warning for {bad!r}"
        assert "GPU_HOURLY_USD_BY_TYPE_JSON" in problem


async def test_read_serving_count_nonzero_no_retry(monkeypatch):
    """A non-zero fresh read is trusted immediately — one RPC, no state probe."""
    fresh = AsyncMock(return_value=5)
    state = AsyncMock(return_value="deployed")
    monkeypatch.setattr(cost_monitor.modalops, "fetch_runner_count_fresh", fresh)
    monkeypatch.setattr(cost_monitor.modalops, "get_app_state", state)
    assert await cost_monitor._read_serving_count("acs-x") == 5
    fresh.assert_awaited_once_with("acs-x")
    state.assert_not_awaited()


async def test_read_serving_count_retries_suspicious_zero(monkeypatch):
    """A fresh 0 on a non-stopped app is re-read once; the retry value wins."""
    fresh = AsyncMock(side_effect=[0, 3])
    state = AsyncMock(return_value="deployed")
    monkeypatch.setattr(cost_monitor.modalops, "fetch_runner_count_fresh", fresh)
    monkeypatch.setattr(cost_monitor.modalops, "get_app_state", state)
    assert await cost_monitor._read_serving_count("acs-x") == 3
    assert fresh.await_count == 2
    state.assert_awaited_once_with("acs-x")


async def test_read_serving_count_no_retry_when_stopped(monkeypatch):
    """A fresh 0 on an explicitly-stopped app is trusted — no wasteful retry."""
    fresh = AsyncMock(return_value=0)
    state = AsyncMock(return_value="stopped")
    monkeypatch.setattr(cost_monitor.modalops, "fetch_runner_count_fresh", fresh)
    monkeypatch.setattr(cost_monitor.modalops, "get_app_state", state)
    assert await cost_monitor._read_serving_count("acs-x") == 0
    fresh.assert_awaited_once_with("acs-x")


async def test_read_serving_count_retries_when_state_unknown(monkeypatch):
    """If the state probe errors, the 0 is treated as suspicious and retried."""
    fresh = AsyncMock(side_effect=[0, 0])
    state = AsyncMock(side_effect=RuntimeError("state RPC blip"))
    monkeypatch.setattr(cost_monitor.modalops, "fetch_runner_count_fresh", fresh)
    monkeypatch.setattr(cost_monitor.modalops, "get_app_state", state)
    # Both reads returned 0 → record 0 (genuine or persistent), but we tried.
    assert await cost_monitor._read_serving_count("acs-x") == 0
    assert fresh.await_count == 2


def test_parse_gpu_shape():
    # Prod labels use the "×" multiplication sign (U+00D7), e.g. "8×H200".
    assert cost_monitor.parse_gpu_shape("8×H200") == (8, "H200")
    assert cost_monitor.parse_gpu_shape("1×L40S") == (1, "L40S")
    # ASCII "x" also accepted (shared-registry property form).
    assert cost_monitor.parse_gpu_shape("8xH200") == (8, "H200")
    assert cost_monitor.parse_gpu_shape("16×H200") == (16, "H200")
    assert cost_monitor.parse_gpu_shape("legacy") is None
    assert cost_monitor.parse_gpu_shape(None) is None


def _model(gpu_shape_label, app_name, status="live", **extra):
    return SimpleNamespace(
        gpu_shape_label=gpu_shape_label, modal_app_name=app_name, status=status, **extra
    )


def test_activation_app_name_derivation():
    from wrapper import modal_ops

    # Keyed on the wrapper model id, NOT modal_app_name: Trinity's serving app
    # kept the pre-rename name (acs-trinity-base) but its activation app is
    # acs-trinity-truebase-activation (probed live, ACS-221).
    assert modal_ops.activation_app_name("trinity-truebase") == "acs-trinity-truebase-activation"
    assert modal_ops.activation_app_name("llama-8b") == "acs-llama-8b-activation"
    # Explicit registry override wins over the derivation.
    entry = SimpleNamespace(activation_app_name="acs-custom-activation")
    assert modal_ops.resolve_activation_app(entry, "llama-8b") == "acs-custom-activation"
    assert (
        modal_ops.resolve_activation_app(SimpleNamespace(), "llama-8b")
        == "acs-llama-8b-activation"
    )


def test_resolve_harvest_app_gate_and_override():
    from wrapper import modal_ops

    # Derived from the MODEL ID, not modal_app_name (ACS-273: Trinity's serving
    # app kept the pre-rename name; its harvester is keyed on the current id).
    entry = SimpleNamespace(modal_app_name="acs-trinity-base")
    assert (
        modal_ops.resolve_harvest_app(entry, "trinity-truebase")
        == "acs-trinity-truebase-harvest"
    )
    # Explicit registry override wins over the derivation.
    entry = SimpleNamespace(modal_app_name="acs-x", harvest_app_name="acs-custom-harvest")
    assert modal_ops.resolve_harvest_app(entry, "llama-8b") == "acs-custom-harvest"
    # No Modal serving app → harvest unsupported → None (the sampler writes no
    # ::harvest row, same gate as the /v1/harvest route).
    assert (
        modal_ops.resolve_harvest_app(SimpleNamespace(modal_app_name=None), "llama-8b")
        is None
    )


async def test_activation_fetch_remembers_winning_tag(monkeypatch):
    """The snapshot-path app (ActivationSnap.*) costs a guaranteed-miss probe on
    the first tag only ONCE — later ticks try the cached winner first. Total
    failure raises one error naming every tag's failure, not just the last."""
    from wrapper import modal_ops

    modal_ops._ACTIVATION_TAG_CACHE.clear()
    calls: list[str] = []

    async def fake_fetch(app_name, function_name=modal_ops.SERVE_FUNCTION_NAME):
        calls.append(function_name)
        if function_name != "ActivationSnap.*":
            raise modal_ops.ModalOpsError(f"{app_name}/{function_name} not found")
        return 1

    monkeypatch.setattr(modal_ops, "_fetch_runner_count", fake_fetch)
    monkeypatch.setattr(modal_ops, "_auth", lambda: None)
    assert await modal_ops.fetch_activation_runner_count_fresh("acs-x-activation") == 1
    assert calls == ["serve_activation", "ActivationSnap.*"]
    calls.clear()
    assert await modal_ops.fetch_activation_runner_count_fresh("acs-x-activation") == 1
    assert calls == ["ActivationSnap.*"]  # cached winner first, no wasted probe

    async def all_fail(app_name, function_name=modal_ops.SERVE_FUNCTION_NAME):
        raise modal_ops.ModalOpsError(f"{function_name} boom")

    monkeypatch.setattr(modal_ops, "_fetch_runner_count", all_fail)
    modal_ops._ACTIVATION_TAG_CACHE.clear()
    with pytest.raises(modal_ops.ModalOpsError) as ei:
        await modal_ops.fetch_activation_runner_count_fresh("acs-x-activation")
    # Both tags' failures are named — a lone last-tag error would blame the
    # wrong lifecycle path.
    assert "serve_activation" in str(ei.value) and "ActivationSnap.*" in str(ei.value)


# ---- sampler writes rows (DB) ----------------------------------------------

@dbtest
async def test_run_cost_sample_writes_rows(monkeypatch):
    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    try:
        # No prior sample → first sample bills one full interval (3600s), so the
        # Trinity figure below is deterministic regardless of other tests/runs.
        from sqlalchemy import delete

        async with session_scope(factory) as s:
            await s.execute(delete(GpuCostSample))
        # Every model reports 1 running container; harvesters are idle.
        monkeypatch.setattr(
            cost_monitor.modalops, "fetch_runner_count_fresh", AsyncMock(return_value=1)
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_harvest_runner_count_fresh",
            AsyncMock(return_value=0),
        )
        models = {
            "trinity-truebase": _model("8×H200", "acs-trinity-base"),
            "llama-8b": _model("1×L40S", "acs-llama-8b"),
        }
        await cost_monitor.run_cost_sample(
            session_factory=factory, settings=_settings(), models=models, period_seconds=3600
        )
        async with session_scope(factory) as s:
            rows = (await s.execute(select(GpuCostSample))).scalars().all()
        by_model = {r.model_id: r for r in rows}
        # Trinity is 8×H200 → 8 × $4.54 = $36.32/container/hr, 1 running, 1 hr.
        assert "trinity-truebase" in by_model
        tr = by_model["trinity-truebase"]
        assert tr.gpu_type == "H200"
        assert tr.gpu_count == 8
        assert tr.hourly_usd_per_container == pytest.approx(36.32)
        assert tr.est_usd == pytest.approx(36.32)
        # llama-8b is a single small GPU → much cheaper than Trinity.
        assert by_model["llama-8b"].est_usd < tr.est_usd
    finally:
        await engine.dispose()


@dbtest
async def test_sample_bills_elapsed_not_fixed_interval(monkeypatch):
    """A sample run shortly after a prior one bills only the gap, not a full
    interval — so boot-on-every-redeploy doesn't over-count."""
    import datetime as dt

    from sqlalchemy import delete, select

    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    try:
        async with session_scope(factory) as s:
            await s.execute(delete(GpuCostSample))
            # A prior sample 600s (10 min) ago.
            s.add(
                GpuCostSample(
                    model_id="seed",
                    gpu_type="H200",
                    gpu_count=8,
                    running_containers=0,
                    hourly_usd_per_container=36.32,
                    period_seconds=3600,
                    est_usd=0.0,
                    ts=dt.datetime.now(tz=dt.UTC) - dt.timedelta(seconds=600),
                )
            )
        monkeypatch.setattr(
            cost_monitor.modalops, "fetch_runner_count_fresh", AsyncMock(return_value=1)
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_harvest_runner_count_fresh",
            AsyncMock(return_value=0),
        )
        await cost_monitor.run_cost_sample(
            session_factory=factory,
            settings=_settings(),
            models={"trinity-truebase": _model("8×H200", "acs-trinity-base")},
            period_seconds=3600,
        )
        async with session_scope(factory) as s:
            row = (
                await s.execute(
                    select(GpuCostSample).where(GpuCostSample.model_id == "trinity-truebase")
                )
            ).scalar_one()
        # ~600s billed (not 3600); cost scales down accordingly (~$6, not $36).
        assert 500 <= row.period_seconds <= 700
        assert row.est_usd == pytest.approx(36.32 * row.period_seconds / 3600, rel=0.02)
    finally:
        await engine.dispose()


@dbtest
async def test_activation_engine_sampled_as_own_series(monkeypatch):
    """An activation-enabled model gets a second ``<id>::activation`` row on the
    same GPU shape/rate (ACS-221); models without activation support get none."""
    from sqlalchemy import delete

    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    try:
        async with session_scope(factory) as s:
            await s.execute(delete(GpuCostSample))
        monkeypatch.setattr(
            cost_monitor.modalops, "fetch_runner_count_fresh", AsyncMock(return_value=1)
        )
        act_fetch = AsyncMock(return_value=2)
        monkeypatch.setattr(
            cost_monitor.modalops, "fetch_activation_runner_count_fresh", act_fetch
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_harvest_runner_count_fresh",
            AsyncMock(return_value=0),
        )
        models = {
            # activation-enabled (like prod llama-8b)
            "llama-8b": _model(
                "1×L40S", "acs-llama-8b", activation_upstream_url="https://act.example"
            ),
            # no activation support → serving + harvest rows only
            "kimi-k2-base": _model("8×H200", "acs-kimi-k2-base"),
        }
        await cost_monitor.run_cost_sample(
            session_factory=factory, settings=_settings(), models=models, period_seconds=3600
        )
        async with session_scope(factory) as s:
            rows = (await s.execute(select(GpuCostSample))).scalars().all()
        by_model = {r.model_id: r for r in rows}
        # Both models have a Modal serving app, so both also get a ::harvest
        # series (ACS-281); only llama-8b is activation-enabled.
        assert set(by_model) == {
            "llama-8b",
            "llama-8b::activation",
            "llama-8b::harvest",
            "kimi-k2-base",
            "kimi-k2-base::harvest",
        }
        act = by_model["llama-8b::activation"]
        # Same GPU shape + rate as serving; 2 runners for 1h at $1.95 → $3.90.
        assert (act.gpu_type, act.gpu_count) == ("L40S", 1)
        assert act.running_containers == 2
        assert act.est_usd == pytest.approx(2 * 1.95)
        # App name derived from the model id (no explicit activation_app_name).
        act_fetch.assert_awaited_once_with("acs-llama-8b-activation")
    finally:
        await engine.dispose()


@dbtest
async def test_activation_fetch_failure_keeps_serving_row(monkeypatch):
    """A dead/undeployed activation app (e.g. kimi today) must not lose the
    serving sample — skip the activation row, keep the model's row."""
    from sqlalchemy import delete

    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    try:
        async with session_scope(factory) as s:
            await s.execute(delete(GpuCostSample))
        monkeypatch.setattr(
            cost_monitor.modalops, "fetch_runner_count_fresh", AsyncMock(return_value=1)
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_activation_runner_count_fresh",
            AsyncMock(side_effect=RuntimeError("app not found")),
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_harvest_runner_count_fresh",
            AsyncMock(side_effect=RuntimeError("harvester not deployed")),
        )
        await cost_monitor.run_cost_sample(
            session_factory=factory,
            settings=_settings(),
            models={
                "llama-8b": _model(
                    "1×L40S", "acs-llama-8b", activation_upstream_url="https://act.example"
                )
            },
            period_seconds=3600,
        )
        async with session_scope(factory) as s:
            rows = (await s.execute(select(GpuCostSample))).scalars().all()
        assert [r.model_id for r in rows] == ["llama-8b"]
    finally:
        await engine.dispose()


@dbtest
async def test_serving_fetch_failure_keeps_activation_row(monkeypatch):
    """The engines are independent Modal apps: a serving control-plane blip
    must not drop the activation sample for that tick (review finding on the
    first cut, which `continue`d past the activation block)."""
    from sqlalchemy import delete

    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    try:
        async with session_scope(factory) as s:
            await s.execute(delete(GpuCostSample))
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_runner_count_fresh",
            AsyncMock(side_effect=RuntimeError("serving RPC blip")),
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_activation_runner_count_fresh",
            AsyncMock(return_value=1),
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_harvest_runner_count_fresh",
            AsyncMock(side_effect=RuntimeError("harvester not deployed")),
        )
        await cost_monitor.run_cost_sample(
            session_factory=factory,
            settings=_settings(),
            models={
                "llama-8b": _model(
                    "1×L40S", "acs-llama-8b", activation_upstream_url="https://act.example"
                )
            },
            period_seconds=3600,
        )
        async with session_scope(factory) as s:
            rows = (await s.execute(select(GpuCostSample))).scalars().all()
        assert [r.model_id for r in rows] == ["llama-8b::activation"]
    finally:
        await engine.dispose()


@dbtest
async def test_harvest_engine_sampled_as_own_series(monkeypatch):
    """A harvest-capable model (Modal serving app present) gets a third
    ``<id>::harvest`` row on the same GPU shape/rate (ACS-281), fetched from
    the app derived from the MODEL ID (not modal_app_name — the Trinity
    rename trap, ACS-273)."""
    from sqlalchemy import delete

    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    try:
        async with session_scope(factory) as s:
            await s.execute(delete(GpuCostSample))
        monkeypatch.setattr(
            cost_monitor.modalops, "fetch_runner_count_fresh", AsyncMock(return_value=1)
        )
        harvest_fetch = AsyncMock(return_value=2)
        monkeypatch.setattr(
            cost_monitor.modalops, "fetch_harvest_runner_count_fresh", harvest_fetch
        )
        await cost_monitor.run_cost_sample(
            session_factory=factory,
            settings=_settings(),
            models={"trinity-truebase": _model("8×H200", "acs-trinity-base")},
            period_seconds=3600,
        )
        async with session_scope(factory) as s:
            rows = (await s.execute(select(GpuCostSample))).scalars().all()
        by_model = {r.model_id: r for r in rows}
        assert set(by_model) == {"trinity-truebase", "trinity-truebase::harvest"}
        hv = by_model["trinity-truebase::harvest"]
        # Same GPU shape + rate as serving; a big-model harvest at its 2-container
        # cap for 1h: 2 × 8 × $4.54 = $72.64.
        assert (hv.gpu_type, hv.gpu_count) == ("H200", 8)
        assert hv.running_containers == 2
        assert hv.est_usd == pytest.approx(2 * 36.32)
        # App derived from the model id, not the pre-rename modal_app_name.
        harvest_fetch.assert_awaited_once_with("acs-trinity-truebase-harvest")
    finally:
        await engine.dispose()


@dbtest
async def test_harvest_fetch_failure_keeps_other_rows(monkeypatch):
    """An unreachable harvester app must not lose the serving or activation
    samples for that tick — engines sample independently (skip-not-zero)."""
    from sqlalchemy import delete

    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    try:
        async with session_scope(factory) as s:
            await s.execute(delete(GpuCostSample))
        monkeypatch.setattr(
            cost_monitor.modalops, "fetch_runner_count_fresh", AsyncMock(return_value=1)
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_activation_runner_count_fresh",
            AsyncMock(return_value=1),
        )
        monkeypatch.setattr(
            cost_monitor.modalops,
            "fetch_harvest_runner_count_fresh",
            AsyncMock(side_effect=RuntimeError("harvester not deployed")),
        )
        await cost_monitor.run_cost_sample(
            session_factory=factory,
            settings=_settings(),
            models={
                "llama-8b": _model(
                    "1×L40S", "acs-llama-8b", activation_upstream_url="https://act.example"
                )
            },
            period_seconds=3600,
        )
        async with session_scope(factory) as s:
            rows = (await s.execute(select(GpuCostSample))).scalars().all()
        assert sorted(r.model_id for r in rows) == ["llama-8b", "llama-8b::activation"]
    finally:
        await engine.dispose()


# ---- cost-spike alert (DB) --------------------------------------------------

@dbtest
async def test_cost_spike_alert_fires_over_threshold(monkeypatch):
    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    try:
        # Isolate from any rows left by other tests/runs so the 24h sum is
        # exactly what we seed here.
        from sqlalchemy import delete

        async with session_scope(factory) as s:
            await s.execute(delete(GpuCostSample))
        # Seed two recent samples summing to $1000 in the last 24h.
        async with session_scope(factory) as s:
            for _ in range(2):
                s.add(
                    GpuCostSample(
                        model_id=f"m-{uuid.uuid4().hex[:6]}",
                        gpu_type="H200",
                        gpu_count=8,
                        running_containers=1,
                        hourly_usd_per_container=500.0,
                        period_seconds=3600,
                        est_usd=500.0,
                    )
                )
        captured = MagicMock()
        import sentry_sdk

        monkeypatch.setattr(sentry_sdk, "capture_message", captured)
        # new_scope context manager — give it a no-op scope.
        monkeypatch.setattr(sentry_sdk, "new_scope", lambda: _NullScope())

        # threshold 800 < 1000 → alert; threshold 2000 → no alert.
        await cost_monitor._maybe_alert(
            session_factory=factory,
            settings=_settings(sentry_dsn="https://x@x.ingest.sentry.io/1", cost_alert_daily_usd=800.0),
        )
        assert captured.called, "expected a cost-spike Sentry alert over threshold"

        captured.reset_mock()
        await cost_monitor._maybe_alert(
            session_factory=factory,
            settings=_settings(sentry_dsn="https://x@x.ingest.sentry.io/1", cost_alert_daily_usd=2000.0),
        )
        assert not captured.called, "must not alert when under threshold"
    finally:
        await engine.dispose()


class _NullScope:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def set_extra(self, *a, **k):
        pass

    fingerprint: list = []
