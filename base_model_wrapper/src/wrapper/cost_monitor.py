"""GPU cost monitoring — periodic spend sampler + cost-spike alert.

Estimates per-model GPU spend by sampling running container counts on a schedule
(APScheduler job wired in ``lifespan``), writes one ``GpuCostSample`` row per
live model — plus one row per activation-enabled model for its separate
``acs-<id>-activation`` engine, under ``model_id = "<id>::activation"``
(ACS-221; same ``::activation`` suffix as the breaker key), plus one row per
harvest-capable model for its ``acs-<id>-harvest`` bulk-harvester app, under
``model_id = "<id>::harvest"`` (ACS-281) — and raises a *grouped* Sentry alert
when rolling-24h spend exceeds ``settings.cost_alert_daily_usd``.

Approximate by design: cost = running-containers-at-tick × per-container hourly
rate × interval. It misses sub-interval container churn, so it's a $/day surface
+ runaway-spend tripwire, not exact billing (the lifetime-log path in
``benchmarks/billing.py`` is the accurate-reconciliation tool).

GPU shape comes from each model's ``gpu_shape_label`` ("8xH200", "1xL40S") in the
wrapper's runtime registry (``app.state.models``) — the same source the rest of
the wrapper uses — so this module needs no direct ``acs_model_registry`` import.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sqlalchemy import func as safunc
from sqlalchemy import select

from . import modal_ops as modalops
from .db import session_scope
from .logging import get_logger
from .models import GpuCostSample

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from .settings import ModelEntry, Settings

log = get_logger()

# gpu_shape_label uses the multiplication sign "×" (U+00D7) in prod
# (e.g. "8×H200"), not ASCII "x" — match both. (This bit us once: the regex
# only matched ASCII "x", so every model was skipped and gpu_cost_sample stayed
# empty.)
_SHAPE_RE = re.compile(r"^\s*(\d+)\s*[x×]\s*(.+?)\s*$", re.IGNORECASE)


def parse_gpu_shape(label: str | None) -> tuple[int, str] | None:
    """Parse a ``gpu_shape_label`` like "8×H200" / "8xH200" → (8, "H200"). None
    if it doesn't match (e.g. the "legacy" placeholder)."""
    m = _SHAPE_RE.match(label or "")
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def estimate_usd(running: int, per_container_hourly: float, period_seconds: int) -> float:
    """est = running containers × per-container $/hr × hours in the period."""
    return running * per_container_hourly * (period_seconds / 3600.0)


def rate_config_problem(settings: Settings) -> str | None:
    """Return a human-readable warning if the GPU-rate config is *set but broken*.

    ``parsed_gpu_hourly_rates()`` returns ``{}`` on malformed JSON (or a JSON
    value that isn't an object, or an object whose values are all non-numeric).
    With no rates every ``per_container`` collapses to 0, so the sampler quietly
    writes ``est_usd = 0`` rows and the spend chart flatlines — a silent failure.

    This distinguishes the *disabled* case (raw empty → intentional, no rates)
    from the *broken* case (raw non-empty but parses to ``{}`` → someone set the
    var and it's wrong). Only the broken case warrants a warning; the caller
    (``lifespan``) logs it at boot. Returns ``None`` when the config is fine or
    intentionally empty.

    Partial drops (one bad value in an otherwise-good object) are *not* flagged
    here — those still surface per-tick as ``cost_sample_no_rate_for_gpu_type``.
    """
    raw = (settings.gpu_hourly_usd_by_type_json or "").strip()
    if not raw:
        return None  # intentional "no rates configured" — not an error
    if settings.parsed_gpu_hourly_rates():
        return None  # parsed to at least one usable rate
    return (
        "GPU_HOURLY_USD_BY_TYPE_JSON is set but parses to no usable rates "
        f"({raw!r}); every GPU cost sample will be $0 until it is fixed. "
        'Expected a JSON object like {"H200": 4.54, "L40S": 1.95}.'
    )


async def _app_maybe_running(app_name: str) -> bool:
    """True unless Modal affirmatively reports the app as *stopped*.

    Used to decide whether a fresh runner count of 0 is worth a second look. An
    explicitly-stopped app genuinely has 0 containers, so we trust that 0; any
    other state (deployed / initializing / unknown) leaves the door open that a
    single control-plane read under-reported, so the zero is "suspicious". State
    comes from the SWR-cached ``get_app_state`` (cheap, pre-warmed), and any
    error there → treat as maybe-running (retry) rather than trust the zero.
    """
    try:
        return (await modalops.get_app_state(app_name)) != "stopped"
    except Exception:  # noqa: BLE001 — can't tell → treat the zero as suspicious
        return True


async def _read_serving_count(app_name: str) -> int:
    """Fresh serving runner count, with one retry on a *suspicious* zero.

    The sampler already reads uncached (``fetch_runner_count_fresh``), so a
    returned 0 is Modal affirmatively reporting ``num_total_tasks == 0`` — not a
    swallowed failure (that path raises). But a single control-plane RPC can
    briefly under-report while containers exist (eventual consistency), and
    recording that 0 would undercount an always-on model's spend. So on a fresh
    0, unless the app is explicitly stopped, read once more before trusting it.
    Raises on RPC failure — the caller skips the row (skip-not-zero policy).

    NB: this is belt-and-braces on top of the earlier cached-0 fix (ACS-50); it
    narrows, not eliminates, the transient-zero window (a second read can still
    return the same 0 if the lag outlasts it).
    """
    count = await modalops.fetch_runner_count_fresh(app_name)
    if count == 0 and await _app_maybe_running(app_name):
        count = await modalops.fetch_runner_count_fresh(app_name)
    return count


async def _effective_period_seconds(
    session_factory: async_sessionmaker[AsyncSession], interval_seconds: int
) -> int:
    """Seconds this sample should bill for = actual time since the last sample,
    clamped to [1, interval].

    Decouples cost from the scheduler cadence: the job runs once at boot and then
    every interval, and the wrapper may redeploy often — billing the real gap (not
    a fixed interval) means frequent reboots don't over-count, while the cap stops
    a long wrapper outage from recording one giant spurious sample. First sample
    ever (no prior row) bills one interval as a sensible default.
    """
    async with session_factory() as s:
        last_ts = (
            await s.execute(select(safunc.max(GpuCostSample.ts)))
        ).scalar_one_or_none()
    if last_ts is None:
        return interval_seconds
    elapsed = (dt.datetime.now(tz=dt.UTC) - last_ts).total_seconds()
    return int(max(1, min(elapsed, interval_seconds)))


async def run_cost_sample(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    models: Mapping[str, ModelEntry],
    period_seconds: int,
) -> None:
    """Sample running container counts for live models, write cost rows, alert."""
    rates = settings.parsed_gpu_hourly_rates()
    effective_period = await _effective_period_seconds(session_factory, period_seconds)
    samples: list[GpuCostSample] = []
    for model_id, entry in models.items():
        if getattr(entry, "status", "live") != "live":
            continue
        shape = parse_gpu_shape(getattr(entry, "gpu_shape_label", None))
        if shape is None:
            log.warning(
                "cost_sample_unparseable_gpu_shape",
                model_id=model_id,
                gpu_shape_label=getattr(entry, "gpu_shape_label", None),
            )
            continue
        gpu_count, gpu_type = shape
        if gpu_type not in rates:
            log.warning(
                "cost_sample_no_rate_for_gpu_type", model_id=model_id, gpu_type=gpu_type
            )
        per_container = rates.get(gpu_type, 0.0) * gpu_count

        # Row builder for this model's engines (serving + activation share the
        # GPU shape/rate). Defined per iteration; only called within it.
        def _sample(series_id: str, running: int) -> GpuCostSample:
            return GpuCostSample(
                model_id=series_id,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                running_containers=running,
                hourly_usd_per_container=per_container,
                period_seconds=effective_period,
                est_usd=estimate_usd(running, per_container, effective_period),
            )

        # Serving engine. Fresh (uncached) read with a suspicious-zero retry
        # (see ``_read_serving_count``): a stale/failure-cached 0 from the SWR
        # cache would mis-bill an always-on model as idle, and even a fresh 0
        # can be a transient control-plane under-report. Skip this ROW on
        # failure rather than record a misleading 0 — the activation engine
        # below is a separate Modal app and is still sampled.
        app_name = getattr(entry, "modal_app_name", None)
        if app_name:
            try:
                running = await _read_serving_count(app_name)
            except Exception as exc:  # noqa: BLE001 — modal raises various RPC/auth errors
                log.warning(
                    "cost_sample_runner_count_failed",
                    model_id=model_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                samples.append(_sample(model_id, running))

        # Activation engine (ACS-221): a separate always-scalable Modal app on
        # the SAME GPU shape as serving (modal_app_activation.py deploys with
        # the registry spec's gpu/count). Sampled only for models that expose
        # activation support; recorded as its own ``<id>::activation`` series
        # (the breaker-key suffix) so serving vs activation spend stays
        # separable on the dashboard. Independent of the serving fetch above —
        # a serving control-plane blip must not drop activation spend.
        if getattr(entry, "activation_upstream_url", None):
            act_app = modalops.resolve_activation_app(entry, model_id)
            act_series = f"{model_id}{modalops.ACTIVATION_KEY_SUFFIX}"
            try:
                act_running = await modalops.fetch_activation_runner_count_fresh(act_app)
            except Exception as exc:  # noqa: BLE001 — same skip-not-zero policy as serving
                log.warning(
                    "cost_sample_runner_count_failed",
                    model_id=act_series,
                    app_name=act_app,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                samples.append(_sample(act_series, act_running))

        # Bulk-harvest engine (ACS-281): the per-model offline harvester
        # (``acs-<id>-harvest``, ACS-245) deploys on the SAME registry GPU
        # shape as serving, so it reuses the model's shape/rate. Containers
        # are ephemeral batch jobs — the tick sampler catches the long
        # multi-GPU harvests (the spend that matters); a short 8B harvest can
        # fall between ticks (the module-level undercount caveat). Gated on
        # ``modal_app_name`` like the /v1/harvest route; independent of the
        # fetches above, same skip-not-zero policy.
        harvest_app = modalops.resolve_harvest_app(entry, model_id)
        if harvest_app:
            harvest_series = f"{model_id}{modalops.HARVEST_KEY_SUFFIX}"
            try:
                harvest_running = await modalops.fetch_harvest_runner_count_fresh(
                    harvest_app
                )
            except Exception as exc:  # noqa: BLE001 — same skip-not-zero policy
                log.warning(
                    "cost_sample_runner_count_failed",
                    model_id=harvest_series,
                    app_name=harvest_app,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                samples.append(_sample(harvest_series, harvest_running))

    if not samples:
        return
    async with session_scope(session_factory) as s:
        s.add_all(samples)
    log.info(
        "cost_sample_written",
        n=len(samples),
        total_est_usd=round(sum(x.est_usd for x in samples), 2),
    )
    await _maybe_alert(session_factory=session_factory, settings=settings)


async def _maybe_alert(
    *, session_factory: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    """Raise a grouped Sentry alert if rolling-24h estimated spend > threshold.

    Stable fingerprint => all spikes group into one Sentry issue (fires the
    Discord notification once, then just bumps the occurrence count) — no spam.
    """
    threshold = settings.cost_alert_daily_usd
    if not threshold or threshold <= 0:
        return
    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=24)
    async with session_factory() as s:
        total = (
            await s.execute(
                select(safunc.coalesce(safunc.sum(GpuCostSample.est_usd), 0.0)).where(
                    GpuCostSample.ts >= since
                )
            )
        ).scalar_one()
    total = float(total or 0.0)
    if total <= threshold:
        return
    log.warning("cost_spike", rolling_24h_usd=round(total, 2), threshold_usd=threshold)
    if not settings.sentry_dsn:
        return
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.fingerprint = ["gpu-cost-spike"]
            scope.set_extra("rolling_24h_usd", round(total, 2))
            scope.set_extra("threshold_usd", threshold)
            sentry_sdk.capture_message(
                f"GPU cost spike: ~${total:,.0f}/24h exceeds ${threshold:,.0f} threshold",
                level="warning",
            )
    except Exception as exc:  # noqa: BLE001 — alerting must never break the job
        log.warning("cost_spike_alert_failed", error=f"{type(exc).__name__}: {exc}")
