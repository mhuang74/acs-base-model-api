"""Startup/shutdown and scheduler wiring for the wrapper app."""

from __future__ import annotations

import datetime as dt
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from sqlalchemy import select

from . import breaker as breakermod
from . import cost_monitor as costmon
from . import docs_build
from . import modal_ops as modalops
from . import probe as probemod
from . import warm_window as warmwin
from .db import make_engine, make_session_factory
from .logging import configure_logging, get_logger
from .models import ChatGeneration, ModelWarmWindow
from .observability import init_sentry
from .runtime import WrapperRuntime, install_runtime, set_scheduler
from .settings import Settings
from .tokenizer import get_token_counter

log = get_logger()


def mirror_modal_credentials_to_env(settings: Settings) -> None:
    """Mirror Modal creds from Settings into ``os.environ`` so the two readers agree.

    There are two readers of the Modal credentials: the admin-page gating
    (``_modal_unavailable_msg`` reads the ``Settings`` object) and the actual
    RPC path (``modal_ops._auth`` + the Modal SDK's ``_Client.from_env`` read
    ``os.environ`` / ``~/.modal.toml``). ``Settings`` loads from env *and* from
    a ``.env`` file, but pydantic does not export ``.env`` values into the
    process env — so creds supplied only via ``.env`` make the page believe
    Modal is available while every RPC fails auth, rendering all models
    ``unknown`` (ACS-126).

    ``setdefault`` keeps real process env vars (prod/Railway) authoritative;
    this only fills the gap when the creds came from ``.env``. Idempotent.
    """
    if settings.modal_token_id and settings.modal_token_secret:
        os.environ.setdefault("MODAL_TOKEN_ID", settings.modal_token_id)
        os.environ.setdefault("MODAL_TOKEN_SECRET", settings.modal_token_secret)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    configure_logging(settings.log_level)
    mirror_modal_credentials_to_env(settings)  # keep Settings/.env ⇄ os.environ in sync (ACS-126)
    init_sentry(settings)  # no-op without SENTRY_DSN; privacy-scrubbed (ACS-40/43)

    engine = make_engine(settings.database_url)
    sessions = make_session_factory(engine)
    http = httpx.AsyncClient(follow_redirects=True)
    models = settings.parsed_models_registry()
    default_model_id = settings.resolve_default_model_id(models)

    # Boot-time GPU-rate-config validation (ACS-112). A non-empty but malformed
    # GPU_HOURLY_USD_BY_TYPE_JSON parses to {} → every cost sample silently
    # records $0. Surface that at boot rather than letting the spend chart
    # flatline unexplained. (Intentionally-empty config → no warning.)
    _rate_problem = costmon.rate_config_problem(settings)
    if _rate_problem:
        log.warning("gpu_rate_config_invalid", detail=_rate_problem)

    # Render the /tutorial markdown tree once at boot (ACS-89). Renderer is
    # synchronous + module-scoped (no I/O beyond ``Path.read_text``); the
    # rendered HTML lives on app.state for the lifetime of the process.
    _docs_api_base = settings.public_base_url.rstrip("/") + "/v1"
    tutorial_pages, tutorial_nav = docs_build.build_docs(
        docs_build.docs_root_default(), api_base=_docs_api_base
    )
    tutorial_combined = docs_build.build_combined_markdown(
        docs_build.docs_root_default(), api_base=_docs_api_base
    )

    install_runtime(
        app,
        WrapperRuntime(
            settings=settings,
            engine=engine,
            sessions=sessions,
            http=http,
            tutorial_pages=tutorial_pages,
            tutorial_nav=tutorial_nav,
            tutorial_combined=tutorial_combined,
            boot_time=dt.datetime.now(tz=dt.UTC),
            models=models,
            default_model_id=default_model_id,
            last_completion_at={},
            key_semaphores={},
            breakers=breakermod.BackendBreakers(),
            generations={},
        ),
    )

    # Wrapper-restart safety: any ChatGeneration row still marked ``running``
    # was orphaned by the previous process's crash/restart. Modal may keep the
    # upstream task alive, but we can't reattach to it from here, so mark them
    # ``failed`` with a recognisable error_message. Cheap; no row resume.
    async with sessions() as _gs:
        try:
            await _gs.execute(
                ChatGeneration.__table__.update()
                .where(ChatGeneration.status == "running")
                .values(
                    status="failed",
                    error_message="wrapper restarted while generation was in progress",
                    ended_at=dt.datetime.now(tz=dt.UTC),
                )
            )
            await _gs.commit()
        except Exception as exc:  # noqa: BLE001 — must not block boot
            log.warning(
                "chat_generations_orphan_cleanup_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
    # Warm each registered tokenizer (fails fast if HF auth is wrong).
    for entry in models.values():
        get_token_counter(entry.tokenizer_repo, settings.hf_token)

    # APScheduler: load the cron from probe_schedule (seeded by migration) and
    # install the probe job. In-memory jobstore is fine for one replica; scaling
    # to N would need SQLAlchemyJobStore so only one replica fires each tick.
    set_scheduler(app, None)
    if not getattr(app.state, "disable_scheduler", False):
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger

            cron_expr = await probemod.load_cron_expression(sessions)
            scheduler = AsyncIOScheduler(timezone="UTC")
            scheduler.add_job(
                _probe_job_wrapper,
                CronTrigger.from_crontab(cron_expr, timezone="UTC"),
                id="capacity_probe",
                replace_existing=True,
                kwargs={"app": app},
            )
            # Per-model warm windows: each enabled row installs two CronTriggers
            # (warm + cool). On invalid cron/tz we skip the bad row and keep
            # booting — the admin UI surfaces a way to fix it.
            async with sessions() as _ws:
                windows = list(
                    (
                        await _ws.execute(
                            select(ModelWarmWindow).where(ModelWarmWindow.enabled.is_(True))
                        )
                    ).scalars().all()
                )
            for w in windows:
                try:
                    _register_warm_window_jobs(scheduler, app, w)
                except Exception as exc:  # noqa: BLE001 — one bad row mustn't block boot
                    log.warning(
                        "warm_window_register_failed",
                        model_id=w.model_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            # GPU cost monitor: sample running container counts on a fixed
            # interval to estimate $/model/day and trip a cost-spike alert.
            from apscheduler.triggers.interval import IntervalTrigger

            cost_interval_min = max(1, int(app.state.settings.cost_sample_interval_minutes))
            cost_warmup_s = max(0, int(app.state.settings.cost_sample_warmup_seconds))
            scheduler.add_job(
                _cost_sample_job_wrapper,
                IntervalTrigger(minutes=cost_interval_min),
                id="gpu_cost_sample",
                replace_existing=True,
                # Fire shortly after boot (IntervalTrigger's first run is otherwise
                # +interval): the wrapper redeploys often, which would reset the
                # interval clock and could mean the sampler ~never runs. Each
                # sample bills only actual elapsed time, so it doesn't over-count.
                # The warm-up delay lets the Modal client get ready so the uncached
                # runner-count read returns true counts, not a cold-start 0.
                next_run_time=dt.datetime.now(dt.UTC) + dt.timedelta(seconds=cost_warmup_s),
                kwargs={"app": app, "period_seconds": cost_interval_min * 60},
            )
            # Scheduled bulk-email batches (ACS-228): check every minute for
            # batches whose scheduled_at has passed and send them off-request.
            scheduler.add_job(
                _bulk_email_job_wrapper,
                IntervalTrigger(minutes=1),
                id="bulk_email_scheduled_send",
                replace_existing=True,
                kwargs={"app": app},
            )
            scheduler.start()
            set_scheduler(app, scheduler)
            log.info(
                "scheduler_started",
                cron=cron_expr,
                warm_windows=[w.model_id for w in windows],
                cost_sample_interval_min=cost_interval_min,
            )
        except Exception as exc:  # noqa: BLE001 — scheduler failures must not block boot
            log.warning("scheduler_start_failed", error=f"{type(exc).__name__}: {exc}")

    # Pre-warm the modal_ops caches in the background so the first workbench
    # render doesn't pay the cold-cache cost (up to ~18s when Modal's control
    # plane is slow).
    try:
        app_names = [
            m.modal_app_name
            for m in models.values()
            if m.modal_app_name
        ]
        if app_names:
            modalops.prewarm_caches(app_names)
            log.info("modal_cache_prewarm_scheduled", app_names=app_names)
    except Exception as exc:  # noqa: BLE001 — startup must not abort on prewarm issues
        log.warning(
            "modal_cache_prewarm_failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    log.info(
        "startup",
        default_model=default_model_id,
        models=sorted(models),
    )
    try:
        yield
    finally:
        scheduler = getattr(app.state, "scheduler", None)
        # ``.running`` guards against a double-shutdown on an already-stopped
        # scheduler (e.g. a nested TestClient lifespan on the shared ``app``
        # overwrites ``app.state.scheduler`` and stops it, then the outer
        # lifespan exits and reads the same stopped object). APScheduler's
        # ``shutdown()`` raises SchedulerNotRunningError when already stopped, so
        # keep the lifespan teardown idempotent (ACS-309).
        if scheduler is not None and scheduler.running:
            scheduler.shutdown(wait=False)
        await http.aclose()
        await engine.dispose()


async def _probe_job_wrapper(app: FastAPI) -> None:
    """APScheduler entry point — pulled out so tests can patch it."""
    await probemod.run_probe(
        session_factory=app.state.sessions,
        settings=app.state.settings,
    )


async def _cost_sample_job_wrapper(app: FastAPI, period_seconds: int) -> None:
    """APScheduler entry point for the GPU cost sampler (patchable in tests)."""
    try:
        await costmon.run_cost_sample(
            session_factory=app.state.sessions,
            settings=app.state.settings,
            models=app.state.models,
            period_seconds=period_seconds,
        )
    except Exception as exc:  # noqa: BLE001 — a sampler failure must not kill the scheduler
        log.warning("cost_sample_job_failed", error=f"{type(exc).__name__}: {exc}")


async def _bulk_email_job_wrapper(app: FastAPI) -> None:
    """APScheduler entry point for due scheduled bulk-email batches (patchable)."""
    from .routes.admin.bulk_emails import send_due_batches

    try:
        await send_due_batches(app)
    except Exception as exc:  # noqa: BLE001 — a send failure must not kill the scheduler
        log.warning("bulk_email_job_failed", error=f"{type(exc).__name__}: {exc}")


async def _warm_window_job_wrapper(app: FastAPI, model_id: str, phase: str) -> None:
    """APScheduler entry point — resolves the model and dispatches warm/cool."""
    entry = app.state.models.get(model_id)
    if entry is None or entry.modal_app_name is None:
        log.warning(
            "warm_window_skipped",
            model_id=model_id,
            phase=phase,
            reason="model not in registry or has no modal_app_name",
        )
        return
    if phase == "warm":
        await warmwin.fire_warm(model_id, entry.modal_app_name)
    elif phase == "cool":
        await warmwin.fire_cool(model_id, entry.modal_app_name)
    else:
        log.warning("warm_window_unknown_phase", model_id=model_id, phase=phase)


def _register_warm_window_jobs(scheduler, app: FastAPI, window: ModelWarmWindow) -> None:
    """Install (or replace) the warm + cool CronTrigger jobs for ``window``."""
    from apscheduler.triggers.cron import CronTrigger

    scheduler.add_job(
        _warm_window_job_wrapper,
        CronTrigger.from_crontab(window.warm_cron, timezone=window.timezone),
        id=f"warm_window:{window.model_id}:warm",
        replace_existing=True,
        kwargs={"app": app, "model_id": window.model_id, "phase": "warm"},
    )
    scheduler.add_job(
        _warm_window_job_wrapper,
        CronTrigger.from_crontab(window.cool_cron, timezone=window.timezone),
        id=f"warm_window:{window.model_id}:cool",
        replace_existing=True,
        kwargs={"app": app, "model_id": window.model_id, "phase": "cool"},
    )


def _unregister_warm_window_jobs(scheduler, model_id: str) -> None:
    """Drop both warm + cool jobs for ``model_id``; idempotent."""
    for suffix in ("warm", "cool"):
        job_id = f"warm_window:{model_id}:{suffix}"
        try:
            scheduler.remove_job(job_id)
        except Exception:  # noqa: BLE001 — APScheduler raises JobLookupError when absent
            pass
