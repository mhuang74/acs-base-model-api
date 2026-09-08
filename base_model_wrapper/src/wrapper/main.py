"""FastAPI app: auth → budget clamp → proxy → usage commit.

See: wiki/common-projects/base-model-hosting/wrapper-implementation-plan.md
"""

from __future__ import annotations

import uuid
from pathlib import Path

import sentry_sdk
from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession

from . import auth as authmod
from . import breaker as breakermod
from . import modal_ops as modalops
from . import proxy as proxymod
from . import web_auth as webauth
from . import workbench_generations as genmod
from .app import create_app
from .db import get_session
from .dependencies import get_settings
from .lifespan import (
    _probe_job_wrapper as _lifespan_probe_job_wrapper,
    _register_warm_window_jobs as _lifespan_register_warm_window_jobs,
    _unregister_warm_window_jobs as _lifespan_unregister_warm_window_jobs,
    _warm_window_job_wrapper as _lifespan_warm_window_job_wrapper,
    lifespan as app_lifespan,
)
from .logging import get_logger
from .models import User
from .rate_limit import limiter
from .schemas import CompletionsRequest, HarvestRequest
from .routes import admin as admin_routes
from .routes import api as api_routes
from .routes import discord as discord_routes
from .routes import loom as loom_routes
from .routes import usage as usage_routes
from .routes import web as web_routes
from .routes import workbench as workbench_routes
from .routes.admin import (
    admin_model_deploy as admin_model_deploy,
    admin_model_keep_warm as admin_model_keep_warm,
    admin_model_release as admin_model_release,
    admin_model_stop as admin_model_stop,
    admin_models_status as admin_models_status,
    admin_probe_schedule as admin_probe_schedule,
    admin_warm_window_delete as admin_warm_window_delete,
    admin_warm_window_upsert as admin_warm_window_upsert,
)
from .routes.usage import usage_tab as usage_tab
from .tokenizer import get_token_counter
from .warm_state import (
    MODAL_SCALEDOWN_WINDOW,
    _is_model_warm,
    _mark_model_warm,
    _resolve_warm_flags,
)

_COMPAT_EXPORT_BINDINGS = (
    authmod,
    breakermod,
    modalops,
    proxymod,
    get_settings,
    admin_model_deploy,
    admin_model_keep_warm,
    admin_model_release,
    admin_model_stop,
    admin_models_status,
    admin_probe_schedule,
    admin_warm_window_delete,
    admin_warm_window_upsert,
    usage_tab,
    MODAL_SCALEDOWN_WINDOW,
    _is_model_warm,
    _mark_model_warm,
    _resolve_warm_flags,
)

__all__ = [
    "app",
    "lifespan",
    "authmod",
    "breakermod",
    "modalops",
    "proxymod",
    "get_settings",
    "get_token_counter",
    "_probe_job_wrapper",
    "_warm_window_job_wrapper",
    "_register_warm_window_jobs",
    "_unregister_warm_window_jobs",
    "MAX_INFLIGHT_PER_KEY",
    "MAX_QUEUED_PER_KEY",
    "COMPLETIONS_RATE_LIMIT",
    "COMPLETIONS_RATE_LIMIT_PER_MINUTE",
    "_get_key_semaphore",
    "MODAL_SCALEDOWN_WINDOW",
    "_mark_model_warm",
    "_is_model_warm",
    "_resolve_warm_flags",
    "GenerationState",
    "LoomGenerationState",
    "_absorb_loom_chunk",
    "_normalise_logprobs",
    "_sse_loom_chunk_frame",
    "GENERATION_EVICT_S",
    "GENERATION_FLUSH_INTERVAL_S",
    "GENERATION_FLUSH_CHARS",
    "HEARTBEAT_INTERVAL_S",
    "SSE_HEADERS",
    "_resolve_model",
    "_unknown_model_response",
    "_validation_message",
    "_check_sequence_length",
    "_apply_extras_headers",
    "_apply_backend_headers",
    "_backend_health_entry",
    "_record_request",
    "_ping_database",
    "aggregate_health",
    "health",
    "models",
    "chat_completions_unsupported",
    "run_completion_nonstream",
    "_budget_error",
    "completions",
    "_serve_stream",
    "_error",
    "admin_model_deploy",
    "admin_model_keep_warm",
    "admin_model_release",
    "admin_model_stop",
    "admin_models_page",
    "admin_models_status",
    "admin_probe_schedule",
    "admin_warm_window_delete",
    "admin_warm_window_upsert",
    "landing",
    "tutorial",
    "login_form",
    "forgot_password_form",
    "forgot_password_submit",
    "reset_password_form",
    "reset_password_submit",
    "signup_form",
    "signup_submit",
    "login_submit",
    "logout",
    "dashboard",
    "me_create_key",
    "me_rename_key",
    "me_pause_key",
    "me_resume_key",
    "me_revoke_key",
    "me_password",
    "submit_feedback",
    "_dashboard_keys",
    "_render_dashboard",
    "_friendly_upstream_error",
    "_derive_title",
    "_workbench_keys",
    "_resolve_workbench_model_states",
    "workbench_models_status",
    "_UI_HIDDEN_MODEL_IDS",
    "_gen_live",
    "_run_generation_task",
    "_sse_status_frame",
    "_sse_error_frame",
    "_sse_replay_frame",
    "_sse_done_frame",
    "_absorb_chunk_text",
    "_parse_upstream_error_message",
    "_session_to_jsonl_record",
    "chat_index",
    "chat_new",
    "chat_export_all",
    "chat_open",
    "chat_rename",
    "chat_delete",
    "chat_revert",
    "chat_stream_gone",
    "chat_generation_start",
    "chat_generation_events",
    "chat_generation_cancel",
    "chat_export_txt",
    "chat_export_jsonl",
    "usage_tab",
]

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

log = get_logger()

GenerationState = genmod.GenerationState
GENERATION_EVICT_S = genmod.GENERATION_EVICT_S
GENERATION_FLUSH_INTERVAL_S = genmod.GENERATION_FLUSH_INTERVAL_S
GENERATION_FLUSH_CHARS = genmod.GENERATION_FLUSH_CHARS
HEARTBEAT_INTERVAL_S = genmod.HEARTBEAT_INTERVAL_S
SSE_HEADERS = genmod.SSE_HEADERS
MAX_INFLIGHT_PER_KEY = api_routes.MAX_INFLIGHT_PER_KEY
MAX_QUEUED_PER_KEY = api_routes.MAX_QUEUED_PER_KEY
COMPLETIONS_RATE_LIMIT = api_routes.COMPLETIONS_RATE_LIMIT
COMPLETIONS_RATE_LIMIT_PER_MINUTE = api_routes.COMPLETIONS_RATE_LIMIT_PER_MINUTE
_get_key_semaphore = api_routes._get_key_semaphore


# Compatibility exports: tests and a few route handlers import these private
# names from wrapper.main. The implementations now live in wrapper.lifespan.
lifespan = app_lifespan
_probe_job_wrapper = _lifespan_probe_job_wrapper
_warm_window_job_wrapper = _lifespan_warm_window_job_wrapper
_register_warm_window_jobs = _lifespan_register_warm_window_jobs
_unregister_warm_window_jobs = _lifespan_unregister_warm_window_jobs


from .settings import Settings as _Settings  # noqa: E402

# CORS is installed at app-creation time (before lifespan runs), so it can't
# read app.state.settings — instantiate Settings once from env to grab the
# allow-origin list. The full Settings used by request handlers is still
# instantiated inside lifespan() and attached to app.state.settings.
_cors_origins = _Settings().cors_allow_origins
app = create_app(
    lifespan=lifespan,
    limiter=limiter,
    cors_allow_origins=_cors_origins,
    # These routes validate their bodies by hand (raw JSON → model_validate),
    # so FastAPI can't infer the request schema from the signature; hand the
    # models to the OpenAPI builder so their field constraints are published.
    request_body_models={
        "/v1/completions": CompletionsRequest,
        "/v1/harvest": HarvestRequest,
    },
)

app.include_router(api_routes.router)
app.include_router(admin_routes.router)
app.include_router(web_routes.router)
app.include_router(discord_routes.router)
app.include_router(workbench_routes.router)
app.include_router(loom_routes.router)
app.include_router(usage_routes.router)

_resolve_model_states = admin_routes._resolve_model_states


async def admin_models_page(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Compatibility wrapper for tests importing this route from wrapper.main.

    Propagate anything tests patched on ``wrapper.main`` (``_resolve_model_states``,
    ``templates``) onto the ``routes.admin`` module globals the real handler
    reads, so patching via this module still reaches the delegated call — then
    restore them afterwards. These are plain module-attribute writes, not
    monkeypatches, so pytest's teardown never undoes them; without the
    try/finally a test that patches the resolver and calls this shim would leave
    a stale fake bound on ``routes.admin.models`` and pollute later test files
    (cross-file isolation bug, ACS-97).
    """
    saved_admin_resolve = admin_routes._resolve_model_states
    saved_models_resolve = admin_routes.models._resolve_model_states
    saved_models_templates = admin_routes.models.templates
    admin_routes._resolve_model_states = _resolve_model_states
    admin_routes.models._resolve_model_states = _resolve_model_states
    admin_routes.models.templates = templates
    try:
        return await admin_routes.admin_models_page(request, admin, session)
    finally:
        admin_routes._resolve_model_states = saved_admin_resolve
        admin_routes.models._resolve_model_states = saved_models_resolve
        admin_routes.models.templates = saved_models_templates


# API compatibility exports. Tests still patch helpers through wrapper.main, so
# wrappers sync those patched names into wrapper.routes.api only for the call.
_resolve_model = api_routes._resolve_model
_unknown_model_response = api_routes._unknown_model_response
_validation_message = api_routes._validation_message
_apply_extras_headers = api_routes._apply_extras_headers
_apply_backend_headers = api_routes._apply_backend_headers
_backend_health_entry = api_routes._backend_health_entry
_record_request = api_routes._record_request
_ping_database = api_routes._ping_database
health = api_routes.health
models = api_routes.models
chat_completions_unsupported = api_routes.chat_completions_unsupported
completions = api_routes.completions

_API_RECORD_REQUEST = api_routes._record_request
_API_GET_TOKEN_COUNTER = api_routes.get_token_counter
_API_PING_DATABASE = api_routes._ping_database
_API_SSE_HEADERS = api_routes.SSE_HEADERS


def _sync_api_compat() -> None:
    api_routes._record_request = _record_request
    api_routes.get_token_counter = get_token_counter
    api_routes._ping_database = _ping_database
    api_routes.SSE_HEADERS = SSE_HEADERS


def _restore_api_compat() -> None:
    api_routes._record_request = _API_RECORD_REQUEST
    api_routes.get_token_counter = _API_GET_TOKEN_COUNTER
    api_routes._ping_database = _API_PING_DATABASE
    api_routes.SSE_HEADERS = _API_SSE_HEADERS


async def _check_sequence_length(*args, **kwargs):
    _sync_api_compat()
    try:
        return await api_routes._check_sequence_length(*args, **kwargs)
    finally:
        _restore_api_compat()


async def aggregate_health(*args, **kwargs):
    _sync_api_compat()
    try:
        return await api_routes.aggregate_health(*args, **kwargs)
    finally:
        _restore_api_compat()


async def run_completion_nonstream(*args, **kwargs):
    _sync_api_compat()
    try:
        return await api_routes.run_completion_nonstream(*args, **kwargs)
    finally:
        _restore_api_compat()


async def _budget_error(*args, **kwargs):
    _sync_api_compat()
    try:
        return await api_routes._budget_error(*args, **kwargs)
    finally:
        _restore_api_compat()


async def _serve_stream(*args, **kwargs):
    _sync_api_compat()
    try:
        return await api_routes._serve_stream(*args, **kwargs)
    finally:
        _restore_api_compat()


async def _error(*args, **kwargs):
    _sync_api_compat()
    try:
        return await api_routes._error(*args, **kwargs)
    finally:
        _restore_api_compat()


# Web compatibility exports.
landing = web_routes.landing
tutorial = web_routes.tutorial
login_form = web_routes.login_form
forgot_password_form = web_routes.forgot_password_form
forgot_password_submit = web_routes.forgot_password_submit
reset_password_form = web_routes.reset_password_form
reset_password_submit = web_routes.reset_password_submit
signup_form = web_routes.signup_form
signup_submit = web_routes.signup_submit
login_submit = web_routes.login_submit
logout = web_routes.logout
dashboard = web_routes.dashboard
me_create_key = web_routes.me_create_key
me_rename_key = web_routes.me_rename_key
me_pause_key = web_routes.me_pause_key
me_resume_key = web_routes.me_resume_key
me_revoke_key = web_routes.me_revoke_key
me_password = web_routes.me_password
submit_feedback = web_routes.submit_feedback
_session_misconfigured = web_routes._session_misconfigured
_reset_invalid_response = web_routes._reset_invalid_response
_signup_disabled_response = web_routes._signup_disabled_response
_dashboard_keys = web_routes._dashboard_keys
_render_dashboard = web_routes._render_dashboard
_feedback_truthy = web_routes._feedback_truthy


# Workbench compatibility exports. A few tests monkeypatch wrapper.main's
# generation constants / hidden model ids, so wrapper calls sync first.
_friendly_upstream_error = workbench_routes._friendly_upstream_error
_derive_title = workbench_routes._derive_title
_workbench_keys = workbench_routes._workbench_keys
_user_chat_sessions = workbench_routes._user_chat_sessions
_get_owned_chat_session = workbench_routes._get_owned_chat_session
_session_snapshots = workbench_routes._session_snapshots
_resolve_workbench_model_states = workbench_routes._resolve_workbench_model_states
_running_generation_for_session = workbench_routes._running_generation_for_session
_render_chat = workbench_routes._render_chat
_session_to_jsonl_record = workbench_routes._session_to_jsonl_record
_sse_status_frame = workbench_routes._sse_status_frame
_sse_error_frame = workbench_routes._sse_error_frame
_sse_replay_frame = workbench_routes._sse_replay_frame
_sse_done_frame = workbench_routes._sse_done_frame
_absorb_chunk_text = workbench_routes._absorb_chunk_text
_parse_upstream_error_message = workbench_routes._parse_upstream_error_message
_UI_HIDDEN_MODEL_IDS = workbench_routes._UI_HIDDEN_MODEL_IDS
# Loom (ACS-148) compat re-exports for tests.
LoomGenerationState = workbench_routes.LoomGenerationState
_absorb_loom_chunk = workbench_routes._absorb_loom_chunk
_normalise_logprobs = workbench_routes._normalise_logprobs
_sse_loom_chunk_frame = workbench_routes._sse_loom_chunk_frame


def _sync_workbench_compat() -> None:
    workbench_routes._UI_HIDDEN_MODEL_IDS = _UI_HIDDEN_MODEL_IDS
    workbench_routes.GENERATION_EVICT_S = GENERATION_EVICT_S
    workbench_routes.GENERATION_FLUSH_INTERVAL_S = GENERATION_FLUSH_INTERVAL_S
    workbench_routes.GENERATION_FLUSH_CHARS = GENERATION_FLUSH_CHARS
    workbench_routes.HEARTBEAT_INTERVAL_S = HEARTBEAT_INTERVAL_S
    workbench_routes.SSE_HEADERS = SSE_HEADERS


async def workbench_models_status(*args, **kwargs):
    _sync_workbench_compat()
    return await workbench_routes.workbench_models_status(*args, **kwargs)


async def _gen_live(*args, **kwargs):
    _sync_workbench_compat()
    async for frame in workbench_routes._gen_live(*args, **kwargs):
        yield frame


async def _run_generation_task(**kwargs) -> None:
    _sync_workbench_compat()
    await workbench_routes._run_generation_task(**kwargs)


chat_index = workbench_routes.chat_index
chat_new = workbench_routes.chat_new
chat_export_all = workbench_routes.chat_export_all
chat_open = workbench_routes.chat_open
chat_rename = workbench_routes.chat_rename
chat_delete = workbench_routes.chat_delete
chat_revert = workbench_routes.chat_revert
chat_stream_gone = workbench_routes.chat_stream_gone
chat_generation_start = workbench_routes.chat_generation_start
chat_generation_events = workbench_routes.chat_generation_events
chat_generation_cancel = workbench_routes.chat_generation_cancel
chat_export_txt = workbench_routes.chat_export_txt
chat_export_jsonl = workbench_routes.chat_export_jsonl


def _wants_html(request: Request) -> bool:
    """Heuristic: did a browser request this page (vs. an SDK / curl)?

    Browsers always send `Accept: text/html` before `application/json`; SDK
    clients usually send `*/*` or only `application/json`.
    """
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    """Unwrap FastAPI's default {"detail": ...} envelope when the handler already
    provided an OpenAI-shaped {"error": {...}} body. SDK clients look for
    `error` at the top level. Forward any headers set on the exception (e.g.
    `Location` on a 303 raised by ``require_admin`` for unauthenticated users).

    For browser requests, render an HTML page for ``admin_required`` 403s so a
    non-admin who navigates to ``/admin`` sees a friendly error rather than raw
    JSON.
    """
    body = exc.detail
    headers = exc.headers or None
    if (
        exc.status_code == status.HTTP_403_FORBIDDEN
        and isinstance(body, dict)
        and body.get("error", {}).get("code") == "admin_required"
        and _wants_html(request)
    ):
        user_row: User | None = None
        try:
            async with request.app.state.sessions() as db:
                user_row = await webauth.current_user(request, db)
        except Exception:  # noqa: BLE001 — exception handler must not raise
            user_row = None
        return templates.TemplateResponse(
            request,
            "admin_required.html",
            {"user": user_row},
            status_code=exc.status_code,
            headers=headers,
        )
    if isinstance(body, dict) and "error" in body:
        return JSONResponse(status_code=exc.status_code, content=body, headers=headers)
    return JSONResponse(status_code=exc.status_code, content={"detail": body}, headers=headers)


@app.exception_handler(RateLimitExceeded)
async def _ratelimit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        headers={"Retry-After": "60"},
        content={
            "error": {
                "message": f"Rate limit exceeded: {exc.detail}",
                "type": "rate_limit_error",
                "code": "rate_limited",
            }
        },
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions (ACS-40).

    Without this, an unhandled error returns Starlette's default 500 — no
    structured body, no `error_kind`, and invisible to our tooling. A registered
    ``Exception`` handler is treated as "handled" by Starlette's
    ServerErrorMiddleware, so Sentry's integration won't auto-capture it — we
    capture explicitly. ``sentry_sdk.capture_exception`` is a no-op when Sentry
    isn't initialized (no DSN), so this is safe in dev/tests.

    The response body is intentionally generic — the exception message / traceback
    never reaches the client (could leak internals or prompt fragments). The
    ``request_id`` ties the 500 to the Railway log line and the Sentry event for
    triage. Reads ``request.state.request_id`` if the request-id middleware set
    one (ACS-41), else mints a fallback so this works before that lands.
    """
    request_id = (
        getattr(getattr(request, "state", None), "request_id", None)
        or f"req_{uuid.uuid4().hex[:20]}"
    )
    sentry_sdk.capture_exception(exc)
    # Log type + path + request_id only — no exc message (may contain internals).
    log.error(
        "unhandled_exception",
        request_id=request_id,
        exc_type=type(exc).__name__,
        # Read from the scope rather than building request.url — this handler
        # runs *while already handling an error*; a partial scope (no "path")
        # would otherwise raise KeyError here and turn a logged 500 into an
        # unhandled crash. Mirrors auth._raise_auth_error (ACS-113).
        path=request.scope.get("path", "unknown"),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error.",
                "type": "internal_error",
                "code": "internal_error",
            },
            "request_id": request_id,
        },
    )


# --- API routes are registered from wrapper.routes.api ----------------------

# --- admin routes are registered from wrapper.routes.admin -------------------

# --- web routes are registered from wrapper.routes.web ----------------------

# --- workbench routes are registered from wrapper.routes.workbench ----------

# --- usage routes are registered from wrapper.routes.usage -------------------
